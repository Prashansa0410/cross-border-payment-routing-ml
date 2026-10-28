"""Model loading and inference module for the payment routing ML system."""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np

from config import ARTIFACTS_DIR, FEATURES
from feature_pipeline import FeaturePipeline
from logger import setup_logger

logger = setup_logger("inference")


class PaymentPredictor:
    """Thread-safe prediction service that loads and caches ML models in memory."""

    _instance: Optional["PaymentPredictor"] = None
    _lock = threading.Lock()

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.clf = None
        self.reg = None
        self.threshold = 0.5
        self.metrics = {}
        self.route_catalog = []
        self.is_loaded = False
        self._load_artifacts()

    @classmethod
    def get_instance(cls, artifacts_dir: Optional[Union[str, Path]] = None) -> "PaymentPredictor":
        """Singleton accessor for thread-safe model caching."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(artifacts_dir)
            return cls._instance

    def _load_artifacts(self) -> None:
        """Loads model artifacts from the artifacts directory."""
        clf_path = self.artifacts_dir / "success_classifier.joblib"
        reg_path = self.artifacts_dir / "settlement_regressor.joblib"
        thresh_path = self.artifacts_dir / "threshold.json"
        metrics_path = self.artifacts_dir / "metrics.json"
        routes_path = self.artifacts_dir / "route_catalog.json"

        logger.info("Loading model artifacts from: %s", self.artifacts_dir)

        if not clf_path.exists() or not reg_path.exists():
            logger.error(
                "Model artifacts missing in %s. Run model_trainer first.", self.artifacts_dir
            )
            self.is_loaded = False
            return

        try:
            self.clf = joblib.load(clf_path)
            self.reg = joblib.load(reg_path)

            if thresh_path.exists():
                with open(thresh_path) as f:
                    self.threshold = float(json.load(f).get("decision_threshold", 0.5))
            else:
                self.threshold = 0.5

            if metrics_path.exists():
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {"model_version": "unknown"}

            if routes_path.exists():
                with open(routes_path) as f:
                    self.route_catalog = json.load(f)
            else:
                self.route_catalog = []

            self.is_loaded = True
            logger.info(
                "Successfully loaded models: version=%s, threshold=%.3f, routes=%d",
                self.metrics.get("model_version"),
                self.threshold,
                len(self.route_catalog),
            )
        except Exception as e:
            logger.error("Failed to load model artifacts: %s", str(e), exc_info=True)
            self.is_loaded = False
            raise

    def predict(
        self, payment: Dict[str, Any], routes: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Performs batch inference across candidate routes for a given payment transaction.

        Args:
            payment: Dictionary of payment transaction features.
            routes: Optional list of candidate route dictionaries. If None, uses loaded route catalog.

        Returns:
            Dictionary containing predictions, model_version, and inference_ms.
        """
        if not self.is_loaded or self.clf is None or self.reg is None:
            logger.error("Inference attempted but models are not loaded.")
            raise RuntimeError("ML models are not loaded or unavailable.")

        candidate_routes = self.route_catalog if routes is None else routes
        if not candidate_routes:
            logger.error("No candidate routes provided or configured.")
            raise ValueError("Candidate routes list cannot be empty.")

        t0 = time.perf_counter()
        try:
            X = FeaturePipeline.prepare_inference_frame(payment, candidate_routes)
        except ValueError as e:
            logger.warning("Input validation error during inference preparation: %s", str(e))
            raise

        logger.debug("Executing model inference for %d candidate routes", len(candidate_routes))
        p_success = self.clf.predict_proba(X)[:, 1]
        t_settle = np.expm1(self.reg.predict(X))

        # Ensure realistic domain bounds
        p_success = np.clip(p_success, 0.0, 1.0)
        t_settle = np.clip(t_settle, 0.0, None)

        predictions = []
        for r, p, t in zip(candidate_routes, p_success, t_settle):
            predictions.append(
                {
                    "route_id": r["route_id"],
                    "sending_institution": r["sending_institution"],
                    "receiving_institution": r["receiving_institution"],
                    "route_cost": float(r["route_cost"]),
                    "predicted_success_probability": float(p),
                    "predicted_settlement_seconds": float(t),
                    "predicted_class": int(p >= self.threshold),
                }
            )

        inference_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Inference successful: %d routes evaluated in %.2f ms (version=%s)",
            len(predictions),
            inference_ms,
            self.metrics.get("model_version", "unknown"),
        )

        return {
            "predictions": predictions,
            "model_version": self.metrics.get("model_version", "unknown"),
            "inference_ms": inference_ms,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Returns safe model metadata without exposing file paths or internal secrets."""
        if not self.is_loaded:
            return {"status": "unloaded", "model_version": None}

        clf_metrics = self.metrics.get("classification", {}).get("gbdt_classifier", {})
        reg_metrics = self.metrics.get("regression_success_only", {}).get("gbdt_regressor", {})

        return {
            "model_version": self.metrics.get("model_version", "unknown"),
            "model_type": "HistGradientBoosting (Classifier & Regressor)",
            "decision_threshold": self.threshold,
            "classifier_metrics": {
                "roc_auc": clf_metrics.get("roc_auc"),
                "f1": clf_metrics.get("f1"),
                "brier": clf_metrics.get("brier"),
                "expected_calibration_error": self.metrics.get("expected_calibration_error"),
            },
            "regressor_metrics": {
                "r2": reg_metrics.get("r2"),
                "mae_seconds": reg_metrics.get("mae_seconds"),
                "mae_minutes": reg_metrics.get("mae_minutes"),
            },
            "supported_features": FEATURES,
            "candidate_routes_count": len(self.route_catalog),
        }
