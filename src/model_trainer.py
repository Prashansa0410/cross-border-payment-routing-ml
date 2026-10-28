"""Model training and evaluation pipeline for the payment routing ML system."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from config import ARTIFACTS_DIR, FIGURES_DIR
from data_ingestion import DataIngestion, DataValidator
from feature_pipeline import FeaturePipeline
from logger import setup_logger

logger = setup_logger("model_trainer")


class ModelEvaluator:
    """Evaluates classification and regression metrics for model quality assessment."""

    @staticmethod
    def evaluate_classifier(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculates comprehensive classification performance metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        fail_recall = float(((y_pred == 0) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))

        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "brier": float(brier_score_loss(y_true, y_proba)),
            "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-6, 1.0 - 1e-6))),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "failure_recall": fail_recall,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }

    @staticmethod
    def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates regression performance metrics on completed payments."""
        mae_sec = float(mean_absolute_error(y_true, y_pred))
        return {
            "mae_seconds": mae_sec,
            "rmse_seconds": float(root_mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mae_minutes": mae_sec / 60.0,
        }

    @staticmethod
    def calculate_calibration_error(
        y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculates Expected Calibration Error (ECE) via quantile binning."""
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="quantile"
        )
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        return ece


class ModelTrainer:
    """Encapsulates training, threshold tuning, and artifact serialization."""

    def __init__(
        self,
        artifacts_dir: Optional[Union[str, Path]] = None,
        figures_dir: Optional[Union[str, Path]] = None,
    ):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self.figures_dir = Path(figures_dir) if figures_dir else FIGURES_DIR
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def train_classifier(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Builds and trains the HistGradientBoostingClassifier pipeline."""
        logger.info("Training payment success classifier on %d samples", len(X_train))
        clf = Pipeline(
            [
                ("prep", FeaturePipeline.build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.03,
                        max_iter=500,
                        max_leaf_nodes=7,
                        min_samples_leaf=200,
                        l2_regularization=1.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42,
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)
        return clf

    def train_regressor(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Builds and trains the HistGradientBoostingRegressor on log1p settlement time."""
        logger.info("Training settlement time regressor on %d completed transactions", len(X_train))
        reg = Pipeline(
            [
                ("prep", FeaturePipeline.build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=600,
                        max_leaf_nodes=31,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=42,
                    ),
                ),
            ]
        )
        reg.fit(X_train, np.log1p(y_train))
        return reg

    def tune_threshold(self, clf: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Tunes decision threshold on validation set to maximize balanced accuracy."""
        proba_val = clf.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.60, 0.98, 0.005)
        scores = [balanced_accuracy_score(y_val, (proba_val >= t).astype(int)) for t in thresholds]
        best_threshold = float(thresholds[int(np.argmax(scores))])
        logger.info("Optimal decision threshold tuned on validation split: %.3f", best_threshold)
        return best_threshold

    def train_and_evaluate(self, data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Full pipeline: Ingest -> Validate -> Split -> Train -> Tune -> Evaluate -> Persist."""
        t0 = time.time()
        ingestor = DataIngestion(data_path)
        df = ingestor.load_data()

        validator = DataValidator()
        quality_report = validator.validate(df)

        train_df, val_df, test_df = validator.create_splits(df)

        X_tr, y_tr_c, y_tr_r = FeaturePipeline.split_features_targets(train_df)
        X_va, y_va_c, y_va_r = FeaturePipeline.split_features_targets(val_df)
        X_te, y_te_c, y_te_r = FeaturePipeline.split_features_targets(test_df)

        # 1. Classification
        t_clf = time.time()
        clf = self.train_classifier(X_tr, y_tr_c)
        clf_train_time = time.time() - t_clf

        threshold = self.tune_threshold(clf, X_va, y_va_c)

        # Baselines
        base_clf = Pipeline(
            [
                ("prep", FeaturePipeline.build_preprocessor()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ).fit(X_tr, y_tr_c)
        majority_clf = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr_c)

        # Evaluate classifiers
        res_clf = {}
        for name, model in [
            ("gbdt_classifier", clf),
            ("logistic_baseline", base_clf),
            ("majority_baseline", majority_clf),
        ]:
            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= threshold).astype(int)
            res_clf[name] = ModelEvaluator.evaluate_classifier(y_te_c, pred, proba)

        # Heuristic baseline
        hist_sr = X_te["historical_route_success_rate"].fillna(
            X_tr["historical_route_success_rate"].median()
        )
        res_clf["historical_success_rate_baseline"] = {
            "roc_auc": float(roc_auc_score(y_te_c, hist_sr)),
            "brier": float(brier_score_loss(y_te_c, hist_sr)),
        }

        proba_te = clf.predict_proba(X_te)[:, 1]
        cm = confusion_matrix(y_te_c, (proba_te >= threshold).astype(int))
        ece = ModelEvaluator.calculate_calibration_error(y_te_c.values, proba_te)

        # 2. Regression (Completed transactions only)
        ok_tr = y_tr_c == 1
        ok_te = y_te_c == 1
        t_reg = time.time()
        reg = self.train_regressor(X_tr[ok_tr], y_tr_r[ok_tr])
        reg_train_time = time.time() - t_reg

        pred_te = np.expm1(reg.predict(X_te[ok_te]))
        base_reg = DummyRegressor(strategy="median").fit(X_tr[ok_tr], y_tr_r[ok_tr])
        hist_lat = X_te.loc[ok_te, "historical_avg_latency_seconds"].fillna(
            X_tr["historical_avg_latency_seconds"].median()
        )

        res_reg = {
            "gbdt_regressor": ModelEvaluator.evaluate_regressor(y_te_r[ok_te], pred_te),
            "historical_latency_baseline": ModelEvaluator.evaluate_regressor(
                y_te_r[ok_te], hist_lat
            ),
            "median_baseline": ModelEvaluator.evaluate_regressor(
                y_te_r[ok_te], base_reg.predict(X_te[ok_te])
            ),
        }

        # 3. Save Artifacts
        joblib.dump(clf, self.artifacts_dir / "success_classifier.joblib")
        joblib.dump(reg, self.artifacts_dir / "settlement_regressor.joblib")
        with open(self.artifacts_dir / "threshold.json", "w") as f:
            json.dump({"decision_threshold": threshold}, f, indent=2)

        route_catalog = (
            df[
                [
                    "route_id",
                    "sending_institution",
                    "receiving_institution",
                    "route_cost",
                    "historical_route_success_rate",
                    "historical_avg_latency_seconds",
                ]
            ]
            .dropna()
            .groupby("route_id")
            .first()
            .reset_index()
        )
        route_catalog.to_json(self.artifacts_dir / "route_catalog.json", orient="records", indent=2)

        metrics = {
            "data_quality": quality_report.to_dict(),
            "split_sizes": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "classification": res_clf,
            "confusion_matrix": cm.tolist(),
            "decision_threshold": threshold,
            "expected_calibration_error": ece,
            "regression_success_only": res_reg,
            "training_time_seconds": {
                "classifier": clf_train_time,
                "regressor": reg_train_time,
            },
            "model_version": time.strftime("v%Y%m%d-%H%M%S"),
            "total_pipeline_seconds": time.time() - t0,
        }

        with open(self.artifacts_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=float)

        logger.info(
            "Training pipeline completed successfully in %.2f seconds",
            metrics["total_pipeline_seconds"],
        )
        return metrics


if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate()
    print("Training finished. Model version:", results["model_version"])
