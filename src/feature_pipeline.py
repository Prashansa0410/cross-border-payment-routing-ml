"""Feature engineering and preprocessing pipeline for the payment routing ML system."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CAT_FEATURES,
    FEATURES,
    NUM_FEATURES,
    REQUIRED_PAYMENT_FIELDS,
    TARGET_CLF,
    TARGET_REG,
)
from logger import setup_logger

logger = setup_logger("feature_pipeline")


class TemporalEnricher:
    """Enriches transaction timestamps into temporal features."""

    @staticmethod
    def enrich(payment: Dict[str, Any], timestamp_str: Optional[str] = None) -> Dict[str, Any]:
        """Extracts hour_of_day, day_of_week, and is_weekend from ISO timestamp."""
        try:
            if timestamp_str:
                ts = datetime.fromisoformat(timestamp_str)
            else:
                ts = datetime.now()
        except Exception as e:
            logger.warning(
                "Failed to parse timestamp '%s', defaulting to now. Error: %s",
                timestamp_str,
                str(e),
            )
            ts = datetime.now()

        enriched = dict(payment)
        enriched["hour_of_day"] = ts.hour
        enriched["day_of_week"] = ts.weekday()
        enriched["is_weekend"] = int(ts.weekday() >= 5)
        return enriched


class FeaturePipeline:
    """Builds and applies scikit-learn transformers for numeric and categorical features."""

    @staticmethod
    def build_preprocessor() -> ColumnTransformer:
        """Constructs the standard ColumnTransformer for numeric and categorical features."""
        logger.debug("Building ColumnTransformer preprocessor.")
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, NUM_FEATURES),
                ("cat", cat_pipe, CAT_FEATURES),
            ]
        )

    @staticmethod
    def split_features_targets(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Separates the feature matrix X, classification target y_c, and regression target y_r."""
        X = df[FEATURES].copy()
        y_c = df[TARGET_CLF].astype(int)
        y_r = df[TARGET_REG].astype(float)
        return X, y_c, y_r

    @staticmethod
    def prepare_inference_frame(
        payment: Dict[str, Any], routes: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Combines a single payment dictionary with a list of candidate routes into a DataFrame."""
        missing = [f for f in REQUIRED_PAYMENT_FIELDS if f not in payment]
        if missing:
            logger.error("Missing required payment fields for inference: %s", missing)
            raise ValueError(f"Missing required payment fields: {missing}")

        if not routes:
            logger.error("Candidate routes list is empty.")
            raise ValueError("Candidate routes list cannot be empty.")

        route_keys = [
            "route_id",
            "sending_institution",
            "receiving_institution",
            "route_cost",
            "historical_route_success_rate",
            "historical_avg_latency_seconds",
        ]

        rows = []
        for r in routes:
            row = dict(payment)
            for k in route_keys:
                if k not in r:
                    logger.error("Candidate route missing field '%s': %s", k, r)
                    raise ValueError(f"Route missing required field: {k}")
                row[k] = r[k]
            rows.append(row)

        df = pd.DataFrame(rows)[FEATURES]
        return df
