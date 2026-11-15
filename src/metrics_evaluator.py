"""metrics_evaluator.py - Model Quality and Data Quality metrics module.
--------------------------------------------------------------------------
Production Model & Data Quality Evaluation Suite:
  * Model Quality: ROC-AUC, F1, Brier Score, Expected Calibration Error (ECE), MAE, R²
  * Data Quality: Schema validity rate, missing value rate, duplicate rate, drift detection (PSI)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from config import NUM_FEATURES
from logger import setup_logger

logger = setup_logger("metrics_evaluator")


def compute_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) to detect data distribution drift.

    Interpretation:
      PSI < 0.1  : No significant distribution change (stable).
      0.1 <= PSI < 0.2: Moderate drift detected (investigate).
      PSI >= 0.2 : Significant data drift (retraining / alerting required).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins from baseline expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5

    # If duplicate bin edges exist (due to discrete/low-variance features)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    # Convert to proportions with smoothing to avoid division by zero
    exp_pct = np.maximum(exp_counts / len(expected), 1e-4)
    act_pct = np.maximum(act_counts / len(actual), 1e-4)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


class DataQualityEvaluator:
    """Calculates quantitative data quality metrics and tests for covariate drift."""

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        self.baseline_df = baseline_df

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Measures data quality indicators across rows, columns, nulls, and distributions."""
        n_rows = len(df)
        if n_rows == 0:
            return {"error": "Empty dataframe provided"}

        # 1. Missing Value Rate
        total_cells = n_rows * df.shape[1]
        missing_cells = int(df.isna().sum().sum())
        missing_rate = missing_cells / total_cells

        # 2. Duplicate Rate
        dup_count = int(df["transaction_id"].duplicated().sum()) if "transaction_id" in df else 0
        dup_rate = dup_count / n_rows

        # 3. Numeric Range Violations
        range_violations = 0
        if "amount" in df:
            range_violations += int((df["amount"] <= 0).sum())
        if "historical_route_success_rate" in df:
            sr = df["historical_route_success_rate"].dropna()
            range_violations += int((~sr.between(0, 1)).sum())
        range_violation_rate = range_violations / max(n_rows, 1)

        # 4. Drift metrics (PSI) compared to baseline
        drift_scores = {}
        if self.baseline_df is not None:
            for col in NUM_FEATURES:
                if col in df and col in self.baseline_df:
                    psi = compute_psi(self.baseline_df[col].values, df[col].values)
                    drift_scores[col] = {
                        "psi": round(psi, 4),
                        "status": (
                            "STABLE"
                            if psi < 0.1
                            else ("MODERATE_DRIFT" if psi < 0.2 else "SIGNIFICANT_DRIFT")
                        ),
                    }

        return {
            "rows_evaluated": n_rows,
            "missing_value_rate": missing_rate,
            "duplicate_record_rate": dup_rate,
            "range_violation_rate": range_violation_rate,
            "drift_assessment": drift_scores,
        }


class ModelQualityEvaluator:
    """Calculates model quality metrics on predictions versus true labels."""

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculates classification quality metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
        ece = float(np.mean(np.abs(prob_true - prob_pred)))

        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "f1": float(f1_score(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "expected_calibration_error": ece,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred)),
        }

    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates regression quality metrics."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mae = float(mean_absolute_error(y_true, y_pred))
        return {
            "mae_seconds": mae,
            "mae_minutes": mae / 60.0,
            "rmse_seconds": float(root_mean_squared_error(y_true, y_pred)),
            "r2_score": float(r2_score(y_true, y_pred)),
        }
