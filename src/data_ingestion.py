"""Data ingestion and validation module for the payment routing ML system."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CAT_FEATURES,
    DATA_PATH,
    DROPPED_REDUNDANT,
    ID_COLS,
    NUM_FEATURES,
    TARGET_CLF,
    TARGET_REG,
)
from logger import setup_logger

logger = setup_logger("data_ingestion")


@dataclass
class DataQualityReport:
    rows: int
    columns: int
    schema_ok: bool
    missing_per_column: Dict[str, int]
    missing_targets: int
    duplicate_transaction_ids: int
    target_balance: Dict[str, int]
    temporal_consistency_ok: bool
    amount_range: List[float]
    success_rate_range_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


class DataIngestion:
    """Handles raw data loading and persistence from local or external sources."""

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else DATA_PATH

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from CSV file with error handling and logging."""
        if not self.data_path.exists():
            logger.error("Dataset not found at expected path: %s", self.data_path)
            raise FileNotFoundError(f"Data file does not exist: {self.data_path}")

        logger.info("Ingesting raw dataset from: %s", self.data_path)
        try:
            df = pd.read_csv(self.data_path)
            logger.info("Successfully ingested %d records with %d columns", len(df), df.shape[1])
            return df
        except Exception as e:
            logger.error("Failed to read dataset from %s: %s", self.data_path, str(e))
            raise


class DataValidator:
    """Validates schema, null values, constraints, and distributions."""

    def __init__(self):
        self.expected_cols = set(
            ID_COLS + CAT_FEATURES + NUM_FEATURES + DROPPED_REDUNDANT + [TARGET_CLF, TARGET_REG]
        )

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """Executes full data quality checks and returns a DataQualityReport."""
        logger.info("Starting data quality validation on dataframe with shape %s", df.shape)

        current_cols = set(df.columns)
        schema_ok = self.expected_cols == current_cols
        if not schema_ok:
            missing_cols = self.expected_cols - current_cols
            unexpected_cols = current_cols - self.expected_cols
            logger.warning(
                "Schema validation discrepancy. Missing: %s, Unexpected: %s",
                missing_cols,
                unexpected_cols,
            )

        missing_per_col = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}
        if missing_per_col:
            logger.warning("Missing values detected in columns: %s", missing_per_col)
        else:
            logger.info("Zero missing values found across all columns.")

        target_cols = [c for c in [TARGET_CLF, TARGET_REG] if c in df.columns]
        missing_targets = int(df[target_cols].isna().sum().sum()) if target_cols else 0
        if missing_targets > 0 or len(target_cols) < 2:
            logger.warning(
                "Target column issue detected. Missing values: %d, present target cols: %s",
                missing_targets,
                target_cols,
            )

        dup_ids = (
            int(df["transaction_id"].duplicated().sum()) if "transaction_id" in df.columns else 0
        )
        if dup_ids > 0:
            logger.warning("%d duplicate transaction_ids found in dataset", dup_ids)

        target_balance = (
            {str(k): int(v) for k, v in df[TARGET_CLF].value_counts().items()}
            if TARGET_CLF in df.columns
            else {}
        )

        # Temporal consistency validation
        temporal_ok = False
        if "timestamp" in df.columns and {"hour_of_day", "day_of_week", "is_weekend"}.issubset(
            df.columns
        ):
            ts = pd.to_datetime(df["timestamp"])
            temporal_ok = bool(
                (ts.dt.hour == df["hour_of_day"]).all()
                and (ts.dt.dayofweek == df["day_of_week"]).all()
                and ((ts.dt.dayofweek >= 5).astype(int) == df["is_weekend"]).all()
            )
            if not temporal_ok:
                logger.warning(
                    "Temporal inconsistency found between timestamp and derived features."
                )

        amount_min = float(df["amount"].min()) if "amount" in df.columns else 0.0
        amount_max = float(df["amount"].max()) if "amount" in df.columns else 0.0
        if amount_min <= 0:
            logger.warning("Unusual amount detected: min amount is %f", amount_min)

        sr_ok = (
            bool(df["historical_route_success_rate"].dropna().between(0, 1).all())
            if "historical_route_success_rate" in df.columns
            else True
        )

        report = DataQualityReport(
            rows=int(len(df)),
            columns=int(df.shape[1]),
            schema_ok=schema_ok,
            missing_per_column=missing_per_col,
            missing_targets=missing_targets,
            duplicate_transaction_ids=dup_ids,
            target_balance=target_balance,
            temporal_consistency_ok=temporal_ok,
            amount_range=[amount_min, amount_max],
            success_rate_range_ok=sr_ok,
        )

        logger.info("Validation completed. Schema OK: %s, Rows: %d", schema_ok, len(df))
        return report

    def create_splits(
        self,
        df: pd.DataFrame,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Creates stratified train, validation, and test splits on payment_success."""
        logger.info(
            "Splitting %d records with ratio %.2f/%.2f/%.2f",
            len(df),
            train_size,
            val_size,
            test_size,
        )
        train_df, rest_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            stratify=df[TARGET_CLF],
            random_state=random_state,
        )
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            rest_df,
            test_size=(1.0 - val_ratio),
            stratify=rest_df[TARGET_CLF],
            random_state=random_state,
        )
        logger.info(
            "Splits generated -> Train: %d, Validation: %d, Test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return train_df, val_df, test_df
