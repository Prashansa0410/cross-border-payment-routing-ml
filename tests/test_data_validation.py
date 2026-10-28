"""Unit and data validation tests."""

import pandas as pd

from config import DATA_PATH, TARGET_CLF
from data_ingestion import DataIngestion, DataValidator
from metrics_evaluator import DataQualityEvaluator


def test_data_ingestion_and_schema():
    """Validates raw data ingestion and expected schema compliance."""
    ingestor = DataIngestion(DATA_PATH)
    df = ingestor.load_data()
    assert not df.empty
    assert len(df) == 50000

    validator = DataValidator()
    report = validator.validate(df)
    assert report.schema_ok is True
    assert report.missing_targets == 0
    assert report.temporal_consistency_ok is True
    assert report.amount_range[0] > 0
    assert report.success_rate_range_ok is True


def test_missing_and_corrupt_data_detection():
    """Verifies that missing and corrupted values are properly caught by the validator."""
    df_corrupted = pd.DataFrame(
        {
            "transaction_id": ["TXN_1", "TXN_1"],  # duplicate
            "amount": [-100.0, 50.0],  # negative amount
            TARGET_CLF: [1, None],  # missing target
        }
    )
    validator = DataValidator()
    report = validator.validate(df_corrupted)
    assert report.schema_ok is False
    assert report.duplicate_transaction_ids == 1
    assert report.missing_targets == 1
    assert report.amount_range[0] == -100.0


def test_stratified_splits():
    """Verifies that train/val/test splits maintain class balance across targets."""
    ingestor = DataIngestion(DATA_PATH)
    df = ingestor.load_data()
    validator = DataValidator()
    train_df, val_df, test_df = validator.create_splits(df)

    assert len(train_df) == 35000
    assert len(val_df) == 7500
    assert len(test_df) == 7500

    # Success rate should be within 1% across splits
    base_sr = df[TARGET_CLF].mean()
    assert abs(train_df[TARGET_CLF].mean() - base_sr) < 0.01
    assert abs(val_df[TARGET_CLF].mean() - base_sr) < 0.01
    assert abs(test_df[TARGET_CLF].mean() - base_sr) < 0.01


def test_data_quality_evaluator_and_drift():
    """Verifies DataQualityEvaluator and PSI drift computation."""
    ingestor = DataIngestion(DATA_PATH)
    df = ingestor.load_data()
    train_df, _, test_df = DataValidator().create_splits(df)

    evaluator = DataQualityEvaluator(baseline_df=train_df)
    report = evaluator.evaluate(test_df)

    assert report["missing_value_rate"] < 0.01
    assert report["duplicate_record_rate"] == 0.0
    assert report["range_violation_rate"] == 0.0
    assert "amount" in report["drift_assessment"]
    assert report["drift_assessment"]["amount"]["status"] == "STABLE"

    # Test drift trigger on drifted feature
    drifted_df = test_df.copy()
    drifted_df["amount"] = drifted_df["amount"] * 100.0
    drift_report = evaluator.evaluate(drifted_df)
    assert drift_report["drift_assessment"]["amount"]["status"] == "SIGNIFICANT_DRIFT"
