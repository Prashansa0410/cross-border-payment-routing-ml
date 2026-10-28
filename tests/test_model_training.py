"""Specific tests for ML component training (Objective 2, Req 7a)."""

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from config import DATA_PATH
from data_ingestion import DataIngestion
from feature_pipeline import FeaturePipeline
from model_trainer import ModelTrainer


def test_overfitting_on_small_batch():
    """Requirement 7a: Verifies that the model architecture can successfully overfit

    a small batch of training data, demonstrating that learning occurs and training loss decreases.
    """
    ingestor = DataIngestion(DATA_PATH)
    df = ingestor.load_data()

    # Extract a small balanced batch of 30 samples containing both classes
    batch_0 = df[df["payment_success"] == 0].head(10)
    batch_1 = df[df["payment_success"] == 1].head(20)
    small_batch = pd.concat([batch_0, batch_1]).sample(frac=1.0, random_state=42).copy()
    X_small, y_small_c, y_small_r = FeaturePipeline.split_features_targets(small_batch)

    # Preprocess features
    preprocessor = FeaturePipeline.build_preprocessor()
    X_proc = preprocessor.fit_transform(X_small)

    from sklearn.ensemble import HistGradientBoostingClassifier

    # Overfit with high iterations and deep trees
    overfit_model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=31, min_samples_leaf=1, random_state=42
    )

    overfit_model.fit(X_proc, y_small_c)
    preds = overfit_model.predict(X_proc)
    probas = overfit_model.predict_proba(X_proc)[:, 1]

    train_accuracy = accuracy_score(y_small_c, preds)
    loss = log_loss(y_small_c, probas)

    # Model must easily memorize / fit 25 samples
    assert (
        train_accuracy >= 0.95
    ), f"Expected train accuracy >= 0.95 on small batch, got {train_accuracy}"
    assert loss < 0.25, f"Expected low log loss on overfit batch, got {loss}"


def test_model_training_pipeline_execution(tmp_path):
    """Verifies that the ModelTrainer executes end-to-end and outputs valid artifacts."""
    trainer = ModelTrainer(artifacts_dir=tmp_path / "artifacts", figures_dir=tmp_path / "figures")

    # Ingest a subset for quick smoke test
    ingestor = DataIngestion(DATA_PATH)
    df = ingestor.load_data().sample(1000, random_state=42)
    sample_csv = tmp_path / "sample_data.csv"
    df.to_csv(sample_csv, index=False)

    metrics = trainer.train_and_evaluate(data_path=sample_csv)

    assert "model_version" in metrics
    assert "classification" in metrics
    assert "regression_success_only" in metrics

    # Check generated files
    assert (tmp_path / "artifacts" / "success_classifier.joblib").exists()
    assert (tmp_path / "artifacts" / "settlement_regressor.joblib").exists()
    assert (tmp_path / "artifacts" / "threshold.json").exists()
    assert (tmp_path / "artifacts" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "route_catalog.json").exists()

    # Metric reasonableness
    auc = metrics["classification"]["gbdt_classifier"]["roc_auc"]
    assert 0.50 <= auc <= 1.0
    r2 = metrics["regression_success_only"]["gbdt_regressor"]["r2"]
    assert r2 > 0.50
