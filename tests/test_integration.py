"""Integration tests for the complete payment routing system."""

from config import DATA_PATH
from data_ingestion import DataIngestion, DataValidator
from decision_engine import Constraints, recommend
from feature_pipeline import TemporalEnricher
from inference import PaymentPredictor


def test_end_to_end_pipeline_integration():
    """Validates the complete flow:

    1. Data Ingestion & Validation
    2. Model Prediction on held-out test transaction
    3. Multi-objective corridor recommendation
    """
    # 1. Ingestion & Validation
    ingestor = DataIngestion(DATA_PATH)
    df = ingestor.load_data()
    validator = DataValidator()
    report = validator.validate(df)
    assert report.schema_ok is True

    # 2. Extract a realistic held-out test sample
    _, _, test_df = validator.create_splits(df)
    sample_row = test_df.iloc[0].to_dict()

    # 3. Enrich payment transaction
    payment_payload = {
        "transaction_id": sample_row["transaction_id"],
        "timestamp": sample_row["timestamp"],
        "source_country": sample_row["source_country"],
        "destination_country": sample_row["destination_country"],
        "source_currency": sample_row["source_currency"],
        "destination_currency": sample_row["destination_currency"],
        "amount": float(sample_row["amount"]),
        "payment_type": sample_row["payment_type"],
    }
    enriched = TemporalEnricher.enrich(payment_payload, payment_payload["timestamp"])

    # 4. Model Inference
    predictor = PaymentPredictor.get_instance()
    pred_res = predictor.predict(enriched)

    assert len(pred_res["predictions"]) >= 6
    assert pred_res["inference_ms"] > 0

    # 5. Route Recommendation
    cands_for_decision = [
        {
            "route_id": p["route_id"],
            "sending_institution": p["sending_institution"],
            "receiving_institution": p["receiving_institution"],
            "route_cost": p["route_cost"],
            "predicted_success_probability": p["predicted_success_probability"],
            "predicted_settlement_seconds": p["predicted_settlement_seconds"],
        }
        for p in pred_res["predictions"]
    ]

    rec_result = recommend(
        candidates=cands_for_decision,
        mode="optimisation",
        weights={"success": 0.6, "latency": 0.25, "cost": 0.15},
        constraints=Constraints(max_cost=4.0),
    )

    assert rec_result["recommended_route"] in [r["route_id"] for r in cands_for_decision]
    assert rec_result["recommendation"]["route_cost"] <= 4.0
    assert "explanation" in rec_result
