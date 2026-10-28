"""API endpoint and contract tests using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_endpoint(client):
    """Verifies GET /health endpoint returns 200 and indicates model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["service"] == "payment-routing-api"
    assert "timestamp" in data


def test_model_info_endpoint(client):
    """Verifies GET /model-info returns metadata and quality metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "decision_threshold" in data
    assert "classifier_metrics" in data
    assert "regressor_metrics" in data
    assert "supported_features" in data
    assert data["candidate_routes_count"] >= 6


def test_predict_endpoint_valid_request(client, valid_payment):
    """Verifies POST /predict with default catalog routes."""
    payload = {"payment": valid_payment}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) >= 6
    assert "model_version" in data
    assert data["inference_ms"] > 0

    first_pred = data["predictions"][0]
    assert 0.0 <= first_pred["predicted_success_probability"] <= 1.0
    assert first_pred["predicted_settlement_seconds"] >= 0.0
    assert first_pred["predicted_class"] in (0, 1)


def test_predict_endpoint_custom_routes(client, valid_payment, sample_routes):
    """Verifies POST /predict with custom candidate routes."""
    payload = {"payment": valid_payment, "routes": sample_routes}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == len(sample_routes)


def test_recommend_route_endpoint(client, valid_payment):
    """Verifies POST /recommend-route with default optimisation mode."""
    payload = {
        "payment": valid_payment,
        "mode": "optimisation",
        "weights": {"success": 0.5, "latency": 0.3, "cost": 0.2},
    }
    response = client.post("/recommend-route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_route" in data
    assert "recommendation" in data
    assert "ranked_routes" in data
    assert "explanation" in data
    assert data["constraints_relaxed"] is False


def test_recommend_route_with_constraints(client, valid_payment):
    """Verifies POST /recommend-route applying hard business constraints."""
    payload = {
        "payment": valid_payment,
        "mode": "optimisation",
        "constraints": {
            "max_cost": 2.0,
            "min_success_probability": 0.85,
            "excluded_routes": ["ROUTE_A"],
        },
    }
    response = client.post("/recommend-route", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_route"] != "ROUTE_A"
    assert data["recommendation"]["route_cost"] <= 2.0


def test_api_validation_errors(client):
    """Verifies that invalid payloads return descriptive HTTP 422 errors."""
    # 1. Missing required fields
    res1 = client.post("/predict", json={"payment": {"amount": 100.0}})
    assert res1.status_code == 422
    assert "error" in res1.json()

    # 2. Negative amount
    res2 = client.post(
        "/predict",
        json={
            "payment": {
                "source_country": "US",
                "destination_country": "India",
                "source_currency": "USD",
                "destination_currency": "INR",
                "amount": -50.0,
                "payment_type": "P2P",
            }
        },
    )
    assert res2.status_code == 422

    # 3. Invalid payment type category
    res3 = client.post(
        "/predict",
        json={
            "payment": {
                "source_country": "US",
                "destination_country": "India",
                "source_currency": "USD",
                "destination_currency": "INR",
                "amount": 100.0,
                "payment_type": "CRYPTO",  # Not in literal enum
            }
        },
    )
    assert res3.status_code == 422


def test_openapi_schema_availability(client):
    """Verifies that OpenAPI documentation endpoints are generated and reachable."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    schema_resp = client.get("/openapi.json")
    assert schema_resp.status_code == 200
    schema = schema_resp.json()
    assert "/predict" in schema["paths"]
    assert "/recommend-route" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/model-info" in schema["paths"]
