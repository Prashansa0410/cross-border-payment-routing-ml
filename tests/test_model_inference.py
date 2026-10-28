"""Specific tests for ML component inference (Objective 2, Req 7b)."""

import copy

import pytest


def test_inference_output_shape_and_range(predictor, valid_payment, sample_routes):
    """Requirement 7b: Verifies output structure, probability bounds [0, 1], and non-negative settlement times."""
    result = predictor.predict(valid_payment, sample_routes)

    assert "predictions" in result
    assert "model_version" in result
    assert "inference_ms" in result
    assert len(result["predictions"]) == len(sample_routes)

    for pred in result["predictions"]:
        # Probability bounds
        assert 0.0 <= pred["predicted_success_probability"] <= 1.0
        # Positive settlement time
        assert pred["predicted_settlement_seconds"] >= 0.0
        # Binary class
        assert pred["predicted_class"] in (0, 1)
        # Required keys
        assert "route_id" in pred
        assert "route_cost" in pred


def test_directional_test_historical_success_rate(predictor, valid_payment):
    """Requirement 7b: Directional Test.

    All else equal, a route with a significantly higher historical success rate
    must produce a higher or equal predicted success probability.
    """
    route_low = {
        "route_id": "ROUTE_LOW",
        "sending_institution": "Bank_A",
        "receiving_institution": "Bank_X",
        "route_cost": 2.0,
        "historical_route_success_rate": 0.70,
        "historical_avg_latency_seconds": 3600.0,
    }
    route_high = {
        "route_id": "ROUTE_HIGH",
        "sending_institution": "Bank_A",
        "receiving_institution": "Bank_X",
        "route_cost": 2.0,
        "historical_route_success_rate": 0.99,
        "historical_avg_latency_seconds": 3600.0,
    }

    result = predictor.predict(valid_payment, [route_low, route_high])
    p_low = result["predictions"][0]["predicted_success_probability"]
    p_high = result["predictions"][1]["predicted_success_probability"]

    assert p_high > p_low, f"Expected p_high ({p_high}) > p_low ({p_low})"


def test_directional_test_temporal_latency_multiplier(predictor, sample_routes):
    """Requirement 7b: Directional Test for settlement latency.

    The model was trained on data with weekend/night latency multipliers.
    A weekend transaction should produce higher or equal predicted latency than a weekday daytime transfer.
    """
    weekday_payment = {
        "source_country": "US",
        "destination_country": "India",
        "source_currency": "USD",
        "destination_currency": "INR",
        "amount": 1000.0,
        "payment_type": "P2P",
        "hour_of_day": 11,
        "day_of_week": 1,  # Tuesday
        "is_weekend": 0,
    }
    weekend_payment = copy.deepcopy(weekday_payment)
    weekend_payment["hour_of_day"] = 2  # Sunday night
    weekend_payment["day_of_week"] = 6
    weekend_payment["is_weekend"] = 1

    res_weekday = predictor.predict(weekday_payment, [sample_routes[0]])
    res_weekend = predictor.predict(weekend_payment, [sample_routes[0]])

    t_weekday = res_weekday["predictions"][0]["predicted_settlement_seconds"]
    t_weekend = res_weekend["predictions"][0]["predicted_settlement_seconds"]

    assert (
        t_weekend >= t_weekday
    ), f"Expected weekend latency ({t_weekend}) >= weekday latency ({t_weekday})"


def test_invariance_test_transaction_id(predictor, valid_payment, sample_routes):
    """Requirement 7b: Invariance Test.

    Varying transaction_id must not change model predictions for identical features.
    """
    p1 = copy.deepcopy(valid_payment)
    p1["transaction_id"] = "TXN_ALPHA"

    p2 = copy.deepcopy(valid_payment)
    p2["transaction_id"] = "TXN_OMEGA"

    res1 = predictor.predict(p1, sample_routes)
    res2 = predictor.predict(p2, sample_routes)

    for pred1, pred2 in zip(res1["predictions"], res2["predictions"]):
        assert pred1["predicted_success_probability"] == pred2["predicted_success_probability"]
        assert pred1["predicted_settlement_seconds"] == pred2["predicted_settlement_seconds"]


def test_invalid_input_rejection(predictor, valid_payment, sample_routes):
    """Verifies that missing features or empty route list trigger descriptive errors."""
    incomplete_payment = {"source_country": "US"}  # missing amount, currency, etc.
    with pytest.raises(ValueError) as exc:
        predictor.predict(incomplete_payment, sample_routes)
    assert "Missing required payment fields" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        predictor.predict(valid_payment, [])
    assert "cannot be empty" in str(exc.value)
