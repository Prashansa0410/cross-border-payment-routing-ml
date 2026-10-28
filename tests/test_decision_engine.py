"""Unit tests for the payment route decision engine."""

import pytest

from decision_engine import Constraints, recommend


def test_recommend_prediction_mode():
    """Verifies that mode='prediction' selects the route with the highest predicted success probability."""
    candidates = [
        {
            "route_id": "ROUTE_CHEAP",
            "sending_institution": "Bank_A",
            "receiving_institution": "Bank_X",
            "route_cost": 1.0,
            "predicted_success_probability": 0.88,
            "predicted_settlement_seconds": 1200.0,
        },
        {
            "route_id": "ROUTE_RELIABLE",
            "sending_institution": "Bank_B",
            "receiving_institution": "Bank_Y",
            "route_cost": 4.0,
            "predicted_success_probability": 0.99,
            "predicted_settlement_seconds": 1200.0,
        },
    ]
    res = recommend(candidates, mode="prediction")
    assert res["recommended_route"] == "ROUTE_RELIABLE"
    assert res["constraints_relaxed"] is False
    assert "highest predicted success probability" in res["explanation"]


def test_recommend_optimisation_mode_with_constraints():
    """Verifies multi-objective optimization under hard business constraints."""
    candidates = [
        {
            "route_id": "ROUTE_EXPENSIVE",
            "sending_institution": "Bank_A",
            "receiving_institution": "Bank_X",
            "route_cost": 5.0,  # exceeds max_cost=2.0
            "predicted_success_probability": 0.98,
            "predicted_settlement_seconds": 600.0,
        },
        {
            "route_id": "ROUTE_AFFORDABLE",
            "sending_institution": "Bank_B",
            "receiving_institution": "Bank_Y",
            "route_cost": 1.5,
            "predicted_success_probability": 0.94,
            "predicted_settlement_seconds": 1800.0,
        },
    ]
    cons = Constraints(max_cost=2.0)
    res = recommend(candidates, mode="optimisation", constraints=cons)

    assert res["recommended_route"] == "ROUTE_AFFORDABLE"
    assert len(res["infeasible_routes"]) == 1
    assert res["infeasible_routes"][0]["route_id"] == "ROUTE_EXPENSIVE"
    assert "max_cost" in res["infeasible_routes"][0]["violated"]


def test_constraints_graceful_relaxation():
    """Verifies that when no routes satisfy constraints, constraints are relaxed to avoid transaction drops."""
    candidates = [
        {
            "route_id": "ROUTE_A",
            "sending_institution": "Bank_A",
            "receiving_institution": "Bank_X",
            "route_cost": 3.0,
            "predicted_success_probability": 0.90,
            "predicted_settlement_seconds": 5000.0,
        }
    ]
    # Impossibly strict constraints
    strict_cons = Constraints(max_cost=1.0, min_success_probability=0.99)
    res = recommend(candidates, mode="optimisation", constraints=strict_cons)

    assert res["constraints_relaxed"] is True
    assert res["recommended_route"] == "ROUTE_A"
    assert "WARNING: no route satisfied all constraints" in res["explanation"]


def test_excluded_routes_constraint():
    """Verifies that excluded routes are marked infeasible."""
    candidates = [
        {
            "route_id": "ROUTE_BANNED",
            "sending_institution": "Bank_A",
            "receiving_institution": "Bank_X",
            "route_cost": 1.0,
            "predicted_success_probability": 0.99,
            "predicted_settlement_seconds": 500.0,
        },
        {
            "route_id": "ROUTE_ALLOWED",
            "sending_institution": "Bank_B",
            "receiving_institution": "Bank_Y",
            "route_cost": 2.0,
            "predicted_success_probability": 0.95,
            "predicted_settlement_seconds": 1000.0,
        },
    ]
    cons = Constraints(excluded_routes=["ROUTE_BANNED"])
    res = recommend(candidates, mode="optimisation", constraints=cons)
    assert res["recommended_route"] == "ROUTE_ALLOWED"


def test_empty_candidates_error():
    """Verifies that an empty candidate list raises a ValueError."""
    with pytest.raises(ValueError):
        recommend([])
