"""Shared pytest fixtures for the payment routing test suite."""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import PaymentPredictor


@pytest.fixture(scope="session")
def valid_payment() -> Dict[str, Any]:
    """Sample valid payment payload."""
    return {
        "transaction_id": "TEST_TXN_001",
        "timestamp": "2026-10-28T14:30:00",
        "source_country": "US",
        "destination_country": "India",
        "source_currency": "USD",
        "destination_currency": "INR",
        "amount": 1250.0,
        "payment_type": "P2P",
        "hour_of_day": 14,
        "day_of_week": 2,
        "is_weekend": 0,
    }


@pytest.fixture(scope="session")
def sample_routes() -> List[Dict[str, Any]]:
    """Sample candidate routes."""
    return [
        {
            "route_id": "ROUTE_A",
            "sending_institution": "Bank_A",
            "receiving_institution": "Bank_X",
            "route_cost": 1.50,
            "historical_route_success_rate": 0.94,
            "historical_avg_latency_seconds": 4500.0,
        },
        {
            "route_id": "ROUTE_B",
            "sending_institution": "Bank_B",
            "receiving_institution": "Bank_Y",
            "route_cost": 4.50,
            "historical_route_success_rate": 0.98,
            "historical_avg_latency_seconds": 600.0,
        },
        {
            "route_id": "ROUTE_C",
            "sending_institution": "Bank_C",
            "receiving_institution": "Bank_Z",
            "route_cost": 2.00,
            "historical_route_success_rate": 0.91,
            "historical_avg_latency_seconds": 18000.0,
        },
    ]


@pytest.fixture(scope="session")
def predictor() -> PaymentPredictor:
    """Pre-loaded predictor instance."""
    return PaymentPredictor.get_instance()
