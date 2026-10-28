"""Central configuration and constants for the Payment Routing ML system."""

from pathlib import Path
from typing import List

# Paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = BASE_DIR / "data.csv"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
FIGURES_DIR: Path = BASE_DIR / "figures"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Feature Definitions
ID_COLS: List[str] = ["transaction_id", "timestamp"]

CAT_FEATURES: List[str] = [
    "source_country",
    "destination_country",
    "source_currency",
    "destination_currency",
    "payment_type",
    "route_id",
    "sending_institution",
    "receiving_institution",
]

NUM_FEATURES: List[str] = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "route_cost",
    "historical_route_success_rate",
    "historical_avg_latency_seconds",
]

DROPPED_REDUNDANT: List[str] = ["historical_route_failure_rate"]
TARGET_CLF: str = "payment_success"
TARGET_REG: str = "settlement_time_seconds"
FEATURES: List[str] = CAT_FEATURES + NUM_FEATURES

# Required payment fields from incoming client requests
REQUIRED_PAYMENT_FIELDS: List[str] = [
    "source_country",
    "destination_country",
    "source_currency",
    "destination_currency",
    "amount",
    "payment_type",
]

# Business Decision Defaults
DEFAULT_WEIGHTS = {"success": 0.6, "latency": 0.25, "cost": 0.15}
