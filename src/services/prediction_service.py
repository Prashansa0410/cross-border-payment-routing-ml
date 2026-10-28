"""
Prediction Service (ML microservice)  - port 8001
---------------------------------------------------
Owns the trained models. Single responsibility: given a payment + candidate
routes, return P(success) and expected settlement time per route.
Can be redeployed/retrained independently of the decision logic.
  POST /predict        {"payment": {...}, "routes": [ {...}, ... ]}
  GET  /model-info
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import joblib
import numpy as np
import pandas as pd

from data_prep import FEATURES
from services.common import serve

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
CLF = joblib.load(f"{ART}/success_classifier.joblib")
REG = joblib.load(f"{ART}/settlement_regressor.joblib")
METRICS = json.load(open(f"{ART}/metrics.json"))
THRESHOLD = json.load(open(f"{ART}/threshold.json"))["decision_threshold"]
REQUIRED_PAYMENT = [
    "source_country",
    "destination_country",
    "source_currency",
    "destination_currency",
    "amount",
    "payment_type",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]
_counters = {"requests": 0, "errors": 0, "latency_ms_total": 0.0}


def _frame(payment, routes):
    rows = [
        {
            **payment,
            **{
                k: r[k]
                for k in [
                    "route_id",
                    "sending_institution",
                    "receiving_institution",
                    "route_cost",
                    "historical_route_success_rate",
                    "historical_avg_latency_seconds",
                ]
            },
        }
        for r in routes
    ]
    return pd.DataFrame(rows)[FEATURES]


def predict(body):
    t0 = time.perf_counter()
    _counters["requests"] += 1
    payment, routes = body.get("payment"), body.get("routes")
    missing = [k for k in REQUIRED_PAYMENT if payment is None or k not in payment]
    if missing or not routes:
        _counters["errors"] += 1
        return 400, {
            "error": f"missing payment fields {missing}" if missing else "routes list is empty"
        }
    X = _frame(payment, routes)
    p_success = CLF.predict_proba(X)[:, 1]
    t_settle = np.expm1(REG.predict(X))
    out = [
        {
            "route_id": r["route_id"],
            "sending_institution": r["sending_institution"],
            "receiving_institution": r["receiving_institution"],
            "route_cost": float(r["route_cost"]),
            "predicted_success_probability": float(p),
            "predicted_settlement_seconds": float(t),
            "predicted_class": int(p >= THRESHOLD),
        }
        for r, p, t in zip(routes, p_success, t_settle)
    ]
    ms = (time.perf_counter() - t0) * 1000
    _counters["latency_ms_total"] += ms
    return 200, {"predictions": out, "model_version": METRICS["model_version"], "inference_ms": ms}


def model_info():
    return {
        "model_version": METRICS["model_version"],
        "decision_threshold": THRESHOLD,
        "classifier_test_metrics": METRICS["classification"]["gbdt_classifier"],
        "regressor_test_metrics": METRICS["regression_success_only"]["gbdt_regressor"],
        "features": FEATURES,
        "runtime_counters": _counters,
    }


if __name__ == "__main__":
    serve("prediction-service", 8001, {"/model-info": model_info}, {"/predict": predict})
