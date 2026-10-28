"""
API Gateway / Payment Orchestrator  - port 8000
-------------------------------------------------
Single entry point for clients. Validates the request, enriches temporal
features from the timestamp, publishes domain EVENTS to the broker
(event-driven pattern) and delegates to the Decision Service (microservices).
  POST /route-payment      {"transaction_id", "timestamp", "source_country", ..., "amount",
                            "payment_type", "mode", "weights", "constraints"}
  GET  /health, GET /services   (aggregated health of all downstream services)
"""

import json
import os
import sys
import time
import urllib.error
import uuid
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from event_bus import EventClient
from services.common import get_json, post_json, serve

DECISION_URL = os.environ.get("DECISION_URL", "http://127.0.0.1:8002")
PREDICTION_URL = os.environ.get("PREDICTION_URL", "http://127.0.0.1:8001")
BROKER_URL = os.environ.get("BROKER_URL", "http://127.0.0.1:8010")
events = EventClient(BROKER_URL)
REQUIRED = [
    "source_country",
    "destination_country",
    "source_currency",
    "destination_currency",
    "amount",
    "payment_type",
]


def _enrich(req):
    ts = datetime.fromisoformat(
        req.get("timestamp") or datetime.now().isoformat(timespec="minutes")
    )
    return {
        **{k: req[k] for k in REQUIRED},
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),
    }


def route_payment(body):
    t0 = time.perf_counter()
    txn_id = body.get("transaction_id") or f"TXN_{uuid.uuid4().hex[:8].upper()}"
    missing = [k for k in REQUIRED if k not in body]
    if missing:
        return 400, {"error": f"missing fields: {missing}", "transaction_id": txn_id}
    try:
        payment = _enrich(body)
    except Exception as e:
        return 400, {"error": f"invalid timestamp: {e}"}
    events.publish(
        "payment.requested",
        {"transaction_id": txn_id, **payment, "mode": body.get("mode", "optimisation")},
    )
    try:
        rec = post_json(
            f"{DECISION_URL}/recommend",
            {
                "payment": payment,
                "mode": body.get("mode", "optimisation"),
                "weights": body.get("weights", {}),
                "constraints": body.get("constraints", {}),
            },
        )
    except (
        urllib.error.HTTPError
    ) as e:  # downstream returned a controlled error (e.g. 502 ML service down)
        detail = json.loads(e.read() or b"{}")
        events.publish(
            "service.error",
            {
                "transaction_id": txn_id,
                "stage": "decision-service",
                "http_status": e.code,
                "error": detail,
            },
        )
        return 503, {
            "error": detail.get("error", "upstream failure"),
            "upstream_status": e.code,
            "detail": detail.get("detail"),
            "transaction_id": txn_id,
        }
    except Exception as e:  # decision service itself unreachable
        events.publish(
            "service.error",
            {"transaction_id": txn_id, "stage": "decision-service", "error": str(e)},
        )
        return 503, {
            "error": "decision service unavailable",
            "detail": str(e),
            "transaction_id": txn_id,
        }
    if "error" in rec:
        events.publish(
            "service.error", {"transaction_id": txn_id, "stage": "decision-service", "error": rec}
        )
        return 502, {**rec, "transaction_id": txn_id}
    events.publish(
        "prediction.completed",
        {
            "transaction_id": txn_id,
            "model_version": rec["model_version"],
            "inference_ms": rec["inference_ms"],
            "routes": [
                {
                    k: r[k]
                    for k in [
                        "route_id",
                        "predicted_success_probability",
                        "predicted_settlement_seconds",
                    ]
                }
                for r in rec["ranked_routes"]
            ],
        },
    )
    total_ms = (time.perf_counter() - t0) * 1000
    events.publish(
        "route.recommended",
        {
            "transaction_id": txn_id,
            "recommended_route": rec["recommended_route"],
            "mode": rec["mode"],
            "score": rec["recommendation"]["score"],
            "predicted_success_probability": rec["recommendation"]["predicted_success_probability"],
            "predicted_settlement_seconds": rec["recommendation"]["predicted_settlement_seconds"],
            "route_cost": rec["recommendation"]["route_cost"],
            "constraints_relaxed": rec["constraints_relaxed"],
            "end_to_end_ms": total_ms,
        },
    )
    return 200, {"transaction_id": txn_id, "payment": payment, **rec, "end_to_end_ms": total_ms}


def services_health():
    out = {}
    for name, url in [
        ("prediction-service", PREDICTION_URL),
        ("decision-service", DECISION_URL),
        ("event-broker", BROKER_URL),
    ]:
        try:
            out[name] = get_json(f"{url}/health", timeout=2)
        except Exception as e:
            out[name] = {"status": "DOWN", "error": str(e)}
    return out


if __name__ == "__main__":
    serve("api-gateway", 8000, {"/services": services_health}, {"/route-payment": route_payment})
