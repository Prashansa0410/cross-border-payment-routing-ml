"""
demo_client.py  -  exercises the running system end-to-end and prints the
outputs that are pasted into the report (each block is labelled SCREENSHOT n).
"""

import json
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.common import get_json, post_json

GW = "http://127.0.0.1:8000"


def show(title, obj):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(json.dumps(obj, indent=2, default=float))


def table(rec):
    print(f"\n{'route':9}{'P(success)':>12}{'settle (h)':>12}{'cost':>7}{'score':>8}  feasible")
    for r in rec["ranked_routes"] + rec["infeasible_routes"]:
        print(
            f"{r['route_id']:9}{r['predicted_success_probability']:>12.3f}{r['predicted_settlement_seconds']/3600:>12.2f}"
            f"{r['route_cost']:>7.2f}{r['score']:>8.3f}  {'yes' if r['feasible'] else 'NO ' + ','.join(r['violated'])}"
        )
    print(">>", rec["explanation"])


show("SCREENSHOT 1 - Health of all microservices (GET /services)", get_json(f"{GW}/services"))
show(
    "SCREENSHOT 2 - Prediction service model info (GET :8001/model-info)",
    get_json("http://127.0.0.1:8001/model-info"),
)

payment = {
    "transaction_id": "DEMO_001",
    "timestamp": "2026-10-04T02:30:00",
    "source_country": "US",
    "destination_country": "India",
    "source_currency": "USD",
    "destination_currency": "INR",
    "amount": 1500.0,
    "payment_type": "P2P",
}
r1 = post_json(f"{GW}/route-payment", {**payment, "mode": "prediction"})
show(
    "SCREENSHOT 3 - Requirement 1: PREDICTION mode (highest success probability), Sunday 02:30 US->India (weekend + night penalty)",
    {
        k: r1[k]
        for k in [
            "transaction_id",
            "payment",
            "mode",
            "recommended_route",
            "model_version",
            "inference_ms",
            "end_to_end_ms",
        ]
    },
)
table(r1)

r2 = post_json(
    f"{GW}/route-payment", {**payment, "transaction_id": "DEMO_002", "mode": "optimisation"}
)
show(
    "SCREENSHOT 4 - Requirement 2: OPTIMISATION mode (success + latency + cost), same payment, default weights",
    {k: r2[k] for k in ["transaction_id", "mode", "weights", "recommended_route"]},
)
table(r2)

r3 = post_json(
    f"{GW}/route-payment",
    {
        **payment,
        "transaction_id": "DEMO_003",
        "mode": "optimisation",
        "weights": {"success": 0.3, "latency": 0.2, "cost": 0.5},
        "constraints": {"max_cost": 2.0},
    },
)
show(
    "SCREENSHOT 5 - Cost-sensitive weights + business constraint max_cost=2.0",
    {k: r3[k] for k in ["transaction_id", "weights", "recommended_route", "constraints_relaxed"]},
)
table(r3)

r4 = post_json(
    f"{GW}/route-payment",
    {
        "transaction_id": "DEMO_004",
        "timestamp": "2026-10-06T11:00:00",
        "source_country": "UK",
        "destination_country": "Nigeria",
        "source_currency": "GBP",
        "destination_currency": "NGN",
        "amount": 60000.0,
        "payment_type": "B2B",
        "mode": "optimisation",
        "constraints": {"max_latency_seconds": 7200, "min_success_probability": 0.9},
    },
)
show(
    "SCREENSHOT 6 - Large B2B UK->Nigeria, constraints max_latency 2h & min P(success) 0.90",
    {k: r4[k] for k in ["transaction_id", "payment", "recommended_route", "constraints_relaxed"]},
)
table(r4)

try:
    bad = post_json(f"{GW}/route-payment", {"source_country": "US", "amount": 10})
except Exception as e:
    bad = json.loads(e.read()) if isinstance(e, urllib.error.HTTPError) else str(e)
show("SCREENSHOT 7 - Input validation (missing fields -> HTTP 400)", bad)

time.sleep(1.5)  # let the asynchronous consumer catch up
show(
    "SCREENSHOT 8 - Event broker topics & consumer-group offsets (GET :8010/topics)",
    get_json("http://127.0.0.1:8010/topics"),
)
show(
    "SCREENSHOT 9 - Audit/monitoring consumer KPIs built purely from events (GET :8003/metrics)",
    get_json("http://127.0.0.1:8003/metrics"),
)
rep = get_json("http://127.0.0.1:8010/replay?topic=route.recommended&from=0")
show(
    "SCREENSHOT 10 - Event replay: first RouteRecommended event from the durable log",
    rep[0] if rep else {},
)
