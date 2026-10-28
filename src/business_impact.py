"""
business_impact.py  -  Measures the Business-View indicators on HELD-OUT test data.
For each of N test transactions the decision service recommends a route; we compare
the recommended route's predicted outcome with the route that was actually used and
with "always pick the cheapest" / "always pick highest historical success" heuristics.
Also measures gateway throughput and p95 end-to-end latency (quality requirement 2).
"""

import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_raw, make_splits
from services.common import post_json

GW = "http://127.0.0.1:8000"
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
W = (
    dict(zip(["success", "latency", "cost"], map(float, sys.argv[2].split(","))))
    if len(sys.argv) > 2
    else {"success": 0.6, "latency": 0.25, "cost": 0.15}
)
TAG = sys.argv[3] if len(sys.argv) > 3 else "default"

df = load_raw(os.path.join(os.path.dirname(__file__), "..", "data.csv"))
_, _, test = make_splits(df)
sample = test.sample(N, random_state=7)
rows, lat = [], []
for _, r in sample.iterrows():
    req = {
        "transaction_id": r.transaction_id,
        "timestamp": r.timestamp,
        "source_country": r.source_country,
        "destination_country": r.destination_country,
        "source_currency": r.source_currency,
        "destination_currency": r.destination_currency,
        "amount": float(r.amount),
        "payment_type": r.payment_type if isinstance(r.payment_type, str) else "P2P",
        "mode": "optimisation",
        "weights": W,
    }
    t0 = time.perf_counter()
    res = post_json(f"{GW}/route-payment", req)
    lat.append((time.perf_counter() - t0) * 1000)
    ranked = {x["route_id"]: x for x in res["ranked_routes"] + res["infeasible_routes"]}
    hist = ranked[r.route_id]
    rec = res["recommendation"]
    cheapest = min(ranked.values(), key=lambda x: x["route_cost"])
    most_reliable = max(ranked.values(), key=lambda x: x["predicted_success_probability"])
    rows.append(
        {
            "txn": r.transaction_id,
            "historical_route": r.route_id,
            "recommended_route": res["recommended_route"],
            "hist_p": hist["predicted_success_probability"],
            "rec_p": rec["predicted_success_probability"],
            "hist_t": hist["predicted_settlement_seconds"],
            "rec_t": rec["predicted_settlement_seconds"],
            "hist_cost": hist["route_cost"],
            "rec_cost": rec["route_cost"],
            "cheapest_p": cheapest["predicted_success_probability"],
            "reliable_p": most_reliable["predicted_success_probability"],
            "reliable_cost": most_reliable["route_cost"],
            "actual_success": int(r.payment_success),
        }
    )
d = pd.DataFrame(rows)
summary = {
    "weights": W,
    "transactions_evaluated": N,
    "historical_route": {
        "expected_success_rate": d.hist_p.mean(),
        "actual_success_rate": d.actual_success.mean(),
        "expected_settlement_hours": d.hist_t.mean() / 3600,
        "avg_cost": d.hist_cost.mean(),
    },
    "ml_optimised_route": {
        "expected_success_rate": d.rec_p.mean(),
        "expected_settlement_hours": d.rec_t.mean() / 3600,
        "avg_cost": d.rec_cost.mean(),
    },
    "always_cheapest_heuristic": {"expected_success_rate": d.cheapest_p.mean()},
    "always_most_reliable": {
        "expected_success_rate": d.reliable_p.mean(),
        "avg_cost": d.reliable_cost.mean(),
    },
    "uplift_vs_historical": {
        "success_rate_points": (d.rec_p.mean() - d.hist_p.mean()) * 100,
        "settlement_time_reduction_pct": (1 - d.rec_t.mean() / d.hist_t.mean()) * 100,
        "cost_change_pct": (d.rec_cost.mean() / d.hist_cost.mean() - 1) * 100,
    },
    "route_changed_pct": (d.historical_route != d.recommended_route).mean() * 100,
    "recommended_route_distribution": d.recommended_route.value_counts().to_dict(),
    "gateway_latency_ms": {
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
        "p99": float(np.percentile(lat, 99)),
        "max": float(max(lat)),
    },
    "throughput_req_per_sec_sequential": N / (sum(lat) / 1000),
}
json.dump(summary, open(f"{ART}/business_impact_{TAG}.json", "w"), indent=2, default=float)
d.to_csv(f"{ART}/business_impact_detail_{TAG}.csv", index=False)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].bar(
    ["Historical\nroute", "ML optimised\nroute", "Always\ncheapest", "Always most\nreliable"],
    [d.hist_p.mean(), d.rec_p.mean(), d.cheapest_p.mean(), d.reliable_p.mean()],
    color=["grey", "#2a6f97", "#e9c46a", "#90be6d"],
)
ax[0].set_ylim(0.8, 1.0)
ax[0].set_title(f"Expected success rate (Indicator 1)\nweights={W}", fontsize=9)
ax[1].bar(
    ["Historical", "ML optimised"],
    [d.hist_t.mean() / 3600, d.rec_t.mean() / 3600],
    color=["grey", "#2a6f97"],
)
ax[1].set_title("Expected settlement time, hours (Indicator 2)")
ax[2].bar(
    ["Historical", "ML optimised", "Most reliable"],
    [d.hist_cost.mean(), d.rec_cost.mean(), d.reliable_cost.mean()],
    color=["grey", "#2a6f97", "#90be6d"],
)
ax[2].set_title("Average route cost")
plt.tight_layout()
plt.savefig(f"{FIG}/business_impact_{TAG}.png", dpi=150)
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(lat, bins=30, color="#2a6f97")
ax.axvline(
    np.percentile(lat, 95), color="r", ls="--", label=f"p95 = {np.percentile(lat,95):.1f} ms"
)
ax.set_xlabel("end-to-end latency per request (ms)")
ax.set_title(f"Gateway latency, {N} requests")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/latency_distribution_{TAG}.png", dpi=150)
print(f"SCREENSHOT 11 - Business impact on {N} held-out test transactions, weights={W}")
print(json.dumps(summary, indent=2, default=float))
