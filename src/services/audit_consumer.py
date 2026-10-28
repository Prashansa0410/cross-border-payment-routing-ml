"""
Audit & Monitoring Consumer (event subscriber)
-----------------------------------------------
Independent process in consumer-group "audit". It never talks to the gateway;
it only reacts to events on the broker, which is what makes the architecture
event-driven: new consumers (fraud screening, billing, dashboards) can be added
without changing any producer. Persists every event to artifacts/event_log.jsonl
and maintains live KPI counters exposed on :8003/metrics for the dashboard.
"""

import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from event_bus import TOPICS, EventClient
from services.common import serve

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
LOG = f"{ART}/event_log.jsonl"
client = EventClient(os.environ.get("BROKER_URL", "http://127.0.0.1:8010"))
kpi = {
    "events_consumed": 0,
    "payments_requested": 0,
    "routes_recommended": 0,
    "errors": 0,
    "route_distribution": Counter(),
    "mode_distribution": Counter(),
    "constraints_relaxed": 0,
    "sum_pred_success": 0.0,
    "sum_end_to_end_ms": 0.0,
    "by_corridor": defaultdict(Counter),
    "last_event_at": None,
}
_pending_corridor = {}


def handle(ev):
    kpi["events_consumed"] += 1
    kpi["last_event_at"] = time.strftime("%H:%M:%S")
    p = ev["payload"]
    t = ev["topic"]
    if t == "payment.requested":
        kpi["payments_requested"] += 1
        _pending_corridor[p["transaction_id"]] = f'{p["source_country"]}-{p["destination_country"]}'
    elif t == "route.recommended":
        kpi["routes_recommended"] += 1
        kpi["route_distribution"][p["recommended_route"]] += 1
        kpi["mode_distribution"][p["mode"]] += 1
        kpi["constraints_relaxed"] += int(p["constraints_relaxed"])
        kpi["sum_pred_success"] += p["predicted_success_probability"]
        kpi["sum_end_to_end_ms"] += p["end_to_end_ms"]
        corr = _pending_corridor.pop(p["transaction_id"], "unknown")
        kpi["by_corridor"][corr][p["recommended_route"]] += 1
    elif t == "service.error":
        kpi["errors"] += 1
    with open(LOG, "a") as f:
        f.write(json.dumps(ev) + "\n")


def poll_loop(interval=0.5):
    print("[audit-consumer] subscribed to", TOPICS, "group=audit", flush=True)
    while True:
        for topic in TOPICS:
            try:
                for ev in client.consume(topic, "audit"):
                    handle(ev)
                    print(
                        f"[audit-consumer] {ev['topic']:<22} offset={ev['offset']} txn={ev['payload'].get('transaction_id')}",
                        flush=True,
                    )
            except Exception as e:
                print("[audit-consumer] broker unreachable:", e, flush=True)
                time.sleep(1)
        time.sleep(interval)


def metrics():
    n = max(kpi["routes_recommended"], 1)
    return {
        **{k: v for k, v in kpi.items() if not k.startswith("sum_") and k != "by_corridor"},
        "route_distribution": dict(kpi["route_distribution"]),
        "mode_distribution": dict(kpi["mode_distribution"]),
        "by_corridor": {k: dict(v) for k, v in kpi["by_corridor"].items()},
        "avg_predicted_success_probability": kpi["sum_pred_success"] / n,
        "avg_end_to_end_ms": kpi["sum_end_to_end_ms"] / n,
    }


if __name__ == "__main__":
    open(LOG, "w").close()
    threading.Thread(target=poll_loop, daemon=True).start()
    serve("audit-consumer", 8003, {"/metrics": metrics})
