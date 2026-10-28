"""
fault_demo.py  -  demonstrates microservice fault isolation + the event-driven error topic.
Stops the Prediction Service, sends a request (gateway stays up, returns a controlled
error, publishes a service.error event), restarts the service, request succeeds again.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.common import get_json, post_json

GW = "http://127.0.0.1:8000"
req = {
    "transaction_id": "FAULT_001",
    "timestamp": "2026-10-07T09:00:00",
    "source_country": "Singapore",
    "destination_country": "Philippines",
    "source_currency": "SGD",
    "destination_currency": "PHP",
    "amount": 250.0,
    "payment_type": "C2B",
}


def call():
    try:
        return post_json(f"{GW}/route-payment", req)
    except urllib.error.HTTPError as e:
        return {"http_status": e.code, **json.loads(e.read())}


print("SCREENSHOT 12 - Fault isolation demo\n" + "=" * 60)
print("1) Stopping prediction-service ...")
subprocess.run(["pkill", "-f", "prediction_service.py"])
time.sleep(1)
print("   gateway /services ->", json.dumps(get_json(f"{GW}/services"), default=str)[:300], "...")
r = call()
print(
    "2) Request while ML service is down ->",
    json.dumps({k: r[k] for k in r if k in ("http_status", "error", "transaction_id")}),
)
print(
    "   service.error topic now has",
    get_json("http://127.0.0.1:8010/topics")["topics"].get("service.error", 0),
    "event(s)",
)
print("3) Restarting prediction-service ...")
subprocess.Popen(
    [sys.executable, os.path.join(os.path.dirname(__file__), "services", "prediction_service.py")],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
for _ in range(40):
    try:
        get_json("http://127.0.0.1:8001/health", timeout=0.5)
        break
    except Exception:
        time.sleep(0.25)
r = call()
print(
    "4) Request after recovery ->",
    r["recommended_route"],
    "P(success)=%.3f" % r["recommendation"]["predicted_success_probability"],
    "end_to_end_ms=%.1f" % r["end_to_end_ms"],
)
