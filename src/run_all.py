"""Start every component of the prototype as separate OS processes (one per microservice)."""

import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SERVICES = [
    ("event-broker", "event_bus.py", 8010),
    ("prediction-service", "services/prediction_service.py", 8001),
    ("decision-service", "services/decision_service.py", 8002),
    ("api-gateway", "services/gateway_service.py", 8000),
    ("audit-consumer", "services/audit_consumer.py", 8003),
]
procs = []
for name, script, port in SERVICES:
    procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, script)]))
    for _ in range(40):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.25)
    print(f"started {name:<20} pid={procs[-1].pid}  http://127.0.0.1:{port}", flush=True)
print("\nAll services up. Press Ctrl+C to stop.", flush=True)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    for p in procs:
        p.send_signal(signal.SIGTERM)
