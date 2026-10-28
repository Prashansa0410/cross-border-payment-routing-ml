"""
event_bus.py  -  Event-Driven Architecture pattern (lightweight message broker)
--------------------------------------------------------------------------------
A topic-based publish/subscribe broker exposed over HTTP, implemented with the
Python standard library only, so the prototype runs anywhere. It provides the
same semantics the design assumes from Kafka/RabbitMQ in production:
  * named topics                        POST /publish      {"topic":..., "event":...}
  * durable append-only log per topic   GET  /consume?topic=T&group=G    (consumer-group offsets)
  * replay                              GET  /replay?topic=T&from=0
  * observability                       GET  /topics, GET /health
Producers and consumers are fully decoupled: the API gateway publishes
PaymentRequested / RouteRecommended events and does not know who consumes them.
"""

import json
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOPICS = ["payment.requested", "prediction.completed", "route.recommended", "service.error"]


class Broker:
    def __init__(self):
        self.log = defaultdict(list)  # topic -> list of events (append-only)
        self.offsets = defaultdict(int)  # (topic, group) -> next index
        self.lock = threading.Lock()

    def publish(self, topic, event):
        with self.lock:
            envelope = {
                "event_id": str(uuid.uuid4()),
                "topic": topic,
                "published_at": time.time(),
                "offset": len(self.log[topic]),
                "payload": event,
            }
            self.log[topic].append(envelope)
            return envelope

    def consume(self, topic, group, max_events=100):
        with self.lock:
            start = self.offsets[(topic, group)]
            events = self.log[topic][start : start + max_events]
            self.offsets[(topic, group)] = start + len(events)
            return events

    def replay(self, topic, start=0):
        with self.lock:
            return list(self.log[topic][start:])

    def stats(self):
        with self.lock:
            return {
                "topics": {t: len(v) for t, v in self.log.items()},
                "consumer_groups": {f"{t}|{g}": o for (t, g), o in self.offsets.items()},
            }


BROKER = Broker()


class BrokerHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/health":
            return self._send(200, {"status": "ok", "service": "event-broker"})
        if u.path == "/topics":
            return self._send(200, BROKER.stats())
        if u.path == "/consume":
            return self._send(
                200,
                BROKER.consume(
                    q["topic"][0], q.get("group", ["default"])[0], int(q.get("max", ["100"])[0])
                ),
            )
        if u.path == "/replay":
            return self._send(200, BROKER.replay(q["topic"][0], int(q.get("from", ["0"])[0])))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/publish":
            return self._send(201, BROKER.publish(body["topic"], body["event"]))
        self._send(404, {"error": "not found"})


# ---------- client helpers used by producers / consumers -----------------
class EventClient:
    def __init__(self, broker_url="http://127.0.0.1:8010"):
        self.url = broker_url

    def publish(self, topic, event):
        data = json.dumps({"topic": topic, "event": event}).encode()
        req = urllib.request.Request(
            f"{self.url}/publish", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.loads(r.read())

    def consume(self, topic, group, max_events=100):
        with urllib.request.urlopen(
            f"{self.url}/consume?topic={topic}&group={group}&max={max_events}", timeout=3
        ) as r:
            return json.loads(r.read())

    def stats(self):
        with urllib.request.urlopen(f"{self.url}/topics", timeout=3) as r:
            return json.loads(r.read())


def serve(port=8010):
    print(f"[event-broker] listening on :{port}  topics={TOPICS}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), BrokerHandler).serve_forever()


if __name__ == "__main__":
    import sys

    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8010)
