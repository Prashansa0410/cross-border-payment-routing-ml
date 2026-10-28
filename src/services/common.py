"""Shared minimal HTTP/JSON service scaffolding (stdlib only)."""

import json
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def post_json(url, payload, timeout=5):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, default=float).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def make_handler(name, routes_get, routes_post):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            print(f"[{name}] {self.address_string()} {fmt % a}", flush=True)

        def _send(self, code, obj):
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200, {"status": "ok", "service": name, "time": time.time()})
            fn = routes_get.get(path)
            if not fn:
                return self._send(404, {"error": f"unknown path {path}"})
            try:
                self._send(200, fn())
            except Exception as e:
                self._send(500, {"error": str(e)})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            fn = routes_post.get(self.path.split("?")[0])
            if not fn:
                return self._send(404, {"error": f"unknown path {self.path}"})
            try:
                code, obj = fn(body)
                self._send(code, obj)
            except Exception as e:
                self._send(500, {"error": str(e)})

    return H


def serve(name, port, routes_get=None, routes_post=None):
    print(f"[{name}] listening on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(name, routes_get or {}, routes_post or {})
    ).serve_forever()
