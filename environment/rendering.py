"""Live 3D renderer for the Imisoro environment.

The environment pushes a JSON snapshot of its state after every step. A tiny
stdlib HTTP server exposes it at ``GET /state`` and serves a Three.js scene
(``environment/web/index.html``) that polls the endpoint and animates the
agent's interventions. The same server exposes ``POST /api/recommend`` when a
policy is attached, which is how a web/mobile frontend would query the agent.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

WEB_DIR = Path(__file__).parent / "web"


class LiveRenderer:
    _instance: "LiveRenderer | None" = None

    def __init__(self, port: int = 8765, open_browser: bool = True):
        self.state: dict = {"seq": 0}
        self.seq = 0
        self.lock = threading.Lock()
        self.recommender: Callable[[list[float]], dict] | None = None
        renderer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal clean for agent logs
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state"):
                    with renderer.lock:
                        body = json.dumps(renderer.state).encode()
                    self._send(200, body, "application/json")
                elif self.path in ("/", "/index.html"):
                    self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/api/recommend" or renderer.recommender is None:
                    self._send(404, b'{"error":"no policy attached"}', "application/json")
                    return
                n = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    out = renderer.recommender(payload["observation"])
                    self._send(200, json.dumps(out).encode(), "application/json")
                except Exception as e:  # noqa: BLE001
                    self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

        for p in range(port, port + 20):
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                self.port = p
                break
            except OSError:
                continue
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.port}/"
        print(f"[render] 3D view live at {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    @classmethod
    def get(cls, **kw) -> "LiveRenderer":
        if cls._instance is None:
            cls._instance = cls(**kw)
        return cls._instance

    def push(self, snapshot: dict):
        with self.lock:
            self.seq += 1
            self.state = {**snapshot, "seq": self.seq}

    def flush(self):
        pass
