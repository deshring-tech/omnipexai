"""GUI server.

Purpose: serve the single-page UI and expose ``omnipex.api`` over local HTTP.
Responsibilities: routing, manifest discovery, JSON marshalling, path safety.
No analysis logic lives here — every result comes from ``omnipex.api``.

Security posture: binds 127.0.0.1 only; requested manifest paths are resolved
against the served project root and rejected if they escape it.

Endpoints:
    GET  /               the single-page UI
    GET  /api/manifests  valid pipeline manifests found under the project root
    POST /api/show       {"path"}                     -> api.show
    POST /api/doctor     {"path","pack"}              -> api.doctor
    POST /api/diff       {"base","candidate","pack"}  -> api.diff
    POST /api/blame      {"base","candidate","pack"}  -> api.blame
"""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import api
from ..core.errors import OmnipexError

_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
              ".pytest_cache", "results", "dist", "build"}
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_SCAN_DEPTH = 3


def find_manifests(root: str) -> list[dict]:
    """Valid pipeline manifests under ``root`` (bounded depth, junk dirs skipped)."""
    root = os.path.abspath(root)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if depth >= _MAX_SCAN_DEPTH:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            full = os.path.join(dirpath, filename)
            try:
                if os.path.getsize(full) > _MAX_MANIFEST_BYTES:
                    continue
                info = api.show(full)
            except (OmnipexError, OSError):
                continue  # not a pipeline manifest; skip silently
            found.append({
                "path": os.path.relpath(full, root).replace(os.sep, "/"),
                "name": info["name"],
                "nodes": len(info["nodes"]),
                "edges": len(info["edges"]),
            })
    return sorted(found, key=lambda m: m["path"])


class _Handler(BaseHTTPRequestHandler):
    server: "UIServer"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict | list) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _resolve(self, rel_path: str) -> str:
        """Resolve a client-supplied path inside the project root, or refuse."""
        full = os.path.abspath(os.path.join(self.server.root, rel_path))
        if os.path.commonpath([full, self.server.root]) != self.server.root:
            raise OmnipexError(f"path escapes the project root: {rel_path!r}")
        return full

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path in ("/", "/index.html"):
            with open(_INDEX, "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif self.path == "/api/manifests":
            self._send_json(200, find_manifests(self.server.root))
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            pack = body.get("pack", "rag")
            if self.path == "/api/show":
                result = api.show(self._resolve(body["path"]))
            elif self.path == "/api/doctor":
                result = api.doctor(self._resolve(body["path"]), pack)
            elif self.path == "/api/diff":
                result = api.diff(self._resolve(body["base"]),
                                  self._resolve(body["candidate"]), pack)
            elif self.path == "/api/blame":
                result = api.blame(self._resolve(body["base"]),
                                   self._resolve(body["candidate"]), pack)
            else:
                self._send_json(404, {"error": "not found"})
                return
        except OmnipexError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except (KeyError, json.JSONDecodeError):
            self._send_json(400, {"error": "malformed request body"})
            return
        self._send_json(200, result)

    def log_message(self, fmt: str, *log_args) -> None:
        pass  # quiet by default; the UI has its own activity log


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root: str, port: int):
        self.root = os.path.abspath(root)
        super().__init__(("127.0.0.1", port), _Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


def serve(root: str = ".", port: int = 7327, open_browser: bool = True) -> None:
    """Run the GUI server until interrupted (Ctrl+C)."""
    server = UIServer(root, port)
    print(f"omnipex ui: {server.url}  (project root: {server.root})")
    print("press Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(server.url,)).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
