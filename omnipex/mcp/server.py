"""MCP (Model Context Protocol) server — JSON-RPC 2.0 over stdio.

Purpose: expose the OmniPex engine to MCP hosts (Claude Code, Cursor, …) so an
engineer can ask "which change broke faithfulness?" in the tool they already have
open, and their assistant calls this engine to answer.

Responsibilities: protocol framing, tool declaration, argument validation, path
safety. All results come from ``omnipex.api`` — no analysis logic lives here.

Implemented with the standard library only: the MCP stdio transport is
newline-delimited JSON-RPC 2.0, so no SDK is required and the package keeps its
zero-dependency guarantee.

Protocol notes:
  * stdout carries protocol messages ONLY. Diagnostics go to stderr; a stray
    print() here would corrupt the stream.
  * Requests carry an ``id`` and expect a response; notifications have no ``id``
    and must not be answered.
  * Tool failures are returned as ``isError`` results (the model can read and
    react to them); protocol faults use JSON-RPC error objects.

Wire it into an MCP host with:
    {"command": "omnipex", "args": ["mcp", "--dir", "/path/to/project"]}
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO

from .. import api
from ..core.errors import OmnipexError

SERVER_NAME = "omnipex"
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC error codes (subset of the spec we can emit).
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

_MANIFEST_ARG = {
    "type": "string",
    "description": "Path to a pipeline manifest JSON file, relative to the "
                   "project root.",
}
_PACK_ARG = {
    "type": "string",
    "description": "Rule-pack naming the AI-system class. Default 'rag'.",
    "default": "rag",
}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "omnipex_show",
        "description": (
            "Show the structure of an AI pipeline manifest: its components "
            "(chunker, embedder, vector store, retriever, prompt, LLM), how they "
            "connect, which artifacts they produce, and which evaluation metrics "
            "are bound to each. Use this to understand a pipeline's shape before "
            "analyzing changes."
        ),
        "inputSchema": _schema({"manifest": _MANIFEST_ARG}, ["manifest"]),
    },
    {
        "name": "omnipex_doctor",
        "description": (
            "Check whether a pipeline's declared configuration matches what was "
            "ACTUALLY built. Catches the classic silent failure where a vector "
            "index was built with one embedding model but the config now names "
            "another — nothing errors, but retrieval quality is silently wrong. "
            "Fast and free: runs no evaluations. Use when a pipeline behaves "
            "oddly, or before trusting any eval result."
        ),
        "inputSchema": _schema(
            {"manifest": _MANIFEST_ARG, "pack": _PACK_ARG}, ["manifest"]),
    },
    {
        "name": "omnipex_diff",
        "description": (
            "Change-impact analysis between two versions of a pipeline. Reports "
            "what changed, the blast radius (which components must be rebuilt, "
            "which need re-evaluation, which are provably unaffected), then "
            "re-runs the affected evaluations and reports which metric changes "
            "are REAL versus statistical noise. Use this as a merge gate, or "
            "when eval scores moved after a change."
        ),
        "inputSchema": _schema({
            "base": dict(_MANIFEST_ARG, description="Baseline manifest (before)."),
            "candidate": dict(_MANIFEST_ARG,
                              description="Candidate manifest (after)."),
            "pack": _PACK_ARG,
        }, ["base", "candidate"]),
    },
    {
        "name": "omnipex_blame",
        "description": (
            "Find WHICH change caused a regression when a commit changed several "
            "things at once. Isolates each suspect change against the baseline, "
            "re-runs only the at-risk metrics, and names the culprit — also "
            "revealing when one change masked another's damage by improving the "
            "same metric. This is git-bisect for AI quality regressions. Use "
            "when a metric dropped and multiple parameters changed together."
        ),
        "inputSchema": _schema({
            "base": dict(_MANIFEST_ARG, description="Baseline manifest (before)."),
            "candidate": dict(_MANIFEST_ARG,
                              description="Candidate manifest (after)."),
            "pack": _PACK_ARG,
        }, ["base", "candidate"]),
    },
]


class MCPServer:
    """Stateless request handler over a project root."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)

    # --- path safety -------------------------------------------------------
    def _resolve(self, rel_path: str) -> str:
        full = os.path.abspath(os.path.join(self.root, rel_path))
        if os.path.commonpath([full, self.root]) != self.root:
            raise OmnipexError(
                f"path escapes the project root: {rel_path!r} "
                f"(root is {self.root})"
            )
        return full

    # --- tool dispatch -----------------------------------------------------
    def call_tool(self, name: str, arguments: dict) -> dict:
        """Run a tool and return its JSON-ready report. Raises OmnipexError."""
        pack = arguments.get("pack", "rag")
        if name == "omnipex_show":
            return api.show(self._resolve(arguments["manifest"]))
        if name == "omnipex_doctor":
            return api.doctor(self._resolve(arguments["manifest"]), pack)
        if name == "omnipex_diff":
            return api.diff(self._resolve(arguments["base"]),
                            self._resolve(arguments["candidate"]), pack)
        if name == "omnipex_blame":
            return api.blame(self._resolve(arguments["base"]),
                             self._resolve(arguments["candidate"]), pack)
        raise OmnipexError(
            f"unknown tool {name!r}; available: "
            f"{', '.join(t['name'] for t in TOOLS)}"
        )

    # --- protocol ----------------------------------------------------------
    def handle(self, message: dict) -> dict | None:
        """Handle one JSON-RPC message. Returns a response, or None for
        notifications (which must never be answered)."""
        method = message.get("method")
        msg_id = message.get("id")
        if msg_id is None:
            return None  # notification, e.g. notifications/initialized

        if method == "initialize":
            params = message.get("params") or {}
            # Echo the client's protocol version when supplied: maximally
            # compatible across host versions.
            version = params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION
            return _result(msg_id, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": _version()},
            })

        if method == "tools/list":
            return _result(msg_id, {"tools": TOOLS})

        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str):
                return _error(msg_id, _INVALID_PARAMS,
                              "tools/call requires a string 'name'")
            try:
                report = self.call_tool(name, arguments)
            except KeyError as exc:
                return _tool_error(msg_id, f"missing required argument: {exc}")
            except OmnipexError as exc:
                return _tool_error(msg_id, str(exc))
            except Exception as exc:  # unexpected: report, never crash the host
                return _error(msg_id, _INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
            return _result(msg_id, {
                "content": [{"type": "text",
                             "text": json.dumps(report, indent=2)}],
                "isError": False,
            })

        if method == "ping":
            return _result(msg_id, {})

        return _error(msg_id, _METHOD_NOT_FOUND, f"unknown method: {method}")


def _version() -> str:
    from .. import __version__
    return __version__


def _result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code,
                                                      "message": message}}


def _tool_error(msg_id: Any, message: str) -> dict:
    """A tool failed. Reported as a result so the model can read and react."""
    return _result(msg_id, {
        "content": [{"type": "text", "text": f"error: {message}"}],
        "isError": True,
    })


def serve(root: str = ".", stdin: TextIO | None = None,
          stdout: TextIO | None = None) -> None:
    """Run the stdio message loop until the input stream closes."""
    server = MCPServer(root)
    source: TextIO = stdin or sys.stdin
    sink: TextIO = stdout or sys.stdout

    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue  # malformed frame: no id to answer, so drop it
        response = server.handle(message)
        if response is not None:
            sink.write(json.dumps(response) + "\n")
            sink.flush()
