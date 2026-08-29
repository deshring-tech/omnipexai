"""Pipeline manifest loading.

Purpose: turn a hand-written JSON manifest into a validated ``Pipeline``.
Responsibility: file I/O, shape checking, and precise error messages — nothing
else. Graph semantics live in ``core.schema``; this module guarantees that any
``Pipeline`` it returns has already passed ``Pipeline.validate()``.

JSON is deliberate (zero dependencies). Extension points: a YAML loader can be
added beside ``load`` without touching callers; Phase 2 replaces manifests
entirely with graph reconstruction from OpenInference/OTel traces.

Every malformation raises ``ManifestError`` naming the file and the offending
element — a manifest author should never see a Python traceback.
"""
from __future__ import annotations

import json
from typing import Any

from .core.errors import ManifestError
from .core.schema import Node, Pipeline

_NODE_FIELDS = {"id", "type", "params", "produces_artifact", "evals",
                "stamp", "requires_env"}


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ManifestError(f"{path}: {message}")


def _parse_node(raw: Any, index: int, path: str) -> Node:
    where = f"nodes[{index}]"
    _require(isinstance(raw, dict), path, f"{where} must be an object")
    unknown = set(raw) - _NODE_FIELDS
    _require(not unknown, path, f"{where} has unknown fields {sorted(unknown)}")
    for field_name in ("id", "type"):
        value = raw.get(field_name)
        _require(isinstance(value, str) and value != "", path,
                 f"{where} needs a non-empty string {field_name!r}")
    params = raw.get("params", {})
    _require(isinstance(params, dict), path, f"{where}.params must be an object")
    evals = raw.get("evals", [])
    _require(isinstance(evals, list) and all(isinstance(e, str) for e in evals),
             path, f"{where}.evals must be a list of strings")
    artifact = raw.get("produces_artifact")
    _require(artifact is None or isinstance(artifact, str), path,
             f"{where}.produces_artifact must be a string")
    stamp = raw.get("stamp")
    _require(stamp is None or isinstance(stamp, str), path,
             f"{where}.stamp must be a string path")
    _require(stamp is None or artifact is not None, path,
             f"{where} declares a stamp but no 'produces_artifact' to verify")
    requires_env = raw.get("requires_env", [])
    _require(isinstance(requires_env, list)
             and all(isinstance(v, str) for v in requires_env),
             path, f"{where}.requires_env must be a list of strings")
    return Node(id=raw["id"], type=raw["type"], params=params,
                produces_artifact=artifact, evals=evals, stamp=stamp,
                requires_env=requires_env)


def _parse_edge(raw: Any, index: int, path: str) -> tuple[str, str]:
    _require(
        isinstance(raw, (list, tuple)) and len(raw) == 2
        and all(isinstance(e, str) for e in raw),
        path, f"edges[{index}] must be a pair of node ids",
    )
    return (raw[0], raw[1])


def load(path: str) -> Pipeline:
    """Load, parse, and validate a pipeline manifest.

    Raises ``ManifestError`` for I/O and shape problems and
    ``PipelineValidationError`` for graph problems (both ``OmnipexError``).
    """
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise ManifestError(f"manifest not found: {path}") from None
    except OSError as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}): "
            f"{exc.msg}"
        ) from None

    _require(isinstance(raw, dict), path, "top level must be an object")
    _require(isinstance(raw.get("nodes"), list) and raw["nodes"], path,
             "requires a non-empty 'nodes' list")
    _require(isinstance(raw.get("edges", []), list), path, "'edges' must be a list")
    name = raw.get("name", "pipeline")
    _require(isinstance(name, str), path, "'name' must be a string")

    pipeline = Pipeline(
        name=name,
        nodes=[_parse_node(n, i, path) for i, n in enumerate(raw["nodes"])],
        edges=[_parse_edge(e, i, path) for i, e in enumerate(raw.get("edges", []))],
    )
    pipeline.validate()
    return pipeline


def load_stamp(path: str) -> dict | None:
    """Load a build stamp. Returns None when the file does not exist — a missing
    stamp is a reportable finding, not an error. Malformed content IS an error."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ManifestError(f"cannot read stamp {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"stamp {path} is not valid JSON (line {exc.lineno}): {exc.msg}"
        ) from None
    _require(isinstance(raw, dict), path, "stamp must be an object")
    _require(isinstance(raw.get("params", {}), dict), path,
             "stamp 'params' must be an object of \"node.param\": value")
    return raw
