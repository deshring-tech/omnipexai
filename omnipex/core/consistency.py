"""Consistency checking — does the declared config match what was actually built?

The invalidation engine tells you an artifact *should* be rebuilt after a change.
It cannot tell you whether it *was*. That gap is the classic silent failure in AI
pipelines: the manifest says one embedding model, the vector index was built with
another, nothing errors, and retrieval quality quietly degrades.

This module closes the loop by comparing a build-emitted **stamp** (what the
artifact was really built with) against the current manifest.

Which parameters matter is NOT hardcoded here — it is derived from the rule-pack
by asking which changes would force a ``RECOMPUTE_REQUIRED`` on that artifact.
So every rule-pack gets consistency checking for free, and the domain knowledge
stays in one place.

Pure logic: no file I/O and no environment access. Callers supply stamps and the
environment mapping (see ``omnipex.api.doctor``).

IMPORTANT: consistency is not health. A fully consistent pipeline can still
produce bad answers — that is what ``diff``/``blame`` are for. Never present a
clean result as "everything is fine".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .invalidation import propagate
from .rulepack import RulePack
from .schema import Change, EffectType, Pipeline

CRITICAL, WARNING, OK = "critical", "warning", "ok"

# Finding kinds
STALE_ARTIFACT = "stale_artifact"
MISSING_STAMP = "missing_stamp"
INCOMPLETE_STAMP = "incomplete_stamp"
MISSING_ENV = "missing_env"
VERIFIED = "verified"


@dataclass
class Finding:
    severity: str
    kind: str
    node_id: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


def determining_params(
    pipeline: Pipeline, artifact_node_id: str, pack: RulePack
) -> list[tuple[str, str]]:
    """The ``(node_id, param)`` pairs whose change would invalidate this artifact.

    Derived from the rule-pack, not hardcoded: a param qualifies when changing it
    propagates ``RECOMPUTE_REQUIRED`` onto ``artifact_node_id``.
    """
    determining: list[tuple[str, str]] = []
    for node in pipeline.nodes:
        for param in node.params:
            # Values are irrelevant to rule lookup; this is a probe, not a change.
            probe = Change(node.id, node.type, param, None, None)
            for effect in propagate(pipeline, probe, pack):
                if (effect.node_id == artifact_node_id
                        and effect.effect is EffectType.RECOMPUTE_REQUIRED):
                    determining.append((node.id, param))
                    break
    return determining


def _check_artifact(
    pipeline: Pipeline, node, stamp: dict | None, pack: RulePack
) -> list[Finding]:
    artifact = node.produces_artifact
    determining = determining_params(pipeline, node.id, pack)

    if stamp is None:
        return [Finding(
            WARNING, MISSING_STAMP, node.id,
            f"no build stamp for '{artifact}' — cannot verify it matches the config",
            {"artifact": artifact,
             "expected_keys": [f"{n}.{p}" for n, p in determining]},
        )]

    recorded = stamp.get("params", {})
    drift, unverified = [], []
    for node_id, param in determining:
        key = f"{node_id}.{param}"
        declared = pipeline.node(node_id).params.get(param)
        if key not in recorded:
            unverified.append(key)
        elif recorded[key] != declared:
            drift.append({"param": key, "built_with": recorded[key],
                          "declared": declared})

    findings: list[Finding] = []
    if drift:
        summary = ", ".join(
            f"{d['param']} built with {d['built_with']!r} but config says "
            f"{d['declared']!r}" for d in drift
        )
        findings.append(Finding(
            CRITICAL, STALE_ARTIFACT, node.id,
            f"'{artifact}' is STALE — {summary}. Rebuild it; until then results "
            f"are silently wrong.",
            {"artifact": artifact, "drift": drift,
             "built_at": stamp.get("built_at")},
        ))
    if unverified:
        findings.append(Finding(
            WARNING, INCOMPLETE_STAMP, node.id,
            f"stamp for '{artifact}' does not record {', '.join(unverified)} — "
            f"those cannot be verified",
            {"artifact": artifact, "missing_keys": unverified},
        ))
    if not findings:
        findings.append(Finding(
            OK, VERIFIED, node.id,
            f"'{artifact}' matches the config on all "
            f"{len(determining)} determining parameter(s)",
            {"artifact": artifact, "built_at": stamp.get("built_at"),
             "verified_keys": [f"{n}.{p}" for n, p in determining]},
        ))
    return findings


def check(
    pipeline: Pipeline,
    stamps: Mapping[str, dict | None],
    env: Mapping[str, str],
    pack: RulePack,
) -> list[Finding]:
    """All consistency findings, most severe first.

    ``stamps`` maps node id -> parsed stamp (or None when unavailable).
    """
    findings: list[Finding] = []
    for node in pipeline.nodes:
        for var in node.requires_env:
            if not env.get(var):
                findings.append(Finding(
                    CRITICAL, MISSING_ENV, node.id,
                    f"required environment variable {var} is not set",
                    {"variable": var},
                ))
        if node.produces_artifact:
            findings.extend(_check_artifact(pipeline, node, stamps.get(node.id), pack))

    order = {CRITICAL: 0, WARNING: 1, OK: 2}
    findings.sort(key=lambda f: order[f.severity])
    return findings
