"""Service layer: every omnipex capability as a function returning a
JSON-serializable dict.

Purpose: single source of truth for all interfaces. The CLI formats these dicts
for the terminal; the GUI server sends them to the browser; the MCP server
returns them to Claude Code / Cursor. No interface may compute results itself.

Responsibilities: orchestrate manifest loading, invalidation analysis, eval
re-runs, and attribution into plain-dict reports. No printing, no I/O beyond
manifest reads. Raises ``OmnipexError`` subtypes for all expected failures.
"""
from __future__ import annotations

import os

from . import manifest
from . import rulepacks  # noqa: F401  (registers built-in packs)
from .core import attribution, consistency, invalidation
from .core.rulepack import get
from .core.schema import Change
from .eval.runner import run_eval
from .eval.stats import control_fdr, paired_bootstrap

# Verdict labels shared by every interface.
REGRESSION, IMPROVED, NOISE = "REGRESSION", "IMPROVED", "noise"
INSUFFICIENT = "INSUFFICIENT"


def _change_dict(c: Change) -> dict:
    return {"node_id": c.node_id, "param": c.param, "old": c.old, "new": c.new,
            "label": str(c)}


def show(path: str) -> dict:
    """Pipeline structure for display."""
    p = manifest.load(path)
    return {
        "name": p.name,
        "nodes": [
            {"id": n.id, "type": n.type, "params": n.params,
             "produces_artifact": n.produces_artifact, "evals": n.evals}
            for n in p.nodes
        ],
        "edges": [list(e) for e in p.edges],
    }


def diff(base_path: str, cand_path: str, pack_name: str = "rag") -> dict:
    """Full change-impact analysis: changes, blast radius, stale-eval re-runs
    with real-vs-noise verdicts, and the CI gate verdict."""
    base = manifest.load(base_path)
    cand = manifest.load(cand_path)
    pack = get(pack_name)
    report = invalidation.analyze(base, cand, pack)

    safe, seen = [], set()
    for eff in report.confirmed_safe:
        if eff.node_id not in seen:
            seen.add(eff.node_id)
            safe.append({"node_id": eff.node_id, "reason": eff.reason})

    # Every metric in one analysis is one family of tests: score them all, then
    # apply family-wise control BEFORE reading off any verdict. Doing it per
    # metric would inflate false alarms with each extra metric.
    scored = [
        (eval_id, node_id, causes,
         paired_bootstrap(eval_id,
                          run_eval(eval_id, base, run_seed=1),
                          run_eval(eval_id, cand, run_seed=2)))
        for eval_id, (node_id, causes) in report.stale_evals.items()
    ]
    control_fdr([res for *_, res in scored])

    evals, regressions, improvements = [], 0, 0
    for eval_id, node_id, causes, res in scored:
        if res.real and res.delta < 0:
            verdict = REGRESSION
            regressions += 1
        elif res.real:
            verdict = IMPROVED
            improvements += 1
        else:
            verdict = INSUFFICIENT if res.verdict == "INSUFFICIENT" else NOISE
        evals.append({
            "metric": eval_id, "node_id": node_id,
            "baseline_mean": res.baseline_mean,
            "candidate_mean": res.candidate_mean,
            "delta": res.delta, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "verdict": verdict, "n_items": res.n_items,
            "causes": [str(c) for c in causes] if res.real else [],
        })

    return {
        "pipeline": cand.name,
        "changes": [_change_dict(c) for c in report.changes],
        "effects": [
            {"node_id": e.node_id, "node_type": e.node_type,
             "effect": e.effect.value, "reason": e.reason,
             "cause": str(e.cause)}
            for e in report.effects
        ],
        "recompute": dict(report.recompute),
        "confirmed_safe": safe,
        "evals": evals,
        "verdict": {
            "status": "block" if regressions else "ok",
            "regressions": regressions,
            "improvements": improvements,
        },
    }


def doctor(path: str, pack_name: str = "rag", env: dict | None = None) -> dict:
    """Consistency check: does the declared config match what was actually built?

    Fast and free — no eval runs, no judge tokens. Stamp paths are resolved
    relative to the manifest's own directory.
    """
    pipeline = manifest.load(path)
    pack = get(pack_name)
    base_dir = os.path.dirname(os.path.abspath(path))

    stamps: dict[str, dict | None] = {}
    for node in pipeline.nodes:
        if node.produces_artifact and node.stamp:
            stamps[node.id] = manifest.load_stamp(
                os.path.join(base_dir, node.stamp)
            )

    findings = consistency.check(
        pipeline, stamps, os.environ if env is None else env, pack
    )
    counts = {level: 0 for level in (consistency.CRITICAL, consistency.WARNING,
                                     consistency.OK)}
    for finding in findings:
        counts[finding.severity] += 1

    if counts[consistency.CRITICAL]:
        status = "inconsistent"
    elif counts[consistency.WARNING]:
        status = "unverified"
    else:
        status = "consistent"

    return {
        "pipeline": pipeline.name,
        "findings": [
            {"severity": f.severity, "kind": f.kind, "node_id": f.node_id,
             "message": f.message, "detail": f.detail}
            for f in findings
        ],
        "counts": counts,
        "status": status,
        # Consistency is not quality: a consistent pipeline can still answer badly.
        "note": "Consistency only means config and artifacts agree. It does not "
                "mean output quality is good — use `diff` for that.",
    }


def blame(base_path: str, cand_path: str, pack_name: str = "rag") -> dict:
    """Bisection attribution: per regressed metric, the isolated effect of each
    at-risk change and the named culprit (or interaction fallback)."""
    base = manifest.load(base_path)
    cand = manifest.load(cand_path)
    pack = get(pack_name)
    report = attribution.attribute(base, cand, pack)

    metrics = []
    for ma in report.metrics:
        isolated = []
        for eff in ma.isolated:
            r = eff.result
            if eff.is_culprit_candidate:
                verdict = "regression"
            elif r.real and r.delta > 0:
                verdict = "helps"
            else:
                verdict = "none"
            isolated.append({
                "change": str(eff.change), "delta": r.delta,
                "ci_low": r.ci_low, "ci_high": r.ci_high, "verdict": verdict,
            })
        masked = [
            str(e.change) for e in ma.isolated
            if e.result.real and e.result.delta > 0
        ] if ma.culprit is not None else []
        metrics.append({
            "metric": ma.metric, "node_id": ma.node_id,
            "total": {"delta": ma.total.delta, "ci_low": ma.total.ci_low,
                      "ci_high": ma.total.ci_high},
            "isolated": isolated,
            "culprit": str(ma.culprit) if ma.culprit is not None else None,
            "masked_by": masked,
            "residual": ma.residual,
        })

    return {
        "metrics": metrics,
        "naive_runs": report.naive_runs,
        "graph_runs": report.graph_runs,
        "saved_runs": report.saved_runs,
        # Split by cause so the graph is never credited with savings that came
        # from simply not bisecting a metric that never regressed.
        "skipped_clean_metric": report.skipped_clean_metric,
        "skipped_by_graph": report.skipped_by_graph,
    }
