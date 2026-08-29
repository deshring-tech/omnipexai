"""Change-set bisection: turn "here are the suspects" into "here is the culprit".

A commit changes K things; a metric regresses. Instead of listing every change
whose blast radius *could* reach the metric (suspects), we isolate each change —
apply it alone to the baseline — re-run the metric, and rank changes by their
*isolated* real effect. The one with the largest real negative isolated delta is
the culprit; changes that turn out flat are exonerated.

Why the graph matters here (this is the whole point):
    Blind bisection costs |changes| x |metrics| eval runs. The invalidation graph
    prunes that to only the (change, metric) pairs where the change can actually
    affect the metric — so attribution stays cheap enough to run on every merge.

Deterministic metrics (latency / cost / token-usage) are intentionally NOT
bisected here: they can be predicted directly from the rule pack without running
anything. That hook lives in the rule pack, not this module.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..eval.runner import SIGNALS, run_eval
from ..eval.stats import DeltaResult, control_fdr, paired_bootstrap
from .invalidation import propagate
from .rulepack import RulePack
from .schema import Change, EffectType, Pipeline, diff_pipelines


@dataclass
class IsolatedEffect:
    change: Change
    result: DeltaResult  # baseline vs (baseline + this change alone)

    @property
    def is_culprit_candidate(self) -> bool:
        return self.result.real and self.result.delta < 0


@dataclass
class MetricAttribution:
    metric: str
    node_id: str
    total: DeltaResult                       # baseline vs full candidate
    isolated: list[IsolatedEffect]           # ranked, worst real regression first
    culprit: Change | None
    residual: float                          # total delta not explained by isolated sum


@dataclass
class AttributionReport:
    """Run accounting is split by CAUSE, not lumped together.

    Bisecting blindly costs |changes| x |metrics| runs. Two different things
    reduce that, and conflating them would overstate what the dependency graph
    contributes:

      * ``skipped_clean_metric`` — metrics that did not regress are not bisected
        at all. Cheap, obvious, and nothing to do with the graph.
      * ``skipped_by_graph`` — (change, metric) pairs the invalidation graph
        proves cannot interact. THIS is the graph's contribution.
    """

    metrics: list[MetricAttribution] = field(default_factory=list)
    naive_runs: int = 0            # |changes| x |metrics| (blind bisection)
    graph_runs: int = 0            # runs actually performed
    skipped_clean_metric: int = 0  # not bisected: the metric never regressed
    skipped_by_graph: int = 0      # pruned: change provably cannot affect metric

    @property
    def saved_runs(self) -> int:
        return self.naive_runs - self.graph_runs


def _apply(base: Pipeline, changes: list[Change]) -> Pipeline:
    """Clone ``base`` and apply a subset of changes (one-hot from baseline)."""
    p = copy.deepcopy(base)
    for c in changes:
        p.node(c.node_id).params[c.param] = c.new
    return p


def _metric_nodes(pipeline: Pipeline) -> dict[str, str]:
    """eval_id -> node_id it is bound to."""
    out: dict[str, str] = {}
    for n in pipeline.nodes:
        for eval_id in n.evals:
            out[eval_id] = n.id
    return out


def _affects(pipeline: Pipeline, change: Change, node_id: str, pack: RulePack) -> bool:
    """Does ``change``'s blast radius put ``node_id`` at behavioural risk?"""
    return any(
        e.node_id == node_id and e.effect is EffectType.BEHAVIOR_MAY_CHANGE
        for e in propagate(pipeline, change, pack)
    )


def attribute(
    base: Pipeline, cand: Pipeline, pack: RulePack, n_items: int = 100
) -> AttributionReport:
    changes = diff_pipelines(base, cand)
    metric_nodes = _metric_nodes(cand)
    report = AttributionReport()

    # Blind bisection would run every change against every metric.
    report.naive_runs = len(changes) * len(metric_nodes)

    seed = 100  # distinct run seed per isolated config -> independent noise
    for metric, node_id in metric_nodes.items():
        if metric not in SIGNALS:
            continue
        base_scores = run_eval(metric, base, n_items=n_items, run_seed=1)
        cand_scores = run_eval(metric, cand, n_items=n_items, run_seed=2)
        total = paired_bootstrap(metric, base_scores, cand_scores)

        # Only bother attributing genuine regressions.
        if not (total.real and total.delta < 0):
            report.skipped_clean_metric += len(changes)
            continue

        # Graph pruning: only changes that can actually reach this metric.
        suspects = [c for c in changes if _affects(cand, c, node_id, pack)]
        report.skipped_by_graph += len(changes) - len(suspects)

        isolated: list[IsolatedEffect] = []
        for c in suspects:
            iso_scores = run_eval(
                metric, _apply(base, [c]), n_items=n_items, run_seed=seed
            )
            seed += 1
            report.graph_runs += 1
            res = paired_bootstrap(metric, base_scores, iso_scores)
            isolated.append(IsolatedEffect(c, res))

        # Each metric's suspects form one family of tests: control the false
        # discovery rate across them before naming anyone the culprit.
        control_fdr([e.result for e in isolated])

        # Rank: real regressions first (most negative), then the rest.
        isolated.sort(key=lambda e: (not e.is_culprit_candidate, e.result.delta))
        culprit = next((e.change for e in isolated if e.is_culprit_candidate), None)
        residual = total.delta - sum(e.result.delta for e in isolated)

        report.metrics.append(
            MetricAttribution(metric, node_id, total, isolated, culprit, residual)
        )

    return report
