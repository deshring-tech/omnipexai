"""Attribution under noise (problem H3).

The hard part of "regression CI for AI" is not running an eval — it is deciding
whether an observed score change is REAL or just LLM/sampling noise. A tool that
cries wolf is worse than no tool. This module does a paired bootstrap on per-item
score differences and only calls a delta REAL when the evidence survives.

Two separate safeguards, because a single 95% interval is not enough:

1. **Per-test interval** — the bootstrap CI of the paired mean difference.
2. **Family-wise control** (``control_fdr``) — one analysis tests many metrics
   at once, and testing k metrics at alpha=0.05 each means a ~1-(0.95)^k chance
   of at least one false alarm (measured at 11.3% for k=3 before this was added).
   Benjamini-Hochberg keeps the *false discovery rate* across the family at q,
   so "OmniPex flagged something" stays trustworthy.

Degenerate inputs are refused rather than answered confidently: with very few
paired items the bootstrap distribution collapses and *any* difference looks
certain, which is precisely how a tool loses trust.

Pure stdlib on purpose: this logic must run anywhere, and it is real even when
the score *source* is simulated (see eval/runner.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Below this many paired items a bootstrap CI is not meaningful: resamples repeat
# the same handful of values, the interval collapses, and noise reads as signal.
MIN_PAIRED_ITEMS = 8

REAL, NOISE, INSUFFICIENT = "REAL", "NOISE", "INSUFFICIENT"


@dataclass
class DeltaResult:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str          # REAL | NOISE | INSUFFICIENT
    p_value: float = 1.0  # two-sided bootstrap p, used for family-wise control
    n_items: int = 0

    @property
    def real(self) -> bool:
        """True only for a verdict that survived every check. INSUFFICIENT is
        deliberately not real: unknown must never be reported as a finding."""
        return self.verdict == REAL


def paired_bootstrap(
    metric: str,
    baseline: list[float],
    candidate: list[float],
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> DeltaResult:
    """Bootstrap CI of the paired mean difference (candidate - baseline).

    ``baseline`` and ``candidate`` must be aligned: index i is the same eval item
    scored under each config (common-random-numbers pairing cancels shared noise).
    """
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be the same length (paired)")
    n = len(baseline)
    if n == 0:
        raise ValueError(f"{metric}: cannot compare empty score lists")

    diffs = [c - b for b, c in zip(baseline, candidate)]
    delta = sum(diffs) / n
    base_mean, cand_mean = sum(baseline) / n, sum(candidate) / n

    if n < MIN_PAIRED_ITEMS:
        # Report honestly instead of producing a confident, meaningless interval.
        return DeltaResult(metric, base_mean, cand_mean, delta, delta, delta,
                           INSUFFICIENT, 1.0, n)

    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()

    # Clamp so extreme alpha values cannot index past the ends.
    lo_idx = min(max(int((alpha / 2) * n_boot), 0), n_boot - 1)
    hi_idx = min(max(int((1 - alpha / 2) * n_boot), 0), n_boot - 1)
    lo, hi = boot_means[lo_idx], boot_means[hi_idx]

    # Two-sided bootstrap p: how much of the resampled mass sits on the far side
    # of zero from the observed effect.
    at_or_below = sum(1 for m in boot_means if m <= 0)
    at_or_above = sum(1 for m in boot_means if m >= 0)
    p_value = min(1.0, 2.0 * min(at_or_below, at_or_above) / n_boot)

    verdict = REAL if (lo > 0 or hi < 0) else NOISE
    return DeltaResult(metric, base_mean, cand_mean, delta, lo, hi,
                       verdict, p_value, n)


def control_fdr(results: list[DeltaResult], q: float = 0.05) -> list[DeltaResult]:
    """Benjamini-Hochberg across one family of tests, in place.

    Every metric examined in a single analysis belongs to one family. Without
    this, more metrics mean more spurious findings — the failure mode that makes
    a regression detector untrustworthy. Results that do not survive are demoted
    REAL -> NOISE; nothing is ever promoted.

    The family is EVERY test performed, not just the ones that already look
    significant. Correcting only the significant subset is a no-op, because those
    all have p <= alpha by construction; the non-significant tests are what make
    the multiplicity penalty real.

    Returns the same list for convenience.
    """
    # INSUFFICIENT tests produced no evidence, so they are not part of the family.
    testable = [r for r in results if r.verdict != INSUFFICIENT]
    m = len(testable)
    if m <= 1:
        return results  # a single test needs no multiplicity correction

    ordered = sorted(testable, key=lambda r: r.p_value)
    survivors = 0
    for rank, result in enumerate(ordered, start=1):
        if result.p_value <= (rank / m) * q:
            survivors = rank  # BH: the largest passing rank carries all below it
    for result in ordered[survivors:]:
        if result.verdict == REAL:
            result.verdict = NOISE
    return results
