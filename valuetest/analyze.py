"""Aggregate session results into the numbers that decide the hypothesis."""
from __future__ import annotations

import json
import os
from collections import defaultdict

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "sessions.jsonl")


def load(path: str = RESULTS_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _adjusted(r: dict, penalty: float | None) -> float:
    """Adjusted time; if ``penalty`` is given, recompute post-hoc from the raw
    wall time and re-run count (so the CI cost can be set from a measured
    number after sessions were recorded)."""
    if penalty is None:
        return r["adjusted_seconds"]
    return r["wall_seconds"] + r["isolation_runs"] * penalty


def _summary(rows: list[dict], penalty: float | None = None) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    return {
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "mean_adjusted_s": sum(_adjusted(r, penalty) for r in rows) / n,
        "mean_wall_s": sum(r["wall_seconds"] for r in rows) / n,
        "mean_confidence": sum(r["confidence"] for r in rows) / n,
        "mean_runs": sum(r["isolation_runs"] for r in rows) / n,
    }


def report(path: str = RESULTS_PATH, penalty: float | None = None) -> None:
    rows = load(path)
    if not rows:
        print("no results yet — run some sessions first "
              "(python -m valuetest.cli run --participant <name>)")
        return

    arms = defaultdict(list)
    for r in rows:
        arms[r["arm"]].append(r)

    print(f"\nsessions: {len(rows)}  "
          f"participants: {len({r['participant'] for r in rows})}")
    if penalty is not None:
        print(f"(adjusted time recomputed post-hoc with measured CI penalty = "
              f"{penalty:.0f}s per re-run)")
    print()
    header = f"{'arm':<12}{'n':>4}{'accuracy':>10}{'time(s)':>10}{'confidence':>12}{'re-runs':>9}"
    print(header)
    print("-" * len(header))
    for arm in ("control", "treatment"):
        s = _summary(arms.get(arm, []), penalty)
        if s["n"] == 0:
            continue
        print(f"{arm:<12}{s['n']:>4}{s['accuracy']:>9.0%}"
              f"{s['mean_adjusted_s']:>10.0f}{s['mean_confidence']:>12.1f}"
              f"{s['mean_runs']:>9.1f}")

    c, t = (_summary(arms.get("control", []), penalty),
            _summary(arms.get("treatment", []), penalty))
    if c["n"] and t["n"]:
        print("\ndelta (treatment vs control):")
        speedup = c["mean_adjusted_s"] / t["mean_adjusted_s"] if t["mean_adjusted_s"] else float("inf")
        print(f"  time-to-culprit:  {c['mean_adjusted_s']:.0f}s -> {t['mean_adjusted_s']:.0f}s "
              f"({speedup:.1f}x faster)")
        print(f"  accuracy:         {c['accuracy']:.0%} -> {t['accuracy']:.0%}")
        print(f"  confidence:       {c['mean_confidence']:.1f} -> {t['mean_confidence']:.1f}")

    # Where the value concentrates — by difficulty.
    print("\nby difficulty (adjusted time, control vs treatment):")
    for diff in ("easy", "medium", "hard"):
        cc = _summary([r for r in arms.get("control", []) if r["difficulty"] == diff], penalty)
        tt = _summary([r for r in arms.get("treatment", []) if r["difficulty"] == diff], penalty)
        if cc["n"] and tt["n"]:
            print(f"  {diff:<8} {cc['mean_adjusted_s']:>6.0f}s -> {tt['mean_adjusted_s']:>6.0f}s "
                  f"| accuracy {cc['accuracy']:.0%} -> {tt['accuracy']:.0%}")

    print("\nDecision rule (see PROTOCOL.md): validated if treatment is meaningfully "
          "faster AND at least as accurate, with the gap widening on hard scenarios.")
