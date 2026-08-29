"""Interactive session harness.

Two arms:

  CONTROL (today's workflow): the participant sees the commit's changes and the
    before/after metric numbers, then investigates by *manually* isolating one
    change at a time (revert-and-rerun). They see raw means only — no
    significance verdict — exactly like eyeballing a CI eval today. Each manual
    re-run costs a configurable simulated CI-time penalty.

  TREATMENT (Omnipex): the participant sees the `blame` attribution — the named
    culprit, the isolated effects with REAL/NOISE verdicts, and any masking — and
    just reads off the answer.

We record time-to-culprit (wall clock until they commit an answer), number of
manual re-runs, an adjusted time that adds the simulated CI cost, plus accuracy
and self-reported confidence.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from omnipex.core import attribution
from omnipex.core.schema import Change, Pipeline
from omnipex.eval.runner import SIGNALS, run_eval

from .scenarios import RAG, Scenario, base_pipeline, change_key

DEFAULT_CI_PENALTY_S = 180  # a real isolated eval re-run costs minutes in CI


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _bound_metrics() -> list[str]:
    """Eval ids bound to nodes in the experiment pipeline."""
    base = base_pipeline()
    out = []
    for n in base.nodes:
        out.extend(e for e in n.evals if e in SIGNALS)
    return out


def _prompt_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  please enter a number in [{lo}, {hi}]")


def _prompt_answer(changes: list[Change]) -> str | None:
    """Return the chosen change key, or None for 'noise / nothing'."""
    while True:
        raw = input("  your answer — change number, or 'n' for nothing/noise: ").strip().lower()
        if raw in ("n", "none", "noise"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(changes):
            return change_key(changes[int(raw) - 1])
        print(f"  enter 1..{len(changes)} or 'n'")


@dataclass
class Result:
    participant: str
    scenario_id: str
    difficulty: str
    arm: str
    metric: str
    answer: str | None
    expected: str | None
    correct: bool
    confidence: int
    wall_seconds: float
    isolation_runs: int
    adjusted_seconds: float


def _show_changes(changes: list[Change]) -> None:
    print("\n  This commit changed:")
    for i, c in enumerate(changes, 1):
        print(f"    {i}. {c.node_id}.{c.param}: {c.old!r} -> {c.new!r}")


def _show_before_after(scenario: Scenario, base: Pipeline, cand: Pipeline) -> None:
    print("\n  Eval results (mean over the test set):")
    print(f"    {'metric':<20}{'before':>9}{'after':>9}{'Δ':>9}")
    for m in _bound_metrics():
        b = _mean(run_eval(m, base, run_seed=1))
        a = _mean(run_eval(m, cand, run_seed=2))
        star = "   <-- investigating" if m == scenario.metric else ""
        print(f"    {m:<20}{b:>9.3f}{a:>9.3f}{a - b:>+9.3f}{star}")


def run_control(scenario: Scenario, penalty: float) -> tuple[str | None, int, float]:
    base = base_pipeline()
    cand = scenario.candidate()
    changes = scenario.change_objects()

    t0 = time.perf_counter()
    print("\n=== WORKFLOW A (manual isolation) ===")
    print(f"Task: after this commit, `{scenario.metric}` looks different. "
          f"Find what caused it (or decide it's noise).")
    _show_changes(changes)
    _show_before_after(scenario, base, cand)
    print("\n  To investigate you can revert ONE change and re-run the evals.")
    print(f"  (Each re-run would take ~{int(penalty)}s in real CI.)")

    runs = 0
    while True:
        print(f"\n  [1..{len(changes)}] isolate that change & re-run   [s] submit answer")
        raw = input("  > ").strip().lower()
        if raw == "s":
            break
        if raw.isdigit() and 1 <= int(raw) <= len(changes):
            c = changes[int(raw) - 1]
            iso = base_pipeline()
            iso.node(c.node_id).params[c.param] = c.new
            runs += 1
            print(f"    re-running with only [{c.node_id}.{c.param}] applied "
                  f"(+{int(penalty)}s CI):")
            for m in _bound_metrics():
                b = _mean(run_eval(m, base, run_seed=1))
                a = _mean(run_eval(m, iso, run_seed=200 + runs))
                print(f"      {m:<20}{b:>9.3f} -> {a:>9.3f}  (Δ{a - b:+.3f})")
        else:
            print("  unrecognised; enter a change number or 's'")

    answer = _prompt_answer(changes)
    wall = time.perf_counter() - t0
    return answer, runs, wall


def _render_blame(scenario: Scenario) -> None:
    base = base_pipeline()
    cand = scenario.candidate()
    report = attribution.attribute(base, cand, RAG)
    ma = next((m for m in report.metrics if m.metric == scenario.metric), None)

    print("\n=== ATTRIBUTION REPORT ===")
    if ma is None:
        print(f"  `{scenario.metric}`: no real regression detected — the change is "
              f"within the noise band. Nothing to attribute.")
        return
    print(f"  `{scenario.metric}` regressed (Δ{ma.total.delta:+.3f}, "
          f"95% CI [{ma.total.ci_low:+.3f}, {ma.total.ci_high:+.3f}]).")
    print("  isolated effect of each at-risk change:")
    for eff in ma.isolated:
        r = eff.result
        if eff.is_culprit_candidate:
            tag = "REGRESSION"
        elif r.real and r.delta > 0:
            tag = "helps"
        else:
            tag = "no effect"
        print(f"    {eff.change.node_id}.{eff.change.param:<16} "
              f"Δ{r.delta:+.3f}  [{tag}]")
    if ma.culprit is not None:
        print(f"  => culprit: {ma.culprit.node_id}.{ma.culprit.param}")
        masked = [e for e in ma.isolated if e.result.real and e.result.delta > 0]
        if masked:
            names = ", ".join(f"{e.change.node_id}.{e.change.param}" for e in masked)
            print(f"     (partially masked by an improvement from: {names})")
    else:
        print("  => no single change explains it — likely an interaction.")


def run_treatment(scenario: Scenario) -> tuple[str | None, int, float]:
    changes = scenario.change_objects()
    t0 = time.perf_counter()
    print("\n=== WORKFLOW B (attribution report) ===")
    print(f"Task: after this commit, `{scenario.metric}` looks different. "
          f"Find what caused it (or decide it's noise).")
    _show_changes(changes)
    _render_blame(scenario)
    answer = _prompt_answer(changes)
    wall = time.perf_counter() - t0
    return answer, 0, wall


def run_scenario(participant: str, scenario: Scenario, arm: str,
                 penalty: float = DEFAULT_CI_PENALTY_S) -> Result:
    if arm == "control":
        answer, runs, wall = run_control(scenario, penalty)
    else:
        answer, runs, wall = run_treatment(scenario)
    adjusted = wall + runs * penalty  # runs == 0 for treatment
    confidence = _prompt_int("  confidence this is correct (1=guess, 5=certain): ", 1, 5)

    expected = scenario.expected
    correct = (answer == expected)
    return Result(
        participant=participant, scenario_id=scenario.id,
        difficulty=scenario.difficulty, arm=arm, metric=scenario.metric,
        answer=answer, expected=expected, correct=correct, confidence=confidence,
        wall_seconds=round(wall, 1), isolation_runs=runs,
        adjusted_seconds=round(adjusted, 1),
    )


def result_dict(r: Result) -> dict:
    return asdict(r)
