"""Value-test CLI.

    python -m valuetest.cli preview [scenario_id] [--arm treatment|control]
    python -m valuetest.cli run --participant <name> [--penalty 180]
    python -m valuetest.cli analyze
    python -m valuetest.cli check      # validate scenarios against the engine

`preview` is non-interactive (for facilitator prep and CI); `run` drives a live
counterbalanced session with a participant.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

from . import analyze, harness, scenarios


def _fnv(s: str) -> int:
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def cmd_check(_args) -> None:
    problems = scenarios.validate_all()
    if problems:
        print("SCENARIO MISMATCHES:")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print(f"all {len(scenarios.SCENARIOS)} scenarios consistent with the engine")


def cmd_preview(args) -> None:
    ids = [args.scenario_id] if args.scenario_id else [s.id for s in scenarios.SCENARIOS]
    for sid in ids:
        s = scenarios.get_scenario(sid)
        print("\n" + "=" * 70)
        print(f"{s.id}  [{s.difficulty}]  {s.title}")
        print(f"facilitator note: {s.note}")
        print(f"ground-truth culprit: {s.expected or 'noise / nothing'}")
        if args.arm == "treatment":
            harness._render_blame(s)
        else:
            base, cand = scenarios.base_pipeline(), s.candidate()
            harness._show_changes(s.change_objects())
            harness._show_before_after(s, base, cand)


def _assign_arm(participant: str, index: int) -> str:
    """Counterbalanced: alternate arms across scenarios, phase set by participant."""
    parity = _fnv(participant) & 1
    return "control" if (parity + index) % 2 == 0 else "treatment"


def cmd_run(args) -> None:
    order = list(scenarios.SCENARIOS)
    random.Random(_fnv(args.participant)).shuffle(order)

    os.makedirs(os.path.dirname(analyze.RESULTS_PATH), exist_ok=True)
    print(f"\nDebugging-workflow study — participant: {args.participant}")
    print("You'll work through several scenarios using one of two workflows (A or B, "
          "assigned per scenario). Work as you naturally would; think aloud if you "
          "can. There's no time pressure beyond doing it realistically.\n")

    with open(analyze.RESULTS_PATH, "a", encoding="utf-8") as fh:
        for i, scenario in enumerate(order):
            arm = args.arm if args.arm != "auto" else _assign_arm(args.participant, i)
            print("\n" + "#" * 70)
            print(f"# scenario {i + 1}/{len(order)}")
            result = harness.run_scenario(args.participant, scenario, arm, args.penalty)
            fh.write(json.dumps(harness.result_dict(result)) + "\n")
            fh.flush()
            # Deliberately no correctness/arm feedback mid-session: telling the
            # participant they were right or wrong would bias later scenarios.
            print("  recorded.")

    print("\nSession complete. Thank you!")
    print("(facilitator: `python -m valuetest.cli analyze` for results — after the "
          "participant is done, not on a shared screen)")


def cmd_analyze(args) -> None:
    analyze.report(args.path, penalty=args.penalty)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(prog="valuetest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preview", help="non-interactive preview of scenarios")
    p.add_argument("scenario_id", nargs="?")
    p.add_argument("--arm", choices=["control", "treatment"], default="treatment")
    p.set_defaults(func=cmd_preview)

    r = sub.add_parser("run", help="drive a live participant session")
    r.add_argument("--participant", required=True)
    r.add_argument("--arm", choices=["auto", "control", "treatment"], default="auto")
    r.add_argument("--penalty", type=float, default=harness.DEFAULT_CI_PENALTY_S,
                   help="simulated CI seconds per manual re-run (control arm)")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="aggregate recorded results")
    a.add_argument("--path", default=analyze.RESULTS_PATH)
    a.add_argument("--penalty", type=float, default=None,
                   help="recompute adjusted time post-hoc with a measured CI "
                        "seconds-per-re-run (overrides the value used at record time)")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("check", help="validate scenarios against the engine")
    c.set_defaults(func=cmd_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
