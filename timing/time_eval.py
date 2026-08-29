"""Standalone DeepEval timing rig — measures the real wall-clock cost of one
evaluation run, to set the value-test CI penalty from data instead of a guess.

DELIBERATELY independent of the omnipex package: this is a stopwatch, not a
product integration. (The live omnipex<->DeepEval backend is a separate, later
milestone, gated on the value test passing.)

Usage:
    pip install deepeval           # and set OPENAI_API_KEY for the judge
    python timing/time_eval.py --dry-run                 # sanity check, no API calls
    python timing/time_eval.py                           # full timed run
    python timing/time_eval.py --judge gpt-4o --items 40 # vary assumptions

Each run appends a fully transparent record (judge model, items, metrics,
wall-clock, host) to timing/results.jsonl. Report the derived penalty together
with those assumptions, never as a bare number. The suggested value-test penalty
is the TOTAL wall time of the run: in the control arm, isolating one change
means re-running the whole suite once.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(HERE, "sample_dataset.json")
RESULTS = os.path.join(HERE, "results.jsonl")

# The four quality metrics mirrored from the value-test scenarios.
METRIC_NAMES = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]


def load_items(path: str, n_items: int | None) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"]
    required = {"input", "actual_output", "expected_output", "retrieval_context"}
    for i, item in enumerate(items):
        missing = required - set(item)
        if missing:
            raise SystemExit(f"dataset item {i} is missing fields: {sorted(missing)}")
    if n_items is not None:
        # repeat the set if the user asks for more items than the file has
        while len(items) < n_items:
            items = items + items
        items = items[:n_items]
    return items


def build(items: list[dict], judge: str):
    """Construct DeepEval test cases + metrics. Kept in one function so an API
    change in deepeval only ever requires editing here."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError:
        raise SystemExit(
            "deepeval is not installed. Run:  pip install deepeval\n"
            "(and set OPENAI_API_KEY, or configure a custom judge model — see "
            "https://docs.confident-ai.com/)"
        )

    test_cases = [
        LLMTestCase(
            input=i["input"],
            actual_output=i["actual_output"],
            expected_output=i["expected_output"],
            retrieval_context=i["retrieval_context"],
        )
        for i in items
    ]
    metrics = [
        FaithfulnessMetric(model=judge, include_reason=False),
        AnswerRelevancyMetric(model=judge, include_reason=False),
        ContextualPrecisionMetric(model=judge, include_reason=False),
        ContextualRecallMetric(model=judge, include_reason=False),
    ]
    return test_cases, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--judge", default="gpt-4o-mini",
                        help="judge model for DeepEval metrics (default gpt-4o-mini)")
    parser.add_argument("--items", type=int, default=None,
                        help="number of eval items (default: whole dataset file)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate dataset + construction, make no API calls")
    args = parser.parse_args()

    items = load_items(args.dataset, args.items)
    print(f"dataset: {args.dataset}  ({len(items)} items)")
    print(f"metrics: {', '.join(METRIC_NAMES)}  |  judge: {args.judge}")

    if args.dry_run:
        try:
            build(items, args.judge)
            print("dry-run OK: deepeval installed, test cases and metrics construct cleanly.")
        except SystemExit as exc:
            print(f"dry-run (partial): dataset OK; {exc}")
        print(f"planned judged (item, metric) pairs: {len(items) * len(METRIC_NAMES)}")
        return

    if not os.environ.get("OPENAI_API_KEY") and "gpt" in args.judge.lower():
        raise SystemExit("OPENAI_API_KEY is not set (required for a GPT judge).")

    test_cases, metrics = build(items, args.judge)
    from deepeval import evaluate  # noqa: PLC0415  (import after guard)

    print(f"\ntiming one full evaluation run "
          f"({len(items)} items x {len(metrics)} metrics)...")
    t0 = time.perf_counter()
    evaluate(test_cases=test_cases, metrics=metrics)
    wall = time.perf_counter() - t0

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "judge": args.judge,
        "n_items": len(items),
        "metrics": METRIC_NAMES,
        "wall_seconds": round(wall, 1),
        "seconds_per_item": round(wall / len(items), 2),
        "host": platform.node(),
        "platform": platform.platform(),
        "dataset": os.path.basename(args.dataset),
    }
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print(f"\nwall-clock: {wall:.1f}s total, {wall / len(items):.2f}s/item")
    print(f"recorded -> {RESULTS}")
    print(f"\nsuggested value-test penalty (one suite re-run): --penalty {wall:.0f}")
    print("report it with its assumptions: "
          f"\"{wall:.0f}s for {len(items)} items x {len(metrics)} metrics, judge={args.judge}\"")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    main()
