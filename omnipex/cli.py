"""omnipex — regression CI for AI pipelines.

    omnipex show   <manifest.json>
    omnipex doctor <manifest.json> [--pack rag]
    omnipex diff   <base.json> <candidate.json> [--pack rag]
    omnipex blame  <base.json> <candidate.json> [--pack rag]
    omnipex ui     [--port 7327] [--dir .] [--no-browser]
    omnipex mcp    [--dir .]                 # MCP server for Claude Code/Cursor

`doctor` is the instant check: does the declared config match what was actually
built? (no eval runs, no cost). `diff` is the CI gate: blast radius, stale-eval
re-runs, real-vs-noise. `blame` is the diagnosis: bisect the diff and name the
culprit change. `ui` opens the local GUI (same engine, browser view).

All results come from ``omnipex.api``; this module only formats them for the
terminal.

Exit codes (CI semantics):
    0  clean — no real regression / no inconsistency detected
    1  gate failure — `diff` found a real regression, or `doctor` found a
       critical inconsistency
    2  input error — bad manifest, unknown rule-pack, unsupported change
`blame` and `ui` are diagnostic and exit 0 unless the input is invalid.
"""
from __future__ import annotations

import argparse
import sys

from . import api
from .core.errors import OmnipexError

# ANSI colours (harmless if the terminal ignores them)
_BOLD, _DIM, _RED, _GRN, _YEL, _CYN, _RST = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m",
)


def _h(text: str) -> str:
    return f"\n{_BOLD}{_CYN}{text}{_RST}"


def cmd_show(args) -> int:
    p = api.show(args.manifest)
    print(_h(f"pipeline: {p['name']}"))
    for n in p["nodes"]:
        art = f"  ->produces {n['produces_artifact']}" if n["produces_artifact"] else ""
        ev = f"  evals={n['evals']}" if n["evals"] else ""
        print(f"  {_BOLD}{n['id']}{_RST} ({n['type']}) {n['params']}{art}{ev}")
    print(_h("edges"))
    for src, dst in p["edges"]:
        print(f"  {src} -> {dst}")
    return 0


_EFFECT_TAGS = {
    "recompute_required": f"{_RED}RECOMPUTE{_RST}",
    "behavior_may_change": f"{_YEL}RE-EVAL  {_RST}",
    "no_effect": f"{_GRN}SAFE     {_RST}",
}


def cmd_diff(args) -> int:
    report = api.diff(args.base, args.candidate, args.pack)

    print(_h("1. detected changes"))
    if not report["changes"]:
        print("  (none)")
        return 0
    for c in report["changes"]:
        print(f"  {_YEL}~{_RST} {c['label']}")

    print(_h("2. blast radius"))
    for eff in report["effects"]:
        tag = _EFFECT_TAGS[eff["effect"]]
        print(f"  [{tag}] {eff['node_id']:<10} {_DIM}{eff['reason']}{_RST}")

    if report["recompute"]:
        print(_h("3. required rebuilds (stale artifacts)"))
        for node_id, artifact in report["recompute"].items():
            print(f"  {_RED}rebuild{_RST} {artifact}  {_DIM}(node {node_id}){_RST}")

    if report["confirmed_safe"]:
        print(_h("4. proven unaffected"))
        for safe in report["confirmed_safe"]:
            print(f"  {_GRN}OK{_RST} {safe['node_id']:<10} {_DIM}{safe['reason']}{_RST}")

    if not report["evals"]:
        print(_h("5. no evals bound to affected nodes — nothing to re-run"))
        return 0

    print(_h("5. re-running stale evals (paired, noise-band attribution)"))
    for ev in report["evals"]:
        if ev["verdict"] == api.REGRESSION:
            verdict = f"{_RED}REGRESSION{_RST}"
        elif ev["verdict"] == api.IMPROVED:
            verdict = f"{_GRN}IMPROVED{_RST}"
        elif ev["verdict"] == api.INSUFFICIENT:
            verdict = f"{_YEL}TOO FEW ITEMS{_RST}"
        else:
            verdict = f"{_DIM}noise{_RST}"
        print(
            f"  {ev['metric']:<18} {ev['baseline_mean']:.3f} -> "
            f"{ev['candidate_mean']:.3f}  (Δ{ev['delta']:+.3f}, "
            f"95% CI [{ev['ci_low']:+.3f}, {ev['ci_high']:+.3f}])  {verdict}"
        )
        if ev["causes"]:
            print(f"      {_DIM}candidate cause(s): {', '.join(ev['causes'])}{_RST}")

    verdict = report["verdict"]
    print(_h("verdict"))
    if verdict["status"] == "block":
        print(f"  {_RED}{_BOLD}BLOCK{_RST}: {verdict['regressions']} real "
              f"regression(s) hidden in this diff.")
        return 1
    if verdict["improvements"]:
        print(f"  {_GRN}{_BOLD}OK{_RST}: only improvements are real; the rest is noise.")
    else:
        print(f"  {_GRN}{_BOLD}OK{_RST}: no real behavioural change detected.")
    return 0


_SEVERITY_TAGS = {
    "critical": f"{_RED}CRITICAL{_RST}",
    "warning": f"{_YEL}WARNING {_RST}",
    "ok": f"{_GRN}OK      {_RST}",
}


def cmd_doctor(args) -> int:
    report = api.doctor(args.manifest, args.pack)

    print(_h(f"consistency check: {report['pipeline']}"))
    for finding in report["findings"]:
        tag = _SEVERITY_TAGS[finding["severity"]]
        print(f"  [{tag}] {finding['node_id']:<10} {finding['message']}")

    counts = report["counts"]
    print(_h("verdict"))
    if report["status"] == "inconsistent":
        print(f"  {_RED}{_BOLD}INCONSISTENT{_RST}: {counts['critical']} "
              f"critical problem(s) — the pipeline is silently wrong.")
    elif report["status"] == "unverified":
        print(f"  {_YEL}{_BOLD}UNVERIFIED{_RST}: nothing broken, but "
              f"{counts['warning']} thing(s) could not be checked.")
    else:
        print(f"  {_GRN}{_BOLD}CONSISTENT{_RST}: config and artifacts agree.")
    print(f"  {_DIM}{report['note']}{_RST}")
    return 1 if report["status"] == "inconsistent" else 0


def cmd_blame(args) -> int:
    report = api.blame(args.base, args.candidate, args.pack)

    if not report["metrics"]:
        print(_h("no real regression to attribute"))
        print("  every affected metric was flat or improved.")
        return 0

    for ma in report["metrics"]:
        total = ma["total"]
        print(_h(f"regression: {ma['metric']}  (Δ{total['delta']:+.3f}, "
                 f"CI [{total['ci_low']:+.3f}, {total['ci_high']:+.3f}])"))
        print(f"  {_DIM}bisecting {len(ma['isolated'])} at-risk change(s), "
              f"isolated from baseline:{_RST}")
        for iso in ma["isolated"]:
            if iso["verdict"] == "regression":
                tag = f"{_RED}REGRESSION{_RST}"
            elif iso["verdict"] == "helps":
                tag = f"{_GRN}helps{_RST}    "
            else:
                tag = f"{_DIM}no effect{_RST}"
            print(f"    {iso['change']:<52} Δ{iso['delta']:+.3f}  [{tag}]")
        if ma["culprit"] is not None:
            print(f"  {_BOLD}{_RED}culprit:{_RST} {ma['culprit']}")
            if ma["masked_by"]:
                print(f"  {_DIM}(partially masked by an improvement from: "
                      f"{', '.join(ma['masked_by'])}){_RST}")
        else:
            print(f"  {_YEL}no single change explains it{_RST} — likely an "
                  f"interaction (residual {ma['residual']:+.3f}).")

    print(_h("efficiency"))
    print(f"  blind bisection: {report['naive_runs']} eval runs; "
          f"actually run: {report['graph_runs']}")
    print(f"    {_DIM}{report['skipped_clean_metric']} skipped (metric never "
          f"regressed) · {report['skipped_by_graph']} pruned by the "
          f"dependency graph{_RST}")
    return 0


def cmd_ui(args) -> int:
    from .ui.server import serve  # local import: keeps CLI startup instant
    serve(root=args.dir, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_mcp(args) -> int:
    from .mcp.server import serve  # local import: keeps CLI startup instant
    serve(root=args.dir)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="omnipex")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show", help="print a pipeline graph")
    s.add_argument("manifest")
    s.set_defaults(func=cmd_show)

    d = sub.add_parser("diff", help="change-impact analysis between two versions")
    d.add_argument("base")
    d.add_argument("candidate")
    d.add_argument("--pack", default="rag")
    d.set_defaults(func=cmd_diff)

    b = sub.add_parser("blame", help="bisect a diff to attribute regressions to a change")
    b.add_argument("base")
    b.add_argument("candidate")
    b.add_argument("--pack", default="rag")
    b.set_defaults(func=cmd_blame)

    doc = sub.add_parser(
        "doctor", help="check declared config against what was actually built")
    doc.add_argument("manifest")
    doc.add_argument("--pack", default="rag")
    doc.set_defaults(func=cmd_doctor)

    u = sub.add_parser("ui", help="open the local GUI (same engine, browser view)")
    u.add_argument("--port", type=int, default=7327)
    u.add_argument("--dir", default=".", help="project directory to browse for manifests")
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(func=cmd_ui)

    m = sub.add_parser(
        "mcp", help="run as an MCP server (stdio) for Claude Code / Cursor")
    m.add_argument("--dir", default=".", help="project root; paths resolve inside it")
    m.set_defaults(func=cmd_mcp)

    # Windows consoles default to cp1252 and choke on the report glyphs.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parser.parse_args()
    try:
        sys.exit(args.func(args) or 0)
    except OmnipexError as exc:
        print(f"{_RED}error:{_RST} {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
