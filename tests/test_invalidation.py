"""Invalidation-engine tests. Run without pytest:  python tests/test_invalidation.py
(or `python -m pytest` if you have it). These pin the domain knowledge that is
the actual IP, so a bad edit to the RAG matrix fails loudly.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnipex import manifest  # noqa: E402
from omnipex.core import attribution, invalidation  # noqa: E402
from omnipex.core.errors import (  # noqa: E402
    ManifestError,
    OmnipexError,
    PipelineValidationError,
    UnsupportedChangeError,
)
from omnipex.core.rulepack import get  # noqa: E402
from omnipex.core.schema import Change, EffectType, Node, Pipeline  # noqa: E402
from omnipex.eval.stats import paired_bootstrap  # noqa: E402
import omnipex.rulepacks  # noqa: E402,F401

RAG = get("rag")


def _pipeline() -> Pipeline:
    return manifest.load(
        os.path.join(os.path.dirname(__file__), "..", "examples",
                     "rag_docs_qa", "manifest.json")
    )


def test_topk_does_not_recompute_index():
    """The whole point: changing top_k must NOT invalidate the vector index."""
    p = _pipeline()
    change = Change("retriever", "retriever", "top_k", 4, 8)
    effects = invalidation.propagate(p, change, RAG)
    store_effects = {e.effect for e in effects if e.node_id == "store"}
    assert EffectType.RECOMPUTE_REQUIRED not in store_effects
    assert EffectType.NO_EFFECT in store_effects  # proven safe


def test_embedder_change_recomputes_index():
    p = _pipeline()
    change = Change("embedder", "embedder", "model", "a", "b")
    effects = invalidation.propagate(p, change, RAG)
    recompute = {e.node_id for e in effects
                 if e.effect is EffectType.RECOMPUTE_REQUIRED}
    assert "embedder" in recompute
    assert "store" in recompute


def test_prompt_change_leaves_retrieval_untouched():
    p = _pipeline()
    change = Change("prompt", "prompt", "template_hash", "x", "y")
    effects = invalidation.propagate(p, change, RAG)
    retr = {e.effect for e in effects if e.node_id == "retriever"}
    assert retr == {EffectType.NO_EFFECT} or retr == set()
    llm = {e.effect for e in effects if e.node_id == "llm"}
    assert EffectType.BEHAVIOR_MAY_CHANGE in llm


def test_downstream_only():
    """Effects never propagate upstream against the data flow."""
    p = _pipeline()
    change = Change("llm", "llm", "model", "gpt-4o", "gpt-4o-mini")
    effects = invalidation.propagate(p, change, RAG)
    behaviour_nodes = {e.node_id for e in effects
                       if e.effect is EffectType.BEHAVIOR_MAY_CHANGE}
    assert behaviour_nodes <= {"llm"}  # nothing upstream of the llm


def test_noise_band_rejects_zero_signal():
    """Identical configs must read NOISE, never REAL."""
    p = _pipeline()
    from omnipex.eval.runner import run_eval
    base = run_eval("faithfulness", p, run_seed=1)
    cand = run_eval("faithfulness", p, run_seed=2)  # same config, different noise
    res = paired_bootstrap("faithfulness", base, cand)
    assert res.verdict == "NOISE", f"false positive: {res}"


def test_analyze_end_to_end():
    base = _pipeline()
    cand = manifest.load(
        os.path.join(os.path.dirname(__file__), "..", "examples",
                     "rag_docs_qa", "manifest_v2.json")
    )
    report = invalidation.analyze(base, cand, RAG)
    assert "store" in report.recompute        # index must rebuild
    assert "faithfulness" in report.stale_evals
    assert len(report.stale_evals["faithfulness"][1]) >= 1  # >=1 candidate cause


def _v2():
    return manifest.load(
        os.path.join(os.path.dirname(__file__), "..", "examples",
                     "rag_docs_qa", "manifest_v2.json")
    )


def _expect(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


# --- input hardening -------------------------------------------------------

def test_missing_manifest_raises_manifest_error():
    _expect(ManifestError, manifest.load, "does_not_exist.json")


def test_malformed_manifest_raises_manifest_error(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('{"nodes": [{"type": "llm"}]}')  # node missing id
        _expect(ManifestError, manifest.load, bad)


def test_duplicate_node_ids_rejected():
    p = Pipeline("dup", [Node("a", "llm"), Node("a", "llm")], [])
    _expect(PipelineValidationError, p.validate)


def test_dangling_edge_rejected():
    p = Pipeline("dangling", [Node("a", "llm")], [("a", "ghost")])
    _expect(PipelineValidationError, p.validate)


def test_cycle_rejected():
    p = Pipeline("cyclic", [Node("a", "llm"), Node("b", "llm")],
                 [("a", "b"), ("b", "a")])
    _expect(PipelineValidationError, p.validate)


def test_structural_diff_rejected_not_silently_skipped():
    base = _pipeline()
    cand = _pipeline()
    cand.nodes.append(Node("reranker", "retriever", {"model": "x"}))
    from omnipex.core.schema import diff_pipelines
    _expect(UnsupportedChangeError, diff_pipelines, base, cand)


def test_unknown_rulepack_raises_omnipex_error():
    _expect(OmnipexError, get, "no_such_pack")


# --- statistical guarantees (the credibility core) --------------------------

def test_empty_scores_rejected_cleanly():
    """No confident answer from no data — and no raw ZeroDivisionError."""
    _expect(ValueError, paired_bootstrap, "m", [], [])


def test_underpowered_sample_is_insufficient_not_real():
    """With too few paired items the CI collapses and any gap looks certain.
    That must report INSUFFICIENT, never REAL."""
    from omnipex.eval.stats import INSUFFICIENT
    res = paired_bootstrap("m", [0.5], [0.1])
    assert res.verdict == INSUFFICIENT
    assert res.real is False


def test_fdr_demotes_lone_borderline_finding():
    """One barely-significant metric among several clean ones is exactly the
    false alarm multiplicity produces — it must not survive."""
    from omnipex.eval.stats import DeltaResult, NOISE, REAL, control_fdr
    family = [
        DeltaResult("borderline", 0, 0, -0.01, -0.02, -0.001, REAL, p_value=0.04, n_items=100),
        DeltaResult("clean1", 0, 0, 0.0, -0.02, 0.02, NOISE, p_value=0.55, n_items=100),
        DeltaResult("clean2", 0, 0, 0.0, -0.02, 0.02, NOISE, p_value=0.80, n_items=100),
    ]
    control_fdr(family)
    assert family[0].verdict == NOISE, "borderline lone finding must be demoted"


def test_fdr_never_promotes():
    from omnipex.eval.stats import DeltaResult, NOISE, control_fdr
    family = [DeltaResult(f"m{i}", 0, 0, 0, -1, 1, NOISE, p_value=0.001, n_items=100)
              for i in range(3)]
    control_fdr(family)
    assert all(r.verdict == NOISE for r in family), "control_fdr must never promote"


def test_fdr_keeps_strong_findings():
    """A genuine, strong regression must survive correction — the fix must not
    make the tool blind."""
    from omnipex.eval.stats import REAL, control_fdr, DeltaResult
    family = [
        DeltaResult("strong", 0, 0, -0.20, -0.25, -0.15, REAL, p_value=0.0002, n_items=100),
        DeltaResult("weak", 0, 0, -0.01, -0.02, -0.001, REAL, p_value=0.049, n_items=100),
    ]
    control_fdr(family)
    assert family[0].verdict == REAL


def test_attribution_run_accounting_is_split_by_cause():
    """The dependency graph must only be credited with savings it caused."""
    report = attribution.attribute(_pipeline(), _v2(), RAG)
    assert (report.skipped_clean_metric + report.skipped_by_graph
            + report.graph_runs == report.naive_runs), "run accounting must balance"


# --- consistency (doctor) ---------------------------------------------------

def test_determining_params_derived_from_rulepack():
    """Which params invalidate the index is derived, never hardcoded."""
    from omnipex.core.consistency import determining_params
    determining = set(determining_params(_pipeline(), "embedder", RAG))
    assert ("embedder", "model") in determining
    assert ("chunker", "size") in determining      # upstream re-chunk forces re-embed
    assert ("retriever", "top_k") not in determining  # never touches the index


def test_doctor_clean_when_stamp_matches():
    from omnipex import api
    root = os.path.join(os.path.dirname(__file__), "..", "examples", "rag_docs_qa")
    report = api.doctor(os.path.join(root, "manifest.json"), env={})
    assert report["status"] == "consistent"
    assert report["counts"]["critical"] == 0


def test_doctor_detects_stale_artifact():
    from omnipex import api
    root = os.path.join(os.path.dirname(__file__), "..", "examples", "rag_docs_qa")
    report = api.doctor(os.path.join(root, "manifest_v2.json"), env={})
    assert report["status"] == "inconsistent"
    stale = [f for f in report["findings"] if f["kind"] == "stale_artifact"]
    assert len(stale) == 1
    drift = stale[0]["detail"]["drift"]
    assert drift[0]["param"] == "embedder.model"
    assert drift[0]["built_with"] == "text-embedding-3-small"


def test_doctor_missing_stamp_warns_not_criticals():
    from omnipex.core.consistency import check
    p = _pipeline()
    findings = check(p, {}, {}, RAG)   # no stamps supplied at all
    assert any(f.kind == "missing_stamp" for f in findings)
    assert all(f.severity != "critical" for f in findings)


def test_doctor_missing_env_is_critical():
    from omnipex.core.consistency import check
    p = _pipeline()
    p.node("llm").requires_env = ["OPENAI_API_KEY"]
    findings = check(p, {}, {}, RAG)
    env_findings = [f for f in findings if f.kind == "missing_env"]
    assert len(env_findings) == 1 and env_findings[0].severity == "critical"
    # ...and satisfied when the variable is present
    ok = check(p, {}, {"OPENAI_API_KEY": "sk-test"}, RAG)
    assert not [f for f in ok if f.kind == "missing_env"]


def test_incomplete_stamp_is_flagged():
    from omnipex.core.consistency import check
    p = _pipeline()
    partial = {"params": {"embedder.model": "text-embedding-3-small"}}
    findings = check(p, {"embedder": partial}, {}, RAG)
    incomplete = [f for f in findings if f.kind == "incomplete_stamp"]
    assert len(incomplete) == 1
    assert "chunker.size" in incomplete[0].detail["missing_keys"]


# --- service layer & GUI server --------------------------------------------

def test_api_diff_contract():
    """The dict every interface (CLI, GUI, future MCP) depends on."""
    from omnipex import api
    root = os.path.join(os.path.dirname(__file__), "..", "examples", "rag_docs_qa")
    report = api.diff(os.path.join(root, "manifest.json"),
                      os.path.join(root, "manifest_v2.json"))
    assert set(report) == {"pipeline", "changes", "effects", "recompute",
                           "confirmed_safe", "evals", "verdict"}
    assert report["verdict"]["status"] == "block"
    assert any(ev["verdict"] == api.REGRESSION for ev in report["evals"])
    import json as _json
    _json.dumps(report)  # must be JSON-serializable end to end


def test_ui_server_endpoints_and_path_guard():
    import json as _json
    import threading
    import urllib.request
    from omnipex.ui.server import UIServer

    root = os.path.join(os.path.dirname(__file__), "..")
    server = UIServer(root, 0)  # ephemeral port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def call(path, body=None):
            req = urllib.request.Request(
                server.url + path,
                data=_json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
                method="POST" if body is not None else "GET",
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, _json.loads(resp.read())
            except urllib.error.HTTPError as err:
                return err.code, _json.loads(err.read())

        status, manifests = call("/api/manifests")
        assert status == 200 and len(manifests) >= 2

        status, report = call("/api/diff", {
            "base": "examples/rag_docs_qa/manifest.json",
            "candidate": "examples/rag_docs_qa/manifest_v2.json"})
        assert status == 200 and report["verdict"]["status"] == "block"

        status, err = call("/api/show", {"path": "../../../etc/passwd"})
        assert status == 400 and "escapes" in err["error"]
    finally:
        server.shutdown()
        server.server_close()


# --- MCP server -------------------------------------------------------------

def _mcp_roundtrip(messages: list[dict]) -> list[dict]:
    """Drive the real MCP server over stdio in a subprocess."""
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    payload = "\n".join(json.dumps(m) for m in messages) + "\n"
    proc = subprocess.run(
        [sys.executable, "-m", "omnipex.cli", "mcp"],
        cwd=root, input=payload, capture_output=True, text=True, timeout=120,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def test_mcp_handshake_and_tool_list():
    responses = _mcp_roundtrip([
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},  # no reply
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ])
    assert len(responses) == 2, "notifications must not be answered"
    init = responses[0]["result"]
    assert init["serverInfo"]["name"] == "omnipex"
    assert init["protocolVersion"] == "2024-11-05"
    names = {t["name"] for t in responses[1]["result"]["tools"]}
    assert names == {"omnipex_show", "omnipex_doctor", "omnipex_diff",
                     "omnipex_blame"}


def test_mcp_blame_names_culprit():
    responses = _mcp_roundtrip([
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "omnipex_blame", "arguments": {
             "base": "examples/rag_docs_qa/manifest.json",
             "candidate": "examples/rag_docs_qa/manifest_v2.json"}}},
    ])
    result = responses[0]["result"]
    assert result["isError"] is False
    report = json.loads(result["content"][0]["text"])
    assert report["metrics"][0]["culprit"].startswith("embedder.model")


def test_mcp_errors_are_reported_not_raised():
    responses = _mcp_roundtrip([
        # bad path -> tool error the model can read
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "omnipex_doctor",
                    "arguments": {"manifest": "../../../etc/passwd"}}},
        # missing required argument
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "omnipex_doctor", "arguments": {}}},
        # unknown method -> JSON-RPC protocol error
        {"jsonrpc": "2.0", "id": 3, "method": "nonsense/method"},
    ])
    assert responses[0]["result"]["isError"] is True
    assert "escapes" in responses[0]["result"]["content"][0]["text"]
    assert responses[1]["result"]["isError"] is True
    assert responses[2]["error"]["code"] == -32601


# --- CLI exit codes (the CI-gate contract) ---------------------------------

def test_cli_exit_codes():
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    base = os.path.join(root, "examples", "rag_docs_qa", "manifest.json")
    v2 = os.path.join(root, "examples", "rag_docs_qa", "manifest_v2.json")

    def run(*argv):
        return subprocess.run(
            [sys.executable, "-m", "omnipex.cli", *argv],
            cwd=root, capture_output=True,
        ).returncode

    assert run("diff", base, base) == 0, "identical configs must pass the gate"
    assert run("diff", base, v2) == 1, "real regression must fail the gate"
    assert run("diff", base, "missing.json") == 2, "bad input must exit 2"
    assert run("diff", base, v2, "--pack", "nope") == 2, "unknown pack must exit 2"
    assert run("doctor", base) == 0, "consistent pipeline must pass"
    assert run("doctor", v2) == 1, "stale artifact must fail the gate"
    assert run("doctor", "missing.json") == 2, "bad input must exit 2"


def test_bisection_names_embedder_as_culprit():
    report = attribution.attribute(_pipeline(), _v2(), RAG)
    by_metric = {m.metric: m for m in report.metrics}
    assert "faithfulness" in by_metric, "faithfulness regression should be attributed"
    culprit = by_metric["faithfulness"].culprit
    assert culprit is not None and culprit.node_id == "embedder"


def test_topk_is_exonerated_for_faithfulness():
    report = attribution.attribute(_pipeline(), _v2(), RAG)
    fa = next(m for m in report.metrics if m.metric == "faithfulness")
    topk = next(e for e in fa.isolated if e.change.param == "top_k")
    assert not topk.is_culprit_candidate  # top_k does not cause the drop


def test_graph_pruning_reduces_runs():
    report = attribution.attribute(_pipeline(), _v2(), RAG)
    assert report.graph_runs < report.naive_runs


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
