"""Evaluation harness: metric math, golden-set integrity, and result reporting."""

import json

import pytest
import yaml

from src.config import ROOT
from src.eval.metrics import chunk_matches_unit, percentile, score_retrieval
from src.eval.run_eval import check_all_items_scored, judge_llm, write_results


def test_chunk_matches_unit_boundaries():
    assert chunk_matches_unit("gdpr_en:art:6:c0", "gdpr_en:art:6")
    assert chunk_matches_unit("gdpr_en:art:6:p1:c2", "gdpr_en:art:6")
    assert chunk_matches_unit("gdpr_en:art:6", "gdpr_en:art:6")
    # art:60 must not match art:6
    assert not chunk_matches_unit("gdpr_en:art:60:c0", "gdpr_en:art:6")


def test_score_retrieval():
    retrieved = [
        ("gdpr_en:art:83:c0", "gdpr_en"),
        ("dsa_en:art:33:c1", "dsa_en"),
        ("eu_ai_act_2024:art:99:c0", "eu_ai_act_2024"),
    ]
    s = score_retrieval(
        retrieved,
        relevant_document_ids=["gdpr_en", "eu_ai_act_2024"],
        relevant_unit_ids=["gdpr_en:art:83", "eu_ai_act_2024:art:99"],
    )
    assert s.doc_hit == 1.0 and s.doc_recall == 1.0
    assert s.unit_hit == 1.0 and s.unit_recall == 1.0
    assert s.mrr == 1.0  # first retrieved chunk is relevant

    s2 = score_retrieval(retrieved, ["gdpr_pl"], ["gdpr_pl:art:83"])
    assert s2.doc_hit == 0.0 and s2.unit_hit == 0.0 and s2.mrr == 0.0


def test_score_retrieval_doc_level_only():
    s = score_retrieval([("x:doc:p0:c0", "x")], ["x"], [])
    assert s.doc_hit == 1.0
    assert s.unit_hit is None and s.mrr is None  # excluded from unit metrics


def test_percentile():
    values = list(map(float, range(1, 101)))
    # nearest-rank: round(p/100 * (n-1)) as 0-based index
    assert percentile(values, 50) == 51.0
    assert percentile(values, 95) == 95.0
    assert percentile([1.0], 95) == 1.0
    assert percentile([], 95) == 0.0


def test_check_all_items_scored_rejects_unjudged_items():
    """A timed-out judge job leaves NaN; averaging it away would understate the
    sample size, so the run must fail rather than publish the mean."""
    pd = pytest.importorskip("pandas")

    complete = pd.DataFrame({"faithfulness": [1.0, 0.5], "answer_relevancy": [0.9, 0.8]})
    check_all_items_scored(complete, "hybrid_rerank")  # no raise

    partial = pd.DataFrame({"faithfulness": [1.0, float("nan")], "answer_relevancy": [0.9, 0.8]})
    with pytest.raises(RuntimeError, match="faithfulness failed on 1/2 items"):
        check_all_items_scored(partial, "hybrid_rerank")


def test_judge_llm_follows_provider(monkeypatch):
    """The judge must track LLM_PROVIDER — a hardwired judge would keep hitting
    the old provider's rate limit after switching to escape it."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    assert judge_llm().model_name == "gpt-4o-mini-2024-07-18"

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="langchain-anthropic"):
        judge_llm()

    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        judge_llm()


def test_write_results_is_incremental(tmp_path, monkeypatch):
    """Results land on disk per config, so a later failure cannot discard the
    minutes of reranking already spent on earlier ones."""
    monkeypatch.setattr("src.eval.run_eval.RESULTS_DIR", tmp_path)
    report = {
        "config": "dense",
        "overall": {
            "n": 2,
            "unit_hit@5": 0.5,
            "unit_recall@5": 0.5,
            "mrr": 0.5,
            "doc_recall@5": 1.0,
        },
        "by_difficulty": {
            d: {"unit_hit@5": 0.5, "mrr": 0.5} for d in ("simple", "cross-lingual", "multi-hop")
        },
        "by_language": {lang: {"unit_hit@5": 0.5, "mrr": 0.5} for lang in ("en", "pl", "uk")},
        "latency_ms": {"p50": 90.0, "p95": 120.0},
        "_results": ["dropped from the json"],
    }

    write_results([report], {}, "")
    assert json.loads((tmp_path / "dense.json").read_text())["overall"]["n"] == 2
    assert "_results" not in json.loads((tmp_path / "dense.json").read_text())
    assert "Not run" in (tmp_path / "results.md").read_text()  # ragas table still empty

    write_results([report], {"dense": {"faithfulness": 0.9, "generation_cost_usd": 0.01}}, "")
    assert "0.9" in (tmp_path / "results.md").read_text()


def test_golden_set_integrity():
    items = yaml.safe_load((ROOT / "src" / "eval" / "golden_set.yaml").read_text())
    assert len(items) >= 50
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    difficulties = {i["difficulty"] for i in items}
    assert difficulties == {"simple", "cross-lingual", "multi-hop"}
    languages = {i["question_language"] for i in items}
    assert languages == {"en", "pl", "uk"}
    for i in items:
        assert i["question"] and i["ground_truth_answer"]
        assert i["relevant_document_ids"]
        # cross-lingual items must point at least one document in another language
        if i["difficulty"] == "multi-hop":
            assert len(i["relevant_document_ids"]) >= 2
