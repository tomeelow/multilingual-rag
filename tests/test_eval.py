"""Evaluation harness: metric math and golden-set integrity."""

import yaml

from src.config import ROOT
from src.eval.metrics import chunk_matches_unit, percentile, score_retrieval


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
