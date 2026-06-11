"""Deterministic retrieval metrics against the golden set.

Relevance is defined at the structural-unit level (articles/recitals): a
retrieved chunk hits a relevant unit if its chunk_id descends from the
unit_id. These metrics need no LLM judge and run offline; ragas adds the
LLM-judged generation metrics on top when an API key is available.
"""

from dataclasses import dataclass


def chunk_matches_unit(chunk_id: str, unit_id: str) -> bool:
    """gdpr_en:art:6:c0 and gdpr_en:art:6:p1:c2 descend from gdpr_en:art:6;
    gdpr_en:art:60:c0 does not."""
    return chunk_id == unit_id or chunk_id.startswith(unit_id + ":")


@dataclass
class RetrievalScores:
    doc_hit: float  # any relevant document retrieved
    doc_recall: float  # fraction of relevant documents retrieved
    unit_hit: float | None  # any relevant unit retrieved (None: no unit ground truth)
    unit_recall: float | None  # fraction of relevant units retrieved
    mrr: float | None  # 1/rank of the first relevant unit


def score_retrieval(
    retrieved: list[tuple[str, str]],  # [(chunk_id, source_id)] in rank order
    relevant_document_ids: list[str],
    relevant_unit_ids: list[str],
) -> RetrievalScores:
    docs_found = {sid for _, sid in retrieved} & set(relevant_document_ids)
    doc_hit = float(bool(docs_found))
    doc_recall = len(docs_found) / len(relevant_document_ids)

    if not relevant_unit_ids:
        return RetrievalScores(doc_hit, doc_recall, None, None, None)

    units_found = {
        u for u in relevant_unit_ids if any(chunk_matches_unit(cid, u) for cid, _ in retrieved)
    }
    mrr = 0.0
    for rank, (cid, _) in enumerate(retrieved, start=1):
        if any(chunk_matches_unit(cid, u) for u in relevant_unit_ids):
            mrr = 1.0 / rank
            break
    return RetrievalScores(
        doc_hit=doc_hit,
        doc_recall=doc_recall,
        unit_hit=float(bool(units_found)),
        unit_recall=len(units_found) / len(relevant_unit_ids),
        mrr=mrr,
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))
    return ordered[idx]
