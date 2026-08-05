"""Evaluation harness: 4-configuration ablation over the golden set.

Configurations (the README comparison table):
  dense          — e5 dense retrieval only, top 5
  hybrid         — dense + BM25 with RRF, top 5
  hybrid_rerank  — RRF top 20 -> cross-encoder -> top 5 (production path)
  hybrid_rerank_hyde — as above with HyDE query vectors (needs an LLM key)

Retrieval metrics (doc/unit recall, MRR, latency percentiles) are
deterministic and run offline. With an LLM key configured, the harness also
generates answers for every item and scores them with ragas
(context_precision, context_recall, faithfulness, answer_relevancy), tracking
token cost per run. All retrieval runs use the open corpus
(cross_lingual=True) — the harder, headline setting.

Usage:
  uv run python -m src.eval.run_eval [--configs dense hybrid ...] [--ragas]
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any

import yaml
from langchain_core.embeddings import Embeddings
from loguru import logger

from src.config import ROOT, env, pipeline_config
from src.eval.metrics import RetrievalScores, mean, percentile, score_retrieval
from src.models import Chunk
from src.pipeline.hyde import hyde_query_vector
from src.pipeline.llm import credentials_available, get_llm
from src.pipeline.prompts import build_prompt
from src.pipeline.rerank import get_reranker
from src.pipeline.retriever import get_retriever
from src.retrieval.embedding import get_embedder

GOLDEN_SET = ROOT / "src" / "eval" / "golden_set.yaml"
RESULTS_DIR = ROOT / "docs" / "eval"

# pinned pricing for cost tracking (USD per 1M tokens, gemini-2.5-flash public
# rates). Tracks answer generation only; the ragas judge calls are not counted.
PRICE_IN, PRICE_OUT = 0.30, 2.50

CONFIGS = ["dense", "hybrid", "hybrid_rerank", "hybrid_rerank_hyde"]


def load_golden_set() -> list[dict[str, Any]]:
    return yaml.safe_load(GOLDEN_SET.read_text())


@dataclass
class ItemResult:
    item_id: str
    difficulty: str
    language: str
    scores: RetrievalScores
    retrieval_ms: float
    chunks: list[Chunk]


def retrieve_for_config(config: str, query: str, final_k: int) -> list[Chunk]:
    retriever = get_retriever()
    if config == "dense":
        ids = retriever.dense(query, k=final_k)
        return [retriever._children[cid] for cid, _ in ids]
    if config == "hybrid":
        return retriever.hybrid(query, cross_lingual=True)[:final_k]
    if config == "hybrid_rerank":
        candidates = retriever.hybrid(query, cross_lingual=True)
        return get_reranker().rerank(query, candidates)
    if config == "hybrid_rerank_hyde":
        vector = hyde_query_vector(query, get_llm())
        candidates = retriever.hybrid(query, cross_lingual=True, query_vector=vector)
        return get_reranker().rerank(query, candidates)
    raise ValueError(config)


def run_config(config: str, items: list[dict]) -> dict[str, Any]:
    final_k = pipeline_config()["retrieval"]["final_k"]
    results: list[ItemResult] = []
    for item in items:
        t0 = time.perf_counter()
        chunks = retrieve_for_config(config, item["question"], final_k)
        elapsed = (time.perf_counter() - t0) * 1000
        scores = score_retrieval(
            [(c.chunk_id, c.source_id) for c in chunks],
            item["relevant_document_ids"],
            item.get("relevant_unit_ids") or [],
        )
        results.append(
            ItemResult(
                item["id"], item["difficulty"], item["question_language"], scores, elapsed, chunks
            )
        )

    def agg(rs: list[ItemResult]) -> dict[str, float]:
        unit_scored = [r for r in rs if r.scores.unit_hit is not None]
        return {
            "n": len(rs),
            "doc_hit@5": round(mean([r.scores.doc_hit for r in rs]), 3),
            "doc_recall@5": round(mean([r.scores.doc_recall for r in rs]), 3),
            "unit_hit@5": round(mean([r.scores.unit_hit for r in unit_scored]), 3),
            "unit_recall@5": round(mean([r.scores.unit_recall for r in unit_scored]), 3),
            "mrr": round(mean([r.scores.mrr for r in unit_scored]), 3),
        }

    latencies = [r.retrieval_ms for r in results]
    return {
        "config": config,
        "overall": agg(results),
        "by_difficulty": {
            d: agg([r for r in results if r.difficulty == d])
            for d in ("simple", "cross-lingual", "multi-hop")
        },
        "by_language": {
            lang: agg([r for r in results if r.language == lang]) for lang in ("en", "pl", "uk")
        },
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
        },
        "_results": results,
    }


class _E5Embeddings(Embeddings):
    """LangChain Embeddings backed by the project's local multilingual-e5 model,
    so ragas needs no embedding API: documents get the 'passage:' prefix and
    queries the 'query:' prefix, matching how the index was built."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in get_embedder().embed_passages(texts)]

    def embed_query(self, text: str) -> list[float]:
        return get_embedder().embed_query(text)


def check_all_items_scored(df: Any, config: str) -> None:
    """Raise unless the judge scored every item on every metric.

    A judge job that exhausts its retries leaves NaN for that row, and pandas
    skips NaN when averaging — so the published mean would quietly cover fewer
    items than the header claims. That is a hole in the results table that
    reads exactly like a real score, so refuse to publish it.
    """
    means = df.mean(numeric_only=True)
    dropped = {m: int(df[m].isna().sum()) for m in means.index if df[m].isna().any()}
    if dropped:
        raise RuntimeError(
            f"ragas judge did not score every item for {config}: "
            + ", ".join(f"{m} failed on {n}/{len(df)} items" for m, n in dropped.items())
            + ". Refusing to publish means over a smaller sample than reported. This is "
            "almost always the provider rate limit — see the 429s above and rerun with "
            "more quota, or use --limit N for a smaller proof run."
        )


def judge_llm() -> Any:
    """LangChain chat model for the ragas judge, following LLM_PROVIDER.

    Pinned to the same provider that generated the answers: a judge hardwired
    to one provider silently ignores LLM_PROVIDER, so switching providers to
    escape a rate limit would still bottleneck on the old one.
    """
    provider = env("LLM_PROVIDER", "openai")
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=env("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=env("GEMINI_API_KEY"),
            temperature=0.0,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=env("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
            api_key=env("OPENAI_API_KEY"),
            temperature=0.0,
        )
    if provider == "anthropic":
        # langchain-anthropic is not a project dependency; add it to judge here
        raise ValueError(
            "LLM_PROVIDER=anthropic has no ragas judge: add `langchain-anthropic` to "
            "pyproject.toml and wire ChatAnthropic in judge_llm(), or set LLM_PROVIDER "
            "to openai or gemini for the eval run."
        )
    raise ValueError(f"unknown LLM_PROVIDER: {provider}")


def run_ragas(config_report: dict, items: list[dict]) -> dict[str, float]:
    """LLM-judged generation metrics + cost tracking. Judge follows LLM_PROVIDER;
    embeddings are the local e5 model (no embedding API)."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    llm = get_llm()
    gen_cfg = pipeline_config()["generation"]
    rows, in_tok, out_tok = [], 0, 0
    by_id = {r.item_id: r for r in config_report["_results"]}
    for item in items:
        chunks = by_id[item["id"]].chunks
        parents = get_retriever().parents_of(chunks)
        result = llm.complete(
            build_prompt(item["question"], parents),
            max_tokens=gen_cfg["max_tokens"],
            temperature=gen_cfg["temperature"],
        )
        in_tok += result.input_tokens
        out_tok += result.output_tokens
        rows.append(
            {
                "question": item["question"],
                "answer": result.text,
                "contexts": [p.text for p in parents],
                "ground_truth": item["ground_truth_answer"].strip(),
            }
        )
    scores = evaluate(
        Dataset.from_list(rows),
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=judge_llm(),
        embeddings=_E5Embeddings(),
        # bounded concurrency + generous timeout: a judge behind provider rate
        # limits 429-storms and times out at the default 16 workers / 180 s
        run_config=RunConfig(max_workers=4, timeout=300),
    )
    cost = (in_tok * PRICE_IN + out_tok * PRICE_OUT) / 1e6
    logger.info(
        "generation cost for {}: ${:.4f} ({} in / {} out tokens)",
        config_report["config"],
        cost,
        in_tok,
        out_tok,
    )
    df = scores.to_pandas()
    check_all_items_scored(df, config_report["config"])
    means = df.mean(numeric_only=True)
    return {k: round(float(v), 3) for k, v in means.items()} | {
        "generation_cost_usd": round(cost, 4)
    }


def write_markdown(reports: list[dict], ragas_scores: dict[str, dict]) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"Golden set: {reports[0]['overall']['n']} items · "
        "all retrieval open-corpus (cross_lingual=True) · "
        f"run {date.today().isoformat()}",
        "",
        "## Retrieval ablation (deterministic, no LLM judge)",
        "",
        "| Configuration | unit hit@5 | unit recall@5 | MRR | doc recall@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        o, lat = r["overall"], r["latency_ms"]
        lines.append(
            f"| {r['config']} | {o['unit_hit@5']} | {o['unit_recall@5']} | {o['mrr']} "
            f"| {o['doc_recall@5']} | {lat['p50']} | {lat['p95']} |"
        )
    lines += [
        "",
        "## By difficulty (unit hit@5 / MRR)",
        "",
        "| Configuration | simple | cross-lingual | multi-hop |",
        "|---|---|---|---|",
    ]
    for r in reports:
        cells = [
            f"{r['by_difficulty'][d]['unit_hit@5']} / {r['by_difficulty'][d]['mrr']}"
            for d in ("simple", "cross-lingual", "multi-hop")
        ]
        lines.append(f"| {r['config']} | {' | '.join(cells)} |")
    lines += [
        "",
        "## By question language (unit hit@5 / MRR)",
        "",
        "| Configuration | en | pl | uk |",
        "|---|---|---|---|",
    ]
    for r in reports:
        cells = [
            f"{r['by_language'][lang]['unit_hit@5']} / {r['by_language'][lang]['mrr']}"
            for lang in ("en", "pl", "uk")
        ]
        lines.append(f"| {r['config']} | {' | '.join(cells)} |")

    lines += ["", "## Generation quality (ragas, LLM-judged)", ""]
    if ragas_scores:
        metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        lines += [
            "| Configuration | " + " | ".join(metrics) + " | cost (USD) |",
            "|---|" + "---|" * (len(metrics) + 1),
        ]
        for config, s in ragas_scores.items():
            cells = [str(s.get(m, "—")) for m in metrics]
            lines.append(
                f"| {config} | " + " | ".join(cells) + f" | {s.get('generation_cost_usd', '—')} |"
            )
    else:
        lines += [
            "_Not run: requires a key for the configured `LLM_PROVIDER` in `.env`. "
            "Run `uv run python -m src.eval.run_eval --ragas` to fill this table._",
        ]
    return "\n".join(lines) + "\n"


def write_results(reports: list[dict], ragas_scores: dict[str, dict], suffix: str) -> None:
    for r in reports:
        slim = {k: v for k, v in r.items() if k != "_results"}
        (RESULTS_DIR / f"{r['config']}{suffix}.json").write_text(json.dumps(slim, indent=2))
    (RESULTS_DIR / f"results{suffix}.md").write_text(write_markdown(reports, ragas_scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None, choices=CONFIGS)
    parser.add_argument("--ragas", action="store_true", help="run LLM-judged metrics")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N golden-set items; outputs go to *.proof.* "
        "files so a partial run never overwrites the full-corpus results",
    )
    args = parser.parse_args()

    has_key = credentials_available()  # active LLM_PROVIDER's key (gemini here)
    configs = args.configs or (CONFIGS if has_key else CONFIGS[:3])
    if "hybrid_rerank_hyde" in configs and not has_key:
        logger.warning("skipping hybrid_rerank_hyde: HyDE needs an LLM key")
        configs = [c for c in configs if c != "hybrid_rerank_hyde"]

    items = load_golden_set()
    if args.limit:
        items = items[: args.limit]
    suffix = ".proof" if args.limit else ""
    logger.info("golden set: {} items{}", len(items), " (proof subset)" if args.limit else "")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reports, ragas_scores = [], {}
    for config in configs:
        logger.info("running config: {}", config)
        report = run_config(config, items)
        logger.info("  {} | latency {}", report["overall"], report["latency_ms"])
        reports.append(report)
        # flush after each stage: a judge that dies on config 3 of 4 (an exhausted
        # rate limit, most likely) then costs only its own scores, not the whole
        # run's retrieval work — minutes of reranking per config
        write_results(reports, ragas_scores, suffix)
        if args.ragas and has_key:
            ragas_scores[config] = run_ragas(report, items)
            write_results(reports, ragas_scores, suffix)
        logger.info("wrote {}", RESULTS_DIR / f"results{suffix}.md")


if __name__ == "__main__":
    main()
