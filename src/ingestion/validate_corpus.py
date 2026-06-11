"""Corpus validation: per-language counts, declared-vs-detected language,
and length outliers. Run after ingestion (and after chunking, once
data/processed/chunks.json exists).

Usage: uv run python -m src.ingestion.validate_corpus
"""

import json
import sys
from collections import Counter

from src.config import DATA_PROCESSED
from src.language import detect_language

CHUNK_MIN_CHARS = 50
CHUNK_MAX_CHARS = 2000


def validate() -> int:
    docs = sorted(p for p in DATA_PROCESSED.glob("*.json") if p.name != "chunks.json")
    if not docs:
        print("no processed documents found — run src.ingestion.run_ingestion first")
        return 1

    problems = 0
    lang_docs: Counter[str] = Counter()
    lang_units: Counter[str] = Counter()
    print(f"{'source_id':28} {'lang':4} {'units':>5} {'kinds':32} lang-check")
    for path in docs:
        doc = json.loads(path.read_text())
        meta, units = doc["meta"], doc["units"]
        lang_docs[meta["language"]] += 1
        lang_units[meta["language"]] += len(units)
        kinds = dict(Counter(u["kind"] for u in units))
        # detect on a body sample, not the title page
        sample = " ".join(u["text"] for u in units[len(units) // 2 : len(units) // 2 + 3])
        detected = detect_language(sample[:2000])
        ok = "ok" if detected == meta["language"] else f"MISMATCH (detected {detected})"
        if detected != meta["language"]:
            problems += 1
        print(f"{meta['source_id']:28} {meta['language']:4} {len(units):5} {kinds!s:32} {ok}")

    print(f"\ndocuments per language: {dict(lang_docs)}")
    print(f"units per language:     {dict(lang_units)}")

    chunks_path = DATA_PROCESSED / "chunks.json"
    if chunks_path.exists():
        # bounds apply to retrieval (child) chunks; parents are ≤1024 tokens by design
        chunks = [c for c in json.loads(chunks_path.read_text()) if c["chunk_type"] == "child"]
        short = [c["chunk_id"] for c in chunks if len(c["text"]) < CHUNK_MIN_CHARS]
        long_ = [c["chunk_id"] for c in chunks if len(c["text"]) > CHUNK_MAX_CHARS]
        print(
            f"\nchunks: {len(chunks)} total, {len(short)} under {CHUNK_MIN_CHARS} chars, "
            f"{len(long_)} over {CHUNK_MAX_CHARS} chars"
        )
        for cid in short[:10]:
            print(f"  short: {cid}")
        for cid in long_[:10]:
            print(f"  long:  {cid}")
        problems += len(short) + len(long_)
    else:
        print("\nchunks.json not found — chunk-level checks skipped (run chunking first)")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(validate())
