"""Parse every corpus PDF into data/processed/{source_id}.json.

Usage: uv run python -m src.ingestion.run_ingestion [source_id ...]
"""

import json
import sys
from datetime import date

from loguru import logger

from src.config import DATA_PROCESSED
from src.ingestion.corpus import CORPUS
from src.ingestion.parse import dehyphenate, parse_pdf
from src.ingestion.structure import parse_units


def ingest_all(source_ids: list[str] | None = None) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for entry in CORPUS:
        if source_ids and entry.meta.source_id not in source_ids:
            continue
        logger.info("parsing {} ({})", entry.meta.source_id, entry.file)
        pages = [dehyphenate(p) for p in parse_pdf(entry)]
        units = parse_units(pages, entry)
        kinds = {k: sum(1 for u in units if u.kind == k) for k in {u.kind for u in units}}
        logger.info("  {} units: {}", len(units), kinds)
        out = {
            "meta": entry.meta.to_json() | {"ingestion_date": date.today().isoformat()},
            "units": [u.to_json() for u in units],
        }
        path = DATA_PROCESSED / f"{entry.meta.source_id}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        logger.info("  wrote {}", path)


if __name__ == "__main__":
    ingest_all(sys.argv[1:] or None)
