"""Project paths, environment, and pipeline parameters (configs/pipeline.yaml)."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
INDEX_DIR = ROOT / "indexes"

load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def pipeline_config() -> dict[str, Any]:
    with open(ROOT / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value else default
