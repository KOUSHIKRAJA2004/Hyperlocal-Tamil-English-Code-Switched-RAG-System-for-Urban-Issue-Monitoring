"""
Build a single cleaned corpus from many raw sources.

Input:
- data/raw/*.json or *.jsonl

Output:
- data/processed/cleaned.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import sys

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS


# ------------------ CLEANING ------------------ #

URL_RE = re.compile(r"http\S+")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


# ------------------ FILE ITERATORS ------------------ #

def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        yield from data["data"]


def iter_raw_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from _iter_jsonl(path)
    elif path.suffix.lower() == ".json":
        yield from _iter_json(path)


# ------------------ DOMAIN FROM FILE ------------------ #

def infer_domain_from_filename(path: Path) -> str | None:
    name = path.stem.lower()

    if "traffic" in name:
        return "traffic"
    if "transport" in name or "bus" in name or "metro" in name:
        return "transport"
    if "water" in name:
        return "water"
    if "power" in name or "electric" in name:
        return "power"
    if "weather" in name:
        return "weather"

    return None


# ------------------ NORMALIZED RECORD ------------------ #

@dataclass(frozen=True)
class NormalizedRecord:
    doc_id: str
    text: str
    source: str
    domain: str | None
    url: str | None = None
    date: str | None = None


def normalize_record(
    obj: dict[str, Any],
    *,
    doc_id: str,
    fallback_source: str,
    fallback_domain: str | None,
) -> NormalizedRecord | None:

    text = (
        obj.get("text")
        or obj.get("content")
        or obj.get("caption")
        or obj.get("body")
    )

    if not isinstance(text, str):
        return None

    text = clean_text(text)
    if not text:
        return None

    return NormalizedRecord(
        doc_id=doc_id,
        text=text,

        # prefer explicit metadata if present
        source=obj.get("source") or fallback_source,
        domain=obj.get("domain") or fallback_domain,

        url=obj.get("url"),
        date=obj.get("date"),
    )


# ------------------ MAIN PIPELINE ------------------ #

def build_corpus() -> list[dict[str, Any]]:
    PATHS.data_processed_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    raw_files = (
        sorted(PATHS.data_raw_dir.glob("*.json"))
        + sorted(PATHS.data_raw_dir.glob("*.jsonl"))
    )

    if not raw_files:
        raise FileNotFoundError(f"No raw files found in {PATHS.data_raw_dir}")

    for path in raw_files:
        fallback_source = path.stem
        fallback_domain = infer_domain_from_filename(path)

        i = 0
        for obj in iter_raw_records(path):
            doc_id = f"{fallback_source}:{i}"
            i += 1

            rec = normalize_record(
                obj,
                doc_id=doc_id,
                fallback_source=fallback_source,
                fallback_domain=fallback_domain,
            )

            if rec:
                records.append(rec.__dict__)

    with PATHS.cleaned_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} cleaned records to {PATHS.cleaned_path}")
    return records


if __name__ == "__main__":
    build_corpus()
