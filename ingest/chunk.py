"""
Chunk cleaned records into overlapping chunks for better retrieval.

Input:
- data/processed/cleaned.json (from ingest/build_corpus.py)

Output:
- data/processed/chunks.json
"""

from __future__ import annotations

import json
from typing import Iterable

import sys
from pathlib import Path

# Ensure project root (with config.py) is on sys.path, even when running from ingest/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS


CHUNK_WORDS = 200
OVERLAP_WORDS = 50


def chunk_words(words: list[str], *, chunk_words: int, overlap_words: int) -> Iterable[list[str]]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be > 0")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        yield window


def chunk_text(text: str) -> Iterable[str]:
    words = text.split()
    for window in chunk_words(words, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
        yield " ".join(window)


def build_chunks() -> list[dict]:
    with PATHS.cleaned_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: list[dict] = []

    for doc in data:
        doc_id = doc.get("doc_id") or ""
        text = doc.get("text") or ""

        j = 0
        for chunk in chunk_text(text):
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}#c{j}",
                    "text": chunk,

                    # ✅ PRESERVE METADATA
                    "domain": doc.get("domain"),
                    "source": doc.get("source"),
                    "date": doc.get("date"),
                    "url": doc.get("url"),
                }
            )
            j += 1

    PATHS.data_processed_dir.mkdir(parents=True, exist_ok=True)
    with PATHS.chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(chunks)} chunks to {PATHS.chunks_path}")
    return chunks


if __name__ == "__main__":
    build_chunks()
