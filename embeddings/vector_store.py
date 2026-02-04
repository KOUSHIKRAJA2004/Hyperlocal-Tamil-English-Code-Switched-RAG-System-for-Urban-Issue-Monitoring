from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import sys


# Ensure project root (with config.py) is on sys.path, even when running from embeddings/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS


@dataclass(frozen=True)
class VectorStore:
    index: faiss.Index
    meta: list[dict[str, Any]]


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def build_and_save(index_vectors: np.ndarray, meta: list[dict[str, Any]]) -> None:
    """
    Saves a cosine-similarity FAISS index (IndexFlatIP) and metadata.
    Assumes vectors are already L2-normalized.
    """
    if index_vectors.ndim != 2:
        raise ValueError("index_vectors must be a 2D array [n, d]")
    if len(meta) != index_vectors.shape[0]:
        raise ValueError("meta length must match number of vectors")

    d = int(index_vectors.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(index_vectors.astype(np.float32, copy=False))

    PATHS.embeddings_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(PATHS.faiss_index_path))

    _ensure_parent_dir(PATHS.meta_path)
    with PATHS.meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load() -> VectorStore:
    if not PATHS.faiss_index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {PATHS.faiss_index_path}. Run `python embeddings/embed.py`."
        )
    if not PATHS.meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {PATHS.meta_path}. Run `python embeddings/embed.py`."
        )

    index = faiss.read_index(str(PATHS.faiss_index_path))
    with PATHS.meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return VectorStore(index=index, meta=meta)

