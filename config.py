from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    cleaned_path: Path = BASE_DIR / "data" / "processed" / "cleaned.json"
    chunks_path: Path = BASE_DIR / "data" / "processed" / "chunks.json"

    embeddings_dir: Path = BASE_DIR / "embeddings"
    faiss_index_path: Path = BASE_DIR / "embeddings" / "index.faiss"
    meta_path: Path = BASE_DIR / "embeddings" / "meta.json"


@dataclass(frozen=True)
class Models:
    # NOTE: For better retrieval quality at scale, consider switching to:
    # - "intfloat/multilingual-e5-base" (use query/passsage prefixes)
    # - "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embed_model_name: str = "sentence-transformers/LaBSE"
    generate_model_name: str = "google/flan-t5-small"

    # When using E5 models, prefix queries/passages as below.
    e5_query_prefix: str = "query: "
    e5_passage_prefix: str = "passage: "


@dataclass(frozen=True)
class Retrieval:
    top_k: int = 5
    dense_candidates_k: int = 30
    top_score_threshold: float = 0.30

    # Hybrid retrieval (dense + BM25). If rank-bm25 is not installed, we fall back to dense-only.
    use_hybrid: bool = True
    dense_weight: float = 0.7
    bm25_weight: float = 0.3


PATHS = Paths()
MODELS = Models()
RETRIEVAL = Retrieval()

