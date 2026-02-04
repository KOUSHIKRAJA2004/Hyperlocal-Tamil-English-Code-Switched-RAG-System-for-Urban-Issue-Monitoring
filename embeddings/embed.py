"""
Build dense embeddings for chunks and write:
- embeddings/index.faiss
- embeddings/meta.json
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, PATHS
from embeddings.vector_store import build_and_save


# ---------- METADATA NORMALIZATION ---------- #

def infer_domain(chunk):
    doc_id = chunk.get("doc_id", "").lower()

    if "traffic" in doc_id:
        return "traffic"
    if "transport" in doc_id or "bus" in doc_id or "metro" in doc_id:
        return "transport"
    if "water" in doc_id:
        return "water"
    if "power" in doc_id or "electric" in doc_id:
        return "power"
    if "weather" in doc_id:
        return "weather"

    return "unknown"


def infer_source(chunk):
    src = chunk.get("source", "").lower()

    if "twitter" in src:
        return "twitter"
    if "forum" in src:
        return "forums"
    if "news" in src:
        return "news"

    return "other"


def normalize_chunks(chunks):
    normalized = []

    for c in chunks:
        nc = c.copy()

        # 1️⃣ DOMAIN — trust chunk if already present
        if "domain" in c and c["domain"]:
            nc["domain"] = c["domain"]
        else:
            nc["domain"] = "unknown"

        # 2️⃣ SOURCE — trust chunk if clean
        if c.get("source") in {"twitter", "news", "forums", "youtube", "govt"}:
            nc["source"] = c["source"]
        else:
            nc["source"] = "other"

        normalized.append(nc)

    return normalized



# ---------- EMBEDDING PIPELINE ---------- #

def create_embeddings():
    chunks_path = PATHS.chunks_path

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {chunks_path}. "
            "Please run ingest/chunk.py first."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("chunks.json is empty.")

    print(f"Loaded {len(chunks)} raw chunks")

    # 🔥 NORMALIZE METADATA HERE
    chunks = normalize_chunks(chunks)
    # 🚨 DROP chunks with unknown / null domain
    chunks = [
        c for c in chunks
        if c.get("domain") not in {None, "", "unknown"}
    ]
    # sanity check
    print("Sample normalized chunk:")
    print({k: chunks[0][k] for k in ["text", "domain", "source"]})

    model = SentenceTransformer(MODELS.embed_model_name)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    vectors = np.asarray(embeddings, dtype=np.float32)

    build_and_save(vectors, chunks)

    print(f"FAISS index saved to {PATHS.faiss_index_path}")
    print(f"Metadata saved to {PATHS.meta_path}")


if __name__ == "__main__":
    create_embeddings()
