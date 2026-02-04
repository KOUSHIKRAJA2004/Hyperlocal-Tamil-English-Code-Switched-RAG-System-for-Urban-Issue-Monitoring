# rag/retrieve.py
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ---------------- LOAD MODEL ---------------- #
model = SentenceTransformer("sentence-transformers/LaBSE")

# ---------------- PATHS ---------------- #
BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "embeddings" / "index.faiss"
META_PATH = BASE_DIR / "embeddings" / "meta.json"

# ---------------- LOAD INDEX ---------------- #
index = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

# ---------------- RETRIEVE ---------------- #
def retrieve(query: str, k: int = 8):
    """
    Returns list of dicts with full metadata
    """
    query_emb = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, ids = index.search(np.array(query_emb, dtype=np.float32), k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue

        m = meta[idx]

        results.append({
            "text": m["text"],
            "domain": m.get("domain"),
            "source": m.get("source"),
            "date": m.get("date"),
            "url": m.get("url"),
            "score": float(score)
        })

    return results
