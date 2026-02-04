# domain_detect.py
from sentence_transformers import SentenceTransformer, util

# Load LaBSE once
model = SentenceTransformer("sentence-transformers/LaBSE")

# Domain labels (THIS is not hardcoding logic, just class names)
DOMAINS = {
    "transport": "bus, metro, train, transport, perundhu, strike, delay",
    "traffic": "traffic, jam, road, congestion, signal, accident",
    "water": "water, thanni, water problem, drinking water, pipeline",
    "power": "power cut, current cut, electricity, tneb",
    "weather": "rain, cyclone, flood, weather, climate"
}

# Precompute embeddings
domain_names = list(DOMAINS.keys())
domain_texts = list(DOMAINS.values())
domain_embeddings = model.encode(domain_texts, normalize_embeddings=True)


def detect_domain(query: str) -> str:
    """
    Detects best matching domain using embedding similarity.
    Returns domain name.
    """
    query_emb = model.encode(query, normalize_embeddings=True)

    scores = util.cos_sim(query_emb, domain_embeddings)[0]
    best_idx = int(scores.argmax())

    return domain_names[best_idx]
