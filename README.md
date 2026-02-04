## Hyperlocal Tamil–English Code-Switched RAG System for Urban Issue Monitoring

This project is a **hyperlocal Retrieval-Augmented Generation (RAG)** system that answers questions about **urban issues in Tamil Nadu** using **Tamil, English, and code‑switched (Tanglish)** queries.

It focuses on five practical city domains:

- **traffic / transport**
- **water supply issues**
- **power / electricity cuts**
- **weather / rain / floods**
- **general transport disruptions (bus, metro, train)**

The system ingests posts and reports from **news, forums, Twitter/X, YouTube captions, etc.**, builds a vector index, and uses a **Gemini-based generator** (plus a Streamlit UI) to produce structured, human‑friendly updates.

---

### 1. High‑Level Architecture

- **Ingestion & Cleaning** (`ingest/`)
  - `ingest/build_corpus.py`  
    - Reads **all** JSON / JSONL files from `data/raw/`.
    - Normalizes fields: `text`, `source`, `domain`, `date`, `url`.
    - Infers `domain` from file name when not explicitly provided (e.g. `transport_traffic.json` → `traffic` / `transport`).
    - Cleans text (removes URLs, extra whitespace).
    - Writes a unified corpus to `data/processed/cleaned.json`.

- **Chunking** (`ingest/chunk.py`)
  - Splits each document into **overlapping word chunks** for better retrieval.
  - Adds metadata per chunk:
    - `doc_id`, `chunk_id`, `text`, `source`, `domain`, `url`, `date`.
  - Writes chunks to `data/processed/chunks.json`.

- **Embeddings & Vector Store** (`embeddings/`)
  - `embeddings/embed.py`
    - Uses **LaBSE (`sentence-transformers/LaBSE`)** to embed all chunks.
    - Normalizes embeddings for **cosine similarity**.
    - Saves a **FAISS `IndexFlatIP`** to `embeddings/index.faiss`.
    - Saves aligned metadata to `embeddings/meta.json`.
  - `embeddings/vector_store.py`
    - Utility for building and loading the FAISS index + metadata.

- **Retrieval & Domain Detection** (`rag/`)
  - `rag/retrieve.py`
    - Loads LaBSE, `index.faiss`, and `meta.json`.
    - Given a query, computes an embedding and retrieves **top‑k** similar chunks.
    - Returns **full metadata** per result:
      - `text`, `domain`, `source`, `date`, `url`, `score`.
  - `rag/domain_detect.py`
    - Uses LaBSE + cosine similarity to classify a query into one of:
      - `transport`, `traffic`, `water`, `power`, `weather`.
    - Embeds short domain descriptions (e.g. “bus, metro, train, transport…”).
    - Picks the **closest domain** to the query embedding.

- **Generation / Answering**
  - **CLI app** (`app.py`)
    - Uses Gemini via `google.genai`.
    - Detects domain, retrieves relevant chunks, filters them by domain, and builds a strong prompt to Gemini.
    - Enforces a **fixed answer format**:
      - `Status:`
      - `Reason:`
      - `Current situation:`
    - Includes guardrails and a fallback summarizer if Gemini output is too short or low‑quality.
  - **Streamlit UI** (`streamlit_app.py`)
    - Simple web interface on top of the same backend (retrieve + domain detect + Gemini).
    - Shows the final answer and an expandable debug section listing retrieved context.

---

### 2. Project Structure

```text
rag_project/
│
├── app.py                  # CLI: Gemini-backed RAG assistant
├── streamlit_app.py        # Streamlit web UI
├── config.py               # Paths, model names, retrieval settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/
│   ├── raw/                # Source JSON / JSONL data
│   │   ├── news.json
│   │   ├── transport_traffic.json
│   │   ├── water.json
│   │   ├── power.json
│   │   ├── weather.json
│   │   └── youtube.json
│   └── processed/
│       ├── cleaned.json    # Unified cleaned corpus
│       └── chunks.json     # Overlapping chunks with metadata
│
├── embeddings/
│   ├── embed.py            # Build FAISS index + metadata
│   ├── vector_store.py     # Vector store utilities
│   ├── index.faiss         # FAISS index (generated)
│   └── meta.json           # Chunk metadata (generated)
│
├── ingest/
│   ├── build_corpus.py     # Main ingest + cleaning pipeline
│   ├── chunk.py            # Chunk builder
│   └── clean.py            # Legacy entry; forwards to build_corpus
│
└── rag/
    ├── retrieve.py         # Dense retrieval over FAISS
    ├── generate.py         # (Older T5-based generator, optional)
    └── domain_detect.py    # Embedding-based domain classifier
```

---

### 3. Data Model & Domains

#### Raw records (`data/raw/*.json` or `*.jsonl`)

Each raw dataset is a list of objects (or JSONL lines). Typical fields:

- `text`: post content (Tamil / English / mixed)
- `source`: e.g. `news`, `twitter`, `forums`, `youtube`
- `domain`: one of `transport`, `traffic`, `water`, `power`, `weather` (optional, can be inferred)
- `date`: `YYYY-MM-DD` (optional but recommended)
- `url`: original link (optional)

`build_corpus.py` also infers `domain` from the **file name** (e.g. `transport_traffic.json` → `traffic` / `transport`) when `domain` is not present per record.

#### Cleaned corpus (`data/processed/cleaned.json`)

Each entry looks roughly like:

```json
{
  "doc_id": "transport_traffic:42",
  "text": "gandhipuram la iniku evening heavy traffic jam, rain + construction nala road romba narrow aagiduchu.",
  "source": "twitter",
  "domain": "traffic",
  "url": "https://twitter.com/...",
  "date": "2024-11-21"
}
```

#### Chunks (`data/processed/chunks.json`)

`chunk.py` splits each `text` into overlapping windows and attaches metadata:

- `doc_id`, `chunk_id`
- `text` (chunk)
- `source`, `domain`, `url`, `date`

This is what gets embedded and stored in FAISS.

---

### 4. Retrieval & Domain Detection

- **Dense retrieval** (`rag/retrieve.py`)
  - Embeds the user query with LaBSE (`normalize_embeddings=True`).
  - Searches the FAISS index for top‑k similar chunks.
  - Returns dicts with:
    - `text`, `domain`, `source`, `date`, `url`, `score`.

- **Domain detection** (`rag/domain_detect.py`)
  - Maintains a small mapping:
    - `transport`: “bus, metro, train, transport, perundhu, strike, delay”
    - `traffic`: “traffic, jam, road, congestion, signal, accident”
    - `water`: “water, thanni, water problem, drinking water, pipeline”
    - `power`: “power cut, current cut, electricity, tneb”
    - `weather`: “rain, cyclone, flood, weather, climate”
  - Embeds these domain descriptions and chooses the closest one to the query.

- **Domain‑aware filtering**
  - `app.py` and `streamlit_app.py` use a `DOMAIN_COMPATIBILITY` map so that:
    - `traffic` queries can also see `transport` chunks (and vice versa).
    - `water`, `power`, `weather` stay more strict.
  - Retrieved chunks are filtered by this domain compatibility before going to the generator.

---

### 5. Answer Generation

There are two main frontends:

- **CLI (`app.py`)**
  - Reads `GOOGLE_API_KEY` from environment.
  - Calls Gemini (`models/gemini-flash-latest`) with a **strict, structured prompt**:
    - `Status:`
    - `Reason:`
    - `Current situation:`
  - Uses a defensive pattern:
    - If Gemini output is too short / low‑quality, falls back to a simple extractive summarizer over top retrieved chunks.

- **Streamlit UI (`streamlit_app.py`)**
  - Uses the same retrieval + domain detection + Gemini backend.
  - Shows:
    - User query input
    - Final answer
    - Optional debug context (top retrieved posts).

> Note: There is also an older `rag/generate.py` that uses `google/flan-t5-small`. The main, stable path uses **Gemini** via `app.py` and `streamlit_app.py`.

---

### 6. Setup & Installation

1. **Create and activate a virtual environment** (recommended).

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Gemini API key** (Windows example):

   ```bash
   setx GOOGLE_API_KEY "YOUR_API_KEY_HERE"
   ```

   Then restart your terminal so `GOOGLE_API_KEY` is visible to Python.

4. **Prepare data**

   - Put your domain data into `data/raw/` as `*.json` or `*.jsonl`.
   - Ensure each record has a `text` field and ideally `source`, `domain`, `date`, `url`.

5. **Build corpus, chunks, and embeddings**

   From the `rag_project/` root:

   ```bash
   # Aggregate + clean all raw sources
   python ingest/build_corpus.py

   # Chunk into overlapping windows
   python ingest/chunk.py

   # Build embeddings + FAISS index
   python embeddings/embed.py
   ```

---

### 7. Running the System

- **CLI mode**:

  ```bash
  python app.py
  ```

  Example queries:

  - `gandhipuram route la traffic irukka?`
  - `iniku night power cut irukka anna nagar la?`
  - `chennai la rain situation epdi irukku?`
  - `kovai la perundhu strike iniku?`

- **Streamlit UI**:

  ```bash
  streamlit run streamlit_app.py
  ```

  Then open the local URL provided by Streamlit in your browser.

---

### 8. Extending to New Domains or Sources

- **Add a new domain** (e.g. `health`):
  - Update `rag/domain_detect.py`:
    - Add `"health": "hospital, clinic, fever, dengue, health, medical"` to `DOMAINS`.
  - Update `DOMAIN_COMPATIBILITY` in `app.py` and `streamlit_app.py`.
  - Add `health_*.json` raw files into `data/raw/` and annotate `domain: "health"` where possible.

- **Add new data sources**:
  - Scrape or export JSON / JSONL.
  - Ensure each record has a `text` field.
  - Drop into `data/raw/` and re‑run:

    ```bash
    python ingest/build_corpus.py
    python ingest/chunk.py
    python embeddings/embed.py
    ```

---

### 9. Design Choices & Limitations

- **Dense retrieval only (FAISS + LaBSE)**:
  - Works well for multilingual and code‑switched text.
  - For very noisy or extremely short queries, a future improvement would be to add a **BM25 (lexical) component** and combine scores (hybrid retrieval).

- **Domain detection via embeddings**:
  - Lightweight and flexible; no separate classifier needed.
  - Relies on good domain descriptions; you can refine the `DOMAINS` text if misclassifications appear.

- **Generation model**:
  - The production path uses **Gemini (flash)** which handles Tamil–English mix better than small open models.
  - Guardrails and fallbacks ensure it doesn’t hallucinate wildly when context is weak.

Despite these limitations, the system is already **realistic and impressive** for:

- Monitoring **traffic, power, water, weather, and transport** issues in Tamil Nadu.
- Handling **Tamil–English code‑switched queries**.
- Producing concise, structured summaries grounded in real posts and reports.

