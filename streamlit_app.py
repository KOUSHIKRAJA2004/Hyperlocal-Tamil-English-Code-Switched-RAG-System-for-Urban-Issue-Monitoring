# streamlit_app.py
# Tamil–English Code-Switched RAG – Streamlit UI

import streamlit as st
import os
from google import genai

# ---------------- IMPORT BACKEND ---------------- #

from rag.retrieve import retrieve
from rag.domain_detect import detect_domain

# ---------------- CONFIG ---------------- #

DOMAIN_COMPATIBILITY = {
    "traffic": {"traffic", "transport"},
    "transport": {"transport", "traffic"},
    "water": {"water"},
    "power": {"power"},
    "weather": {"weather"}
}

VALID_DOMAINS = set(DOMAIN_COMPATIBILITY.keys())

# ---------------- GEMINI CLIENT ---------------- #

client = genai.Client(
    api_key=os.environ.get("GOOGLE_API_KEY")
)

MODEL_NAME = "models/gemini-flash-latest"

# ---------------- HELPERS ---------------- #

def filter_by_domain(docs, detected_domain):
    allowed = DOMAIN_COMPATIBILITY.get(detected_domain, {detected_domain})
    return [d for d in docs if d.get("domain") in allowed]


def build_prompt(query, docs):
    context = "\n".join(f"- {d['text']}" for d in docs[:5])

    return f"""
You are a hyperlocal Tamil Nadu city update assistant.

TASK:
Summarize the real-world situation from the reports.

RULES:
- Do NOT copy sentences
- Do NOT repeat the question
- Combine multiple reports
- Mention locations if present
- Natural Tamil or Tanglish
- 2–3 sentences only

User question:
{query}

Reports:
{context}

Answer:
""".strip()


def clean_answer(answer: str) -> str:
    lines = []
    for line in answer.splitlines():
        if ":" in line:
            lines.append(line.split(":", 1)[1].strip())
        else:
            lines.append(line.strip())
    return " ".join(lines)


def generate_answer(query, docs):
    if not docs:
        return "No relevant update found."

    prompt = build_prompt(query, docs)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        raw = response.text.strip()
        if len(raw.split()) < 4:
            return docs[0]["text"]

        return clean_answer(raw)

    except Exception as e:
        return docs[0]["text"]


# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(
    page_title="Tamil–English Code-Switched RAG",
    page_icon="🚌",
    layout="centered"
)

st.title("🗣️ Tamil–English Code-Switched RAG")
st.caption("Hyperlocal city updates using AI + RAG")

query = st.text_input(
    "Ask about traffic, water, transport, power, weather 👇",
    placeholder="gandhipuram route la traffic irukka?"
)

if query:
    with st.spinner("Analyzing reports..."):

        detected_domain = detect_domain(query)

        if detected_domain not in VALID_DOMAINS:
            st.warning("No relevant domain detected.")
        else:
            docs = retrieve(query, k=8)
            docs = filter_by_domain(docs, detected_domain)

            answer = generate_answer(query, docs)

            st.subheader("📍 Answer")
            st.success(answer)

            with st.expander("🔍 Debug / Retrieved Context"):
                st.write(f"**Detected domain:** `{detected_domain}`")
                for d in docs[:5]:
                    st.write("•", d["text"])
