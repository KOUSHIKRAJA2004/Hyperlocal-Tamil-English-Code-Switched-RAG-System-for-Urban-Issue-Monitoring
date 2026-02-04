# app.py
# Tamil–English Code-Switched Hybrid RAG (Gemini – STABLE & CLEAN)

import sys
import os
from google import genai

# ---------------- CONFIG ---------------- #

DOMAIN_COMPATIBILITY = {
    "traffic": {"traffic", "transport"},
    "transport": {"transport", "traffic"},
    "water": {"water"},
    "power": {"power"},
    "weather": {"weather"}
}

VALID_DOMAINS = set(DOMAIN_COMPATIBILITY.keys())

# ---------------- IMPORTS ---------------- #

from rag.retrieve import retrieve
from rag.domain_detect import detect_domain

# ---------------- GEMINI SETUP ---------------- #

# ✅ SAFEST WAY: read from env, but allow fallback for testing
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY not found in environment")
    print("👉 Run this once in CMD (then restart terminal):")
    print('   setx GOOGLE_API_KEY "YOUR_API_KEY_HERE"')
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "models/gemini-flash-latest"  # ✅ VERIFIED WORKING

# ---------------- HELPERS ---------------- #

def is_valid_question(query: str) -> bool:
    q = query.strip()
    return len(q) >= 4 and not q.isnumeric()


def filter_by_domain(docs, detected_domain):
    allowed = DOMAIN_COMPATIBILITY.get(detected_domain, {detected_domain})
    return [d for d in docs if d.get("domain") in allowed]


# ---------------- PROMPT ---------------- #

def build_prompt(query, docs):
    context = "\n".join(f"- {d['text']}" for d in docs[:5])

    return f"""
You are a hyperlocal city update assistant for Tamil Nadu.

You MUST answer in the following format.
Do not change the format.

FORMAT:
Status:
Reason:
Current situation:

Rules:
- Each field must be a full sentence
- Use natural Tamil or Tanglish
- Do NOT stop early
- Do NOT copy reports directly

User question:
{query}

Reports:
{context}

Answer now using the exact format above:
""".strip()



# ---------------- ANSWER GENERATION ---------------- #

def generate_answer(query, docs):
    if not docs:
        return "No relevant update found."

    prompt = build_prompt(query, docs)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.5,
                "top_p": 0.95,
                "max_output_tokens": 180,
  
            }
        )

        # 🔍 DEBUG (you can remove later)
        print("\n🔹 RAW GEMINI OUTPUT 🔹")
        print(response.text)
        print("🔹 END 🔹\n")

        answer = (response.text or "").strip()

        # 🚨 HARD GUARDRAILS
        if (
            len(answer.split()) < 5
            or answer.lower().startswith(("bro", "enna", "anyone", "-"))
        ):
            return summarize_fallback(docs)

        return answer

    except Exception as e:
        print("❌ Gemini error:", e)
        return summarize_fallback(docs)


def summarize_fallback(docs):
    texts = [d["text"] for d in docs[:2]]
    return " ".join(texts)



# ---------------- MAIN LOOP ---------------- #

def main():
    print("Tamil–English Code-Switched RAG")
    print("Type 'exit' to quit\n")

    while True:
        query = input("Ask (or type exit): ").strip()

        if query.lower() == "exit":
            sys.exit(0)

        if not is_valid_question(query):
            print("\nAnswer:\n Ask a clear, meaningful question.\n")
            continue

        detected_domain = detect_domain(query)
        print("Detected domain:", detected_domain)
        print("Allowed domains:", DOMAIN_COMPATIBILITY.get(detected_domain))

        if detected_domain not in VALID_DOMAINS:
            print("\nAnswer:\n No relevant update found.\n")
            continue

        docs = retrieve(query, k=8)
        docs = filter_by_domain(docs, detected_domain)

        print("Docs after filtering:", len(docs))

        print("\n--- FINAL CONTEXT ---")
        for d in docs[:5]:
            print("-", d["text"])
        print("---------------------")

        answer = generate_answer(query, docs)
        print("\nAnswer:\n", answer, "\n")


if __name__ == "__main__":
    main()
