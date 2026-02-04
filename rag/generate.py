from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def generate_answer(context, query):
    prompt = f"""
You are given a factual context.
Answer the question by stating the fact clearly.
Do NOT repeat the question.
Do NOT rephrase the question.
If the context contains a single fact, restate that fact.
Reply in natural Tamil-English mix, 1 sentence, no extra symbols like '-' or '.'.

Context:
{context}

Question:
{query}

Answer:
"""




    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
    **inputs,
    max_length=100,
    min_length=10,
    do_sample=False,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)


    return tokenizer.decode(outputs[0], skip_special_tokens=True)
