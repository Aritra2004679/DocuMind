import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(query, retrieved_chunks, model="openai/gpt-oss-20b"):
    context = "\n\n".join(
        f"[Source: {c.metadata.get('source', 'unknown')}, Page: {c.metadata.get('page', 0) + 1}]\n{c.page_content}"
        for c in retrieved_chunks
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=query)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content