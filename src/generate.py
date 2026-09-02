import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Prompt for the final grounded answer ---
PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Conversation so far:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""

# --- Prompt for rewriting follow-up questions into standalone queries ---
CONDENSE_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that includes all necessary context from the history. If the follow-up question is already standalone (doesn't depend on prior context), return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


def condense_question(question, chat_history, model="openai/gpt-oss-20b"):
    """Rewrite a follow-up question into a standalone query using chat history.
    Returns the original question unchanged if there's no prior history."""
    if not chat_history:
        return question

    history_str = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in chat_history
    )
    prompt = CONDENSE_PROMPT_TEMPLATE.format(chat_history=history_str, question=question)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic rewriting, no creativity needed
    )
    return response.choices[0].message.content.strip()


def generate_answer(query, retrieved_chunks, chat_history=None, model="openai/gpt-oss-20b"):
    """Generate a grounded answer from retrieved chunks, aware of prior conversation turns."""
    context = "\n\n".join(
        f"[Source: {c.metadata.get('source', 'unknown')}, Page: {c.metadata.get('page', 0) + 1}]\n{c.page_content}"
        for c in retrieved_chunks
    )

    history_str = "\n".join(
        f"User: {t['question']}\nAssistant: {t['answer']}" for t in (chat_history or [])
    ) or "(no prior conversation)"

    prompt = PROMPT_TEMPLATE.format(chat_history=history_str, context=context, question=query)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content