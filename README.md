---
title: DocuMind
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.61.1"
app_file: app.py
pinned: false
---

# 📄 DocuMind — Advanced RAG System with Hybrid Retrieval & Evaluation

DocuMind is a Retrieval-Augmented Generation (RAG) application that lets users upload their own PDF documents and ask natural-language questions answered strictly from the uploaded content — with source and page-level citations.

This project was built incrementally to demonstrate a production-style RAG pipeline: starting from a working dense-retrieval baseline and extending through conversational memory, hybrid retrieval, cross-encoder reranking, and quantitative RAG evaluation with RAGAS.

> **Status:** Weeks 1–3 complete. Dense retrieval, conversational memory, hybrid retrieval + reranking, and RAGAS-based evaluation are all fully functional. Deployment to Hugging Face Spaces is in progress (see [Roadmap](#roadmap)).

---

## Features

- 📤 Upload multiple PDFs directly through a Streamlit web interface (no manual file placement required)
- ✂️ Automatic chunking with configurable size/overlap using `RecursiveCharacterTextSplitter`
- 🔎 Dense semantic retrieval using `BAAI/bge-small-en-v1.5` embeddings via `sentence-transformers`
- 🔀 Hybrid retrieval combining dense (FAISS) and sparse (BM25) search via Reciprocal Rank Fusion
- 🎯 Cross-encoder reranking (`bge-reranker-base`) for improved top-k precision
- 💬 Conversational memory — multi-turn chat with automatic follow-up question condensation
- ⚡ Fast, free-tier LLM generation via the [Groq API](https://console.groq.com) (`openai/gpt-oss-20b`)
- 📚 Accurate source and page-number citations for every answer
- 🚫 Explicit "I don't have enough information" fallback instead of hallucinating when the answer isn't in the uploaded documents
- 📊 Quantitative RAG evaluation with RAGAS (faithfulness, answer relevancy, context precision/recall) comparing dense-only, hybrid, and hybrid+reranked pipeline variants
- ✅ Layered unit and integration test suite (`pytest`) covering ingestion, retrieval, hybrid fusion, and RAG quality

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Vector Store | FAISS |
| Sparse Retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder (`bge-reranker-base`) |
| LLM Generation | Groq API (`openai/gpt-oss-20b`) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision/recall) |
| UI | Streamlit |
| Testing | Pytest |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
DocuMind/
├── eval/
│   ├── qa_testset.json      # hand-curated evaluation question set
│   └── results/              # per-variant RAGAS score CSVs
├── src/
│   ├── ingest.py             # PDF loading (from Streamlit uploads) + chunking
│   ├── retrieval.py          # FAISS index build/load + dense retrieval
│   ├── hybrid.py              # BM25 index + Reciprocal Rank Fusion
│   ├── rerank.py              # cross-encoder reranking
│   ├── generate.py            # Groq LLM call, prompt templates, query condensation
│   └── evaluate.py            # RAGAS evaluation pipeline across pipeline variants
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   ├── test_bm25.py
│   ├── test_rrf.py
│   ├── test_hybrid.py
│   ├── test_rag_quality.py
│   └── test_edge_cases.py
├── app.py                    # Streamlit application entry point
├── requirements.txt
└── README.md
```

---

## Architecture

```
PDF Upload (Streamlit)
        │
        ▼
   Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
   Embedding (bge-small-en-v1.5) + BM25 Indexing
        │
        ▼
   Dense (FAISS) + Sparse (BM25) Retrieval
        │
        ▼
   Reciprocal Rank Fusion (top_n)
        │
        ▼
   Cross-Encoder Reranking (bge-reranker-base → top_k)
        │
        ▼
   Query Condensation (multi-turn follow-ups only)
        │
        ▼
   Groq LLM (openai/gpt-oss-20b) + Prompt Template
        │
        ▼
   Answer + Source Citations (rendered in Streamlit)
```

---

## Setup & Installation

### 1. Clone and create a virtual environment
```bash
git clone <your-repo-url>
cd documind
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Usage
1. Upload one or more PDFs in the sidebar
2. Click **Build Index**
3. Ask a question in the main panel — follow-up questions are automatically condensed using conversation history
4. Review the answer and its cited sources (page-accurate, reranked for relevance)

---

## Evaluation

DocuMind includes a RAGAS-based evaluation harness (`src/evaluate.py`) that runs three pipeline variants against a shared hand-curated QA test set (`eval/qa_testset.json`):

1. **Dense-only** — baseline FAISS retrieval
2. **Hybrid (RRF)** — BM25 + FAISS fusion
3. **Hybrid + Reranked** — RRF followed by cross-encoder reranking

Each variant is scored on faithfulness, answer relevancy, context precision, and context recall, using Groq (`openai/gpt-oss-20b`) as the judge LLM. Results are saved per-variant to `eval/results/*.csv`.

Run a single variant:
```bash
python -m src.evaluate --variant reranked > eval_reranked_log.txt 2>&1
```

---

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Testing approach follows a layered strategy:
- **Unit tests** — deterministic components (chunking, metadata handling, BM25 scoring, RRF fusion)
- **Integration tests** — retrieval pipeline correctness (relevance ranking, result counts, hybrid fusion behavior)
- **RAG quality tests** — RAGAS-based thresholds for faithfulness and answer relevancy
- **Edge case tests** — out-of-context questions, malformed uploads

---

## Roadmap

- [x] **Week 1** — Dense RAG baseline: ingestion, FAISS retrieval, Groq generation, citations, core tests
- [x] **Week 2** — Conversational memory (query condensation), hybrid retrieval (BM25 + dense fusion via RRF), cross-encoder reranking
- [x] **Week 3** — RAG evaluation with RAGAS (faithfulness, answer relevancy, context precision/recall); comparison across dense-only, hybrid, and hybrid+reranked variants
- [ ] **Week 4** — Deployment to Hugging Face Spaces, documentation polish, optional extensions

---

## Known Limitations

- Index is rebuilt (not appended) on each "Build Index" click; upload all relevant documents together in one batch.
- Code-generation queries (e.g., "write a function to...") can create ambiguity under the strict "answer only from context" grounding constraint — a documented open design decision rather than a bug.
- Deployed instance is subject to Groq's free-tier rate limits (requests/tokens per minute and per day); heavy concurrent usage may return rate-limit errors.

---

## License

This project is for educational and portfolio purposes.