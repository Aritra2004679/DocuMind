# 📄 DocuMind — Advanced RAG System with Hybrid Retrieval & Evaluation

DocuMind is a Retrieval-Augmented Generation (RAG) application that lets users upload their own PDF documents and ask natural-language questions answered strictly from the uploaded content — with source and page-level citations.

This project is being built incrementally to demonstrate a production-style RAG pipeline: starting from a working dense-retrieval baseline and extending toward hybrid retrieval, reranking, and quantitative RAG evaluation.

> **Status:** Week 1 complete — dense retrieval baseline (FAISS + Groq) is fully functional. Hybrid retrieval, reranking, and RAGAS-based evaluation are in progress (see [Roadmap](#roadmap)).

---

## Features (Week 1)

- 📤 Upload multiple PDFs directly through a Streamlit web interface (no manual file placement required)
- ✂️ Automatic chunking with configurable size/overlap using `RecursiveCharacterTextSplitter`
- 🔎 Dense semantic retrieval using `BAAI/bge-small-en-v1.5` embeddings via `sentence-transformers`
- ⚡ Fast, free-tier LLM generation via the [Groq API](https://console.groq.com) (`openai/gpt-oss-20b`)
- 📚 Accurate source and page-number citations for every answer
- 🚫 Explicit "I don't have enough information" fallback instead of hallucinating when the answer isn't in the uploaded documents
- ✅ Unit and integration test suite (`pytest`) covering chunking and retrieval correctness

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Vector Store | FAISS |
| LLM Generation | Groq API (`openai/gpt-oss-20b`) |
| UI | Streamlit |
| Testing | Pytest |
| Planned (Week 2–3) | BM25 (`rank_bm25`), cross-encoder reranking (`bge-reranker-base`), RAGAS evaluation |

---

## Project Structure

```
DocuMind/
├── data/                    # (optional local storage — primary flow is via Streamlit upload)
├── eval/
│   └── qa_testset.json      # evaluation question set (Week 3)
├── src/
│   ├── ingest.py            # PDF loading (from Streamlit uploads) + chunking
│   ├── retrieval.py         # FAISS index build/load + dense retrieval
│   ├── generate.py          # Groq LLM call + prompt template
│   ├── rerank.py            # cross-encoder reranking (Week 2)
│   └── evaluate.py          # RAGAS evaluation pipeline (Week 3)
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   ├── test_rag_quality.py
│   └── test_edge_cases.py
├── app.py                   # Streamlit application entry point
├── requirements.txt
└── README.md
```

---

## Architecture (Current — Week 1)

```
PDF Upload (Streamlit)
        │
        ▼
   Chunking (RecursiveCharacterTextSplitter)
        │
        ▼
   Embedding (bge-small-en-v1.5)
        │
        ▼
   FAISS Vector Store
        │
        ▼
   Dense Similarity Search (top-k)
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
3. Ask a question in the main panel
4. Review the answer and its cited sources

---

## Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Run only the components completed so far:
```bash
pytest tests/test_ingest.py tests/test_retrieval.py -v
```

Testing approach follows a layered strategy:
- **Unit tests** — deterministic components (chunking, metadata handling)
- **Integration tests** — retrieval pipeline correctness (relevance ranking, result counts)
- **RAG quality tests** *(Week 3)* — RAGAS-based thresholds for faithfulness and answer relevancy
- **Edge case tests** *(in progress)* — out-of-context questions, malformed uploads

---

## Roadmap

- [x] **Week 1** — Dense RAG baseline: ingestion, FAISS retrieval, Groq generation, citations, core tests
- [ ] **Week 2** — Hybrid retrieval (BM25 + dense fusion via Reciprocal Rank Fusion) and cross-encoder reranking
- [ ] **Week 3** — RAG evaluation with RAGAS (faithfulness, answer relevancy, context precision/recall); comparison across pipeline variants
- [ ] **Week 4** — UI polish, deployment (Hugging Face Spaces), documentation, and optional extensions (conversational memory, query rewriting)

---

## Known Limitations (Week 1)

- Retrieval is dense-only; multi-topic comparison queries (e.g., across two uploaded documents) may retrieve an imbalanced number of chunks per source — this is a known weakness of naive dense retrieval and is addressed in Week 2 (hybrid retrieval).
- Index is rebuilt (not appended) on each "Build Index" click; upload all relevant documents together in one batch.

---

## License

This project is for educational and portfolio purposes.