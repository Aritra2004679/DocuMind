import streamlit as st
import os
from src.ingest import load_documents_from_uploads, chunk_documents
from src.retrieval import build_vectorstore, load_vectorstore, dense_retrieve, load_chunks_for_bm25, EMBEDDING_MODEL
from src.hybrid import build_bm25_index, bm25_retrieve, reciprocal_rank_fusion
from src.rerank import rerank
from src.generate import generate_answer, condense_question

st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("📄 DocuMind — Conversational RAG over your documents")


# --- Cached model loader: loads bge-small-en-v1.5 once per session,   ---
# --- instead of once per chat message                                 ---
@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# --- Session state: holds the running conversation ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question", "answer", "sources"}

# --- Sidebar: upload + build index ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build Index") and uploaded_files:
        with st.spinner("Processing documents..."):
            docs = load_documents_from_uploads(uploaded_files)
            chunks = chunk_documents(docs)
            build_vectorstore(chunks, embeddings=get_embeddings())  # also saves chunks for BM25
        st.success(f"Index built from {len(uploaded_files)} file(s)!")
        st.session_state.chat_history = []  # fresh index = fresh conversation

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []

# --- Main: chat interface ---
st.header("Chat")

# Render every past turn
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Sources"):
            for s in turn["sources"]:
                st.write(f"- **{s}**")

# Input box for the current turn
question = st.chat_input("Ask a question about your documents...")

if question:
    if not os.path.exists("faiss_index"):
        st.error("Please upload documents and build the index first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            standalone_question = condense_question(question, st.session_state.chat_history)

            vectorstore = load_vectorstore(embeddings=get_embeddings())
            chunks = load_chunks_for_bm25()
            bm25 = build_bm25_index(chunks)

            dense_results = dense_retrieve(vectorstore, standalone_question, k=10)
            bm25_results = bm25_retrieve(bm25, chunks, standalone_question, k=10)

            fused_candidates = reciprocal_rank_fusion(
                dense_results, bm25_results, top_n=10
            )

            retrieved = rerank(standalone_question, fused_candidates, top_k=5)

            # --- TEMPORARY DEBUG OUTPUT: remove once reranking is confirmed working ---
            print("\n" + "=" * 60)
            print(f"QUERY: {standalone_question}")
            print("=" * 60)
            print("FUSED CANDIDATES (before rerank) - TOP 10")
            for i, doc in enumerate(fused_candidates, start=1):
                print(f"{i}. {doc.metadata.get('source')}, Page {doc.metadata.get('page', 0) + 1}")

            print("\nRERANKED - TOP 5")
            for i, doc in enumerate(retrieved, start=1):
                print(f"{i}. {doc.metadata.get('source')}, Page {doc.metadata.get('page', 0) + 1}")
            print("=" * 60 + "\n")
            # --- END TEMPORARY DEBUG OUTPUT ---

            answer = generate_answer(question, retrieved, chat_history=st.session_state.chat_history)

        sources = [
            f"{r.metadata.get('source')}, page {r.metadata.get('page', 0) + 1}"
            for r in retrieved
        ]

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"- **{s}**")

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
        })