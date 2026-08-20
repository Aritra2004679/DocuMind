import streamlit as st
import os
from src.ingest import load_documents_from_uploads, chunk_documents
from src.retrieval import build_vectorstore, load_vectorstore, dense_retrieve
from src.generate import generate_answer

st.set_page_config(page_title="DocuMind", page_icon="📄")
st.title("📄 DocuMind — RAG over your documents")

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
            build_vectorstore(chunks)
        st.success(f"Index built from {len(uploaded_files)} file(s)!")
        st.session_state["index_ready"] = True

# --- Main: ask questions ---
st.header("Ask a Question")
question = st.text_input("Your question:")

if st.button("Ask") and question:
    if not os.path.exists("faiss_index"):
        st.error("Please upload documents and build the index first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            vectorstore = load_vectorstore()
            retrieved = dense_retrieve(vectorstore, question, k=5)
            answer = generate_answer(question, retrieved)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for r in retrieved:
            page_num = r.metadata.get('page', 0) + 1
            st.write(f"- **{r.metadata.get('source')}**, page {page_num}")