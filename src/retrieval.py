import os
import pickle

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def build_vectorstore(
    chunks,
    save_path="faiss_index",
    bm25_path="bm25_store/chunks.pkl",
    embeddings=None
):
    """Embed chunks, build a FAISS index, and save chunks for BM25."""

    # -----------------------------
    # Build FAISS dense index
    # -----------------------------
    embeddings = embeddings or HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(save_path)

    print(f"FAISS index saved to {save_path}")

    # -----------------------------
    # Save chunks for BM25
    # -----------------------------
    bm25_dir = os.path.dirname(bm25_path)

    if bm25_dir:
        os.makedirs(bm25_dir, exist_ok=True)

    with open(bm25_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Chunks saved for BM25 to {bm25_path}")

    return vectorstore


def load_vectorstore(save_path="faiss_index", embeddings=None):
    """Load an existing FAISS index."""

    embeddings = embeddings or HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_chunks_for_bm25(
    bm25_path="bm25_store/chunks.pkl"
):
    """Load the original document chunks saved for BM25."""

    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"BM25 chunk store not found at: {bm25_path}. "
            "Please build the document index first."
        )

    with open(bm25_path, "rb") as f:
        chunks = pickle.load(f)

    return chunks


def dense_retrieve(vectorstore, query, k=5):
    """Retrieve top-k chunks via dense similarity search."""

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results