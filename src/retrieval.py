from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def build_vectorstore(chunks, save_path="faiss_index"):
    """Embed chunks and build a FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    return vectorstore

def load_vectorstore(save_path="faiss_index"):
    """Load an existing FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

def dense_retrieve(vectorstore, query, k=5):
    """Retrieve top-k chunks via dense similarity search."""
    results = vectorstore.similarity_search(query, k=k)
    return results