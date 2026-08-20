# tests/test_retrieval.py
'''Layer 2: Integration tests (retrieval pipeline)
Test that retrieval actually returns relevant results — not exact answers, but structural correctness (right count, right type, non-empty).'''

from src.retrieval import build_vectorstore, dense_retrieve
from langchain_core.documents import Document

def test_retrieval_returns_k_results():
    docs = [Document(page_content=f"Test content number {i}", metadata={"source": "test.pdf"}) for i in range(10)]
    vectorstore = build_vectorstore(docs, save_path="test_index")
    results = dense_retrieve(vectorstore, "test content", k=3)
    assert len(results) == 3

def test_retrieval_relevance_ordering():
    docs = [
        Document(page_content="Python is a programming language", metadata={"source": "a.pdf"}),
        Document(page_content="Bananas are yellow fruit", metadata={"source": "b.pdf"}),
    ]
    vectorstore = build_vectorstore(docs, save_path="test_index")
    results = dense_retrieve(vectorstore, "Tell me about Python programming", k=1)
    assert "Python" in results[0].page_content