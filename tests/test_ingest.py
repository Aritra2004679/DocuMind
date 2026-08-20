# tests/test_ingest.py
'''Layer 1: Unit tests (deterministic components)
Test the parts that aren't AI — chunking logic, metadata handling, file parsing. These should have exact, reproducible expected outputs.'''

from src.ingest import chunk_documents
from langchain_core.documents import Document

def test_chunking_respects_size():
    doc = Document(page_content="A" * 2000, metadata={"source": "test.pdf"})
    chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
    assert all(len(c.page_content) <= 500 for c in chunks)

def test_chunking_preserves_metadata():
    doc = Document(page_content="Some test content here.", metadata={"source": "test.pdf"})
    chunks = chunk_documents([doc])
    assert chunks[0].metadata["source"] == "test.pdf"

def test_empty_document_list():
    chunks = chunk_documents([])
    assert chunks == []