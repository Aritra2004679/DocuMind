# tests/test_edge_cases.py
'''Layer 4: Edge case / robustness tests
Things that break naive RAG systems — worth explicitly testing and mentioning in interviews.'''

def test_question_not_in_documents_returns_dont_know(rag_pipeline):
    """System should admit ignorance, not hallucinate."""
    answer = rag_pipeline.query("What is the capital of a fictional planet Zorbon?")
    assert "don't have enough information" in answer.lower() or "not" in answer.lower()

def test_empty_pdf_upload_handled_gracefully():
    # should not crash, should show a clear error to the user
    pass

def test_very_long_question_handled():
    # stress test with a 500-word question
    pass