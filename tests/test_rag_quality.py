# tests/test_rag_quality.py
'''Layer 3: RAG evaluation tests (the AI-specific part — this is your Week 3 work, formalized)
This is what makes it a real AI testing suite, not just software testing. Use RAGAS metrics as pass/fail thresholds.'''

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

def test_faithfulness_above_threshold(eval_dataset):
    """Answers should be grounded in retrieved context, not hallucinated."""
    results = evaluate(eval_dataset, metrics=[faithfulness])
    assert results["faithfulness"] >= 0.7  # threshold, tune based on your data

def test_answer_relevancy_above_threshold(eval_dataset):
    results = evaluate(eval_dataset, metrics=[answer_relevancy])
    assert results["answer_relevancy"] >= 0.7