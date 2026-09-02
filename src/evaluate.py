import argparse
import json
import os

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval import load_vectorstore, dense_retrieve, load_chunks_for_bm25
from src.hybrid import build_bm25_index, bm25_retrieve, reciprocal_rank_fusion
from src.rerank import rerank
from src.generate import generate_answer

RAGAS_MODEL = "openai/gpt-oss-20b"  # judge LLM used by RAGAS metrics
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # same embedding model as retrieval, for consistency


def load_qa_testset(path="eval/qa_testset.json"):
    """Load the hand-curated question/ground-truth pairs."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_dense_only(query, vectorstore, k=5):
    """Phase 1 baseline: dense retrieval only."""
    return dense_retrieve(vectorstore, query, k=k)


def run_hybrid(query, vectorstore, bm25, chunks, dense_k=10, bm25_k=10, top_n=5):
    """Phase 2B: dense + BM25 fused via RRF, no reranking."""
    dense_results = dense_retrieve(vectorstore, query, k=dense_k)
    bm25_results = bm25_retrieve(bm25, chunks, query, k=bm25_k)
    return reciprocal_rank_fusion(dense_results, bm25_results, top_n=top_n)


def run_hybrid_reranked(query, vectorstore, bm25, chunks, dense_k=10, bm25_k=10, rrf_top_n=10, final_k=5):
    """Phase 2B: dense + BM25 fused via RRF, then cross-encoder reranked."""
    dense_results = dense_retrieve(vectorstore, query, k=dense_k)
    bm25_results = bm25_retrieve(bm25, chunks, query, k=bm25_k)
    fused = reciprocal_rank_fusion(dense_results, bm25_results, top_n=rrf_top_n)
    return rerank(query, fused, top_k=final_k)


def build_ragas_dataset(qa_pairs, retrieve_fn, vectorstore, bm25=None, chunks=None):
    """Run one retrieval variant over the full test set and assemble a
    RAGAS-compatible dataset: question, answer, retrieved contexts, and
    ground truth for each example."""
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in qa_pairs:
        question = item["question"]
        ground_truth = item["ground_truth"]

        if bm25 is not None:
            retrieved = retrieve_fn(question, vectorstore, bm25, chunks)
        else:
            retrieved = retrieve_fn(question, vectorstore)

        contexts = [doc.page_content for doc in retrieved]
        answer = generate_answer(question, retrieved)

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def evaluate_variant(name, dataset, llm, embeddings):
    """Run RAGAS metrics on a single variant's dataset and return the scores."""
    print(f"\nEvaluating variant: {name}...")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = result.to_pandas()
    print(f"\n{name} — mean scores:")
    print(scores[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    return result, scores


def get_variant_builders(qa_pairs, vectorstore, bm25, chunks):
    """Return a dict mapping variant name -> a zero-arg callable that builds
    that variant's RAGAS dataset. Built lazily so we only construct the
    dataset for the variant(s) actually being run."""
    return {
        "dense": ("Dense-Only (Phase 1)", lambda: build_ragas_dataset(qa_pairs, run_dense_only, vectorstore)),
        "hybrid": ("Hybrid RRF (Phase 2B)", lambda: build_ragas_dataset(qa_pairs, run_hybrid, vectorstore, bm25, chunks)),
        "reranked": ("Hybrid + Reranked (Phase 2B)", lambda: build_ragas_dataset(qa_pairs, run_hybrid_reranked, vectorstore, bm25, chunks)),
    }


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on one or more DocuMind retrieval pipeline variants.")
    parser.add_argument(
        "--variant",
        choices=["dense", "hybrid", "reranked", "all"],
        default="all",
        help="Which pipeline variant to evaluate. Default: all (runs all three back-to-back). "
             "Use a single variant to avoid re-burning API quota on variants already evaluated.",
    )
    args = parser.parse_args()

    qa_pairs = load_qa_testset()

    vectorstore = load_vectorstore()
    chunks = load_chunks_for_bm25()
    bm25 = build_bm25_index(chunks)

    # bypass_n=True stops RAGAS from requesting n=3 completions per call,
    # which Groq's API for openai/gpt-oss-20b rejects outright
    # ("'n' : number must be at most 1"). Without this, a meaningful chunk
    # of RAGAS's judge-LLM calls fail before ever reaching the model.
    raw_llm = ChatGroq(model=RAGAS_MODEL, temperature=0.0)
    llm = LangchainLLMWrapper(raw_llm, bypass_n=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    all_builders = get_variant_builders(qa_pairs, vectorstore, bm25, chunks)

    if args.variant == "all":
        selected_keys = ["dense", "hybrid", "reranked"]
    else:
        selected_keys = [args.variant]

    all_results = {}
    for key in selected_keys:
        name, build_fn = all_builders[key]
        dataset = build_fn()
        result, scores = evaluate_variant(name, dataset, llm, embeddings)
        all_results[name] = scores

    print("\n" + "=" * 70)
    print("SUMMARY: MEAN SCORES")
    print("=" * 70)
    for name, scores in all_results.items():
        print(f"\n{name}")
        print(scores[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())

    os.makedirs("eval/results", exist_ok=True)
    for name, scores in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        scores.to_csv(f"eval/results/{safe_name}.csv", index=False)
    print("\nPer-example scores saved to eval/results/*.csv")


if __name__ == "__main__":
    main()