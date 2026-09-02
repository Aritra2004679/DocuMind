from src.retrieval import load_chunks_for_bm25
from src.hybrid import build_bm25_index, bm25_retrieve


def test_bm25_retrieval():
    chunks = load_chunks_for_bm25()

    bm25 = build_bm25_index(chunks)

    results = bm25_retrieve(
        bm25,
        chunks,
        "What are solids?",
        k=5
    )

    assert len(results) == 5
    assert all(hasattr(result, "page_content") for result in results)


if __name__ == "__main__":
    chunks = load_chunks_for_bm25()

    print(f"\nTotal chunks: {len(chunks)}")

    bm25 = build_bm25_index(chunks)

    results = bm25_retrieve(
        bm25,
        chunks,
        "What are solids?",
        k=5
    )

    print("\nBM25 Top-5 Results:")
    print("-" * 70)

    for i, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        page = result.metadata.get("page", 0) + 1
        text = result.page_content[:200].replace("\n", " ")

        print(f"\n{i}. {source}, Page {page}")
        print(f"   {text}...")


    print("\nBM25 retrieval test completed successfully.")