from src.retrieval import load_vectorstore, dense_retrieve, load_chunks_for_bm25
from src.hybrid import (
    build_bm25_index,
    bm25_retrieve,
    reciprocal_rank_fusion
)


def main():
    query = "What are solids?"

    # -----------------------------
    # Dense retrieval
    # -----------------------------
    vectorstore = load_vectorstore()

    dense_results = dense_retrieve(
        vectorstore,
        query,
        k=5
    )

    # -----------------------------
    # BM25 retrieval
    # -----------------------------
    chunks = load_chunks_for_bm25()

    bm25 = build_bm25_index(chunks)

    bm25_results = bm25_retrieve(
        bm25,
        chunks,
        query,
        k=5
    )

    # -----------------------------
    # RRF fusion
    # -----------------------------
    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        k=60,
        top_n=5
    )

    # -----------------------------
    # Display results
    # -----------------------------
    print("\n" + "=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL - TOP 5")
    print("=" * 70)

    for i, result in enumerate(dense_results, start=1):
        source = result.metadata.get("source", "unknown")
        page = result.metadata.get("page", 0) + 1
        text = result.page_content[:200].replace("\n", " ")

        print(f"\n{i}. {source}, Page {page}")
        print(f"   {text}...")

    print("\n" + "=" * 70)
    print("BM25 RETRIEVAL - TOP 5")
    print("=" * 70)

    for i, result in enumerate(bm25_results, start=1):
        source = result.metadata.get("source", "unknown")
        page = result.metadata.get("page", 0) + 1
        text = result.page_content[:200].replace("\n", " ")

        print(f"\n{i}. {source}, Page {page}")
        print(f"   {text}...")

    print("\n" + "=" * 70)
    print("RRF HYBRID RETRIEVAL - TOP 5")
    print("=" * 70)

    for i, result in enumerate(hybrid_results, start=1):
        source = result.metadata.get("source", "unknown")
        page = result.metadata.get("page", 0) + 1
        text = result.page_content[:200].replace("\n", " ")

        print(f"\n{i}. {source}, Page {page}")
        print(f"   {text}...")

    print("\n" + "=" * 70)
    print("RRF TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()