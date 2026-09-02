from rank_bm25 import BM25Okapi


def build_bm25_index(chunks):
    """Build a BM25 index from document chunks."""

    tokenized_corpus = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def bm25_retrieve(
    bm25,
    chunks,
    query,
    k=5,
    min_score=0.0
):
    """Retrieve top-k chunks using BM25.

    Chunks with a score <= min_score are dropped before ranking, so
    that queries with no lexical overlap don't pad the result list
    with irrelevant chunks (which would otherwise still get a rank
    and contribute to RRF fusion downstream).
    """

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    candidate_indices = [
        i for i in range(len(scores))
        if scores[i] > min_score
    ]

    ranked_indices = sorted(
        candidate_indices,
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [
        chunks[i]
        for i in ranked_indices
    ]


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
    top_n=5
):
    """
    Combine dense and BM25 rankings using
    Reciprocal Rank Fusion.
    """

    scores = {}
    documents = {}

    # -----------------------------
    # Dense ranking
    # -----------------------------

    for rank, doc in enumerate(dense_results, start=1):

        doc_id = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content
        )

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (k + rank)
        )

        documents[doc_id] = doc

    # -----------------------------
    # BM25 ranking
    # -----------------------------

    for rank, doc in enumerate(bm25_results, start=1):

        doc_id = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content
        )

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (k + rank)
        )

        documents[doc_id] = doc

    # -----------------------------
    # Sort by fused score
    # -----------------------------

    ranked_docs = sorted(
        documents.keys(),
        key=lambda doc_id: scores[doc_id],
        reverse=True
    )

    return [
        documents[doc_id]
        for doc_id in ranked_docs[:top_n]
    ]