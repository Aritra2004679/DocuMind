from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-base"

_model = None  # lazy-loaded singleton, so the reranker model is only
                # loaded into memory once per process, not once per query


def load_reranker(model_name=RERANKER_MODEL):
    """Load (or return the already-loaded) cross-encoder reranker model."""
    global _model
    if _model is None:
        _model = CrossEncoder(model_name)
    return _model


def rerank(query, candidates, top_k=5, model=None):
    """Rerank a candidate list of chunks against the query using a
    cross-encoder, and return the top-k most relevant chunks.

    Unlike BM25 or dense retrieval, which score the query and each
    chunk independently and then compare vectors/scores, a cross-encoder
    scores the (query, chunk) pair jointly in a single forward pass,
    which is typically far more accurate at judging true relevance --
    at the cost of being too slow to run over an entire corpus, which
    is why it's applied only to the small candidate pool retrieval has
    already narrowed down.
    """
    if not candidates:
        return []

    reranker = model or load_reranker()

    pairs = [
        (query, candidate.page_content)
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    scored_candidates = sorted(
        zip(candidates, scores),
        key=lambda pair: pair[1],
        reverse=True
    )

    return [
        candidate
        for candidate, score in scored_candidates[:top_k]
    ]