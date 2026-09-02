from langchain_core.documents import Document

from src.hybrid import build_bm25_index, bm25_retrieve, reciprocal_rank_fusion


def make_doc(text, source="test.pdf", page=0):
    return Document(page_content=text, metadata={"source": source, "page": page})


class TestBM25Retrieval:

    def test_finds_keyword_relevant_chunk(self):
        chunks = [
            make_doc("Solids have a fixed shape and volume.", page=0),
            make_doc("Liquids take the shape of their container.", page=1),
            make_doc("Gases expand to fill their container.", page=2),
        ]
        bm25 = build_bm25_index(chunks)
        results = bm25_retrieve(bm25, chunks, "What are solids?", k=2)

        assert len(results) == 2
        assert any("Solids" in r.page_content for r in results)

    def test_respects_k(self):
        chunks = [make_doc(f"chunk {i} about solids") for i in range(10)]
        bm25 = build_bm25_index(chunks)
        results = bm25_retrieve(bm25, chunks, "solids", k=3)

        assert len(results) == 3


class TestReciprocalRankFusion:

    def test_fused_results_are_subset_of_inputs(self):
        docs = [make_doc(f"doc {i}", page=i) for i in range(5)]
        dense_results = docs[0:3]
        bm25_results = docs[2:5]

        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=5)

        fused_ids = {(d.metadata["source"], d.metadata["page"]) for d in fused}
        input_ids = {(d.metadata["source"], d.metadata["page"]) for d in dense_results + bm25_results}
        assert fused_ids.issubset(input_ids)

    def test_respects_top_n(self):
        docs = [make_doc(f"doc {i}", page=i) for i in range(10)]
        fused = reciprocal_rank_fusion(docs[:5], docs[5:], k=60, top_n=3)

        assert len(fused) == 3

    def test_doc_in_both_rankings_outranks_doc_in_one(self):
        docs = [make_doc(f"doc {i}", page=i) for i in range(3)]
        # doc0 is rank 1 in both lists -> should win the fused top spot
        dense_results = [docs[0], docs[1]]
        bm25_results = [docs[0], docs[2]]

        fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60, top_n=3)

        assert fused[0].metadata["page"] == 0

    def test_no_duplicates_in_fused_results(self):
        docs = [make_doc(f"doc {i}", page=i) for i in range(3)]
        dense_results = [docs[0], docs[1]]
        bm25_results = [docs[0], docs[2]]

        fused = reciprocal_rank_fusion(dense_results, bm25_results, top_n=10)

        ids = [(d.metadata["source"], d.metadata["page"]) for d in fused]
        assert len(ids) == len(set(ids))

    def test_empty_inputs_return_empty(self):
        assert reciprocal_rank_fusion([], [], top_n=5) == []