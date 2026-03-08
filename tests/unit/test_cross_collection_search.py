"""Tests for cross-collection search functionality."""

import pytest
from simplevecdb import VectorDB, Quantization


class TestListCollections:
    def test_empty_db_returns_empty_list(self):
        db = VectorDB(":memory:")
        assert db.list_collections() == []

    def test_returns_initialized_collections(self):
        db = VectorDB(":memory:")
        db.collection("users")
        db.collection("products")
        db.collection("orders")

        result = db.list_collections()
        assert set(result) == {"users", "products", "orders"}

    def test_includes_default_collection(self):
        db = VectorDB(":memory:")
        db.collection("default")
        db.collection("other")

        result = db.list_collections()
        assert "default" in result
        assert "other" in result


class TestSearchCollections:
    @pytest.fixture
    def db_with_collections(self):
        db = VectorDB(":memory:")
        c1 = db.collection("c1", quantization=Quantization.FLOAT)
        c2 = db.collection("c2", quantization=Quantization.FLOAT)
        c3 = db.collection("c3", quantization=Quantization.FLOAT)

        c1.add_texts(["doc1_a", "doc1_b"], embeddings=[[0.1, 0.2], [0.15, 0.25]])
        c2.add_texts(["doc2_a", "doc2_b"], embeddings=[[0.9, 0.8], [0.85, 0.75]])
        c3.add_texts(["doc3_a"], embeddings=[[0.5, 0.5]])

        return db

    def test_search_all_collections(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.1, 0.2], k=10)

        assert len(results) == 5
        doc, score, coll_name = results[0]
        assert doc.page_content == "doc1_a"
        assert coll_name == "c1"
        assert 0 < score <= 1

    def test_search_specific_collections(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.9, 0.8], collections=["c2"], k=5)

        assert len(results) == 2
        for doc, score, coll_name in results:
            assert coll_name == "c2"

    def test_search_multiple_specific_collections(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.5, 0.5], collections=["c1", "c3"], k=10)

        collection_names = {r[2] for r in results}
        assert collection_names == {"c1", "c3"}

    def test_k_limits_results(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.5, 0.5], k=2)

        assert len(results) == 2

    def test_empty_collections_list_returns_empty(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.5, 0.5], collections=[])

        assert results == []

    def test_no_initialized_collections_returns_empty(self):
        db = VectorDB(":memory:")
        results = db.search_collections([0.5, 0.5])

        assert results == []

    def test_nonexistent_collection_raises_keyerror(self, db_with_collections):
        db = db_with_collections
        with pytest.raises(KeyError, match="not initialized"):
            db.search_collections([0.5, 0.5], collections=["nonexistent"])

    def test_normalized_scores_in_zero_one_range(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.1, 0.2], k=10, normalize_scores=True)

        for doc, score, coll_name in results:
            assert 0 < score <= 1, f"Score {score} not in (0, 1]"

    def test_results_sorted_by_score_descending(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.1, 0.2], k=10)

        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filter_applies_across_collections(self):
        db = VectorDB(":memory:")
        c1 = db.collection("c1", quantization=Quantization.FLOAT)
        c2 = db.collection("c2", quantization=Quantization.FLOAT)

        c1.add_texts(
            ["match1", "nomatch1"],
            embeddings=[[0.1, 0.2], [0.11, 0.21]],
            metadatas=[{"category": "A"}, {"category": "B"}],
        )
        c2.add_texts(
            ["match2", "nomatch2"],
            embeddings=[[0.9, 0.8], [0.91, 0.81]],
            metadatas=[{"category": "A"}, {"category": "B"}],
        )

        results = db.search_collections([0.5, 0.5], k=10, filter={"category": "A"})

        assert len(results) == 2
        for doc, score, coll_name in results:
            assert "match" in doc.page_content

    def test_parallel_vs_sequential_same_results(self, db_with_collections):
        db = db_with_collections
        query = [0.3, 0.4]

        parallel_results = db.search_collections(query, k=5, parallel=True)
        sequential_results = db.search_collections(query, k=5, parallel=False)

        parallel_docs = {r[0].page_content for r in parallel_results}
        sequential_docs = {r[0].page_content for r in sequential_results}
        assert parallel_docs == sequential_docs

    def test_unnormalized_scores(self, db_with_collections):
        db = db_with_collections
        results = db.search_collections([0.1, 0.2], k=10, normalize_scores=False)

        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_single_collection_same_as_direct_search(self):
        db = VectorDB(":memory:")
        coll = db.collection("test", quantization=Quantization.FLOAT)
        coll.add_texts(["a", "b", "c"], embeddings=[[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])

        query = [0.5, 0.5]
        direct_results = coll.similarity_search(query, k=3)
        cross_results = db.search_collections(query, collections=["test"], k=3)

        direct_docs = [r[0].page_content for r in direct_results]
        cross_docs = [r[0].page_content for r in cross_results]
        assert direct_docs == cross_docs


class TestDimensionValidation:
    def test_mismatched_dimensions_raises_valueerror(self):
        db = VectorDB(":memory:")
        c1 = db.collection("c1", quantization=Quantization.FLOAT)
        c2 = db.collection("c2", quantization=Quantization.FLOAT)

        c1.add_texts(["doc1"], embeddings=[[0.1, 0.2]])
        c2.add_texts(["doc2"], embeddings=[[0.1, 0.2, 0.3]])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            db.search_collections([0.1, 0.2], collections=["c1", "c2"])

    def test_empty_collection_dimension_ignored(self):
        db = VectorDB(":memory:")
        c1 = db.collection("c1", quantization=Quantization.FLOAT)
        db.collection("c2", quantization=Quantization.FLOAT)

        c1.add_texts(["doc1"], embeddings=[[0.1, 0.2]])

        results = db.search_collections([0.1, 0.2], collections=["c1", "c2"])
        assert len(results) == 1
        assert results[0][0].page_content == "doc1"


class TestAsyncCrossCollectionSearch:
    @pytest.fixture
    def async_db(self):
        from simplevecdb.async_core import AsyncVectorDB

        return AsyncVectorDB(":memory:")

    @pytest.mark.asyncio
    async def test_list_collections(self, async_db):
        async_db.collection("users")
        async_db.collection("products")

        result = async_db.list_collections()
        assert set(result) == {"users", "products"}

    @pytest.mark.asyncio
    async def test_search_collections(self, async_db):
        c1 = async_db.collection("c1", quantization=Quantization.FLOAT)
        c2 = async_db.collection("c2", quantization=Quantization.FLOAT)

        await c1.add_texts(["doc1"], embeddings=[[0.1, 0.2]])
        await c2.add_texts(["doc2"], embeddings=[[0.9, 0.8]])

        results = await async_db.search_collections([0.5, 0.5], k=10)

        assert len(results) == 2
        docs = {r[0].page_content for r in results}
        assert docs == {"doc1", "doc2"}
