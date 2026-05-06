"""Tests targeting uncovered lines in search.py.

Missing lines: 97, 138, 150-194, 273, 348, 358, 364, 376
"""

import pytest
from unittest.mock import patch

from simplevecdb import VectorDB


@pytest.fixture
def db_3d(tmp_path):
    """DB with 3D vectors and metadata for filter tests."""
    db = VectorDB(str(tmp_path / "test.db"))
    col = db.collection("test")
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0],
    ]
    metadatas = [
        {"cat": "A", "score": 10},
        {"cat": "B", "score": 20},
        {"cat": "A", "score": 30},
        {"cat": "B", "score": 40},
    ]
    col.add_texts(["doc1", "doc2", "doc3", "doc4"], embeddings=embeddings, metadatas=metadatas)
    return db


class TestSimilaritySearchKeyNotInDocsMap:
    """Line 97: key not in docs_map during similarity_search."""

    def test_key_missing_from_catalog(self, db_3d):
        """When index has a key that catalog doesn't, it should be skipped."""
        col = db_3d.collection("test")
        # Directly manipulate catalog to simulate missing doc
        original_get = col._search._catalog.get_documents_by_ids

        def patched_get(ids):
            result = original_get(ids)
            # Remove first key to simulate missing doc
            if result:
                first_key = next(iter(result))
                del result[first_key]
            return result

        col._search._catalog.get_documents_by_ids = patched_get
        results = col.similarity_search([1.0, 0.0, 0.0], k=4)
        # Should still return results, just missing the removed one
        assert len(results) < 4


class TestBatchSimilaritySearch:
    """Lines 138, 150-194: batch similarity search paths."""

    def test_empty_queries(self, db_3d):
        """Line 138: empty queries list returns empty list."""
        col = db_3d.collection("test")
        results = col.similarity_search_batch([], k=2)
        assert results == []

    def test_batch_above_threshold(self, db_3d):
        """Lines 150-194: batch search with queries > USEARCH_BATCH_THRESHOLD."""
        col = db_3d.collection("test")
        # Need > 10 queries to exceed threshold
        queries = [[1.0, 0.0, 0.0]] * 12
        with patch("simplevecdb.engine.search.constants") as mock_constants:
            # Set threshold to 1 so even 2 queries triggers batch path
            mock_constants.USEARCH_BATCH_THRESHOLD = 1
            mock_constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER = 3
            mock_constants.DEFAULT_K = 5
            results = col.similarity_search_batch(queries, k=2)
        assert len(results) == 12
        for r in results:
            assert len(r) > 0

    def test_batch_with_filter(self, db_3d):
        """Batch search with filter triggers overfetch and filtering."""
        col = db_3d.collection("test")
        queries = [[1.0, 0.0, 0.0]] * 3
        with patch("simplevecdb.engine.search.constants") as mock_constants:
            mock_constants.USEARCH_BATCH_THRESHOLD = 1
            mock_constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER = 3
            mock_constants.DEFAULT_K = 5
            results = col.similarity_search_batch(queries, k=2, filter={"cat": "A"})
        assert len(results) == 3
        for r in results:
            for doc, _ in r:
                assert doc.metadata["cat"] == "A"

    def test_batch_key_not_in_docs_map(self, db_3d):
        """Batch search skips keys not in docs_map."""
        col = db_3d.collection("test")
        original_get = col._search._catalog.get_documents_by_ids

        def patched_get(ids):
            result = original_get(ids)
            if result:
                first_key = next(iter(result))
                del result[first_key]
            return result

        col._search._catalog.get_documents_by_ids = patched_get
        queries = [[1.0, 0.0, 0.0]] * 3
        with patch("simplevecdb.engine.search.constants") as mock_constants:
            mock_constants.USEARCH_BATCH_THRESHOLD = 1
            mock_constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER = 3
            mock_constants.DEFAULT_K = 5
            results = col.similarity_search_batch(queries, k=4)
        assert len(results) == 3

    def test_batch_single_query_reshape(self, db_3d):
        """Lines 163-165: ndim==1 reshape for single query in batch."""
        col = db_3d.collection("test")
        # Force batch path with single query
        with patch("simplevecdb.engine.search.constants") as mock_constants:
            mock_constants.USEARCH_BATCH_THRESHOLD = 0
            mock_constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER = 3
            mock_constants.DEFAULT_K = 5
            results = col.similarity_search_batch([[1.0, 0.0, 0.0]], k=2)
        assert len(results) == 1
        assert len(results[0]) > 0


class TestHybridSearchEmptyQuery:
    """Line 273: hybrid_search with whitespace-only query."""

    def test_whitespace_query_returns_empty(self, db_3d):
        """Empty/whitespace query returns empty list."""
        col = db_3d.collection("test")
        results = col.hybrid_search("   ", query_vector=[1.0, 0.0, 0.0])
        assert results == []


class TestMMRSearch:
    """Lines 348, 358, 364, 376: MMR search edge cases."""

    def test_mmr_empty_index(self, tmp_path):
        """Line 348: empty index returns empty list."""
        db = VectorDB(str(tmp_path / "empty.db"))
        col = db.collection("test")
        # No documents added, search should return []
        col.add_texts(["x"], embeddings=[[1.0, 0.0, 0.0]])
        # Remove all docs from index to simulate empty
        col._search._index.remove([1])
        results = col.max_marginal_relevance_search([1.0, 0.0, 0.0], k=2)
        assert results == []

    def test_mmr_key_not_in_docs_and_embs(self, db_3d):
        """Line 358: key not in docs_and_embs during MMR."""
        col = db_3d.collection("test")
        original_get = col._search._catalog.get_documents_and_embeddings_by_ids

        def patched_get(ids):
            result = original_get(ids)
            if result:
                first_key = next(iter(result))
                del result[first_key]
            return result

        col._search._catalog.get_documents_and_embeddings_by_ids = patched_get
        results = col.max_marginal_relevance_search([1.0, 0.0, 0.0], k=2, fetch_k=4)
        assert len(results) >= 1

    def test_mmr_with_filter(self, db_3d):
        """Line 364: metadata filter applied during MMR candidate building."""
        col = db_3d.collection("test")
        results = col.max_marginal_relevance_search(
            [1.0, 0.0, 0.0], k=2, fetch_k=4, filter={"cat": "A"}
        )
        for doc in results:
            assert doc.metadata["cat"] == "A"

    def test_mmr_candidates_fewer_than_k(self, db_3d):
        """Line 376: when candidates <= k, return all candidates directly."""
        col = db_3d.collection("test")
        # Request k=10 but only 4 docs exist, so candidates <= k
        results = col.max_marginal_relevance_search(
            [1.0, 0.0, 0.0], k=10, fetch_k=4
        )
        assert len(results) <= 4
        assert len(results) > 0

    def test_mmr_candidates_fewer_than_k_with_filter(self, db_3d):
        """Line 376: with filter reducing candidates below k."""
        col = db_3d.collection("test")
        # Filter to "A" gives 2 docs, request k=5
        results = col.max_marginal_relevance_search(
            [1.0, 0.0, 0.0], k=5, fetch_k=4, filter={"cat": "A"}
        )
        assert len(results) <= 2
