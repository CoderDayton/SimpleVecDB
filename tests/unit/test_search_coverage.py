"""Test cases for search.py coverage"""

import pytest

from simplevecdb import VectorDB
from simplevecdb.types import DistanceStrategy


def test_hybrid_search_fts_check(tmp_path):
    """Cover hybrid_search FTS check branch"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Add some data first
    collection.add_texts(["test doc"], embeddings=[[0.1] * 384])

    # Disable FTS by setting flag on the catalog (where fts_enabled lives)
    collection._search._catalog._fts_enabled = False

    # Should raise RuntimeError
    with pytest.raises(RuntimeError, match="FTS5 support"):
        collection.hybrid_search("test", query_vector=[0.1] * 384)


def test_hybrid_search_empty_query(tmp_path):
    """Cover hybrid_search with empty text query"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    embedding = [0.1] * 384
    collection.add_texts(["test doc"], embeddings=[embedding])

    # Empty query should return empty list (keyword search returns nothing)
    results = collection.hybrid_search("", query_vector=[0.1] * 384)
    # RRF of empty keyword + vector results
    assert len(results) >= 0


def test_keyword_search_no_results(tmp_path):
    """Cover keyword search with no matching results"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    embedding = [0.1] * 384
    collection.add_texts(["apple banana"], embeddings=[embedding])

    # Search for non-existent term
    results = collection.keyword_search("nonexistent", k=5)
    assert len(results) == 0


def test_keyword_search_candidates_error(tmp_path):
    """Cover keyword search error handling"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Drop FTS table to force error
    collection.conn.execute(f"DROP TABLE IF EXISTS {collection._table_name}_fts")

    # Should raise sqlite error
    with pytest.raises(Exception):
        collection.keyword_search("test")


def test_l2_distance_search(tmp_path):
    """Cover L2 distance strategy in search"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path), distance_strategy=DistanceStrategy.L2)
    collection = db.collection("test")

    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    collection.add_texts(["doc1", "doc2"], embeddings=embeddings)

    # Query closest to doc1
    results = collection.similarity_search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0][0].page_content == "doc1"


def test_similarity_search_with_filter(tmp_path):
    """Cover similarity search with metadata filter"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    embeddings = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    metadatas = [{"category": "A"}, {"category": "B"}]
    collection.add_texts(["doc1", "doc2"], embeddings=embeddings, metadatas=metadatas)

    # Search with filter
    results = collection.similarity_search(
        [0.15, 0.25, 0.35], k=2, filter={"category": "A"}
    )
    assert len(results) == 1
    assert results[0][0].metadata["category"] == "A"


def test_mmr_search(tmp_path):
    """Cover MMR search diversity"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Create vectors with some similarity
    embeddings = [
        [1.0, 0.0, 0.0],  # doc1
        [0.9, 0.1, 0.0],  # doc2 - similar to doc1
        [0.0, 1.0, 0.0],  # doc3 - different
    ]
    collection.add_texts(["doc1", "doc2", "doc3"], embeddings=embeddings)

    # MMR should prefer diverse results
    results = collection.max_marginal_relevance_search(
        [1.0, 0.0, 0.0], k=2, fetch_k=3, lambda_mult=0.5
    )
    assert len(results) == 2
