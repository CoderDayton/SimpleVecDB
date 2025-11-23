"""Test cases for uncovered lines in search.py"""

import numpy as np
import pytest

from simplevecdb import VectorDB
from simplevecdb.types import DistanceStrategy


def test_hybrid_search_fts_check(tmp_path):
    """Cover hybrid_search FTS check branch"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Disable FTS by setting flag
    collection._search._fts_enabled = False

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

    # Empty query should return empty list
    results = collection.hybrid_search("", query_vector=[0.1] * 384)
    assert len(results) == 0


def test_vector_search_candidates_exception(tmp_path):
    """Cover exception path in _vector_search_candidates"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Force exception by dropping vec table
    collection.conn.execute(f"DROP TABLE IF EXISTS {collection._vec_table_name}")

    # Should fallback to brute force (which returns empty)
    results = collection.similarity_search([0.1] * 384, k=1)
    assert results == []


def test_brute_force_search_empty_table(tmp_path):
    """Cover brute force search on empty table"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Drop vec table to force brute force path
    collection.conn.execute(f"DROP TABLE IF EXISTS {collection._vec_table_name}")

    results = collection._search._brute_force_search(
        np.array([0.1] * 384),
        k=5,
        filter=None,
        filter_builder=None,
        batch_size=100,
    )
    assert results == []


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


def test_brute_force_l1_distance(tmp_path):
    """Cover L1 distance strategy in brute force search"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test", distance_strategy=DistanceStrategy.L1)

    # Add docs but use small dimension for faster test
    from simplevecdb.engine.quantization import QuantizationStrategy

    collection._search._quantizer = QuantizationStrategy(collection.quantization)

    # Use actual brute force by creating vec table manually
    embeddings = [[0.1] * 384, [0.2] * 384]
    collection.add_texts(["doc1", "doc2"], embeddings=embeddings)

    # Invoke brute force directly
    query_vec = np.array([0.15] * 384)
    results = collection._search._brute_force_search(
        query_vec, k=2, filter=None, filter_builder=None, batch_size=100
    )
    assert len(results) == 2


def test_brute_force_l2_distance(tmp_path):
    """Cover L2 distance strategy in brute force search"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test", distance_strategy=DistanceStrategy.L2)

    embeddings = [[0.1] * 384, [0.2] * 384]
    collection.add_texts(["doc1", "doc2"], embeddings=embeddings)

    # Invoke brute force directly
    query_vec = np.array([0.15] * 384)
    results = collection._search._brute_force_search(
        query_vec, k=2, filter=None, filter_builder=None, batch_size=100
    )
    assert len(results) == 2


def test_brute_force_with_filter(tmp_path):
    """Cover brute force search with metadata filter"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    embeddings = [[0.1] * 384, [0.2] * 384]
    metadatas = [{"category": "A"}, {"category": "B"}]
    collection.add_texts(["doc1", "doc2"], embeddings=embeddings, metadatas=metadatas)

    # Invoke brute force with filter
    def build_filter(f):
        return collection._catalog.build_filter_clause(f)

    query_vec = np.array([0.15] * 384)
    results = collection._search._brute_force_search(
        query_vec,
        k=2,
        filter={"category": "A"},
        filter_builder=build_filter,
        batch_size=100,
    )
    assert len(results) == 1


def test_hydrate_documents_missing_row(tmp_path):
    """Cover _hydrate_documents when row not found"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    # Add one doc
    collection.add_texts(["doc1"], embeddings=[[0.1] * 384])

    # Request non-existent rowid
    docs = collection._search._hydrate_documents([(999, 0.5)])
    assert len(docs) == 0


def test_brute_force_invalid_distance_strategy(tmp_path):
    """Cover invalid distance strategy error"""
    db_path = tmp_path / "test.db"
    db = VectorDB(str(db_path))
    collection = db.collection("test")

    embeddings = [[0.1] * 384]
    collection.add_texts(["doc1"], embeddings=embeddings)

    # Monkey patch to invalid value
    original = collection._search.distance_strategy
    try:
        collection._search.distance_strategy = "INVALID"  # type: ignore

        query_vec = np.array([0.1] * 384)
        with pytest.raises(ValueError, match="Unsupported distance strategy"):
            collection._search._brute_force_search(
                query_vec, k=1, filter=None, filter_builder=None, batch_size=100
            )
    finally:
        collection._search.distance_strategy = original
