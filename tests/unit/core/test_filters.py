"""Metadata filtering and query building tests."""

import pytest

from simplevecdb import VectorDB


def test_build_filter_clause_like():
    """Test _build_filter_clause with string LIKE pattern."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    filter_dict = {"name": "test*"}
    clause, params = collection._build_filter_clause(filter_dict)
    assert "LIKE" in clause
    assert "%test*%" in params


def test_build_filter_clause_in_list():
    """Test _build_filter_clause with list (IN clause)."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    filter_dict = {"color": ["red", "blue", "green"]}
    clause, params = collection._build_filter_clause(filter_dict)
    assert "IN" in clause
    assert "red" in params and "blue" in params and "green" in params


def test_build_filter_clause_unsupported_type():
    """Test _build_filter_clause with unsupported value type."""
    db = VectorDB(":memory:")
    collection = db.collection("default")
    filter_dict = {"key": {"nested": "dict"}}  # Dict is not supported
    with pytest.raises(ValueError, match="must be int, float, str, or list"):
        collection._build_filter_clause(filter_dict)


def test_filter_advanced():
    """Test advanced metadata filtering with list values and exact match."""
    import numpy as np

    db = VectorDB(":memory:")
    collection = db.collection("default")

    # Generate embeddings matching dimension 384
    embeddings = np.random.randn(2, 384).tolist()
    collection.add_texts(
        ["apple", "banana"],
        embeddings=embeddings,
        metadatas=[{"likes": 10}, {"likes": 20}],
    )

    # Filter with list of values
    results = collection.similarity_search([0.1] * 384, k=2, filter={"likes": [10, 20]})
    assert len(results) == 2

    # Filter with exact value that doesn't match
    results_no_match = collection.similarity_search([0.1] * 384, filter={"likes": 15})
    assert len(results_no_match) == 0

    db.close()
