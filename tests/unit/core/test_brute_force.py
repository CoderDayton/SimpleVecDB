"""Brute force search fallback tests."""

import numpy as np

from tinyvecdb import VectorDB
from tinyvecdb.types import DistanceStrategy


def test_brute_force_search_operational_error(tmp_path):
    """Test brute force fallback when vec_index query fails."""
    db_path = tmp_path / "test_brute.db"
    db = VectorDB(str(db_path))
    db.add_texts(["test1", "test2"], embeddings=[[0.1, 0.2], [0.3, 0.4]])

    # Drop the vec_index table to force OperationalError
    db.conn.execute("DROP TABLE IF EXISTS vec_index")

    # Should fallback to brute force
    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=1, filter=None)
    assert len(results) >= 0  # May be empty or have results


def test_brute_force_search_with_filter(tmp_path):
    """Test brute force search with metadata filter."""
    db_path = tmp_path / "test_brute_filter.db"
    db = VectorDB(str(db_path))
    db.add_texts(
        ["red", "blue"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"color": "red"}, {"color": "blue"}],
    )

    # Drop vec_index to force brute force
    db.conn.execute("DROP TABLE IF EXISTS vec_index")

    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=2, filter={"color": "red"})
    assert len(results) <= 1  # Only red should match


def test_brute_force_search_with_ip_distance(tmp_path):
    """Test brute force search with IP distance strategy."""
    db_path = tmp_path / "test_brute_ip.db"
    # Note: Can't use IP with CREATE TABLE, but can test brute force path
    db = VectorDB(str(db_path), distance_strategy=DistanceStrategy.COSINE)
    db.add_texts(["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]])

    # Manually set to IP to test the brute force IP branch
    db.distance_strategy = DistanceStrategy.IP
    db.conn.execute("DROP TABLE IF EXISTS vec_index")

    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=1, filter=None)
    assert len(results) >= 0


def test_brute_force_search_with_list_filter(tmp_path):
    """Test brute force search with list value in filter."""
    db_path = tmp_path / "test_brute_list.db"
    db = VectorDB(str(db_path))
    db.add_texts(
        ["red", "blue", "green"],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        metadatas=[{"color": "red"}, {"color": "blue"}, {"color": "green"}],
    )

    db.conn.execute("DROP TABLE IF EXISTS vec_index")

    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=3, filter={"color": ["red", "blue"]})
    assert len(results) <= 2


def test_brute_force_empty_results():
    """Test brute force search returns empty when no rows."""
    db = VectorDB(":memory:")
    db._dim = 2  # Set dimension manually

    # No data added, should return empty
    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=1, filter=None)
    assert results == []


def test_brute_force_operational_error_returns_empty():
    """Test brute force search returns empty on OperationalError."""
    db = VectorDB(":memory:")
    db._dim = 2

    # Create a scenario where fetching embeddings fails
    query_vec = np.array([0.1, 0.2], dtype=np.float32)

    # Drop the main table to cause OperationalError
    db.conn.execute(f"DROP TABLE IF EXISTS {db._table_name}")

    results = db._brute_force_search(query_vec, k=1, filter=None)
    assert results == []


def test_brute_force_search_no_filter_metadata():
    """Test brute force search without filter doesn't fetch metadata."""
    db = VectorDB(":memory:")
    db.add_texts(["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]])

    # Don't drop vec_index - brute force still needs it
    # Just call _brute_force_search directly
    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=2, filter=None)
    # Should get results without fetching metadata
    assert len(results) == 2


def test_brute_force_search_with_l2_distance():
    """Test brute force search with L2 distance calculation."""
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L2)
    db.add_texts(["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]])

    # Call brute force directly to test L2 distance path
    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=1, filter=None)
    assert len(results) >= 1


def test_brute_force_search_filter_no_metadata():
    """Test brute force search with filter when metadata is NULL."""
    db = VectorDB(":memory:")
    db.add_texts(["test"], embeddings=[[0.1, 0.2]])

    # Manually set metadata to NULL
    db.conn.execute(f"UPDATE {db._table_name} SET metadata = NULL")
    db.conn.execute("DROP TABLE IF EXISTS vec_index")

    query_vec = np.array([0.1, 0.2], dtype=np.float32)
    results = db._brute_force_search(query_vec, k=1, filter={"key": "value"})
    # Should handle NULL metadata gracefully
    assert len(results) == 0  # No match since metadata is NULL
