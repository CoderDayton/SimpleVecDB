"""Additional coverage tests for VectorDB core."""

import sqlite3
import sys
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simplevecdb.core import (
    VectorDB,
    Quantization,
    _batched,
    get_optimal_batch_size,
)


def test_batched_handles_sequence():
    """Ensure _batched slices sequences without iterator fallback."""
    batches = list(_batched([1, 2, 3, 4], 3))
    assert batches == [[1, 2, 3], [4]]


def test_get_optimal_batch_size_arm_many_cores_returns_16():
    """ARM machines with many cores should return the largest branch."""
    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 12
    mock_psutil.virtual_memory.return_value.available = 16 * 1024**3

    with (
        patch.dict(
            sys.modules,
            {"onnxruntime": None, "torch": None, "psutil": mock_psutil},
        ),
        patch("platform.machine", return_value="arm64"),
    ):
        assert get_optimal_batch_size() == 16


def test_ensure_virtual_table_dimension_mismatch(tmp_path):
    """_ensure_virtual_table should reject mismatched dims once set."""
    db = VectorDB(str(tmp_path / "ensure_dim.db"))
    collection = db.collection("default")
    collection._dim = 3

    with pytest.raises(ValueError):
        collection._catalog.ensure_virtual_table(2)

    db.close()


def test_add_texts_uses_local_embedder_numpy(tmp_path):
    """When embeddings are missing, local embedder should run and accept numpy."""
    db_path = tmp_path / "auto_embed.db"
    embed_returns = [
        [np.array([1.0, 0.0, 0.0], dtype=np.float32)],
        [np.array([0.5, 0.5, 0.5], dtype=np.float32)],
    ]

    with patch("simplevecdb.embeddings.models.embed_texts", side_effect=embed_returns):
        db = VectorDB(str(db_path))
        collection = db.collection("default")
        first_ids = collection.add_texts(["alpha"], metadatas=[{"idx": 1}])
        second_ids = collection.add_texts(["beta"], metadatas=[{"idx": 2}])

    assert len(first_ids) == 1
    assert len(second_ids) == 1
    assert collection._dim == 3
    db.close()


def test_similarity_search_bruteforce_filter(tmp_path):
    """Force sqlite-vec failure to exercise brute-force fallback with filters."""
    db = VectorDB(str(tmp_path / "bf.db"), quantization=Quantization.FLOAT)
    collection = db.collection("default")
    texts = ["tech doc", "sports doc"]
    metas = [{"category": "news", "tag": "tech"}, {"category": "news", "tag": "sports"}]
    embeds = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    collection.add_texts(texts, metadatas=metas, embeddings=embeds)

    filter_dict = {"category": "news", "tag": ["tech", "ai"]}

    class FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *params):
            if "embedding MATCH" in sql:
                raise sqlite3.OperationalError("vec extension unavailable")
            return self._conn.execute(sql, *params)

        def executemany(self, sql, params):
            return self._conn.executemany(sql, params)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def fake_batched(iterable, n):
        rows = list(iterable)
        yield []
        step = max(1, n)
        for i in range(0, len(rows), step):
            yield rows[i : i + step]

    db.conn = cast(sqlite3.Connection, FailingConnection(db.conn))
    # Re-initialize collection to use the mocked connection if needed,
    # but collection.conn is a reference to db.conn, so it should be fine if we updated db.conn in place.
    # However, db.conn was replaced with a wrapper. We need to update collection.conn too.
    collection.conn = db.conn

    with patch("simplevecdb.core._batched", new=fake_batched):
        results = collection.similarity_search([1.0, 0.0, 0.0], k=2, filter=filter_dict)

    assert len(results) == 1
    assert results[0][0].metadata["tag"] == "tech"
    db.close()


def test_bruteforce_invalid_distance_strategy(tmp_path):
    """_brute_force_search should error on unsupported distance metrics."""
    db = VectorDB(str(tmp_path / "bf_invalid.db"))
    collection = db.collection("default")
    collection.add_texts(["only"], embeddings=[[1.0, 0.0, 0.0]])
    db.distance_strategy = "invalid"  # type: ignore[assignment]
    collection.distance_strategy = "invalid"
    collection._search.distance_strategy = "invalid"  # type: ignore[assignment]

    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        collection._brute_force_search(query_vec, 1, None)
    db.close()


def test_remove_texts_requires_criteria(tmp_path):
    """remove_texts should demand either texts or filters."""
    db = VectorDB(str(tmp_path / "remove_none.db"))
    collection = db.collection("default")
    with pytest.raises(ValueError):
        collection.remove_texts()
    db.close()


def test_remove_texts_combines_text_and_filter(tmp_path):
    """Removal should deduplicate IDs gathered from texts and filters."""
    db = VectorDB(str(tmp_path / "remove.db"))
    collection = db.collection("default")
    collection.add_texts(
        ["dup", "filter", "keep"],
        metadatas=[{"topic": "target"}, {"topic": "filter"}, {"topic": "keep"}],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    removed = collection.remove_texts(texts=["dup"], filter={"topic": "filter"})
    assert removed == 2

    remaining = db.conn.execute(
        f"SELECT text FROM {collection._table_name} ORDER BY id"
    ).fetchall()
    assert [row[0] for row in remaining] == ["keep"]
    db.close()


def test_vector_db_del_swallows_close_error(tmp_path):
    """__del__ should ignore close failures."""
    db = VectorDB(str(tmp_path / "del.db"))
    db.close = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]

    db.__del__()  # Should not raise
    assert db.close.called
