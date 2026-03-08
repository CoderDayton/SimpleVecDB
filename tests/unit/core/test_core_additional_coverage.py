"""Additional coverage tests for VectorDB core."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simplevecdb.core import (
    VectorDB,
    _batched,
    get_optimal_batch_size,
)


def test_batched_handles_sequence():
    """Ensure _batched slices sequences without iterator fallback."""
    batches = list(_batched([1, 2, 3, 4], 3))
    assert batches == [[1, 2, 3], [4]]


def test_batched_handles_iterator():
    """Ensure _batched works with non-sequence iterators (lines 60-65)."""

    def gen():
        yield 1
        yield 2
        yield 3
        yield 4
        yield 5

    batches = list(_batched(gen(), 2))
    assert batches == [[1, 2], [3, 4], [5]]


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


class TestCatalogEdgeCases:
    """Test empty-input edge cases in CatalogManager methods."""

    def test_delete_by_ids_empty(self, tmp_path):
        """delete_by_ids with empty list returns empty."""
        db = VectorDB(str(tmp_path / "empty_delete.db"))
        collection = db.collection("test")
        result = collection._catalog.delete_by_ids([])
        assert result == []
        db.close()

    def test_get_documents_by_ids_empty(self, tmp_path):
        """get_documents_by_ids with empty list returns empty dict."""
        db = VectorDB(str(tmp_path / "empty_get.db"))
        collection = db.collection("test")
        result = collection._catalog.get_documents_by_ids([])
        assert result == {}
        db.close()

    def test_get_embeddings_by_ids_empty(self, tmp_path):
        """get_embeddings_by_ids with empty list returns empty dict."""
        db = VectorDB(str(tmp_path / "empty_emb.db"))
        collection = db.collection("test")
        result = collection._catalog.get_embeddings_by_ids([])
        assert result == {}
        db.close()

    def test_find_ids_by_texts_empty(self, tmp_path):
        """find_ids_by_texts with empty list returns empty."""
        db = VectorDB(str(tmp_path / "empty_find.db"))
        collection = db.collection("test")
        result = collection._catalog.find_ids_by_texts([])
        assert result == []
        db.close()

    def test_find_ids_by_filter_empty(self, tmp_path):
        """find_ids_by_filter with empty dict returns empty."""
        db = VectorDB(str(tmp_path / "empty_filter.db"))
        collection = db.collection("test")
        result = collection._catalog.find_ids_by_filter(
            {}, collection._catalog.build_filter_clause
        )
        assert result == []
        db.close()

    def test_get_embeddings_with_none_values(self, tmp_path):
        """get_embeddings_by_ids handles None embeddings (line 362)."""
        db = VectorDB(str(tmp_path / "null_emb.db"))
        collection = db.collection("test")
        # Add document without embedding stored in DB (embedding stored in index only)
        ids = collection.add_texts(
            ["test doc"],
            embeddings=[[1.0, 0.0, 0.0]],
        )
        # Manually set embedding to NULL to test the None branch
        db.conn.execute(
            f"UPDATE {collection._table_name} SET embedding = NULL WHERE id = ?",
            (ids[0],),
        )
        db.conn.commit()

        result = collection._catalog.get_embeddings_by_ids(ids)
        assert ids[0] in result
        assert result[ids[0]] is None
        db.close()


class TestFilterClauseCoverage:
    """Test build_filter_clause edge cases."""

    def test_filter_with_numeric_value(self, tmp_path):
        """Filter with int/float values (lines 464-466)."""
        db = VectorDB(str(tmp_path / "numeric_filter.db"))
        collection = db.collection("test")
        collection.add_texts(
            ["doc1", "doc2", "doc3"],
            metadatas=[{"score": 10}, {"score": 20}, {"score": 30}],
            embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        # Search with numeric filter
        results = collection.similarity_search([1.0, 0.0], k=10, filter={"score": 10})
        assert len(results) == 1
        assert results[0][0].page_content == "doc1"
        db.close()

    def test_filter_with_unsupported_type_raises(self, tmp_path):
        """Filter with unsupported type raises ValueError."""
        db = VectorDB(str(tmp_path / "unsupported_filter.db"))
        collection = db.collection("test")
        collection.add_texts(
            ["doc"],
            embeddings=[[1.0, 0.0]],
        )
        # dict as filter value is not supported
        with pytest.raises(ValueError, match="must be int, float, str, or list"):
            collection._catalog.build_filter_clause({"key": {"nested": "dict"}})
        db.close()

    def test_keyword_search_empty_query(self, tmp_path):
        """Keyword search with empty/whitespace query returns empty (line 418)."""
        db = VectorDB(str(tmp_path / "empty_kw.db"))
        collection = db.collection("test")
        collection.add_texts(["hello world"], embeddings=[[1.0, 0.0]])
        results = collection.keyword_search("   ", k=10)
        assert results == []
        db.close()

    def test_keyword_search_with_filter(self, tmp_path):
        """Keyword search with metadata filter (line 423)."""
        db = VectorDB(str(tmp_path / "kw_filter.db"))
        collection = db.collection("test")
        collection.add_texts(
            ["hello world", "hello mars"],
            metadatas=[{"planet": "earth"}, {"planet": "mars"}],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
        results = collection.keyword_search("hello", k=10, filter={"planet": "mars"})
        assert len(results) == 1
        assert results[0][0].page_content == "hello mars"
        db.close()
