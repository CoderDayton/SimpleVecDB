"""Tests targeting uncovered lines in usearch_index.py.

Missing lines: 76-78, 141-145, 168, 203, 213, 235, 244-248, 328-329,
               368, 372, 380-381, 388-390, 410, 413, 432, 441-448, 459-460
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from simplevecdb import VectorDB, DistanceStrategy


class TestUnpackBits:
    """Lines 76-78: _unpack_bits function."""

    def test_unpack_bits_roundtrip(self):
        from simplevecdb.engine.usearch_index import _pack_bits, _unpack_bits

        vectors = np.array([[1.0, -0.5, 0.3, -0.8, 0.0, 0.1, -0.2, 0.9]], dtype=np.float32)
        packed = _pack_bits(vectors)
        unpacked = _unpack_bits(packed, ndim=8)
        # Positive -> 1 -> +1, negative/zero -> 0 -> -1
        expected_signs = np.array([[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]])
        np.testing.assert_array_equal(unpacked, expected_signs)

    def test_unpack_bits_multi_row(self):
        from simplevecdb.engine.usearch_index import _pack_bits, _unpack_bits

        vectors = np.array([
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
        ], dtype=np.float32)
        packed = _pack_bits(vectors)
        unpacked = _unpack_bits(packed, ndim=4)
        assert unpacked.shape == (2, 4)
        assert unpacked.dtype == np.float32


class TestMemoryMappedView:
    """Lines 141-145: memory-mapped view path for large indexes."""

    def test_large_index_uses_mmap(self, tmp_path):
        """Large index file triggers memory-mapped view."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        db_path = tmp_path / "large.db"
        db = VectorDB(str(db_path))
        col = db.collection("test")

        # Add vectors and save to create index file
        n = 50
        embeddings = np.random.randn(n, 4).astype(np.float32).tolist()
        texts = [f"doc{i}" for i in range(n)]
        col.add_texts(texts, embeddings=embeddings)
        index_path = col._search._index._path
        col._search._index.save()

        # Patch the constants module that _load_or_create imports
        import simplevecdb.constants as real_constants
        original_threshold = real_constants.USEARCH_MMAP_THRESHOLD
        try:
            real_constants.USEARCH_MMAP_THRESHOLD = 0  # Everything is "large"
            idx = UsearchIndex(
                index_path=index_path,
                ndim=4,
                distance_strategy=DistanceStrategy.COSINE,
            )
            assert idx.is_memory_mapped is True
            idx.close()
        finally:
            real_constants.USEARCH_MMAP_THRESHOLD = original_threshold


class TestCreateIndexOnInit:
    """Line 168: _create_index called when ndim provided but no file exists."""

    def test_new_index_with_ndim(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "new.usearch",
            ndim=8,
            distance_strategy=DistanceStrategy.COSINE,
        )
        assert idx.ndim == 8
        assert idx.size == 0
        idx.close()


class TestNdimProperty:
    """Line 203: ndim property."""

    def test_ndim_none_before_add(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "lazy.usearch",
        )
        assert idx.ndim is None
        idx.close()


class TestIsMemoryMapped:
    """Line 213: is_memory_mapped property."""

    def test_not_memory_mapped(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "small.usearch",
            ndim=4,
        )
        assert idx.is_memory_mapped is False
        idx.close()


class TestDimensionMismatch:
    """Line 235: ValueError on dimension mismatch during add."""

    def test_add_wrong_dimension_raises(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "dim.usearch",
            ndim=4,
        )
        keys = np.array([1], dtype=np.uint64)
        vectors = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # dim=3 != 4
        with pytest.raises(ValueError, match="dimension"):
            idx.add(keys, vectors)
        idx.close()


class TestViewModeUpgrade:
    """Lines 244-248: upgrading from view to writable mode for add."""

    def test_upgrade_view_on_add(self, tmp_path):
        """Adding to a memory-mapped index upgrades to writable."""
        from simplevecdb.engine.usearch_index import UsearchIndex
        import simplevecdb.constants as real_constants

        db_path = tmp_path / "upgrade.db"
        db = VectorDB(str(db_path))
        col = db.collection("test")

        embeddings = np.random.randn(20, 4).astype(np.float32).tolist()
        texts = [f"doc{i}" for i in range(20)]
        col.add_texts(texts, embeddings=embeddings)
        index_path = col._search._index._path
        col._search._index.save()

        # Reload with mmap by lowering threshold
        original_threshold = real_constants.USEARCH_MMAP_THRESHOLD
        try:
            real_constants.USEARCH_MMAP_THRESHOLD = 0
            idx = UsearchIndex(
                index_path=index_path,
                ndim=4,
                distance_strategy=DistanceStrategy.COSINE,
            )
            assert idx.is_memory_mapped is True

            # Add should upgrade from view
            new_keys = np.array([999], dtype=np.uint64)
            new_vecs = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            idx.add(new_keys, new_vecs)
            assert idx.is_memory_mapped is False
            assert idx.size == 21
            idx.close()
        finally:
            real_constants.USEARCH_MMAP_THRESHOLD = original_threshold


class TestBatchQueryNormalization:
    """Lines 328-329: batch query normalization for cosine."""

    def test_batch_query_cosine_normalization(self, tmp_path):
        db = VectorDB(str(tmp_path / "batch.db"))
        col = db.collection("test")

        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        col.add_texts(["doc1", "doc2"], embeddings=embeddings)

        # Batch query (2D array) through search
        queries = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        keys, dists = col._search._index.search(queries, k=2)
        assert len(keys) > 0


class TestRemoveKeyNotFound:
    """Line 368: KeyError during remove (key not in index)."""

    def test_remove_with_key_error(self, tmp_path):
        """Mock the index to raise KeyError for missing keys."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "rm.usearch",
            ndim=3,
        )
        keys = np.array([1, 2], dtype=np.uint64)
        vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        idx.add(keys, vecs)

        # Patch the underlying index.remove to raise KeyError for key 999
        original_remove = idx._index.remove

        def mock_remove(key):
            if key == 999:
                raise KeyError(f"Key {key} not found")
            return original_remove(key)

        idx._index.remove = mock_remove
        removed = idx.remove(np.array([1, 999], dtype=np.uint64))
        assert removed == 1  # Only key 1 was removed
        idx.close()


class TestContainsNoneIndex:
    """Line 372: contains with None index."""

    def test_contains_none_index(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "none.usearch",
        )
        # No ndim, so _index is None
        assert idx.contains(42) is False


class TestSaveEdgeCases:
    """Lines 380-381, 388-390: save with various states."""

    def test_save_no_index(self, tmp_path):
        """Save with None index does nothing."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "nosave.usearch",
        )
        # _index is None, should be no-op
        idx.save()
        assert not (tmp_path / "nosave.usearch").exists()

    def test_save_not_dirty(self, tmp_path):
        """Save with clean index does nothing."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "clean.usearch",
            ndim=3,
        )
        idx._dirty = False
        idx.save()
        assert not (tmp_path / "clean.usearch").exists()

    def test_save_dirty_index(self, tmp_path):
        """Save with dirty index writes to disk."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "dirty.usearch",
            ndim=3,
        )
        keys = np.array([1], dtype=np.uint64)
        vecs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        idx.add(keys, vecs)
        assert idx._dirty is True
        idx.save()
        assert idx._dirty is False
        assert (tmp_path / "dirty.usearch").exists()
        idx.close()


class TestCloseAndDunder:
    """Lines 410, 413: close and __len__."""

    def test_close_saves_and_clears(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "close.usearch",
            ndim=3,
        )
        keys = np.array([1], dtype=np.uint64)
        vecs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        idx.add(keys, vecs)
        idx.close()
        assert idx._index is None

    def test_len(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "len.usearch",
            ndim=3,
        )
        assert len(idx) == 0
        keys = np.array([1, 2], dtype=np.uint64)
        vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        idx.add(keys, vecs)
        assert len(idx) == 2
        idx.close()

    def test_contains_dunder(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "contains.usearch",
            ndim=3,
        )
        keys = np.array([42], dtype=np.uint64)
        vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        idx.add(keys, vecs)
        assert 42 in idx
        assert 999 not in idx
        idx.close()


class TestKeysNoneIndex:
    """Line 432: keys with None index."""

    def test_keys_none_index(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "nokeys.usearch",
        )
        assert idx.keys() == []


class TestGetVectors:
    """Lines 441-448: get vectors with missing keys."""

    def test_get_empty(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "get.usearch",
            ndim=3,
        )
        result = idx.get(np.array([], dtype=np.uint64))
        assert result.shape == (0, 3)
        idx.close()

    def test_get_missing_keys_returns_zeros(self, tmp_path):
        """Mock the index.get to raise KeyError for missing keys."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "getmissing.usearch",
            ndim=3,
        )
        keys = np.array([1], dtype=np.uint64)
        vecs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        idx.add(keys, vecs)

        # Patch the underlying index.get to raise KeyError for key 999
        original_get = idx._index.get

        def mock_get(key):
            if key == 999:
                raise KeyError(f"Key {key} not found")
            return original_get(key)

        idx._index.get = mock_get

        result = idx.get(np.array([1, 999], dtype=np.uint64))
        assert result.shape == (2, 3)
        # First should be non-zero (the stored vector)
        assert not np.allclose(result[0], np.zeros(3))
        # Second should be zeros (missing key fallback)
        np.testing.assert_array_equal(result[1], np.zeros(3))
        idx.close()

    def test_get_none_index(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "getnone.usearch",
        )
        result = idx.get(np.array([1], dtype=np.uint64))
        assert result.shape == (0, 1)


class TestDel:
    """Lines 459-460: __del__ method."""

    def test_del_calls_close(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "del.usearch",
            ndim=3,
        )
        keys = np.array([1], dtype=np.uint64)
        vecs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        idx.add(keys, vecs)
        # __del__ should not raise
        idx.__del__()
        assert idx._index is None

    def test_del_handles_exception(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "delerr.usearch",
            ndim=3,
        )
        # Break _index so close raises
        idx._index = MagicMock()
        idx._index.save.side_effect = RuntimeError("boom")
        idx._dirty = True
        # Should not raise
        idx.__del__()


class TestRemoveEmptyIndex:
    """Line 368: remove from None index."""

    def test_remove_none_index(self, tmp_path):
        from simplevecdb.engine.usearch_index import UsearchIndex

        idx = UsearchIndex(
            index_path=tmp_path / "rmnone.usearch",
        )
        assert idx.remove([1, 2, 3]) == 0
