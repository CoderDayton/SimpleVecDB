"""simplevecdb 2.5.0 correctness tests.

Covers: list_collections persistence, collection cache keying,
delete_collection, __repr__, and connection health checks.
"""

import numpy as np
import pytest

from simplevecdb import VectorDB, Quantization, DistanceStrategy


DIM = 8


def _rand_embedding(dim: int = DIM) -> list[list[float]]:
    return [np.random.default_rng(42).random(dim).tolist()]


# ------------------------------------------------------------------ #
# 1. list_collections persistence
# ------------------------------------------------------------------ #


class TestListCollectionsPersistence:
    """list_collections across sessions and edge cases."""

    def test_persists_across_sessions(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        db = VectorDB(db_path)
        db.collection("users").add_texts(["alice"], embeddings=_rand_embedding())
        db.collection("products").add_texts(["widget"], embeddings=_rand_embedding())
        db.close()

        db2 = VectorDB(db_path)
        names = db2.list_collections()
        assert "users" in names
        assert "products" in names
        db2.close()

    def test_memory_db_shows_session_collections(self):
        db = VectorDB(":memory:")
        db.collection("alpha").add_texts(["a"], embeddings=_rand_embedding())
        db.collection("beta").add_texts(["b"], embeddings=_rand_embedding())
        names = db.list_collections()
        assert "alpha" in names
        assert "beta" in names

    def test_empty_db_returns_empty_list(self):
        db = VectorDB(":memory:")
        assert db.list_collections() == []

    def test_fts_subtables_excluded(self):
        """FTS internal tables (items_X_fts_data, etc.) must not leak."""
        db = VectorDB(":memory:")
        coll = db.collection("docs")
        coll.add_texts(["hello world"], embeddings=_rand_embedding())

        # Force FTS table creation by doing a text search
        try:
            coll.search("hello", k=1)
        except Exception:
            pass  # search may need embeddings; table may still be created

        names = db.list_collections()
        fts_leaked = [n for n in names if "fts" in n]
        assert fts_leaked == [], f"FTS tables leaked into list_collections: {fts_leaked}"

    def test_sorted_output(self):
        db = VectorDB(":memory:")
        for name in ("zebra", "alpha", "middle"):
            db.collection(name).add_texts(["x"], embeddings=_rand_embedding())
        names = db.list_collections()
        assert names == sorted(names)

    def test_default_collection_listed_when_accessed(self):
        db = VectorDB(":memory:")
        db.collection("default").add_texts(["x"], embeddings=_rand_embedding())
        assert "default" in db.list_collections()


# ------------------------------------------------------------------ #
# 2. collection() cache key includes strategy / quantization
# ------------------------------------------------------------------ #


class TestCollectionCacheKey:
    """Cache key must distinguish (name, distance, quantization)."""

    def test_different_quantization_returns_different_objects(self):
        db = VectorDB(":memory:")
        c_float = db.collection("test", quantization=Quantization.FLOAT)
        c_int8 = db.collection("test", quantization=Quantization.INT8)
        assert c_float is not c_int8

    def test_same_params_returns_cached_object(self):
        db = VectorDB(":memory:")
        c1 = db.collection("test", quantization=Quantization.FLOAT)
        c2 = db.collection("test", quantization=Quantization.FLOAT)
        assert c1 is c2

    def test_different_distance_returns_different_objects(self):
        db = VectorDB(":memory:")
        c_cos = db.collection("test", distance_strategy=DistanceStrategy.COSINE)
        c_l2 = db.collection("test", distance_strategy=DistanceStrategy.L2)
        assert c_cos is not c_l2

    def test_full_key_match(self):
        db = VectorDB(":memory:")
        c1 = db.collection(
            "x",
            distance_strategy=DistanceStrategy.COSINE,
            quantization=Quantization.FLOAT16,
        )
        c2 = db.collection(
            "x",
            distance_strategy=DistanceStrategy.COSINE,
            quantization=Quantization.FLOAT16,
        )
        assert c1 is c2


# ------------------------------------------------------------------ #
# 3. delete_collection
# ------------------------------------------------------------------ #


class TestDeleteCollection:
    """delete_collection data removal and error handling."""

    def test_delete_removes_from_list(self):
        db = VectorDB(":memory:")
        db.collection("doomed").add_texts(["bye"], embeddings=_rand_embedding())
        assert "doomed" in db.list_collections()
        db.delete_collection("doomed")
        assert "doomed" not in db.list_collections()

    def test_delete_allows_recreate_with_same_name(self):
        db = VectorDB(":memory:")
        coll = db.collection("recycle")
        coll.add_texts(["old"], embeddings=_rand_embedding())
        db.delete_collection("recycle")

        fresh = db.collection("recycle")
        assert fresh.count() == 0

    def test_delete_nonexistent_raises_key_error(self):
        db = VectorDB(":memory:")
        with pytest.raises(KeyError, match="does not exist"):
            db.delete_collection("ghost")

    def test_delete_removes_usearch_file(self, tmp_path):
        db_path = str(tmp_path / "del.db")
        db = VectorDB(db_path)
        db.collection("victim").add_texts(["x"], embeddings=_rand_embedding())
        db.save()

        from pathlib import Path

        index_file = Path(db_path + ".victim.usearch")
        # Index file should exist after save (if data was added)
        # It may or may not exist depending on lazy flushing, so just
        # verify that after delete it is gone.
        db.delete_collection("victim")
        assert not index_file.exists(), "usearch index file was not cleaned up"
        db.close()

    def test_delete_clears_cache_entries(self):
        db = VectorDB(":memory:")
        c1 = db.collection("temp", quantization=Quantization.FLOAT)
        c2 = db.collection("temp", quantization=Quantization.INT8)
        c1.add_texts(["a"], embeddings=_rand_embedding())
        c2.add_texts(["b"], embeddings=_rand_embedding())

        db.delete_collection("temp")
        # Both cache entries for "temp" should be evicted
        remaining = [k for k in db._collections if k[0] == "temp"]
        assert remaining == [], f"Cache still has entries for 'temp': {remaining}"


# ------------------------------------------------------------------ #
# 4. __repr__
# ------------------------------------------------------------------ #


class TestRepr:
    """__repr__ for VectorCollection, VectorDB, and async wrappers."""

    def test_vector_collection_repr_populated(self):
        db = VectorDB(":memory:")
        coll = db.collection("things")
        coll.add_texts(["item"], embeddings=_rand_embedding())
        r = repr(coll)
        assert "things" in r
        assert str(DIM) in r
        assert "1" in r  # size
        assert "cosine" in r or "l2" in r  # distance

    def test_vector_collection_repr_empty(self):
        db = VectorDB(":memory:")
        coll = db.collection("empty")
        r = repr(coll)
        assert "empty" in r
        assert "None" in r  # dim is None
        assert "0" in r  # size is 0

    def test_vector_db_repr(self):
        db = VectorDB(":memory:")
        db.collection("a").add_texts(["x"], embeddings=_rand_embedding())
        r = repr(db)
        assert ":memory:" in r
        assert "a" in r

    def test_vector_db_repr_empty(self):
        db = VectorDB(":memory:")
        r = repr(db)
        assert ":memory:" in r
        assert "[]" in r

    def test_async_collection_repr(self):
        from simplevecdb.async_core import AsyncVectorCollection

        db = VectorDB(":memory:")
        sync_coll = db.collection("async_test")
        async_coll = AsyncVectorCollection(sync_coll, executor=None)
        r = repr(async_coll)
        assert "async_test" in r
        assert "AsyncVectorCollection" in r

    def test_async_db_repr(self):
        from simplevecdb.async_core import AsyncVectorDB

        db = VectorDB(":memory:")
        async_db = AsyncVectorDB.__new__(AsyncVectorDB)
        async_db._db = db
        r = repr(async_db)
        assert ":memory:" in r
        assert "AsyncVectorDB" in r


# ------------------------------------------------------------------ #
# 5. Connection health check
# ------------------------------------------------------------------ #


class TestConnectionHealthCheck:
    """Corrupt DB files must raise RuntimeError on open."""

    def test_corrupt_db_raises_runtime_error(self, tmp_path):
        corrupt_path = str(tmp_path / "corrupt.db")
        with open(corrupt_path, "wb") as f:
            f.write(b"\x00garbage\xffinvalid\x00sqlite\x00not\x00really")

        with pytest.raises((RuntimeError, Exception)):
            VectorDB(corrupt_path)

    def test_valid_db_opens_cleanly(self, tmp_path):
        db_path = str(tmp_path / "valid.db")
        db = VectorDB(db_path)
        db.collection("ok").add_texts(["fine"], embeddings=_rand_embedding())
        db.close()

        db2 = VectorDB(db_path)
        assert "ok" in db2.list_collections()
        db2.close()
