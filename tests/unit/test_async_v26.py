"""AsyncVectorDB 2.6.0 changes.

Covers:
- ``AsyncVectorDB.collection(store_embeddings=...)`` — previously absent;
  forced async users to drop into the sync API to enable embedding storage.
- ``AsyncVectorCollection.cluster(algorithm=...)`` — runtime validation
  produces a clear ValueError for invalid algorithms instead of a
  confusing internal failure deep in sync code.
- ``AsyncVectorDB.close()`` — drains executor with ``wait=True`` so pool
  threads finish before the SQLite connection is closed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from simplevecdb.async_core import AsyncVectorDB


pytestmark = pytest.mark.asyncio


class TestAsyncCollectionStoreEmbeddings:
    async def test_collection_accepts_store_embeddings_true(self):
        db = AsyncVectorDB(":memory:")
        col = db.collection("with_emb", store_embeddings=True)
        # The flag must propagate to the underlying sync collection so
        # rebuild_index() can later replay vectors out of SQLite.
        assert col._collection._store_embeddings is True
        await db.close()

    async def test_collection_default_store_embeddings_false(self):
        db = AsyncVectorDB(":memory:")
        col = db.collection("default")
        assert col._collection._store_embeddings is False
        await db.close()

    async def test_collection_cache_key_includes_store_embeddings(self):
        # Same name with different store_embeddings must produce distinct
        # AsyncVectorCollection wrappers (otherwise the cached collection
        # would silently override the requested mode).
        db = AsyncVectorDB(":memory:")
        a = db.collection("dup", store_embeddings=False)
        b = db.collection("dup", store_embeddings=True)
        assert a is not b
        await db.close()

    async def test_rebuild_index_works_when_store_embeddings_true(self):
        db = AsyncVectorDB(":memory:")
        col = db.collection("rebuild", store_embeddings=True)
        emb = np.random.RandomState(0).randn(384).astype(np.float32).tolist()
        await col.add_texts(["hello"], embeddings=[emb])
        # rebuild_index() requires store_embeddings=True; if the kwarg
        # didn't propagate, this raises.
        await col.rebuild_index()
        await db.close()


class TestAsyncClusterAlgorithmValidation:
    async def test_invalid_algorithm_raises_value_error(self):
        db = AsyncVectorDB(":memory:")
        col = db.collection("c")
        with pytest.raises(ValueError, match="algorithm must be one of"):
            await col.cluster(algorithm="not-a-real-algo")
        await db.close()

    async def test_valid_algorithm_does_not_raise_validation_error(self):
        db = AsyncVectorDB(":memory:")
        col = db.collection("c")
        # Without data, sklearn raises its own error; we just need the
        # validation path to NOT raise the algorithm validation error.
        try:
            await col.cluster(algorithm="kmeans")
        except ValueError as exc:
            # If our validator fires, the message contains "algorithm must"
            # — bubble up failure. Other ValueErrors (empty data, etc.)
            # are fine.
            assert "algorithm must be one of" not in str(exc)
        except Exception:
            pass  # any non-ValueError is fine; we tested validator path
        await db.close()


class TestAsyncCloseDrainsExecutor:
    async def test_close_shuts_down_owned_executor(self):
        db = AsyncVectorDB(":memory:")
        executor = db._executor
        assert db._owns_executor is True
        await db.close()
        # After shutdown(wait=True), submitting new work raises.
        with pytest.raises(RuntimeError):
            executor.submit(lambda: 1)

    async def test_close_does_not_shutdown_external_executor(self):
        # If the user passed in their own executor, close() must NOT
        # shut it down — they own its lifecycle.
        external = ThreadPoolExecutor(max_workers=2)
        try:
            db = AsyncVectorDB(":memory:", executor=external)
            assert db._owns_executor is False
            await db.close()
            # External executor still alive.
            fut = external.submit(lambda: 42)
            assert fut.result() == 42
        finally:
            external.shutdown(wait=True)

    async def test_close_is_idempotent(self):
        db = AsyncVectorDB(":memory:")
        await db.close()
        # Second close() must not raise (executor already shut down).
        await db.close()
