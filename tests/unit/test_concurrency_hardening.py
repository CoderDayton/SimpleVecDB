"""Regressions for crashes and races found auditing the storage layer.

Each test here corresponds to a defect that was reachable from ordinary API
use, not a hypothetical:

- Removing from a memory-mapped index segfaulted the process.
- Retrying a locked write inside a transaction duplicated rows.
"""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from simplevecdb import VectorDB
from simplevecdb.async_core import AsyncVectorDB
from simplevecdb.engine.catalog import _TxState
from simplevecdb.utils import DatabaseLockedError, retry_on_lock


class TestMemoryMappedRemove:
    """`remove()` on a `view=True` index mutated a read-only mapping.

    usearch does not raise on that — it segfaults, taking the process with
    it. Any database whose index file passes the mmap threshold and then
    sees a delete was exposed.
    """

    def test_delete_against_mapped_index_does_not_crash(self, tmp_path):
        # Run out-of-process: the failure mode is SIGSEGV, which no
        # in-process assertion can catch.
        script = f"""
import numpy as np
from simplevecdb import constants
constants.USEARCH_MMAP_THRESHOLD = 1  # force the mmap path on a small index
from simplevecdb import VectorDB

path = {str(tmp_path / "v.db")!r}
db = VectorDB(path)
coll = db.collection("docs")
coll.add_texts(
    [f"t{{i}}" for i in range(20)],
    embeddings=np.random.rand(20, 8).astype(np.float32).tolist(),
)
coll.save()
db.close()

db2 = VectorDB(path)
coll2 = db2.collection("docs")
assert coll2._index._is_view, "expected the index to load memory-mapped"
coll2.delete_by_ids([1, 2, 3])
print(coll2.count(), coll2._index.size)
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )

        assert result.returncode == 0, (
            f"expected clean exit, got {result.returncode} "
            f"(-11 is SIGSEGV). stderr: {result.stderr[-500:]}"
        )
        assert result.stdout.split() == ["17", "17"]

    def test_mapped_index_is_upgraded_in_place(self, tmp_path, monkeypatch):
        from simplevecdb import constants

        monkeypatch.setattr(constants, "USEARCH_MMAP_THRESHOLD", 1)

        db = VectorDB(str(tmp_path / "v.db"))
        coll = db.collection("docs")
        coll.add_texts(
            [f"t{i}" for i in range(20)],
            embeddings=np.random.rand(20, 8).astype(np.float32).tolist(),
        )
        coll.save()
        db.close()

        db2 = VectorDB(str(tmp_path / "v.db"))
        coll2 = db2.collection("docs")
        assert coll2._index._is_view

        coll2.delete_by_ids([1])

        # The remove must have taken the index out of view mode, not
        # mutated the mapping underneath it.
        assert not coll2._index._is_view
        assert coll2._index.size == 19
        db2.close()


class TestSearchIsolation:
    """Searches must not surface another thread's uncommitted transaction.

    All collections share one SQLite connection and a connection has no
    isolation between its own operations, so raw `db.conn` reads *can* see
    uncommitted rows. Vector search cannot, for two reasons worth pinning
    down: a transaction's vector writes are buffered until commit, so an
    uncommitted row has no index entry to be found by; and the catalog reads
    a search makes each take the DB lock already.

    These tests exist so that neither property can be removed silently — drop
    the write buffering and the first one fails.
    """

    def test_search_does_not_see_a_rolled_back_write(self):
        db = VectorDB(":memory:")
        coll = db.collection("docs")
        coll.add_texts(["committed"], embeddings=[[1.0, 0.0]])

        started = threading.Event()
        seen: dict[str, int] = {}

        def reader():
            assert started.wait(5)
            seen["hits"] = len(coll.similarity_search([1.0, 0.0], k=10))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            with coll.tx() as tx:
                tx.add_texts(["doomed"], embeddings=[[0.0, 1.0]])
                started.set()
                # Give the reader time to reach the search and block on it.
                thread.join(0.3)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        thread.join(5)

        assert seen["hits"] == 1
        assert coll.count() == 1
        db.close()

    def test_search_inside_its_own_transaction_does_not_deadlock(self):
        """The DB lock is reentrant, so the owning thread may still search."""
        db = VectorDB(":memory:")
        coll = db.collection("docs")
        coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        with coll.tx() as tx:
            tx.add_texts(["inside"], embeddings=[[0.0, 1.0]])
            hits = tx.similarity_search([1.0, 0.0], k=5)

        # Only the committed row is visible: the transaction's own vector
        # write is buffered until commit.
        assert len(hits) == 1
        assert coll.count() == 2
        db.close()


@pytest.mark.asyncio
class TestAsyncSearchIsolation:
    """The same guarantees must hold through the executor.

    Async searches run in pool threads while a transaction runs in another,
    so this covers the same two properties as the sync case across the
    executor boundary. `atomic()` never awaits inside its body, so the lock
    holder always makes progress on its own worker and cannot be starved by
    queued searches.
    """

    async def test_async_search_does_not_see_a_rolled_back_write(self):
        db = AsyncVectorDB(":memory:")
        coll = db.collection("docs")
        await coll.add_texts(["committed"], embeddings=[[1.0, 0.0]])

        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def tx_body(sync_coll):
            sync_coll.add_texts(["doomed"], embeddings=[[0.0, 1.0]])
            loop.call_soon_threadsafe(started.set)
            time.sleep(0.3)  # hold the transaction open across the search
            raise RuntimeError("boom")

        async def reader():
            await started.wait()
            return len(await coll.similarity_search([1.0, 0.0], k=10))

        tx = asyncio.create_task(coll.atomic(tx_body))
        read = asyncio.create_task(reader())

        with pytest.raises(RuntimeError):
            await tx
        assert await read == 1
        assert await coll.count() == 1
        await db.close()

    async def test_saturating_the_pool_with_searches_and_transactions(self):
        db = AsyncVectorDB(":memory:")
        coll = db.collection("docs")
        await coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        def write(sync_coll):
            sync_coll.add_texts(["x"], embeddings=[[0.3, 0.7]])

        async def task(i):
            if i % 2:
                return await coll.similarity_search([1.0, 0.0], k=3)
            return await coll.atomic(write)

        # Far more tasks than the default 4 workers.
        results = await asyncio.wait_for(
            asyncio.gather(*(task(i) for i in range(24))), timeout=30
        )

        assert len(results) == 24
        assert await coll.count() == 13  # seed + 12 writers
        assert coll._collection._index.size == 13
        await db.close()

    async def test_search_inside_atomic_does_not_deadlock(self):
        db = AsyncVectorDB(":memory:")
        coll = db.collection("docs")
        await coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        def body(sync_coll):
            sync_coll.add_texts(["inner"], embeddings=[[0.0, 1.0]])
            return len(sync_coll.similarity_search([1.0, 0.0], k=5))

        hits = await asyncio.wait_for(coll.atomic(body), timeout=10)

        assert hits == 1  # the transaction's own vector write is deferred
        assert await coll.count() == 2
        await db.close()


class TestRetryInsideTransaction:
    """`@retry_on_lock` must not re-run a body inside a caller's transaction.

    Outside one, the write helper enters the connection context and a failed
    attempt is rolled back before the retry. Inside one it does not, so the
    retry would re-execute statements that already applied — duplicating the
    auto-id INSERT.
    """

    @staticmethod
    def _locking_op():
        calls: list[int] = []

        class Fake:
            _tx_state = _TxState()

            @retry_on_lock(max_retries=3, base_delay=0.001)
            def op(self):
                calls.append(1)
                raise sqlite3.OperationalError("database is locked")

        return Fake(), calls

    def test_retries_when_no_transaction_is_open(self):
        fake, calls = self._locking_op()

        with pytest.raises(DatabaseLockedError):
            fake.op()

        assert len(calls) == 4  # initial attempt + 3 retries

    def test_does_not_retry_inside_the_owning_thread_transaction(self):
        fake, calls = self._locking_op()
        fake._tx_state.owner = threading.get_ident()
        fake._tx_state.depth = 1

        # The raw error must surface so the transaction rolls back as a unit.
        with pytest.raises(sqlite3.OperationalError):
            fake.op()

        assert len(calls) == 1

    def test_still_retries_for_a_thread_outside_the_transaction(self):
        fake, calls = self._locking_op()
        # Another thread owns the transaction; this caller is not in it.
        fake._tx_state.owner = threading.get_ident() + 1
        fake._tx_state.depth = 1

        with pytest.raises(DatabaseLockedError):
            fake.op()

        assert len(calls) == 4
