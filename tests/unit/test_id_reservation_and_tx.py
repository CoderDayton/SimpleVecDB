"""Id reservation, explicit-id collision policy, and transactional vectors.

Covers:
- ``VectorCollection.reserve_ids`` — hand out ids before the rows exist, so a
  self-referential or grouped batch goes in with one write.
- ``add_texts(on_conflict=...)`` — an explicit id that already exists is an
  error by default instead of a silent overwrite.
- ``tx()`` covering the HNSW index, not just SQLite: a rollback must undo the
  vectors along with the rows.
- ``AsyncVectorCollection.atomic`` — the callback form that keeps a
  transaction's acquire and release on one executor thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading

import pytest

from simplevecdb import VectorDB
from simplevecdb.async_core import AsyncVectorDB


VEC_A = [1.0, 0.0]
VEC_B = [0.0, 1.0]


@pytest.fixture
def collection():
    # The VectorDB has to stay referenced for the whole test: dropping it
    # closes the SQLite connection out from under the collection.
    db = VectorDB(":memory:")
    coll = db.collection("docs")
    coll.add_texts(["first"], embeddings=[VEC_A])
    yield coll
    db.close()


class TestReserveIds:
    def test_returns_consecutive_ids_without_writing_rows(self, collection):
        reserved = collection.reserve_ids(3)

        assert reserved == [reserved[0], reserved[0] + 1, reserved[0] + 2]
        # Reserving is not inserting.
        assert collection.count() == 1

    def test_reserved_ids_are_never_auto_assigned_later(self, collection):
        reserved = collection.reserve_ids(5)

        auto_id = collection.add_texts(["later"], embeddings=[VEC_B])[0]

        assert auto_id > reserved[-1]

    def test_reserved_ids_are_usable_for_self_referential_rows(self, collection):
        reserved = collection.reserve_ids(2)
        group = reserved[0]

        written = collection.add_texts(
            ["a", "b"],
            metadatas=[{"id": i, "group_id": group} for i in reserved],
            embeddings=[VEC_A, VEC_B],
            ids=reserved,
        )

        assert written == reserved
        docs = collection.get_documents(filter_dict={"group_id": group})
        assert {meta["id"] for _, _, meta in docs} == set(reserved)

    def test_reservation_clears_explicitly_inserted_ids(self, collection):
        collection.add_texts(["high"], embeddings=[VEC_A], ids=[9_000])

        reserved = collection.reserve_ids(2)

        assert reserved[0] > 9_000

    def test_works_on_an_empty_collection(self):
        db = VectorDB(":memory:")
        coll = db.collection("empty")

        assert coll.reserve_ids(2) == [1, 2]

    @pytest.mark.parametrize("count", [0, -1])
    def test_rejects_non_positive_count(self, collection, count):
        with pytest.raises(ValueError, match="must be positive"):
            collection.reserve_ids(count)


class TestOnConflict:
    def test_existing_id_raises_by_default(self, collection):
        with pytest.raises(ValueError, match="already exist"):
            collection.add_texts(["clobber"], embeddings=[VEC_B], ids=[1])

    def test_rejected_call_writes_nothing(self, collection):
        with contextlib.suppress(ValueError):
            collection.add_texts(
                ["clobber", "innocent"],
                embeddings=[VEC_B, VEC_B],
                ids=[1, 4_242],
            )

        assert collection.count() == 1
        assert collection.get_documents()[0][1] == "first"

    def test_replace_overwrites(self, collection):
        collection.add_texts(
            ["replaced"],
            embeddings=[VEC_B],
            ids=[1],
            metadatas=[{"tag": "new"}],
            on_conflict="replace",
        )

        docs = collection.get_documents()
        assert len(docs) == 1
        assert docs[0][1] == "replaced"
        assert docs[0][2]["tag"] == "new"

    def test_duplicate_ids_within_one_call_raise(self, collection):
        with pytest.raises(ValueError, match="duplicate"):
            collection.add_texts(["a", "b"], embeddings=[VEC_A, VEC_B], ids=[500, 500])

    def test_unknown_policy_raises(self, collection):
        with pytest.raises(ValueError, match="on_conflict must be one of"):
            collection.add_texts(
                ["a"], embeddings=[VEC_A], ids=[500], on_conflict="upsert"
            )

    def test_fresh_explicit_ids_still_insert(self, collection):
        written = collection.add_texts(["new"], embeddings=[VEC_B], ids=[77])

        assert written == [77]
        assert collection.count() == 2


class TestTransactionCoversTheIndex:
    def test_commit_applies_both_stores(self, collection):
        with collection.tx() as coll:
            coll.add_texts(["second"], embeddings=[VEC_B])

        assert collection.count() == 2
        assert collection._index.size == 2

    def test_rollback_undoes_both_stores(self, collection):
        with pytest.raises(RuntimeError):
            with collection.tx() as coll:
                coll.add_texts(["doomed"], embeddings=[VEC_B])
                raise RuntimeError("boom")

        assert collection.count() == 1
        # The regression this guards: the row rolled back but the vector
        # stayed, leaving the index keyed to a row that no longer exists.
        assert collection._index.size == 1

    def test_rollback_undoes_deletes(self, collection):
        with pytest.raises(RuntimeError):
            with collection.tx() as coll:
                coll.delete_by_ids([1])
                raise RuntimeError("boom")

        assert collection.count() == 1
        assert collection._index.size == 1

    def test_vector_writes_are_deferred_until_commit(self, collection):
        with collection.tx() as coll:
            coll.add_texts(["second"], embeddings=[VEC_B])
            # Documented caveat: the index does not see the transaction's
            # own writes until it commits.
            assert coll._index.size == 1

        assert collection._index.size == 2

    def test_inner_rollback_keeps_outer_writes(self, collection):
        with collection.tx() as outer:
            outer.add_texts(["outer"], embeddings=[VEC_B])
            with contextlib.suppress(RuntimeError):
                with collection.tx() as inner:
                    inner.add_texts(["inner"], embeddings=[VEC_A])
                    raise RuntimeError("boom")

        assert collection.count() == 2
        assert collection._index.size == 2

    def test_another_thread_is_not_captured_by_an_open_transaction(self, collection):
        """A writer outside the transaction must not have its vectors buffered.

        Its rows are already committed, so buffering them into someone else's
        transaction means a rollback silently drops the vectors and leaves the
        index short — the exact divergence the buffering exists to prevent.
        """
        import numpy as np

        started, release = threading.Event(), threading.Event()

        def owner():
            with contextlib.suppress(RuntimeError):
                with collection.tx():
                    started.set()
                    release.wait(5)
                    raise RuntimeError("boom")

        thread = threading.Thread(target=owner)
        thread.start()
        assert started.wait(5)

        # Stands in for a non-transactional writer past its catalog commit.
        collection._index_add(
            np.array([99], dtype=np.uint64),
            np.array([VEC_B], dtype=np.float32),
        )

        release.set()
        thread.join(5)

        assert collection._index.size == 2

    def test_buffer_does_not_leak_between_transactions(self, collection):
        with pytest.raises(RuntimeError):
            with collection.tx() as coll:
                coll.add_texts(["doomed"], embeddings=[VEC_B])
                raise RuntimeError("boom")

        with collection.tx() as coll:
            coll.add_texts(["kept"], embeddings=[VEC_B])

        assert collection.count() == 2
        assert collection._index.size == 2


@pytest.mark.asyncio
class TestAsyncAtomic:
    async def _collection(self):
        """Return (db, collection); the caller must keep `db` alive."""
        db = AsyncVectorDB(":memory:")
        coll = db.collection("docs")
        await coll.add_texts(["first"], embeddings=[VEC_A])
        return db, coll

    async def test_commits_and_returns_the_callback_result(self):
        db, coll = await self._collection()

        def body(sync_coll):
            sync_coll.add_texts(["second"], embeddings=[VEC_B])
            return "done"

        assert await coll.atomic(body) == "done"
        assert await coll.count() == 2
        assert coll._collection._index.size == 2

    async def test_rolls_back_both_stores(self):
        db, coll = await self._collection()

        def body(sync_coll):
            sync_coll.add_texts(["doomed"], embeddings=[VEC_B])
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await coll.atomic(body)

        assert await coll.count() == 1
        assert coll._collection._index.size == 1

    async def test_concurrent_atomics_do_not_break_the_lock(self):
        """Two-phase `async with` would release the RLock cross-thread here."""
        db, coll = await self._collection()

        def body(sync_coll):
            sync_coll.add_texts(["batch"], embeddings=[VEC_B])
            return sync_coll.count()

        # More tasks than the default 4-worker pool, so they queue.
        counts = await asyncio.gather(*(coll.atomic(body) for _ in range(8)))

        assert sorted(counts) == list(range(2, 10))
        assert await coll.count() == 9

    async def test_reserve_ids_is_exposed(self):
        db, coll = await self._collection()

        reserved = await coll.reserve_ids(2)

        assert len(reserved) == 2
        assert await coll.count() == 1
