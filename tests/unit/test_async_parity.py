"""The async surface must mirror the sync one, permanently.

The async API used to be hand-written wrapper by wrapper, so it drifted:
eight collection methods and four database methods had no async counterpart,
and the sub-namespaces were flattened into inconsistent names
(`ttl.sweep()` became `sweep_ttl()`).

Async namespaces are now generic proxies over the sync ones, so a method
added to a sync namespace is reachable from async with no wrapper to write.
These tests hold the line for the parts that are still declared by hand: add
a public sync method without an async counterpart and this file fails.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from simplevecdb.async_core import AsyncVectorCollection, AsyncVectorDB
from simplevecdb.core import VectorCollection, VectorDB

# Nothing is sync-only: every public sync method has an async counterpart.
# Add a name here only with the reason it cannot be mirrored.
INTENTIONALLY_SYNC_ONLY: set[str] = set()

NAMESPACES = ("edges", "events", "ttl", "pending", "maintenance", "counters")

# `events.subscribe` is an async generator polling with `asyncio.sleep`
# instead of a blocking one polling with `time.sleep`; its return annotation
# and `poll_interval` default legitimately differ from the sync method.
SIGNATURE_EXEMPT = {"subscribe"}


def public_api(cls: type) -> set[str]:
    """Public methods and properties declared on a class."""
    return {
        name
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_")
        and (inspect.isfunction(value) or isinstance(value, property))
    }


class TestSurfaceParity:
    def test_every_collection_method_has_an_async_counterpart(self):
        missing = public_api(VectorCollection) - public_api(AsyncVectorCollection)

        assert missing <= INTENTIONALLY_SYNC_ONLY, (
            f"AsyncVectorCollection is missing {sorted(missing - INTENTIONALLY_SYNC_ONLY)}. "
            "Add a wrapper, or add the name to INTENTIONALLY_SYNC_ONLY with a reason."
        )

    def test_every_database_method_has_an_async_counterpart(self):
        missing = public_api(VectorDB) - public_api(AsyncVectorDB)

        assert not missing, (
            f"AsyncVectorDB is missing {sorted(missing)}. "
            "Add a wrapper, or document the omission."
        )

    def test_both_transaction_forms_are_available(self):
        """`tx()` mirrors sync; `atomic()` is the deadlock-proof alternative."""
        assert "tx" in public_api(AsyncVectorCollection)
        assert "atomic" in public_api(AsyncVectorCollection)
        assert "transaction" in public_api(AsyncVectorDB)


@pytest.mark.asyncio
class TestNamespaceParity:
    @staticmethod
    def _collection():
        db = AsyncVectorDB(":memory:")
        return db, db.collection("docs")

    @pytest.mark.parametrize("namespace", NAMESPACES)
    async def test_namespace_exposes_every_sync_method(self, namespace):
        db, coll = self._collection()
        sync_ns = getattr(coll._collection, namespace)
        async_ns = getattr(coll, namespace)

        for name in dir(sync_ns):
            if name.startswith("_"):
                continue
            assert hasattr(async_ns, name), (
                f"coll.{namespace}.{name} exists on the sync namespace "
                f"but not the async one"
            )
        await db.close()

    @pytest.mark.parametrize("namespace", NAMESPACES)
    async def test_namespace_methods_are_awaitable(self, namespace):
        db, coll = self._collection()
        async_ns = getattr(coll, namespace)
        sync_ns = getattr(coll._collection, namespace)

        for name in dir(sync_ns):
            if name.startswith("_") or not callable(getattr(sync_ns, name)):
                continue
            attr = getattr(async_ns, name)
            assert inspect.iscoroutinefunction(attr) or inspect.isasyncgenfunction(
                attr
            ), f"coll.{namespace}.{name} should be awaitable, got {attr!r}"
        await db.close()

    @pytest.mark.parametrize("namespace", NAMESPACES)
    async def test_namespace_methods_keep_their_signature(self, namespace):
        """functools.wraps must carry the sync signature onto the proxy."""
        db, coll = self._collection()
        async_ns = getattr(coll, namespace)
        sync_ns = getattr(coll._collection, namespace)

        for name in dir(sync_ns):
            sync_attr = getattr(sync_ns, name)
            if (
                name.startswith("_")
                or not callable(sync_attr)
                or name in SIGNATURE_EXEMPT
            ):
                continue
            assert inspect.signature(getattr(async_ns, name)) == inspect.signature(
                sync_attr
            ), f"coll.{namespace}.{name} signature drifted from sync"
        await db.close()

    async def test_async_with_tx_commits(self):
        db, coll = self._collection()
        await coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        async with coll.tx() as scoped:
            await scoped.add_texts(["committed"], embeddings=[[0.0, 1.0]])

        assert await coll.count() == 2
        assert coll._collection._index.size == 2
        await db.close()

    async def test_async_with_tx_rolls_back_both_stores(self):
        db, coll = self._collection()
        await coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        with pytest.raises(RuntimeError):
            async with coll.tx() as scoped:
                await scoped.add_texts(["doomed"], embeddings=[[0.0, 1.0]])
                raise RuntimeError("boom")

        assert await coll.count() == 1
        assert coll._collection._index.size == 1
        await db.close()

    async def test_nested_async_with_tx_reuses_the_pinned_thread(self):
        """A nested tx must not pin a second thread, or it self-deadlocks."""
        db, coll = self._collection()
        await coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        async def nested():
            async with coll.tx() as outer:
                await outer.add_texts(["outer"], embeddings=[[0.1, 0.9]])
                async with outer.tx() as inner:
                    await inner.add_texts(["inner"], embeddings=[[0.9, 0.1]])

        await asyncio.wait_for(nested(), timeout=15)

        assert await coll.count() == 3
        assert coll._collection._index.size == 3
        await db.close()

    async def test_cancelling_a_transaction_releases_the_lock(self):
        """Teardown must not await, or the savepoint and DB lock leak.

        `async with` is driven by an async generator: cancelling the task
        throws GeneratorExit in at the yield, and suspending there raises
        "async generator ignored GeneratorExit" — leaving the transaction
        open and every other writer wedged behind its lock.
        """
        db, coll = self._collection()
        await coll.add_texts(["seed"], embeddings=[[1.0, 0.0]])

        entered = asyncio.Event()

        async def body():
            async with coll.tx() as scoped:
                await scoped.add_texts(["cancelled"], embeddings=[[0.0, 1.0]])
                entered.set()
                await asyncio.sleep(30)

        task = asyncio.create_task(body())
        await asyncio.wait_for(entered.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Would hang if the lock leaked, and the cancelled write must be gone.
        async def after():
            async with coll.tx() as scoped:
                await scoped.add_texts(["after"], embeddings=[[0.3, 0.7]])

        await asyncio.wait_for(after(), timeout=15)

        assert await coll.count() == 2
        assert coll._collection._index.size == 2
        await db.close()

    async def test_namespaces_work_through_a_scoped_transaction(self):
        db, coll = self._collection()

        async with coll.tx() as scoped:
            ids = await scoped.add_texts(["a"], embeddings=[[1.0, 0.0]])
            await scoped.counters.increment(ids[0], {"hits": 1})

        docs = await coll.get_documents()
        assert docs[0][2]["hits"] == 1
        await db.close()

    async def test_events_subscribe_is_a_real_async_generator(self):
        """The sync generator blocks between polls; async must not."""
        db, coll = self._collection()

        assert inspect.isasyncgenfunction(coll.events.subscribe)

        await coll.add_texts(["a"], embeddings=[[1.0, 0.0]])
        await coll.events.append("marker")

        seen = []
        async for event in coll.events.subscribe(since=0, poll_interval=0.01):
            seen.append(event.kind)
            break

        assert seen
        await db.close()
