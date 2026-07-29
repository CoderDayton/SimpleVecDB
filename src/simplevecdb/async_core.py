"""
Async API wrappers for SimpleVecDB.

Provides async versions of VectorDB and VectorCollection for use in
async/await contexts. Uses ThreadPoolExecutor to wrap synchronous
SQLite operations.

Example:
    >>> import asyncio
    >>> from simplevecdb.async_core import AsyncVectorDB
    >>>
    >>> async def main():
    ...     db = AsyncVectorDB("data.db")
    ...     collection = db.collection("docs")
    ...     ids = await collection.add_texts(
    ...         ["Hello world"],
    ...         embeddings=[[0.1] * 384]
    ...     )
    ...     results = await collection.similarity_search([0.1] * 384, k=5)
    ...     return results
    >>>
    >>> results = asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, TypeVar

import logging

from .core import VectorDB, VectorCollection, _DBTransaction
from .types import Document, DistanceStrategy, OnConflict, Quantization

T = TypeVar("T")


class _AsyncNamespace:
    """Awaitable mirror of a sync sub-namespace (`collection.edges`, …).

    Every public callable on the wrapped namespace is re-exposed as a
    coroutine that runs the sync call in the executor. Async code therefore
    reads exactly like sync code with `await` in front —
    `await coll.edges.upsert(...)` against `coll.edges.upsert(...)` — and a
    method added to a sync namespace is reachable from async immediately,
    with no wrapper to write and no way for the two surfaces to drift.

    Non-callable attributes pass through unchanged.
    """

    __slots__ = ("_namespace", "_run")

    def __init__(self, namespace: Any, run: Callable[..., Any]) -> None:
        self._namespace = namespace
        self._run = run

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._namespace, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        async def _in_executor(*args: Any, **kwargs: Any) -> Any:
            return await self._run(attr, *args, **kwargs)

        return _in_executor

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._namespace)) | set(object.__dir__(self)))

    def __repr__(self) -> str:
        return f"Async{type(self._namespace).__name__.lstrip('_')}"


class _AsyncEventsNamespace(_AsyncNamespace):
    """Events namespace with a real async `subscribe`.

    The sync `subscribe` is a blocking generator that sleeps between polls;
    driving it from async would stall the event loop, so the loop is
    reimplemented here over the async `read` with `asyncio.sleep`.
    """

    async def subscribe(
        self,
        *,
        since: int = 0,
        kind: str | None = None,
        poll_interval: float | None = None,
        batch: int = 500,
    ) -> "AsyncIterator[Any]":
        """Async generator yielding events as they appear. Caller controls exit."""
        from . import constants

        interval = (
            constants.EVENTS_POLL_INTERVAL_S if poll_interval is None else poll_interval
        )
        last = int(since)
        while True:
            events = await self._run(
                self._namespace.read, since=last, kind=kind, limit=batch
            )
            if events:
                for event in events:
                    yield event
                last = events[-1].seq
                if len(events) == batch:
                    # Drained a full batch; go again without sleeping.
                    continue
            await asyncio.sleep(interval)


_logger = logging.getLogger(__name__)


class AsyncVectorCollection:
    """
    Async wrapper for VectorCollection.

    All methods are async versions of the synchronous VectorCollection methods,
    executed in a thread pool to avoid blocking the event loop.
    """

    def __init__(
        self,
        sync_collection: VectorCollection,
        executor: ThreadPoolExecutor,
        *,
        tx_pinned: bool = False,
    ):
        self._collection = sync_collection
        self._executor = executor
        # True when `executor` is a transaction's private single-worker pool,
        # so a nested tx() reuses that thread instead of pinning a new one.
        self._tx_pinned = tx_pinned
        self._namespaces: dict[str, _AsyncNamespace] = {}

    @property
    def name(self) -> str:
        """Collection name."""
        return self._collection.name

    def __repr__(self) -> str:
        return f"AsyncVectorCollection(name={self._collection.name!r})"

    async def _run(self, fn, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
        )

    def _namespace(self, name: str) -> _AsyncNamespace:
        """Awaitable mirror of the sync namespace `name`, created once."""
        proxy = self._namespaces.get(name)
        if proxy is None:
            cls = _AsyncEventsNamespace if name == "events" else _AsyncNamespace
            proxy = cls(getattr(self._collection, name), self._run)
            self._namespaces[name] = proxy
        return proxy

    @property
    def edges(self) -> _AsyncNamespace:
        """Awaitable mirror of `VectorCollection.edges`."""
        return self._namespace("edges")

    @property
    def events(self) -> _AsyncNamespace:
        """Awaitable mirror of `VectorCollection.events`."""
        return self._namespace("events")

    @property
    def ttl(self) -> _AsyncNamespace:
        """Awaitable mirror of `VectorCollection.ttl`."""
        return self._namespace("ttl")

    @property
    def pending(self) -> _AsyncNamespace:
        """Awaitable mirror of `VectorCollection.pending`."""
        return self._namespace("pending")

    @property
    def maintenance(self) -> _AsyncNamespace:
        """Awaitable mirror of `VectorCollection.maintenance`."""
        return self._namespace("maintenance")

    @property
    def counters(self) -> _AsyncNamespace:
        """Awaitable mirror of `VectorCollection.counters`."""
        return self._namespace("counters")

    async def add_texts_streaming(
        self,
        items: Iterable[tuple[str, dict | None, Sequence[float] | None]],
        *,
        batch_size: int | None = None,
        threads: int = 0,
        on_progress: Any = None,
    ) -> list[int]:
        """Stream documents in batches, returning every inserted id.

        The sync method is a generator yielding per-batch progress; the whole
        drain runs in one executor task here, so `on_progress` fires from
        that thread rather than the event loop.

        See VectorCollection.add_texts_streaming for full documentation.
        """

        def _drain(coll: VectorCollection) -> list[int]:
            ids: list[int] = []
            for progress in coll.add_texts_streaming(
                items,
                batch_size=batch_size,
                threads=threads,
                on_progress=on_progress,
            ):
                ids.extend(progress["batch_ids"])
            return ids

        return await self._run(_drain, self._collection)

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["AsyncVectorCollection"]:
        """Async mirror of `VectorCollection.tx()`.

            async with collection.tx() as coll:
                await coll.delete_by_ids([1])
                await coll.add_texts(["replacement"], embeddings=[vec])

        A transaction holds a `threading.RLock` for its lifetime, so its
        enter and exit must happen on one thread. The shared pool cannot
        promise that — two executor tasks may land on different workers, and
        releasing an RLock from a thread that never acquired it raises
        `RuntimeError: cannot release un-acquired lock`. So the transaction
        gets a private single-worker executor: one thread, every step on it.

        **Operate through the yielded handle.** It is bound to the pinned
        thread; the outer collection is not. Awaiting work on the outer
        handle inside the block sends it to the shared pool, where it blocks
        on the DB lock this transaction holds — and that lock is only
        released when the block exits, which cannot happen while it is
        awaiting. Use `atomic()` if you want that mistake to be
        unrepresentable: its body is synchronous and cannot await at all.
        """
        loop = asyncio.get_running_loop()
        # A nested tx must run on the thread that already holds the lock: the
        # RLock is reentrant per thread, so pinning a second thread here
        # would block forever waiting on the outer transaction.
        reuse = self._tx_pinned
        pinned = (
            self._executor
            if reuse
            else ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"simplevecdb-tx-{self.name}"
            )
        )

        def _on_pinned(fn: Callable[..., Any], *args: Any) -> Any:
            return loop.run_in_executor(pinned, functools.partial(fn, *args))

        try:
            manager = self._collection.tx()
            scoped_sync = await _on_pinned(manager.__enter__)
            scoped = AsyncVectorCollection(scoped_sync, pinned, tx_pinned=True)
            try:
                yield scoped
            except (GeneratorExit, asyncio.CancelledError) as exc:
                # Teardown paths where awaiting is unsafe. Suspending while a
                # GeneratorExit is in flight raises "async generator ignored
                # GeneratorExit", and a cancelled task's next await can be
                # cancelled again — either way the savepoint would stay open
                # and the DB lock would never be released, wedging every other
                # writer. Drive the exit on the pinned thread without
                # suspending.
                pinned.submit(
                    manager.__exit__, type(exc), exc, exc.__traceback__
                ).result()
                raise
            except BaseException as exc:
                await _on_pinned(manager.__exit__, type(exc), exc, exc.__traceback__)
                raise
            await _on_pinned(manager.__exit__, None, None, None)
        finally:
            if not reuse:
                # Non-blocking: the transaction is over, so waiting here would
                # only stall the event loop on work the body queued and never
                # awaited — work that must not run now that the savepoint has
                # closed. cancel_futures drops exactly that.
                pinned.shutdown(wait=False, cancel_futures=True)

    async def atomic(self, fn: Callable[[VectorCollection], T]) -> T:
        """Run `fn` inside a transaction on this collection.

        `fn` is an ordinary synchronous callable and receives the underlying
        `VectorCollection`; everything it does — catalog writes and vector
        writes alike — commits or rolls back as one unit.

            async def swap(coll):
                coll.delete_by_ids([1])
                coll.add_texts(["replacement"], embeddings=[vec])

            await collection.atomic(swap)

        This is a callback rather than `async with collection.tx()` on
        purpose. The transaction holds a `threading.RLock` for its whole
        lifetime, and an `async with` would enter and exit in two separate
        executor tasks: the pool is free to run them on different threads,
        and releasing an RLock from a thread that did not acquire it raises
        `RuntimeError: cannot release un-acquired lock`. Landing on the same
        thread by luck would be no better — every other call on this
        collection would sit on that lock across each `await` in the body,
        which starves a pool this small. Running the whole body in one
        executor task keeps acquire and release paired on one thread.

        Because `fn` runs off the event loop, it must not await; use the
        sync collection API inside it.

        Args:
            fn: Callable invoked with the sync collection.

        Returns:
            Whatever `fn` returns.
        """
        return await self._run(self._in_tx, fn)

    def _in_tx(self, fn: Callable[[VectorCollection], T]) -> T:
        """Body of `atomic`, run wholly inside one executor thread."""
        with self._collection.tx() as coll:
            return fn(coll)

    async def reserve_ids(self, count: int) -> list[int]:
        """Reserve document ids without writing rows.

        See VectorCollection.reserve_ids for full documentation.
        """
        return await self._run(self._collection.reserve_ids, count)

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
        *,
        parent_ids: Sequence[int | None] | None = None,
        threads: int = 0,
        on_conflict: OnConflict = "error",
    ) -> list[int]:
        """Add texts with optional embeddings and metadata.

        See VectorCollection.add_texts for full documentation.
        """
        return await self._run(
            self._collection.add_texts,
            texts,
            metadatas,
            embeddings,
            ids,
            parent_ids=parent_ids,
            threads=threads,
            on_conflict=on_conflict,
        )

    async def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
        threads: int = 0,
    ) -> list[tuple[Document, float]]:
        """
        Search for most similar vectors.

        See VectorCollection.similarity_search for full documentation.
        """
        return await self._run(
            self._collection.similarity_search,
            query,
            k,
            filter,
            exact=exact,
            threads=threads,
        )

    async def similarity_search_batch(
        self,
        queries: Sequence[Sequence[float]],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
        threads: int = 0,
    ) -> list[list[tuple[Document, float]]]:
        """
        Batch search for multiple query vectors.

        See VectorCollection.similarity_search_batch for full documentation.
        """
        return await self._run(
            self._collection.similarity_search_batch,
            queries,
            k,
            filter,
            exact=exact,
            threads=threads,
        )

    async def keyword_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search using BM25 keyword ranking.

        See VectorCollection.keyword_search for full documentation.
        """
        return await self._run(
            self._collection.keyword_search,
            query,
            k,
            filter,
        )

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        query_vector: Sequence[float] | None = None,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        """
        Combine keyword and vector search using Reciprocal Rank Fusion.

        See VectorCollection.hybrid_search for full documentation.
        """
        return await self._run(
            self._collection.hybrid_search,
            query,
            k,
            filter,
            query_vector=query_vector,
            vector_k=vector_k,
            keyword_k=keyword_k,
            rrf_k=rrf_k,
        )

    async def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search with diversity using Max Marginal Relevance.

        See VectorCollection.max_marginal_relevance_search for full documentation.
        """
        return await self._run(
            self._collection.max_marginal_relevance_search,
            query,
            k,
            fetch_k,
            lambda_mult,
            filter,
        )

    async def delete_by_ids(self, ids: Sequence[int]) -> None:
        """
        Delete documents by their IDs.

        See VectorCollection.delete_by_ids for full documentation.
        """
        await self._run(self._collection.delete_by_ids, ids)

    async def get_documents(
        self,
        filter_dict: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """Get documents with text content and metadata.

        See VectorCollection.get_documents for full documentation.
        """
        return await self._run(
            self._collection.get_documents,
            filter_dict=filter_dict,
            limit=limit,
            offset=offset,
        )

    async def get_embeddings_by_ids(self, ids: Sequence[int]) -> dict[int, Any]:
        """Fetch stored embeddings by document IDs.

        See VectorCollection.get_embeddings_by_ids for full documentation.
        """
        return await self._run(
            self._collection.get_embeddings_by_ids,
            list(ids),
        )

    async def update_metadata(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        """Update metadata for multiple documents (shallow merge).

        See VectorCollection.update_metadata for full documentation.
        """
        return await self._run(self._collection.update_metadata, updates)

    async def count(self) -> int:
        """Count documents in collection."""
        return await self._run(self._collection.count)

    async def save(self) -> None:
        """Save collection to disk."""
        await self._run(self._collection.save)

    @property
    def dim(self) -> int | None:
        """Vector dimension (None if no vectors added yet)."""
        return self._collection.dim

    async def remove_texts(
        self,
        texts: Sequence[str] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> int:
        """
        Remove documents by text content or metadata filter.

        See VectorCollection.remove_texts for full documentation.
        """
        return await self._run(self._collection.remove_texts, texts, filter)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6.1 — pending vectors, counters, edges, events, TTL (Async)
    # ─────────────────────────────────────────────────────────────────────────

    async def update_embedding(
        self,
        doc_id: int,
        vector: Any,
        *,
        source: str | None = None,
    ) -> None:
        """Buffer a vector update; promoted to HNSW on flush_pending()."""
        await self._run(
            self._collection.update_embedding,
            doc_id,
            vector,
            source=source,
        )

    async def flush_pending(self, *, max_batch: int | None = None) -> int:
        """Flush buffered vector updates into the HNSW index."""
        return await self._run(self._collection.pending.flush, max_batch=max_batch)

    async def increment_metadata(
        self,
        doc_id: int,
        deltas: dict[str, int | float],
    ) -> int:
        """Atomically apply numeric deltas to JSON metadata counters.

        Returns 1 if the row existed and was updated, 0 otherwise.
        """
        return await self._run(self._collection.increment_metadata, doc_id, deltas)

    async def add_edge(
        self,
        src: int,
        dst: int,
        *,
        kind: str = "",
        weight: float = 0.0,
        bonus: float = 0.0,
        hits: int = 0,
        metadata: dict | None = None,
    ) -> int:
        return await self._run(
            self._collection.edges.add_edge,
            src,
            dst,
            kind=kind,
            weight=weight,
            bonus=bonus,
            hits=hits,
            metadata=metadata,
        )

    async def update_edge(
        self,
        src: int,
        dst: int,
        *,
        kind: str = "",
        weight: float | None = None,
        bonus: float | None = None,
        hits: int | None = None,
        metadata: dict | None = None,
        dweight: float = 0.0,
        dbonus: float = 0.0,
        dhits: int = 0,
    ) -> int:
        return await self._run(
            self._collection.edges.update_edge,
            src,
            dst,
            kind=kind,
            weight=weight,
            bonus=bonus,
            hits=hits,
            metadata=metadata,
            dweight=dweight,
            dbonus=dbonus,
            dhits=dhits,
        )

    async def delete_edge(
        self,
        src: int,
        dst: int,
        *,
        kind: str = "",
    ) -> int:
        return await self._run(
            self._collection.edges.delete_edge,
            src,
            dst,
            kind=kind,
        )

    async def get_edges(
        self,
        src: int | None = None,
        dst: int | None = None,
        *,
        kind: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list:
        return await self._run(
            self._collection.edges.get_edges,
            src=src,
            dst=dst,
            kind=kind,
            filter=filter,
            limit=limit,
        )

    async def set_ttl(
        self,
        doc_id: int,
        *,
        seconds: float | None = None,
        expires_at: float | None = None,
        on_expire: str = "delete",
    ) -> None:
        await self._run(
            self._collection.ttl.set,
            doc_id,
            seconds=seconds,
            expires_at=expires_at,
            on_expire=on_expire,
        )

    async def clear_ttl(self, doc_id: int) -> None:
        await self._run(self._collection.ttl.clear, doc_id)

    async def sweep_ttl(
        self,
        *,
        now: float | None = None,
        limit: int = 1000,
    ) -> tuple[list[int], list[int]]:
        return await self._run(
            self._collection.ttl.sweep,
            now=now,
            limit=limit,
        )

    async def read_events(
        self,
        *,
        since: int = 0,
        kind: str | None = None,
        limit: int = 500,
    ) -> list:
        return await self._run(
            self._collection.events.read,
            since=since,
            kind=kind,
            limit=limit,
        )

    async def last_event_seq(self) -> int:
        return await self._run(self._collection.events.last_seq)

    async def rebuild_if_needed(
        self,
        *,
        max_pending: int | None = None,
        max_deleted: int | None = None,
    ) -> bool:
        kwargs: dict[str, Any] = {}
        if max_pending is not None:
            kwargs["max_pending"] = max_pending
        if max_deleted is not None:
            kwargs["max_deleted"] = max_deleted
        return await self._run(
            self._collection.maintenance.rebuild_if_needed,
            **kwargs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Index & Hierarchy (Async)
    # ─────────────────────────────────────────────────────────────────────────

    async def rebuild_index(
        self,
        *,
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
    ) -> int:
        """Rebuild the HNSW index. See VectorCollection.rebuild_index."""
        return await self._run(
            self._collection.rebuild_index,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )

    async def get_children(self, doc_id: int) -> list:
        """Get direct children of a document."""
        return await self._run(self._collection.get_children, doc_id)

    async def get_parent(self, doc_id: int):
        """Get parent document, or None."""
        return await self._run(self._collection.get_parent, doc_id)

    async def get_descendants(self, doc_id: int, max_depth: int | None = None) -> list:
        """Get all descendants recursively."""
        return await self._run(
            self._collection.get_descendants,
            doc_id,
            max_depth,
        )

    async def get_ancestors(self, doc_id: int, max_depth: int | None = None) -> list:
        """Get all ancestors to root."""
        return await self._run(
            self._collection.get_ancestors,
            doc_id,
            max_depth,
        )

    async def set_parent(self, doc_id: int, parent_id: int | None) -> bool:
        """Set or remove parent relationship."""
        return await self._run(
            self._collection.set_parent,
            doc_id,
            parent_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Clustering Methods (Async)
    # ─────────────────────────────────────────────────────────────────────────

    async def cluster(
        self,
        n_clusters: int | None = None,
        algorithm: str = "minibatch_kmeans",
        *,
        filter: dict[str, Any] | None = None,
        sample_size: int | None = None,
        min_cluster_size: int = 5,
        random_state: int | None = None,
    ) -> Any:
        """
        Cluster documents by their embeddings (async).

        See VectorCollection.cluster for full documentation.
        """
        # Runtime-validate algorithm so we can drop the prior ``# type: ignore``
        # and produce a clear ValueError instead of a confusing internal
        # failure deep in the sync code.
        valid_algorithms = ("kmeans", "minibatch_kmeans", "hdbscan")
        if algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms!r}; got {algorithm!r}"
            )

        from typing import cast, Literal

        narrowed = cast(Literal["kmeans", "minibatch_kmeans", "hdbscan"], algorithm)

        return await self._run(
            self._collection.cluster,
            n_clusters,
            narrowed,
            filter=filter,
            sample_size=sample_size,
            min_cluster_size=min_cluster_size,
            random_state=random_state,
        )

    async def auto_tag(
        self,
        cluster_result: Any,
        *,
        method: str = "keywords",
        n_keywords: int = 5,
        custom_callback: Any = None,
    ) -> dict[int, str]:
        """
        Generate descriptive tags for clusters (async).

        See VectorCollection.auto_tag for full documentation.
        """
        return await self._run(
            self._collection.auto_tag,
            cluster_result,
            method=method,
            n_keywords=n_keywords,
            custom_callback=custom_callback,
        )

    async def assign_cluster_metadata(
        self,
        cluster_result: Any,
        tags: dict[int, str] | None = None,
        *,
        metadata_key: str = "cluster",
        tag_key: str = "cluster_tag",
    ) -> int:
        """
        Persist cluster assignments to metadata (async).

        See VectorCollection.assign_cluster_metadata for full documentation.
        """
        return await self._run(
            self._collection.assign_cluster_metadata,
            cluster_result,
            tags,
            metadata_key=metadata_key,
            tag_key=tag_key,
        )

    async def get_cluster_members(
        self,
        cluster_id: int,
        *,
        metadata_key: str = "cluster",
    ) -> list[Document]:
        """
        Get all documents in a cluster (async).

        See VectorCollection.get_cluster_members for full documentation.
        """
        return await self._run(
            self._collection.get_cluster_members,
            cluster_id,
            metadata_key=metadata_key,
        )

    async def save_cluster(
        self,
        name: str,
        cluster_result: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save cluster state for later assignment (async).

        See VectorCollection.save_cluster for full documentation.
        """
        await self._run(
            self._collection.save_cluster,
            name,
            cluster_result,
            metadata=metadata,
        )

    async def load_cluster(
        self,
        name: str,
    ) -> tuple[Any, dict[str, Any]] | None:
        """
        Load saved cluster state (async).

        See VectorCollection.load_cluster for full documentation.
        """
        return await self._run(self._collection.load_cluster, name)

    async def list_clusters(self) -> list[dict[str, Any]]:
        """
        List all saved cluster configurations (async).

        See VectorCollection.list_clusters for full documentation.
        """
        return await self._run(self._collection.list_clusters)

    async def delete_cluster(self, name: str) -> bool:
        """
        Delete a saved cluster configuration (async).

        See VectorCollection.delete_cluster for full documentation.
        """
        return await self._run(self._collection.delete_cluster, name)

    async def assign_to_cluster(
        self,
        name: str,
        doc_ids: list[int],
        *,
        metadata_key: str = "cluster",
    ) -> int:
        """
        Assign documents to a saved cluster (async).

        See VectorCollection.assign_to_cluster for full documentation.
        """
        return await self._run(
            self._collection.assign_to_cluster,
            name,
            doc_ids,
            metadata_key=metadata_key,
        )


class AsyncVectorDB:
    """
    Async wrapper for VectorDB.

    Creates a thread pool executor for running synchronous SQLite operations
    without blocking the async event loop.

    Example:
        >>> async def main():
        ...     db = AsyncVectorDB("my_vectors.db")
        ...     collection = db.collection("documents")
        ...     await collection.add_texts(["hello"], embeddings=[[0.1]*384])
        ...     results = await collection.similarity_search([0.1]*384)
        ...     await db.close()

    Args:
        path: Path to SQLite database file. Use ":memory:" for in-memory DB.
        distance_strategy: Distance metric (COSINE, L2, or L1).
        quantization: Vector quantization (FLOAT, INT8, or BIT).
        max_workers: Number of threads in executor pool. Default 4.
        **kwargs: Additional arguments passed to VectorDB.
    """

    def __init__(
        self,
        path: str = ":memory:",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        quantization: Quantization = Quantization.FLOAT,
        max_workers: int = 4,
        *,
        executor: ThreadPoolExecutor | None = None,
        **kwargs: Any,
    ):
        self._db = VectorDB(
            path=path,
            distance_strategy=distance_strategy,
            quantization=quantization,
            **kwargs,
        )
        self._owns_executor = executor is None
        self._executor = (
            executor
            if executor is not None
            else ThreadPoolExecutor(max_workers=max_workers)
        )
        self._collections: dict[tuple, AsyncVectorCollection] = {}
        self._collections_lock = Lock()  # Thread-safe collection caching

    def collection(
        self,
        name: str = "default",
        distance_strategy: DistanceStrategy | None = None,
        quantization: Quantization | None = None,
        store_embeddings: bool = False,
    ) -> AsyncVectorCollection:
        """
        Get or create a named vector collection.

        Args:
            name: Collection name (alphanumeric + underscore only).
            distance_strategy: Override database-level distance metric.
            quantization: Override database-level quantization.
            store_embeddings: If True, store embeddings as BLOBs in SQLite
                alongside the usearch index. Required for ``rebuild_index()``.
                Mirrors ``VectorDB.collection``; without this argument async
                callers had no way to enable embedding storage.

        Returns:
            AsyncVectorCollection instance.
        """
        cache_key = (name, distance_strategy, quantization, store_embeddings)
        with self._collections_lock:
            if cache_key not in self._collections:
                sync_collection = self._db.collection(
                    name,
                    distance_strategy=distance_strategy,
                    quantization=quantization,
                    store_embeddings=store_embeddings,
                )
                self._collections[cache_key] = AsyncVectorCollection(
                    sync_collection, self._executor
                )
            return self._collections[cache_key]

    async def _run(self, fn, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
        )

    async def save(self) -> None:
        """Persist every collection's index to disk.

        See VectorDB.save for full documentation.
        """
        await self._run(self._db.save)

    async def transaction(self, fn: Callable[["_DBTransaction"], T]) -> T:
        """Run `fn` inside a database-wide transaction.

        The DB-level counterpart to `AsyncVectorCollection.atomic`: `fn` is a
        synchronous callable receiving the transaction handle, so it can span
        collections via `tx["name"]`. Everything it does — catalog and vector
        writes across every collection — commits or rolls back together.

            def move(tx):
                tx["archive"].add_texts(texts, embeddings=vecs)
                tx["inbox"].delete_by_ids(ids)

            await db.transaction(move)

        A callback rather than `async with` for the same reason as `atomic`:
        the transaction holds a `threading.RLock` across its lifetime and
        must acquire and release it on one thread. `fn` must not await.
        """

        def _in_tx() -> T:
            with self._db.transaction() as tx:
                return fn(tx)

        return await self._run(_in_tx)

    def as_langchain(self, embeddings: Any = None, collection_name: str = "default"):
        """Return a LangChain-compatible vector store over the sync database.

        Synchronous by design: LangChain drives its own async surface.
        """
        return self._db.as_langchain(embeddings, collection_name)

    def as_llama_index(self, collection_name: str = "default"):
        """Return a LlamaIndex-compatible vector store over the sync database.

        Synchronous by design: LlamaIndex drives its own async surface.
        """
        return self._db.as_llama_index(collection_name)

    def list_collections(self) -> list[str]:
        """Return names of all persisted collections in the database."""
        return self._db.list_collections()

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        await self._run(self._db.delete_collection, name)
        # Evict from async-level cache too — match any tuple whose first
        # element is this name (the cache key now includes store_embeddings).
        with self._collections_lock:
            keys_to_remove = [k for k in self._collections if k[0] == name]
            for k in keys_to_remove:
                del self._collections[k]

    async def search_collections(
        self,
        query: Sequence[float],
        collections: list[str] | None = None,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        *,
        normalize_scores: bool = True,
        parallel: bool = True,
    ) -> list[tuple[Document, float, str]]:
        """
        Search across multiple collections with merged, ranked results.

        See VectorDB.search_collections for full documentation.
        """
        return await self._run(
            self._db.search_collections,
            query,
            collections,
            k,
            filter,
            normalize_scores=normalize_scores,
            parallel=parallel,
        )

    async def vacuum(self, checkpoint_wal: bool = True) -> None:
        """
        Reclaim disk space by rebuilding the database file.

        Async wrapper for VectorDB.vacuum(). See sync version for details.

        Args:
            checkpoint_wal: If True (default), also truncate the WAL file.
        """
        await self._run(self._db.vacuum, checkpoint_wal)

    def __repr__(self) -> str:
        return f"AsyncVectorDB(path={self._db.path!r})"

    async def close(self) -> None:
        """Close the database connection and shutdown executor.

        Drains in-flight tasks (`wait=True`) before closing the SQLite
        connection. Otherwise pool threads can still hold cursors against
        ``self._db.conn`` when ``self._db.close()`` runs the connection's
        close, producing use-after-close races and silent data loss.
        Pending (not-yet-started) work is cancelled.
        """
        try:
            if self._owns_executor:
                # cancel_futures=True cancels work that hasn't started yet;
                # wait=True drains anything already executing so the SQLite
                # connection is not closed under live threads.
                self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            _logger.warning("Executor shutdown failed", exc_info=True)
        finally:
            self._db.close()

    async def __aenter__(self) -> "AsyncVectorDB":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
