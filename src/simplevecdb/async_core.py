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
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Sequence
from threading import Lock
from typing import Any

import logging

from . import constants
from .core import VectorDB, VectorCollection
from .types import Document, DistanceStrategy, Quantization

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
    ):
        self._collection = sync_collection
        self._executor = executor

    @property
    def name(self) -> str:
        """Collection name."""
        return self._collection.name

    def __repr__(self) -> str:
        return f"AsyncVectorCollection(name={self._collection.name!r})"

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
        *,
        parent_ids: Sequence[int | None] | None = None,
        threads: int = 0,
    ) -> list[int]:
        """Add texts with optional embeddings and metadata.

        See VectorCollection.add_texts for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.add_texts(
                texts, metadatas, embeddings, ids,
                parent_ids=parent_ids, threads=threads,
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.similarity_search(
                query, k, filter, exact=exact, threads=threads
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.similarity_search_batch(
                queries, k, filter, exact=exact, threads=threads
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.keyword_search(query, k, filter),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.hybrid_search(
                query,
                k,
                filter,
                query_vector=query_vector,
                vector_k=vector_k,
                keyword_k=keyword_k,
                rrf_k=rrf_k,
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.max_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter
            ),
        )

    async def delete_by_ids(self, ids: Sequence[int]) -> None:
        """
        Delete documents by their IDs.

        See VectorCollection.delete_by_ids for full documentation.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._collection.delete_by_ids(ids),
        )

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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_documents(
                filter_dict=filter_dict, limit=limit, offset=offset
            ),
        )

    async def get_embeddings_by_ids(self, ids: Sequence[int]) -> dict[int, Any]:
        """Fetch stored embeddings by document IDs.

        See VectorCollection.get_embeddings_by_ids for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_embeddings_by_ids(list(ids)),
        )

    async def update_metadata(
        self, updates: list[tuple[int, dict[str, Any]]]
    ) -> int:
        """Update metadata for multiple documents (shallow merge).

        See VectorCollection.update_metadata for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.update_metadata(updates),
        )

    async def count(self) -> int:
        """Count documents in collection."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._collection.count,
        )

    async def save(self) -> None:
        """Save collection to disk."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._collection.save,
        )

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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.remove_texts(texts, filter),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.rebuild_index(
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search,
            ),
        )

    async def get_children(self, doc_id: int) -> list:
        """Get direct children of a document."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_children(doc_id),
        )

    async def get_parent(self, doc_id: int):
        """Get parent document, or None."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_parent(doc_id),
        )

    async def get_descendants(
        self, doc_id: int, max_depth: int | None = None
    ) -> list:
        """Get all descendants recursively."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_descendants(doc_id, max_depth),
        )

    async def get_ancestors(
        self, doc_id: int, max_depth: int | None = None
    ) -> list:
        """Get all ancestors to root."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_ancestors(doc_id, max_depth),
        )

    async def set_parent(self, doc_id: int, parent_id: int | None) -> bool:
        """Set or remove parent relationship."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.set_parent(doc_id, parent_id),
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

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.cluster(
                n_clusters,
                algorithm,  # type: ignore[arg-type]
                filter=filter,
                sample_size=sample_size,
                min_cluster_size=min_cluster_size,
                random_state=random_state,
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.auto_tag(
                cluster_result,
                method=method,
                n_keywords=n_keywords,
                custom_callback=custom_callback,
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.assign_cluster_metadata(
                cluster_result,
                tags,
                metadata_key=metadata_key,
                tag_key=tag_key,
            ),
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.get_cluster_members(
                cluster_id, metadata_key=metadata_key
            ),
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
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._collection.save_cluster(
                name, cluster_result, metadata=metadata
            ),
        )

    async def load_cluster(
        self,
        name: str,
    ) -> tuple[Any, dict[str, Any]] | None:
        """
        Load saved cluster state (async).

        See VectorCollection.load_cluster for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.load_cluster(name),
        )

    async def list_clusters(self) -> list[dict[str, Any]]:
        """
        List all saved cluster configurations (async).

        See VectorCollection.list_clusters for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._collection.list_clusters,
        )

    async def delete_cluster(self, name: str) -> bool:
        """
        Delete a saved cluster configuration (async).

        See VectorCollection.delete_cluster for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.delete_cluster(name),
        )

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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.assign_to_cluster(
                name, doc_ids, metadata_key=metadata_key
            ),
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
        self._db = VectorDB(path=path, distance_strategy=distance_strategy, quantization=quantization, **kwargs)
        self._owns_executor = executor is None
        self._executor = executor if executor is not None else ThreadPoolExecutor(max_workers=max_workers)
        self._collections: dict[tuple, AsyncVectorCollection] = {}
        self._collections_lock = Lock()  # Thread-safe collection caching

    def collection(
        self,
        name: str = "default",
        distance_strategy: DistanceStrategy | None = None,
        quantization: Quantization | None = None,
    ) -> AsyncVectorCollection:
        """
        Get or create a named vector collection.

        Args:
            name: Collection name (alphanumeric + underscore only).
            distance_strategy: Override database-level distance metric.
            quantization: Override database-level quantization.

        Returns:
            AsyncVectorCollection instance.
        """
        cache_key = (name, distance_strategy, quantization)
        with self._collections_lock:
            if cache_key not in self._collections:
                sync_collection = self._db.collection(
                    name,
                    distance_strategy=distance_strategy,
                    quantization=quantization,
                )
                self._collections[cache_key] = AsyncVectorCollection(
                    sync_collection, self._executor
                )
            return self._collections[cache_key]

    def list_collections(self) -> list[str]:
        """Return names of all persisted collections in the database."""
        return self._db.list_collections()

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor, lambda: self._db.delete_collection(name)
        )
        # Evict from async-level cache too
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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._db.search_collections(
                query,
                collections,
                k,
                filter,
                normalize_scores=normalize_scores,
                parallel=parallel,
            ),
        )

    async def vacuum(self, checkpoint_wal: bool = True) -> None:
        """
        Reclaim disk space by rebuilding the database file.

        Async wrapper for VectorDB.vacuum(). See sync version for details.

        Args:
            checkpoint_wal: If True (default), also truncate the WAL file.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor, lambda: self._db.vacuum(checkpoint_wal)
        )

    def __repr__(self) -> str:
        return f"AsyncVectorDB(path={self._db.path!r})"

    async def close(self) -> None:
        """Close the database connection and shutdown executor."""
        try:
            if self._owns_executor:
                # cancel_futures=True cancels pending tasks; wait=False returns
                # immediately so we don't hang if a running task is stuck.
                self._executor.shutdown(wait=False, cancel_futures=True)
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
