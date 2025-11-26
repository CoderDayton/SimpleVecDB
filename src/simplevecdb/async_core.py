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
from typing import Any

from .core import VectorDB, VectorCollection
from .types import Document, DistanceStrategy, Quantization


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

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        """
        Add texts with optional embeddings and metadata.

        See VectorCollection.add_texts for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.add_texts(texts, metadatas, embeddings, ids),
        )

    async def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for most similar vectors.

        See VectorCollection.similarity_search for full documentation.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collection.similarity_search(query, k, filter),
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
                query, k, fetch_k, filter
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
        **kwargs: Any,
    ):
        self._db = VectorDB(
            path=path,
            distance_strategy=distance_strategy,
            quantization=quantization,
            **kwargs,
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._collections: dict[tuple, AsyncVectorCollection] = {}

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

    async def close(self) -> None:
        """Close the database connection and shutdown executor."""
        self._executor.shutdown(wait=True)
        self._db.close()

    async def __aenter__(self) -> "AsyncVectorDB":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
