"""
SearchEngine: Vector and hybrid search operations for SimpleVecDB.

Handles similarity search using usearch HNSW index, keyword search using
SQLite FTS5, and hybrid search combining both with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import numpy as np
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

from ..types import Document, DistanceStrategy
from ..utils import validate_filter
from .. import constants

if TYPE_CHECKING:
    from .usearch_index import UsearchIndex
    from .catalog import CatalogManager


class SearchEngine:
    """
    Handles all search operations for a VectorCollection.

    Provides:
    - Vector similarity search via usearch HNSW index
    - Keyword search via SQLite FTS5
    - Hybrid search with Reciprocal Rank Fusion
    - Max Marginal Relevance (MMR) for diversity

    Args:
        index: UsearchIndex for vector operations
        catalog: CatalogManager for metadata/FTS operations
        distance_strategy: Distance metric for result interpretation
    """

    def __init__(
        self,
        index: UsearchIndex,
        catalog: CatalogManager,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    ):
        self._index = index
        self._catalog = catalog
        self._distance_strategy = distance_strategy

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = constants.DEFAULT_K,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
        threads: int = 0,
    ) -> list[tuple[Document, float]]:
        """
        Perform vector similarity search.

        Args:
            query: Query vector or text (auto-embedded if string)
            k: Number of results to return
            filter: Optional metadata filter
            exact: Force search mode. None=adaptive (brute-force for <10k vectors),
                   True=always brute-force (perfect recall), False=always HNSW.
            threads: Number of threads for parallel search (0=auto)

        Returns:
            List of (Document, distance) tuples sorted by distance (lower = more similar)
        """
        # Validate filter structure early
        validate_filter(filter)

        query_vec = self._resolve_query_vector(query)

        if not filter:
            # No filter: simple fetch
            keys, distances = self._index.search(
                query_vec, k, exact=exact, threads=threads
            )
            if len(keys) == 0:
                return []
            keys_list = keys.tolist()
            dist_list = distances.tolist()
            docs_map = self._catalog.get_documents_by_ids(keys_list)
            return [
                (Document(page_content=text, metadata=metadata), float(dist))
                for key, dist in zip(keys_list, dist_list)
                if key in docs_map
                for text, metadata in [docs_map[key]]
            ][:k]

        # Filtered search: iterative deepening to ensure k results
        multiplier = constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER
        max_multiplier = 30
        index_size = self._index.size
        added_keys: set[int] = set()
        results: list[tuple[Document, float]] = []

        while len(results) < k and multiplier <= max_multiplier:
            fetch_k = min(k * multiplier, index_size) if index_size > 0 else k * multiplier
            keys, distances = self._index.search(
                query_vec, fetch_k, exact=exact, threads=threads
            )
            if len(keys) == 0:
                break

            keys_list = keys.tolist()
            dist_list = distances.tolist()

            # Fetch docs for keys not yet processed
            new_keys = [key for key in keys_list if key not in added_keys]
            if not new_keys:
                break
            docs_map = self._catalog.get_documents_by_ids(new_keys)

            for key, dist in zip(keys_list, dist_list):
                if key in added_keys:
                    continue
                added_keys.add(key)

                if key not in docs_map:
                    continue

                text, metadata = docs_map[key]
                if not self._matches_filter(metadata, filter):
                    continue

                results.append((Document(page_content=text, metadata=metadata), float(dist)))
                if len(results) >= k:
                    break

            if len(results) >= k or fetch_k >= index_size:
                break
            multiplier *= 2

        return results[:k]

    def similarity_search_batch(
        self,
        queries: Sequence[Sequence[float]],
        k: int = constants.DEFAULT_K,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
        threads: int = 0,
    ) -> list[list[tuple[Document, float]]]:
        """
        Perform batch vector similarity search for multiple queries.

        Automatically uses usearch's native batch search for ~10x throughput
        compared to sequential single-query searches.

        Args:
            queries: List of query vectors
            k: Number of results per query
            filter: Optional metadata filter (applied to all queries)
            exact: Force search mode. None=adaptive, True=brute-force, False=HNSW.
            threads: Number of threads for parallel search (0=auto)

        Returns:
            List of result lists, one per query. Each result is (Document, distance).
        """
        if not queries:
            return []

        validate_filter(filter)

        # For small query counts, sequential search avoids batch overhead
        if len(queries) <= constants.USEARCH_BATCH_THRESHOLD:
            return [
                self.similarity_search(q, k, filter, exact=exact, threads=threads)
                for q in queries
            ]

        # Stack queries into batch array
        query_array = np.array(queries, dtype=np.float32)

        # Over-fetch for filtering
        fetch_k = k * constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER if filter else k

        # Batch search - usearch handles this efficiently
        keys_batch, distances_batch = self._index.search(
            query_array, fetch_k, exact=exact, threads=threads
        )

        # Handle batch results shape: (n_queries, k)
        if keys_batch.ndim == 1:
            # Single query case
            keys_batch = keys_batch.reshape(1, -1)
            distances_batch = distances_batch.reshape(1, -1)

        # Collect all unique keys via numpy (avoids per-row tolist)
        all_keys_arr = np.unique(keys_batch.ravel())
        docs_map = self._catalog.get_documents_by_ids(all_keys_arr.tolist())

        # Convert batch arrays to Python lists once
        keys_lists = keys_batch.tolist()
        dist_lists = distances_batch.tolist()

        # Build results for each query
        all_results: list[list[tuple[Document, float]]] = []
        for keys_row, dists_row in zip(keys_lists, dist_lists):
            results: list[tuple[Document, float]] = []
            for key, dist in zip(keys_row, dists_row):
                if key not in docs_map:
                    continue

                text, metadata = docs_map[key]

                if filter and not self._matches_filter(metadata, filter):
                    continue

                results.append((Document(page_content=text, metadata=metadata), float(dist)))

                if len(results) >= k:
                    break

            all_results.append(results)

        return all_results

    def keyword_search(
        self,
        query: str,
        k: int = constants.DEFAULT_K,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform BM25 keyword search using FTS5.

        Args:
            query: Text query (supports FTS5 syntax)
            k: Maximum number of results
            filter: Optional metadata filter

        Returns:
            List of (Document, bm25_score) tuples sorted by relevance

        Raises:
            RuntimeError: If FTS5 not available
        """
        # Validate filter structure early
        validate_filter(filter)

        candidates = self._catalog.keyword_search(
            query, k, filter, self._catalog.build_filter_clause
        )

        if not candidates:
            return []

        ids = [cid for cid, _ in candidates]
        docs_map = self._catalog.get_documents_by_ids(ids)

        results: list[tuple[Document, float]] = []
        for cid, score in candidates:
            if cid in docs_map:
                text, metadata = docs_map[cid]
                doc = Document(page_content=text, metadata=metadata)
                results.append((doc, float(score)))

        return results

    def hybrid_search(
        self,
        query: str,
        k: int = constants.DEFAULT_K,
        filter: dict[str, Any] | None = None,
        *,
        query_vector: Sequence[float] | None = None,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        rrf_k: int = constants.DEFAULT_RRF_K,
    ) -> list[tuple[Document, float]]:
        """
        Combine vector and keyword search using Reciprocal Rank Fusion.

        Args:
            query: Text query for keyword search
            k: Final number of results after fusion
            filter: Optional metadata filter
            query_vector: Optional pre-computed query embedding
            vector_k: Number of vector search candidates
            keyword_k: Number of keyword search candidates
            rrf_k: RRF constant parameter (default: 60)

        Returns:
            List of (Document, rrf_score) tuples sorted by fused score

        Raises:
            RuntimeError: If FTS5 not available
        """
        if not self._catalog.fts_enabled:
            raise RuntimeError(
                "hybrid_search requires SQLite compiled with FTS5 support"
            )

        if not query.strip():
            return []

        dense_k = vector_k or max(k, 10)
        sparse_k = keyword_k or max(k, 10)

        # Vector search — recover document IDs so RRF dedupes by ID, not by
        # page_content. Two distinct documents with identical text would
        # otherwise be silently merged into a single result with inflated
        # score, dropping one of them.
        vector_input = query_vector if query_vector is not None else query
        vector_query_vec = self._resolve_query_vector(vector_input)
        vector_keys, vector_dists = self._index.search(vector_query_vec, dense_k)
        vector_keys_list = [int(k_) for k_ in vector_keys.tolist()]

        # Keyword search candidates already carry IDs.
        keyword_candidates = self._catalog.keyword_search(
            query, sparse_k, filter, self._catalog.build_filter_clause
        )

        all_ids = list({*vector_keys_list, *(cid for cid, _ in keyword_candidates)})
        docs_map = self._catalog.get_documents_by_ids(all_ids) if all_ids else {}

        rrf_scores: dict[int, float] = {}
        doc_lookup: dict[int, Document] = {}

        # Vector ranks
        rank = 0
        for key in vector_keys_list:
            if key not in docs_map:
                continue
            text, metadata = docs_map[key]
            if filter and not self._matches_filter(metadata, filter):
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_lookup[key] = Document(page_content=text, metadata=metadata)
            rank += 1

        # Keyword ranks (BM25 candidates respect the filter via build_filter_clause)
        for kw_rank, (cid, _) in enumerate(keyword_candidates):
            if cid not in docs_map:
                continue
            text, metadata = docs_map[cid]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + kw_rank + 1)
            if cid not in doc_lookup:
                doc_lookup[cid] = Document(page_content=text, metadata=metadata)

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        return [(doc_lookup[cid], rrf_scores[cid]) for cid in sorted_ids[:k]]

    def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = constants.DEFAULT_K,
        fetch_k: int = constants.DEFAULT_FETCH_K,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search with diversity using Max Marginal Relevance algorithm.

        Uses stored embeddings to compute pairwise similarity for diversity.

        Args:
            query: Query vector or text (auto-embedded if string)
            k: Number of diverse results to return
            fetch_k: Number of candidates to consider (should be >= k)
            lambda_mult: Diversity trade-off (0=max diversity, 1=max relevance)
            filter: Optional metadata filter

        Returns:
            List of Documents ordered by MMR selection (no scores)
        """
        # Validate filter structure early
        validate_filter(filter)

        query_vec = self._resolve_query_vector(query)

        # Over-fetch for filtering, then apply MMR
        actual_fetch = (
            fetch_k * constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER
            if filter
            else fetch_k
        )

        keys, distances = self._index.search(query_vec, actual_fetch)

        if len(keys) == 0:
            return []

        # Fetch documents with embeddings in a single SQL round-trip
        keys_list = keys.tolist()
        docs_and_embs = self._catalog.get_documents_and_embeddings_by_ids(keys_list)

        # If catalog has no stored embeddings, retrieve from usearch index
        has_catalog_embs = any(
            emb is not None for _, _, emb in docs_and_embs.values()
        )
        index_embs: np.ndarray | None = None
        if not has_catalog_embs:
            keys_arr = np.array(keys_list, dtype=np.uint64)
            index_embs = self._index.get(keys_arr)

        # Build candidates list with filtering, pre-normalize embeddings
        candidates: list[tuple[int, Document, float, np.ndarray | None]] = []
        for i, (key, dist) in enumerate(zip(keys_list, distances.tolist())):
            if key not in docs_and_embs:
                continue

            text, metadata, emb = docs_and_embs[key]

            # Fall back to index embeddings if catalog has none
            if emb is None and index_embs is not None:
                emb = index_embs[i]

            # Apply metadata filter
            if filter and not self._matches_filter(metadata, filter):
                continue

            doc = Document(page_content=text, metadata=metadata)
            # Pre-normalize embedding once (avoid redundant renorm in MMR loop)
            if emb is not None:
                emb = emb / (np.linalg.norm(emb) + 1e-12)
            candidates.append((key, doc, float(dist), emb))

            if len(candidates) >= fetch_k:
                break

        if len(candidates) <= k:
            return [doc for _, doc, _, _ in candidates]

        # MMR selection with vectorized pairwise similarity. Maintain
        # ``sel_matrix`` incrementally so we don't ``np.stack`` the growing
        # list of selected embeddings on every outer iteration (previously
        # O(k²·d) wasted allocations).
        selected: list[Document] = []
        lambda_comp = 1.0 - lambda_mult
        unselected = list(range(len(candidates)))

        # First selection: most relevant (lowest distance)
        first_idx = unselected.pop(0)
        _, doc, _, emb = candidates[first_idx]
        selected.append(doc)
        sel_matrix: np.ndarray | None = (
            emb[np.newaxis, :].copy() if emb is not None else None
        )

        while len(selected) < k and unselected:
            best_score = -float("inf")
            best_pos = 0

            for pos, idx in enumerate(unselected):
                _, _, dist, emb = candidates[idx]

                # Relevance: convert distance to similarity (lower distance = higher similarity)
                # For cosine distance in [0, 2], similarity = 1 - distance/2
                relevance = 1.0 - dist / 2.0

                # Redundancy: max similarity to any already-selected doc
                redundancy = 0.0
                if emb is not None and sel_matrix is not None:
                    sims = sel_matrix @ emb
                    redundancy = float(sims.max())

                mmr_score = lambda_mult * relevance - lambda_comp * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = pos

            best_idx = unselected.pop(best_pos)

            _, doc, _, emb = candidates[best_idx]
            selected.append(doc)
            if emb is not None:
                if sel_matrix is None:
                    sel_matrix = emb[np.newaxis, :].copy()
                else:
                    sel_matrix = np.vstack([sel_matrix, emb[np.newaxis, :]])

        return selected

    def _resolve_query_vector(self, query: str | Sequence[float]) -> np.ndarray:
        """Convert query to vector, embedding text if necessary."""
        if isinstance(query, str):
            try:
                from ..embeddings.models import embed_texts

                query_embedding = embed_texts([query])[0]
                return np.asarray(query_embedding, dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    "Text queries require embeddings – install with [server] extra "
                    "or provide vector query"
                ) from e
        else:
            return np.asarray(query, dtype=np.float32)

    def _matches_filter(self, metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria."""
        for key, value in filter.items():
            meta_value = metadata.get(key)

            if isinstance(value, list):
                # List filter: meta_value must be in the list
                if meta_value not in value:
                    return False
            elif isinstance(value, str):
                # String filter: exact match (consistent with SQL build_filter_clause)
                if meta_value != value:
                    return False
            else:
                # Exact match for int/float
                if meta_value != value:
                    return False

        return True
