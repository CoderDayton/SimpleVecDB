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
    ) -> list[tuple[Document, float]]:
        """
        Perform vector similarity search.

        Args:
            query: Query vector or text (auto-embedded if string)
            k: Number of results to return
            filter: Optional metadata filter
            exact: Force search mode. None=adaptive (brute-force for <10k vectors),
                   True=always brute-force (perfect recall), False=always HNSW.

        Returns:
            List of (Document, distance) tuples sorted by distance (lower = more similar)
        """
        # Validate filter structure early
        validate_filter(filter)

        query_vec = self._resolve_query_vector(query)

        # Over-fetch for filtering
        fetch_k = k * constants.USEARCH_FILTER_OVERFETCH_MULTIPLIER if filter else k

        keys, distances = self._index.search(query_vec, fetch_k, exact=exact)

        if len(keys) == 0:
            return []

        # Fetch documents and apply filter
        docs_map = self._catalog.get_documents_by_ids(keys.tolist())

        results: list[tuple[Document, float]] = []
        for key, dist in zip(keys.tolist(), distances.tolist()):
            if key not in docs_map:
                continue

            text, metadata = docs_map[key]

            # Apply metadata filter
            if filter and not self._matches_filter(metadata, filter):
                continue

            doc = Document(page_content=text, metadata=metadata)
            results.append((doc, float(dist)))

            if len(results) >= k:
                break

        return results

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

        # Vector search
        vector_input = query_vector if query_vector is not None else query
        vector_results = self.similarity_search(vector_input, dense_k, filter)

        # Keyword search
        keyword_results = self.keyword_search(query, sparse_k, filter)

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}  # Use text as key for deduplication
        doc_lookup: dict[str, Document] = {}

        for rank, (doc, _) in enumerate(vector_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_lookup[key] = doc

        for rank, (doc, _) in enumerate(keyword_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_lookup[key] = doc

        # Sort by RRF score
        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        results: list[tuple[Document, float]] = []
        for key in sorted_keys[:k]:
            results.append((doc_lookup[key], rrf_scores[key]))

        return results

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

        # Fetch documents with embeddings
        keys_list = keys.tolist()
        docs_map = self._catalog.get_documents_by_ids(keys_list)
        embs_map = self._catalog.get_embeddings_by_ids(keys_list)

        # Build candidates list with filtering
        candidates: list[tuple[int, Document, float, np.ndarray | None]] = []
        for key, dist in zip(keys_list, distances.tolist()):
            if key not in docs_map:
                continue

            text, metadata = docs_map[key]

            # Apply metadata filter
            if filter and not self._matches_filter(metadata, filter):
                continue

            doc = Document(page_content=text, metadata=metadata)
            emb = embs_map.get(key)
            candidates.append((key, doc, float(dist), emb))

            if len(candidates) >= fetch_k:
                break

        if len(candidates) <= k:
            return [doc for _, doc, _, _ in candidates]

        # MMR selection with proper pairwise similarity
        selected: list[Document] = []
        selected_embs: list[np.ndarray] = []
        unselected = list(range(len(candidates)))

        # First selection: most relevant (lowest distance)
        first_idx = unselected.pop(0)
        _, doc, _, emb = candidates[first_idx]
        selected.append(doc)
        if emb is not None:
            selected_embs.append(emb / (np.linalg.norm(emb) + 1e-12))

        while len(selected) < k and unselected:
            mmr_scores: list[tuple[float, int]] = []

            for idx in unselected:
                _, _, dist, emb = candidates[idx]

                # Relevance: convert distance to similarity (lower distance = higher similarity)
                # For cosine distance in [0, 2], similarity = 1 - distance/2
                relevance = 1.0 - dist / 2.0

                # Redundancy: max similarity to any already-selected doc
                redundancy = 0.0
                if emb is not None and selected_embs:
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
                    for sel_emb in selected_embs:
                        sim = float(np.dot(emb_norm, sel_emb))
                        redundancy = max(redundancy, sim)

                # MMR: balance relevance vs diversity
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                mmr_scores.append((mmr_score, idx))

            # Pick highest MMR score
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best_idx = mmr_scores[0][1]

            _, doc, _, emb = candidates[best_idx]
            selected.append(doc)
            if emb is not None:
                selected_embs.append(emb / (np.linalg.norm(emb) + 1e-12))
            unselected.remove(best_idx)

        return selected

    def _resolve_query_vector(self, query: str | Sequence[float]) -> np.ndarray:
        """Convert query to vector, embedding text if necessary."""
        if isinstance(query, str):
            try:
                from ..embeddings.models import embed_texts

                query_embedding = embed_texts([query])[0]
                return np.array(query_embedding, dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    "Text queries require embeddings – install with [server] extra "
                    "or provide vector query"
                ) from e
        else:
            return np.array(query, dtype=np.float32)

    def _matches_filter(self, metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria."""
        for key, value in filter.items():
            meta_value = metadata.get(key)

            if isinstance(value, list):
                # List filter: meta_value must be in the list
                if meta_value not in value:
                    return False
            elif isinstance(value, str):
                # String filter: substring match
                if meta_value is None or value not in str(meta_value):
                    return False
            else:
                # Exact match for int/float
                if meta_value != value:
                    return False

        return True
