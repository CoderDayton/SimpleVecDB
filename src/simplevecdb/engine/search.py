from __future__ import annotations

import json
import numpy as np
from typing import Any, TYPE_CHECKING, Callable
from collections.abc import Sequence

from ..types import Document, DistanceStrategy
from .quantization import normalize_l2

if TYPE_CHECKING:
    import sqlite3
    from ..types import Quantization


class SearchEngine:
    """Handles all search operations for a VectorCollection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        vec_table_name: str,
        fts_table_name: str,
        fts_enabled: bool,
        distance_strategy: DistanceStrategy,
        quantization: Quantization,
        quantizer: Any,  # QuantizationStrategy
        dim_getter: Callable[[], int | None],
    ):
        self.conn = conn
        self._table_name = table_name
        self._vec_table_name = vec_table_name
        self._fts_table_name = fts_table_name
        self._fts_enabled = fts_enabled
        self.distance_strategy = distance_strategy
        self.quantization = quantization
        self._quantizer = quantizer
        self._get_dim = dim_getter

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        filter_builder: Callable | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform vector similarity search.

        Args:
            query: Query vector or text (auto-embedded if string)
            k: Number of results to return
            filter: Optional metadata filter
            filter_builder: Function to build SQL WHERE clause

        Returns:
            List of (Document, distance_score) tuples sorted by distance
        """
        candidates = self._vector_search_candidates(query, k, filter, filter_builder)
        return self._hydrate_documents(candidates)

    def keyword_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        filter_builder: Callable | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform BM25 keyword search using FTS5.

        Args:
            query: Text query (supports FTS5 syntax)
            k: Maximum number of results
            filter: Optional metadata filter
            filter_builder: Function to build SQL WHERE clause

        Returns:
            List of (Document, bm25_score) tuples sorted by relevance

        Raises:
            RuntimeError: If FTS5 not available
        """
        candidates = self._keyword_search_candidates(query, k, filter, filter_builder)
        return self._hydrate_documents(candidates)

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        query_vector: Sequence[float] | None = None,
        vector_k: int | None = None,
        keyword_k: int | None = None,
        rrf_k: int = 60,
        filter_builder: Callable | None = None,
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
            filter_builder: Function to build SQL WHERE clause

        Returns:
            List of (Document, rrf_score) tuples sorted by fused score

        Raises:
            RuntimeError: If FTS5 not available
        """
        if not self._fts_enabled:
            raise RuntimeError(
                "hybrid_search requires SQLite compiled with FTS5 support"
            )

        if not query.strip():
            return []

        dense_k = vector_k or max(k, 10)
        sparse_k = keyword_k or max(k, 10)

        vector_input: str | Sequence[float]
        if query_vector is not None:
            vector_input = query_vector
        else:
            vector_input = query

        dense_candidates = self._vector_search_candidates(
            vector_input, dense_k, filter, filter_builder
        )
        sparse_candidates = self._keyword_search_candidates(
            query, sparse_k, filter, filter_builder
        )

        fused_candidates = self._reciprocal_rank_fusion(
            dense_candidates, sparse_candidates, rrf_k
        )

        return self._hydrate_documents(fused_candidates[:k])

    def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        fetch_k: int = 20,
        filter: dict[str, Any] | None = None,
        filter_builder: Callable | None = None,
    ) -> list[Document]:
        """
        Search with diversity using Max Marginal Relevance algorithm.

        Args:
            query: Query vector or text (auto-embedded if string)
            k: Number of diverse results to return
            fetch_k: Number of candidates to consider (should be >= k)
            filter: Optional metadata filter
            filter_builder: Function to build SQL WHERE clause

        Returns:
            List of Documents ordered by MMR selection (no scores)
        """
        candidates_with_scores = self.similarity_search(
            query, k=fetch_k, filter=filter, filter_builder=filter_builder
        )
        candidates = [doc for doc, _ in candidates_with_scores]

        if len(candidates) <= k:
            return candidates

        selected = []
        unselected = candidates.copy()
        selected.append(unselected.pop(0))

        while len(selected) < k:
            mmr_scores = []
            for candidate in unselected:
                relevance = next(
                    score for doc, score in candidates_with_scores if doc == candidate
                )
                diversity = min(
                    next(
                        score
                        for doc, score in candidates_with_scores
                        if doc == selected_doc
                    )
                    for selected_doc in selected
                )
                mmr_score = 0.5 * relevance - 0.5 * diversity
                mmr_scores.append((mmr_score, candidate))

            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            selected.append(mmr_scores[0][1])
            unselected.remove(mmr_scores[0][1])

        return selected

    def _vector_search_candidates(
        self,
        query: str | Sequence[float],
        k: int,
        filter: dict[str, Any] | None,
        filter_builder: Callable | None = None,
    ) -> list[tuple[int, float]]:
        dim = self._get_dim()
        if dim is None:
            return []

        if isinstance(query, str):
            try:
                from ..embeddings.models import embed_texts

                query_embedding = embed_texts([query])[0]
                query_vec = np.array(query_embedding, dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    "Text queries require embeddings – install with [server] extra or provide vector query"
                ) from e
        else:
            query_vec = np.array(query, dtype=np.float32)

        if len(query_vec) != dim:
            raise ValueError(f"Query dim {len(query_vec)} != collection dim {dim}")

        if self.distance_strategy == DistanceStrategy.COSINE:
            query_vec = normalize_l2(query_vec)

        blob = self._quantizer.serialize(query_vec)

        if filter_builder:
            filter_clause, filter_params = filter_builder(
                filter, metadata_column="ti.metadata"
            )
        else:
            filter_clause, filter_params = "", []

        match_placeholder = "?"
        if self.quantization.value == "int8":
            match_placeholder = "vec_int8(?)"
        elif self.quantization.value == "bit":
            match_placeholder = "vec_bit(?)"

        try:
            sql = f"""
                SELECT ti.id, distance
                FROM {self._vec_table_name} vi
                JOIN {self._table_name} ti ON vi.rowid = ti.id
                WHERE embedding MATCH {match_placeholder}
                AND k = ?
                {filter_clause}
                ORDER BY distance
            """
            rows = self.conn.execute(
                sql, (blob,) + (k,) + tuple(filter_params)
            ).fetchall()
        except Exception:
            # Fall back to brute force
            from ..core import get_optimal_batch_size

            rows = self._brute_force_search(
                query_vec, k, filter, filter_builder, get_optimal_batch_size()
            )

        return [(int(cid), float(dist)) for cid, dist in rows[:k]]

    def _keyword_search_candidates(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None,
        filter_builder: Callable | None = None,
    ) -> list[tuple[int, float]]:
        if not self._fts_enabled:
            raise RuntimeError(
                "keyword_search requires SQLite compiled with FTS5 support"
            )
        if not query.strip():
            return []

        if filter_builder:
            filter_clause, filter_params = filter_builder(
                filter, metadata_column="ti.metadata"
            )
        else:
            filter_clause, filter_params = "", []

        sql = f"""
            SELECT ti.id, bm25({self._fts_table_name}) as score
            FROM {self._fts_table_name} f
            JOIN {self._table_name} ti ON ti.id = f.rowid
            WHERE {self._fts_table_name} MATCH ?
            {filter_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        params = (query,) + tuple(filter_params) + (k,)
        rows = self.conn.execute(sql, params).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def _reciprocal_rank_fusion(
        self,
        dense: Sequence[tuple[int, float]],
        sparse: Sequence[tuple[int, float]],
        rrf_k: int,
    ) -> list[tuple[int, float]]:
        rank_scores: dict[int, float] = {}

        def _accumulate(items: Sequence[tuple[int, float]]):
            for rank, (doc_id, _) in enumerate(items):
                rank_scores[doc_id] = rank_scores.get(doc_id, 0.0) + 1.0 / (
                    rrf_k + rank + 1
                )

        _accumulate(dense)
        _accumulate(sparse)

        fused = sorted(rank_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(doc_id, score) for doc_id, score in fused]

    def _brute_force_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: dict[str, Any] | None,
        filter_builder: Callable | None,
        batch_size: int,
    ) -> list[tuple[int, float]]:
        from ..core import _batched
        import sqlite3

        try:
            cursor = self.conn.execute(
                f"SELECT rowid, embedding FROM {self._vec_table_name}"
            )
        except sqlite3.OperationalError:
            return []

        top_k_candidates: list[tuple[int, float]] = []

        for batch in _batched(cursor, batch_size):
            if not batch:
                continue

            ids, blobs = zip(*batch)
            dim = self._get_dim()
            vectors = np.array([self._quantizer.deserialize(b, dim) for b in blobs])

            if self.distance_strategy == DistanceStrategy.COSINE:
                dots = np.dot(vectors, query_vec)
                norms = np.linalg.norm(vectors, axis=1)
                similarities = dots / (norms * np.linalg.norm(query_vec) + 1e-12)
                distances = 1 - similarities
            elif self.distance_strategy == DistanceStrategy.L2:
                distances = np.linalg.norm(vectors - query_vec, axis=1)
            elif self.distance_strategy == DistanceStrategy.L1:
                distances = np.sum(np.abs(vectors - query_vec), axis=1)
            else:
                raise ValueError(
                    f"Unsupported distance strategy: {self.distance_strategy}"
                )

            for cid, dist in zip(ids, distances):
                top_k_candidates.append((int(cid), float(dist)))

        top_k_candidates.sort(key=lambda x: x[1])
        top_k = top_k_candidates[:k]

        # Apply filter if needed
        if filter and filter_builder:
            filtered = []
            ids_to_check = [cid for cid, _ in top_k]
            if ids_to_check:
                placeholders = ",".join("?" for _ in ids_to_check)
                meta_rows = self.conn.execute(
                    f"SELECT id, metadata FROM {self._table_name} WHERE id IN ({placeholders})",
                    tuple(ids_to_check),
                ).fetchall()
                meta_map = {r[0]: r[1] for r in meta_rows}

                for cid, dist in top_k:
                    meta_json = meta_map.get(cid)
                    if meta_json:
                        import json

                        meta = json.loads(meta_json)
                        if all(
                            meta.get(k) == v
                            if not isinstance(v, list)
                            else meta.get(k) in v
                            for k, v in filter.items()
                        ):
                            filtered.append((cid, dist))
                return filtered

        return top_k

    def _hydrate_documents(
        self, candidates: Sequence[tuple[int, float]]
    ) -> list[tuple[Document, float]]:
        results: list[tuple[Document, float]] = []
        for cid, score in candidates:
            row = self.conn.execute(
                f"SELECT text, metadata FROM {self._table_name} WHERE id = ?", (cid,)
            ).fetchone()
            if not row:
                continue
            text, meta_json = row
            meta = json.loads(meta_json) if meta_json else {}
            results.append((Document(page_content=text, metadata=meta), score))
        return results
