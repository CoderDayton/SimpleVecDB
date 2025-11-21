from __future__ import annotations

import sqlite3
import struct
import json
import numpy as np
from collections.abc import Iterable, Sequence
from typing import Any, TYPE_CHECKING
import sqlite_vec
from pathlib import Path

from .types import Document, DistanceStrategy, StrEnum

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from .integrations.langchain import TinyVecDBVectorStore
    from .integrations.llamaindex import TinyVecDBLlamaStore

Quantization = StrEnum("Quantization", ["FLOAT", "INT8", "BIT"])


def _serialize_vector(vector: np.ndarray, quant: Quantization) -> bytes:
    """Serialize a normalized float vector according to quantization mode."""
    if quant == Quantization.FLOAT:
        return struct.pack("<%sf" % len(vector), *(float(x) for x in vector))

    elif quant == Quantization.INT8:
        # Scalar quantization: scale to [-128, 127]
        scaled = np.clip(np.round(vector * 127), -128, 127).astype(np.int8)
        return scaled.tobytes()

    elif quant == Quantization.BIT:
        # Binary quantization: threshold at 0 → pack bits
        bits = (vector > 0).astype(np.uint8)
        packed = np.packbits(bits)
        return packed.tobytes()

    raise ValueError(f"Unsupported quantization: {quant}")


def _dequantize_vector(blob: bytes, dim: int | None, quant: Quantization) -> np.ndarray:
    """Reverse serialization for fallback path."""
    if quant == Quantization.FLOAT:
        return np.frombuffer(blob, dtype=np.float32)

    elif quant == Quantization.INT8:
        return np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0

    elif quant == Quantization.BIT and dim is not None:
        unpacked = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
        v = unpacked[:dim].astype(np.float32)
        return np.where(v == 1, 1.0, -1.0)

    raise ValueError(f"Unsupported quantization: {quant} or unknown dim {dim}")


def _normalize_l2(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


class VectorDB:
    """
    Dead-simple local vector database powered by sqlite-vec.
    One SQLite file = one collection. Chroma-style API with quantization.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        quantization: Quantization = Quantization.FLOAT,
    ):
        self.path = str(path)
        self.distance_strategy = distance_strategy
        self.quantization = quantization

        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.enable_load_extension(True)
        try:
            sqlite_vec.load(self.conn)
            self._extension_available = True
        except sqlite3.OperationalError:
            self._extension_available = False

        self.conn.enable_load_extension(False)

        self._dim: int | None = None
        self._table_name = "tinyvec_items"
        self._create_table()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_table(self) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT
            )
            """
        )

    def _ensure_virtual_table(self, dim: int) -> None:
        if self._dim is not None and self._dim != dim:
            raise ValueError(f"Dimension mismatch: existing {self._dim}, got {dim}")
        if self._dim is None:
            # First insert – recreate virtual table with correct dimension
            self._dim = dim
            self.conn.execute("DROP TABLE IF EXISTS vec_index")

            storage_dim = dim
            if self.quantization == Quantization.BIT:
                storage_dim = ((dim + 7) // 8) * 8

            vec_type = {
                Quantization.FLOAT: f"float[{dim}]",
                Quantization.INT8: f"int8[{dim}]",
                Quantization.BIT: f"bit[{storage_dim}]",
            }[self.quantization]

            # BIT quantization implies Hamming distance; others need explicit metric
            distance_clause = ""
            if self.quantization != Quantization.BIT:
                distance_clause = f"distance_metric={self.distance_strategy.value}"

            self.conn.execute(
                f"""
                CREATE VIRTUAL TABLE vec_index USING vec0(
                embedding {vec_type} {distance_clause}
                )
                """
            )

    # ------------------------------------------------------------------ #
    # Filtering helpers
    # ------------------------------------------------------------------ #
    def _build_filter_clause(
        self, filter_dict: dict[str, Any] | None
    ) -> tuple[str, list[Any]]:
        if not filter_dict:
            return "", []

        clauses = []
        params = []
        for key, value in filter_dict.items():
            json_path = f"$.{key}"
            if isinstance(value, (int, float)):
                clauses.append("json_extract(metadata, ?) = ?")
                params.extend([json_path, value])
            elif isinstance(value, str):
                clauses.append("json_extract(metadata, ?) LIKE ?")
                params.extend([json_path, f"%{value}%"])
            elif isinstance(value, list):
                placeholders = ",".join("?" for _ in value)
                clauses.append(f"json_extract(metadata, ?) IN ({placeholders})")
                params.extend([json_path] + value)
            else:
                raise ValueError(f"Unsupported filter value type for {key}")
        where = " AND ".join(clauses)
        return f"AND ({where})" if where else "", params

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
        batch_size: int = 1000,
    ) -> list[int]:
        """
        Add texts with optional pre-computed embeddings.
        Returns the assigned integer IDs.
        """
        if not texts:
            return []

        # If embeddings are not provided, we need to generate them.
        embed_func = None
        if embeddings is None:
            try:
                from tinyvecdb.embeddings.models import embed_texts
                embed_func = embed_texts
            except Exception as e:
                raise ValueError(
                    "No embeddings provided and local embedder failed – install with [server] extra"
                ) from e

        all_ids = []
        n_total = len(texts)

        for start_idx in range(0, n_total, batch_size):
            end_idx = min(start_idx + batch_size, n_total)
            
            batch_texts = texts[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx] if metadatas else [{} for _ in batch_texts]
            batch_ids = ids[start_idx:end_idx] if ids else [None] * len(batch_texts)
            
            if embeddings is not None:
                batch_embeddings = embeddings[start_idx:end_idx]
            else:
                batch_embeddings = embed_func(list(batch_texts))

            # Ensure table exists (idempotent check)
            if self._dim is None:
                dim = len(batch_embeddings[0])
                self._ensure_virtual_table(dim)

            # Normalize for cosine before quantization
            emb_np = np.array(batch_embeddings, dtype=np.float32)
            if self.distance_strategy == DistanceStrategy.COSINE:
                norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
                emb_np = emb_np / np.maximum(norms, 1e-12)

            serialized = [_serialize_vector(vec, self.quantization) for vec in emb_np]

            rows = []
            for txt, meta, uid in zip(batch_texts, batch_metadatas, batch_ids):
                rows.append((uid, txt, json.dumps(meta)))

            with self.conn:
                # Insert main table
                self.conn.executemany(
                    f"""
                    INSERT INTO {self._table_name}(id, text, metadata)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text=excluded.text,
                        metadata=excluded.metadata
                    """,
                    rows,
                )
                # Sync vec_index with correct rowids
                batch_real_ids = [
                    r[0]
                    for r in self.conn.execute(
                        f"SELECT id FROM {self._table_name} ORDER BY id DESC LIMIT ?",
                        (len(batch_texts),),
                    )
                ]
                batch_real_ids.reverse()  # Align with input order
                
                real_vec_rows = [
                    (real_id, ser) for real_id, ser in zip(batch_real_ids, serialized)
                ]
                
                insert_placeholder = "?"
                if self.quantization == Quantization.INT8:
                    insert_placeholder = "vec_int8(?)"
                elif self.quantization == Quantization.BIT:
                    insert_placeholder = "vec_bit(?)"

                # Delete existing rows in vec_index to handle upserts
                placeholders = ",".join("?" for _ in batch_real_ids)
                self.conn.execute(
                    f"DELETE FROM vec_index WHERE rowid IN ({placeholders})",
                    tuple(batch_real_ids)
                )

                self.conn.executemany(
                    f"INSERT INTO vec_index(rowid, embedding) VALUES (?, {insert_placeholder})", real_vec_rows
                )
                
                all_ids.extend(batch_real_ids)

        return all_ids

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Return top-k documents with distances.
        Supports vector queries (text queries require embeddings integration).
        Optional metadata filter as dict (e.g., {"category": "fruit"}).
        """
        if self._dim is None:
            return []  # empty collection

        if isinstance(query, str):
            try:
                from .embeddings.models import embed_texts

                query_embedding = embed_texts([query])[0]
                query_vec = np.array(query_embedding, dtype=np.float32)
            except Exception as e:
                raise ValueError(
                    "Text queries require embeddings – install with [server] extra or provide vector query"
                ) from e
        else:
            query_vec = np.array(query, dtype=np.float32)
        if len(query_vec) != self._dim:
            raise ValueError(
                f"Query dim {len(query_vec)} != collection dim {self._dim}"
            )

        if self.distance_strategy == DistanceStrategy.COSINE:
            query_vec = _normalize_l2(query_vec)

        blob = _serialize_vector(query_vec, self.quantization)

        filter_clause, filter_params = self._build_filter_clause(filter)

        match_placeholder = "?"
        if self.quantization == Quantization.INT8:
            match_placeholder = "vec_int8(?)"
        elif self.quantization == Quantization.BIT:
            match_placeholder = "vec_bit(?)"

        try:
            sql = f"""
                SELECT ti.id, distance
                FROM vec_index vi
                JOIN {self._table_name} ti ON vi.rowid = ti.id
                WHERE embedding MATCH {match_placeholder}
                AND k = ?
                {filter_clause}
                ORDER BY distance
            """
            candidates = self.conn.execute(
                sql, (blob,) + (k,) + tuple(filter_params)
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback brute-force
            candidates = self._brute_force_search(query_vec, k, filter)

        results = []
        for cid, dist in candidates[:k]:
            text, meta_json = self.conn.execute(
                f"SELECT text, metadata FROM {self._table_name} WHERE id = ?", (cid,)
            ).fetchone()
            meta = json.loads(meta_json) if meta_json else {}
            results.append((Document(page_content=text, metadata=meta), float(dist)))

        return results

    def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        fetch_k: int = 20,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        MMR search to diversify results.
        Returns k documents selected from top fetch_k candidates.
        0.5 trade-off between relevance and diversity.
        """
        # First get top fetch_k candidates
        candidates_with_scores = self.similarity_search(query, k=fetch_k, filter=filter)
        candidates = [doc for doc, _ in candidates_with_scores]

        if len(candidates) <= k:
            return candidates

        # MMR selection
        selected = []
        unselected = candidates.copy()

        # Start with the most relevant document
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

            # Select the candidate with the highest MMR score
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            selected.append(mmr_scores[0][1])
            unselected.remove(mmr_scores[0][1])

        return selected

    def _brute_force_search(
        self,
        query_vec: np.ndarray,
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[tuple[int, float]]:
        # Fetch embeddings from vec_index since we don't store them in main table
        rows = self.conn.execute(
            f"SELECT rowid, embedding FROM vec_index"
        ).fetchall()
        
        if not rows:
            return []

        ids, blobs = zip(*rows)
        
        # Fetch metadata only if needed for filtering
        metas = []
        if filter:
            # This might be slow for large DBs, but it's brute force fallback anyway
            placeholders = ",".join("?" for _ in ids)
            meta_rows = self.conn.execute(
                f"SELECT id, metadata FROM {self._table_name} WHERE id IN ({placeholders})",
                ids
            ).fetchall()
            meta_map = {r[0]: r[1] for r in meta_rows}
            metas = [meta_map.get(i) for i in ids]
        else:
            metas = [None] * len(ids)

        vectors = np.array(
            [_dequantize_vector(b, self._dim, self.quantization) for b in blobs]
        )

        if self.distance_strategy == DistanceStrategy.COSINE:
            dots = np.dot(vectors, query_vec)
            norms = np.linalg.norm(vectors, axis=1)
            similarities = dots / (norms * np.linalg.norm(query_vec) + 1e-12)
            distances = 1 - similarities
        elif self.distance_strategy == DistanceStrategy.L2:
            distances = np.linalg.norm(vectors - query_vec, axis=1)
        else:  # IP
            distances = -np.dot(vectors, query_vec)

        # Apply filter if any
        if filter:
            filtered = []
            for i, (cid, dist, meta_json) in enumerate(zip(ids, distances, metas)):
                meta = json.loads(meta_json) if meta_json else {}
                if all(
                    meta.get(k) == v if not isinstance(v, list) else meta.get(k) in v
                    for k, v in filter.items()
                ):
                    filtered.append((cid, dist))
            distances = np.array([d for _, d in filtered])
            ids = [i for i, _ in filtered]
        else:
            filtered = list(zip(ids, distances))

        indices = np.argsort(distances)[:k]
        # Ensure pure Python primitives (int, float) for type compatibility
        return [(int(ids[i]), float(distances[i])) for i in indices]

    def delete_by_ids(self, ids: Iterable[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self.conn:
            self.conn.execute(
                f"DELETE FROM {self._table_name} WHERE id IN ({placeholders})",
                tuple(ids),
            )
            self.conn.execute(
                f"DELETE FROM vec_index WHERE rowid IN ({placeholders})", tuple(ids)
            )
        self.conn.execute("VACUUM")

    # ------------------------------------------------------------------ #
    # Integrations
    # ------------------------------------------------------------------ #
    def as_langchain(
        self, embeddings: Embeddings | None = None
    ) -> TinyVecDBVectorStore:
        from .integrations.langchain import TinyVecDBVectorStore

        return TinyVecDBVectorStore(db_path=self.path, embedding=embeddings)

    def as_llama_index(self) -> TinyVecDBLlamaStore:
        from .integrations.llamaindex import TinyVecDBLlamaStore

        return TinyVecDBLlamaStore(db_path=self.path)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
