"""
SimpleVecDB Core Module.

Provides VectorDB and VectorCollection classes for local vector search
using usearch HNSW index with SQLite metadata storage.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import tempfile
import numpy as np
import uuid
from collections.abc import Iterable, Sequence
from typing import Any, TYPE_CHECKING
from pathlib import Path
import platform
import multiprocessing
import itertools

from .types import Document, DistanceStrategy, Quantization
from .utils import _import_optional
from .engine.quantization import QuantizationStrategy
from .engine.search import SearchEngine
from .engine.catalog import CatalogManager
from .engine.usearch_index import UsearchIndex
from . import constants

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from .integrations.langchain import SimpleVecDBVectorStore
    from .integrations.llamaindex import SimpleVecDBLlamaStore

_logger = logging.getLogger("simplevecdb.core")


def _batched(iterable: Iterable[Any], n: int) -> Iterable[Sequence[Any]]:
    """Batch data into lists of length n. The last batch may be shorter."""
    if isinstance(iterable, Sequence):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]
    else:
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                return
            yield batch


def get_optimal_batch_size() -> int:
    """
    Automatically determine optimal batch size based on hardware.

    Detection hierarchy:
    1. CUDA GPU (NVIDIA) - High batch sizes for desktop/server GPUs
    2. ROCm GPU (AMD) - Similar to CUDA for high-end cards
    3. MPS (Apple Metal Performance Shaders) - Apple Silicon optimization
    4. ONNX Runtime GPU (CUDA/TensorRT/DirectML)
    5. CPU - Scale with cores and architecture

    Returns:
        Optimal batch size for the detected hardware.
    """
    # 1. Try PyTorch detection first
    torch = _import_optional("torch")
    if torch is not None:
        # Check for NVIDIA CUDA GPU
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)

            for vram_threshold, batch_size in sorted(
                constants.BATCH_SIZE_VRAM_THRESHOLDS.items(), reverse=True
            ):
                if vram_gb >= vram_threshold:
                    return batch_size
            return 64

        # Check for AMD ROCm GPU
        if hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore
            return constants.DEFAULT_AMD_ROCM_BATCH_SIZE

        # Check for Apple Metal (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            machine = platform.machine().lower()
            if "arm" in machine or "aarch64" in machine:
                try:
                    import subprocess

                    chip_info = subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                    ).lower()

                    if "m3" in chip_info or "m4" in chip_info:
                        return constants.DEFAULT_APPLE_M3_M4_BATCH_SIZE
                    elif "max" in chip_info or "ultra" in chip_info:
                        return constants.DEFAULT_APPLE_MAX_ULTRA_BATCH_SIZE
                    else:
                        return constants.DEFAULT_APPLE_M1_M2_BATCH_SIZE
                except Exception:
                    return constants.DEFAULT_APPLE_M1_M2_BATCH_SIZE

    # 2. Try ONNX Runtime detection
    ort = _import_optional("onnxruntime")
    if ort is not None:
        providers = ort.get_available_providers()
        if (
            "CUDAExecutionProvider" in providers
            or "TensorrtExecutionProvider" in providers
        ):
            return 128
        if "DmlExecutionProvider" in providers:
            return 64
        if "CoreMLExecutionProvider" in providers:
            return 32

    # 3. CPU fallback
    psutil = _import_optional("psutil")
    if psutil is not None:
        cpu_count = psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
    else:
        cpu_count = multiprocessing.cpu_count()
        available_ram_gb = 8.0

    machine = platform.machine().lower()

    if "arm" in machine or "aarch64" in machine:
        if cpu_count <= 4:
            return constants.DEFAULT_ARM_MOBILE_BATCH_SIZE
        elif cpu_count <= 8:
            return constants.DEFAULT_ARM_PI_BATCH_SIZE
        else:
            return constants.DEFAULT_ARM_SERVER_BATCH_SIZE

    base_batch = constants.DEFAULT_CPU_FALLBACK_BATCH_SIZE

    for core_threshold, batch_size in sorted(
        constants.CPU_BATCH_SIZE_BY_CORES.items(), reverse=True
    ):
        if cpu_count >= core_threshold:
            base_batch = batch_size
            break

    if available_ram_gb < 2.0:
        return min(base_batch, 4)
    elif available_ram_gb < 4.0:
        return min(base_batch, 8)
    elif available_ram_gb < 8.0:
        return min(base_batch, 16)

    return base_batch


class VectorCollection:
    """
    Represents a single vector collection within the database.

    Handles vector storage via usearch HNSW index and metadata via SQLite.
    Uses a facade pattern to delegate operations to specialized engine
    components (catalog, search, usearch_index).

    Note:
        Collections are created via `VectorDB.collection()`. Do not instantiate directly.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        name: str,
        distance_strategy: DistanceStrategy,
        quantization: Quantization,
    ):
        self.conn = conn
        self._db_path = db_path
        self.name = name
        self.distance_strategy = distance_strategy
        self.quantization = quantization
        self._quantizer = QuantizationStrategy(quantization)

        # Sanitize name to prevent issues
        if not re.match(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError(
                f"Invalid collection name '{name}'. Must be alphanumeric + underscores."
            )

        # Table names
        if name == "default":
            self._table_name = "tinyvec_items"
            self._legacy_vec_table = "vec_index"  # For migration
        else:
            self._table_name = f"items_{name}"
            self._legacy_vec_table = f"vectors_{name}"  # For migration

        self._fts_table_name = f"{self._table_name}_fts"

        # Usearch index path: {db_path}.{collection}.usearch
        if db_path == ":memory:":
            self._index_path = None  # In-memory index
        else:
            self._index_path = f"{db_path}.{name}.usearch"

        # Initialize components
        self._catalog = CatalogManager(
            conn=self.conn,
            table_name=self._table_name,
            fts_table_name=self._fts_table_name,
        )
        self._catalog.create_tables()

        # Create usearch index
        self._index = UsearchIndex(
            index_path=self._index_path
            or os.path.join(
                tempfile.gettempdir(), f"simplevecdb_{uuid.uuid4().hex}.usearch"
            ),
            ndim=None,  # Will be set on first add
            distance_strategy=self.distance_strategy,
            quantization=self.quantization,
        )

        # Create search engine
        self._search = SearchEngine(
            index=self._index,
            catalog=self._catalog,
            distance_strategy=self.distance_strategy,
        )

        # Check for and perform migration from sqlite-vec
        self._migrate_from_sqlite_vec_if_needed()

    def _migrate_from_sqlite_vec_if_needed(self) -> None:
        """Auto-migrate from sqlite-vec to usearch on first connection."""
        if not self._catalog.check_legacy_sqlite_vec(self._legacy_vec_table):
            return

        _logger.info(
            "Detected legacy sqlite-vec data in collection '%s'. Migrating to usearch...",
            self.name,
        )

        try:
            # Get legacy vectors
            legacy_data = self._catalog.get_legacy_vectors(self._legacy_vec_table)
            if not legacy_data:
                _logger.warning("No vectors found in legacy table")
                self._catalog.drop_legacy_vec_table(self._legacy_vec_table)
                return

            # Deserialize and add to usearch
            keys = []
            vectors = []
            for rowid, blob in legacy_data:
                vec = np.frombuffer(blob, dtype=np.float32)
                keys.append(rowid)
                vectors.append(vec)

            keys_arr = np.array(keys, dtype=np.uint64)
            vectors_arr = np.array(vectors, dtype=np.float32)

            self._index.add(keys_arr, vectors_arr)
            self._index.save()

            # Drop legacy table
            self._catalog.drop_legacy_vec_table(self._legacy_vec_table)

            _logger.info(
                "Migration complete: %d vectors migrated to usearch", len(keys)
            )

        except Exception as e:
            _logger.error("Migration failed: %s", e)
            raise RuntimeError(
                f"Failed to migrate from sqlite-vec: {e}. "
                "You may need to manually migrate or restore from backup."
            ) from e

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        """
        Add texts with optional embeddings and metadata to the collection.

        Automatically infers vector dimension from first batch. Supports upsert
        (update on conflict) when providing existing IDs. For COSINE distance,
        vectors are L2-normalized automatically by usearch.

        Args:
            texts: Document text content to store.
            metadatas: Optional metadata dicts (one per text).
            embeddings: Optional pre-computed embeddings (one per text).
                If None, attempts to use local embedding model.
            ids: Optional document IDs for upsert behavior.

        Returns:
            List of inserted/updated document IDs.

        Raises:
            ValueError: If embedding dimensions don't match, or if no embeddings
                provided and local embedder not available.
        """
        if not texts:
            return []

        # Resolve embeddings
        if embeddings is None:
            try:
                from simplevecdb.embeddings.models import embed_texts as embed_fn

                embeddings = embed_fn(list(texts))
            except Exception as e:
                raise ValueError(
                    "No embeddings provided and local embedder failed – "
                    "install with [server] extra"
                ) from e

        # Normalize metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Process in batches
        from simplevecdb import config

        batch_size = config.EMBEDDING_BATCH_SIZE
        all_ids: list[int] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_metas = metadatas[batch_start:batch_end]
            batch_embeds = embeddings[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end] if ids else None

            # Add to SQLite metadata store (with embeddings for MMR support)
            doc_ids = self._catalog.add_documents(
                batch_texts, list(batch_metas), batch_ids, embeddings=batch_embeds
            )

            # Prepare vectors
            emb_np = np.array(batch_embeds, dtype=np.float32)

            # Add to usearch index
            self._index.add(np.array(doc_ids, dtype=np.uint64), emb_np)

            all_ids.extend(doc_ids)

        return all_ids

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for most similar vectors using HNSW approximate nearest neighbor.

        For COSINE distance, returns distance in [0, 2] (lower = more similar).
        For L2/L1, returns raw distance (lower = more similar).

        Args:
            query: Query vector or text string (auto-embedded if string).
            k: Number of nearest neighbors to return.
            filter: Optional metadata filter.
            exact: Force search mode. None=adaptive (brute-force for <10k vectors),
                   True=always brute-force (perfect recall), False=always HNSW.

        Returns:
            List of (Document, distance) tuples, sorted by ascending distance.
        """
        return self._search.similarity_search(query, k, filter, exact=exact)

    def keyword_search(
        self, query: str, k: int = 5, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """
        Search using BM25 keyword ranking (full-text search).

        Uses SQLite's FTS5 extension for BM25-based ranking.

        Args:
            query: Text query using FTS5 syntax.
            k: Maximum number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of (Document, bm25_score) tuples, sorted by descending relevance.

        Raises:
            RuntimeError: If FTS5 is not available.
        """
        return self._search.keyword_search(query, k, filter)

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
    ) -> list[tuple[Document, float]]:
        """
        Combine BM25 keyword search with vector similarity using Reciprocal Rank Fusion.

        Args:
            query: Text query for keyword search.
            k: Final number of results after fusion.
            filter: Optional metadata filter.
            query_vector: Optional pre-computed query embedding.
            vector_k: Number of vector search candidates.
            keyword_k: Number of keyword search candidates.
            rrf_k: RRF constant parameter (default: 60).

        Returns:
            List of (Document, rrf_score) tuples, sorted by descending RRF score.

        Raises:
            RuntimeError: If FTS5 is not available.
        """
        return self._search.hybrid_search(
            query,
            k,
            filter,
            query_vector=query_vector,
            vector_k=vector_k,
            keyword_k=keyword_k,
            rrf_k=rrf_k,
        )

    def max_marginal_relevance_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Search with diversity - return relevant but non-redundant results.

        Args:
            query: Query vector or text string.
            k: Number of diverse results to return.
            fetch_k: Number of candidates to consider.
            lambda_mult: Diversity trade-off (0=diverse, 1=relevant).
            filter: Optional metadata filter.

        Returns:
            List of Documents ordered by MMR selection.
        """
        return self._search.max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, filter
        )

    def delete_by_ids(self, ids: Iterable[int]) -> None:
        """
        Delete documents by their IDs.

        Removes documents from both usearch index and SQLite metadata.
        Does NOT auto-vacuum; call `VectorDB.vacuum()` separately.

        Args:
            ids: Document IDs to delete
        """
        ids_list = list(ids)
        if not ids_list:
            return

        # Delete from usearch
        self._index.remove(ids_list)

        # Delete from SQLite
        self._catalog.delete_by_ids(ids_list)

    def remove_texts(
        self,
        texts: Sequence[str] | None = None,
        filter: dict[str, Any] | None = None,
    ) -> int:
        """
        Remove documents by text content or metadata filter.

        Args:
            texts: Optional list of exact text strings to remove
            filter: Optional metadata filter dict

        Returns:
            Number of documents deleted

        Raises:
            ValueError: If neither texts nor filter provided
        """
        if texts is None and filter is None:
            raise ValueError("Must provide either texts or filter to remove")

        ids_to_delete: list[int] = []

        if texts:
            ids_to_delete.extend(self._catalog.find_ids_by_texts(texts))

        if filter:
            ids_to_delete.extend(
                self._catalog.find_ids_by_filter(
                    filter, self._catalog.build_filter_clause
                )
            )

        unique_ids = list(set(ids_to_delete))
        if unique_ids:
            self.delete_by_ids(unique_ids)

        return len(unique_ids)

    def save(self) -> None:
        """Save the usearch index to disk."""
        self._index.save()

    def rebuild_index(
        self,
        *,
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
    ) -> int:
        """
        Rebuild the usearch HNSW index from embeddings stored in SQLite.

        Useful for:
        - Recovering from index corruption
        - Tuning HNSW parameters (connectivity, expansion)
        - Reclaiming space after many deletions

        Args:
            connectivity: HNSW M parameter (edges per node). Default: 16
            expansion_add: efConstruction (build quality). Default: 128
            expansion_search: ef (search quality). Default: 64

        Returns:
            Number of vectors rebuilt

        Raises:
            RuntimeError: If no embeddings found in SQLite
        """
        _logger.info("Rebuilding usearch index for collection '%s'...", self.name)

        # Get all document IDs
        all_ids = self.conn.execute(f"SELECT id FROM {self._table_name}").fetchall()
        all_ids = [row[0] for row in all_ids]

        if not all_ids:
            _logger.warning("No documents found in collection")
            return 0

        # Fetch embeddings from SQLite
        embeddings_map = self._catalog.get_embeddings_by_ids(all_ids)

        # Filter to only docs with embeddings
        valid_pairs = [
            (doc_id, emb)
            for doc_id in all_ids
            if (emb := embeddings_map.get(doc_id)) is not None
        ]

        if not valid_pairs:
            raise RuntimeError(
                "No embeddings found in SQLite. Cannot rebuild index. "
                "This may happen if documents were added before v2.0.0."
            )

        keys = np.array([doc_id for doc_id, _ in valid_pairs], dtype=np.uint64)
        vectors = np.array([emb for _, emb in valid_pairs], dtype=np.float32)

        # Determine dimension
        ndim = vectors.shape[1]

        # Close old index
        old_path = self._index._path
        self._index.close()

        # Delete old index file
        if old_path.exists():
            old_path.unlink()
            _logger.debug("Deleted old index file: %s", old_path)

        # Create new index with optional custom parameters
        from .engine.usearch_index import (
            DEFAULT_CONNECTIVITY,
            DEFAULT_EXPANSION_ADD,
            DEFAULT_EXPANSION_SEARCH,
        )

        self._index = UsearchIndex(
            index_path=str(old_path),
            ndim=ndim,
            distance_strategy=self.distance_strategy,
            quantization=self.quantization,
            connectivity=connectivity or DEFAULT_CONNECTIVITY,
            expansion_add=expansion_add or DEFAULT_EXPANSION_ADD,
            expansion_search=expansion_search or DEFAULT_EXPANSION_SEARCH,
        )

        # Re-add all vectors
        self._index.add(keys, vectors)
        self._index.save()

        # Update search engine reference
        self._search._index = self._index

        _logger.info("Rebuilt index with %d vectors", len(keys))
        return len(keys)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._catalog.count()

    @property
    def _dim(self) -> int | None:
        """Vector dimension (None if no vectors added yet)."""
        return self._index.ndim


class VectorDB:
    """
    Dead-simple local vector database powered by usearch HNSW.

    SQLite stores metadata and text; usearch stores vectors in separate
    .usearch files per collection. Provides Chroma-like API with built-in
    quantization for storage efficiency.

    Storage layout:
    - {path} - SQLite database (metadata, text, FTS)
    - {path}.{collection}.usearch - usearch HNSW index per collection
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        quantization: Quantization = Quantization.FLOAT,
    ):
        """Initialize the vector database.

        Args:
            path: Database file path or ":memory:" for in-memory database.
            distance_strategy: Default distance metric for similarity search.
            quantization: Default vector compression strategy.
        """
        self.path = str(path)
        self.distance_strategy = distance_strategy
        self.quantization = quantization
        self._collections: dict[str, VectorCollection] = {}

        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    def collection(
        self,
        name: str = "default",
        distance_strategy: DistanceStrategy | None = None,
        quantization: Quantization | None = None,
    ) -> VectorCollection:
        """
        Get or create a named collection.

        Collections provide isolated namespaces within a single database.
        Each collection has its own usearch index file.

        Args:
            name: Collection name (alphanumeric + underscore only).
            distance_strategy: Override database-level distance metric.
            quantization: Override database-level quantization.

        Returns:
            VectorCollection instance.

        Raises:
            ValueError: If collection name contains invalid characters.
        """
        cache_key = name
        if cache_key not in self._collections:
            self._collections[cache_key] = VectorCollection(
                conn=self.conn,
                db_path=self.path,
                name=name,
                distance_strategy=distance_strategy or self.distance_strategy,
                quantization=quantization or self.quantization,
            )
        return self._collections[cache_key]

    # ------------------------------------------------------------------ #
    # Integrations
    # ------------------------------------------------------------------ #
    def as_langchain(
        self, embeddings: Embeddings | None = None, collection_name: str = "default"
    ) -> SimpleVecDBVectorStore:
        """Return a LangChain-compatible vector store interface."""
        from .integrations.langchain import SimpleVecDBVectorStore

        return SimpleVecDBVectorStore(
            db_path=self.path, embedding=embeddings, collection_name=collection_name
        )

    def as_llama_index(self, collection_name: str = "default") -> SimpleVecDBLlamaStore:
        """Return a LlamaIndex-compatible vector store interface."""
        from .integrations.llamaindex import SimpleVecDBLlamaStore

        return SimpleVecDBLlamaStore(db_path=self.path, collection_name=collection_name)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @staticmethod
    def check_migration(path: str | Path) -> dict[str, Any]:
        """
        Check if a database needs migration from sqlite-vec (dry-run).

        Use this before opening a v1.x database to understand what will
        be migrated. Does not modify the database.

        Args:
            path: Path to the SQLite database file

        Returns:
            Dict with migration info:
            - needs_migration: bool
            - collections: list of collection names with legacy data
            - total_vectors: estimated total vector count
            - estimated_size_mb: approximate usearch index size
            - rollback_notes: instructions for reverting if needed

        Example:
            >>> info = VectorDB.check_migration("mydb.db")
            >>> if info["needs_migration"]:
            ...     print(f"Will migrate {info['total_vectors']} vectors")
            ...     print(info["rollback_notes"])
        """
        path = str(path)
        if path == ":memory:" or not Path(path).exists():
            return {
                "needs_migration": False,
                "collections": [],
                "total_vectors": 0,
                "estimated_size_mb": 0.0,
                "rollback_notes": "",
            }

        conn = sqlite3.connect(path, check_same_thread=False)
        try:
            # Check for legacy sqlite-vec tables
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}

            legacy_collections = []
            total_vectors = 0
            total_bytes = 0

            # Check default collection
            if "vec_index" in table_names:
                try:
                    count = conn.execute("SELECT COUNT(*) FROM vec_index").fetchone()[0]
                    if count > 0:
                        legacy_collections.append("default")
                        total_vectors += count
                        # Estimate: rowid(8) + embedding blob
                        row = conn.execute(
                            "SELECT embedding FROM vec_index LIMIT 1"
                        ).fetchone()
                        if row and row[0]:
                            dim = len(row[0]) // 4
                            total_bytes += count * dim * 4  # float32
                except Exception:
                    pass

            # Check named collections (vectors_{name})
            for table in table_names:
                if table.startswith("vectors_") and table != "vec_index":
                    collection_name = table[8:]  # Remove "vectors_" prefix
                    try:
                        count = conn.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                        if count > 0:
                            legacy_collections.append(collection_name)
                            total_vectors += count
                            row = conn.execute(
                                f"SELECT embedding FROM {table} LIMIT 1"
                            ).fetchone()
                            if row and row[0]:
                                dim = len(row[0]) // 4
                                total_bytes += count * dim * 4
                    except Exception:
                        pass

            estimated_mb = total_bytes / (1024 * 1024)

            rollback_notes = ""
            if legacy_collections:
                rollback_notes = f"""
MIGRATION ROLLBACK INSTRUCTIONS:
================================
1. BEFORE upgrading, backup your database:
   cp {path} {path}.backup

2. If migration fails or you need to revert:
   - Delete the new .usearch files: {path}.*.usearch
   - Restore from backup: cp {path}.backup {path}
   - Downgrade to simplevecdb<2.0.0

3. After successful migration, the legacy sqlite-vec tables are dropped.
   Keep your backup until you've verified the migration worked correctly.

4. New storage layout after migration:
   - {path} (SQLite: metadata, text, FTS, embeddings)
   - {path}.<collection>.usearch (usearch HNSW index per collection)
"""

            return {
                "needs_migration": len(legacy_collections) > 0,
                "collections": legacy_collections,
                "total_vectors": total_vectors,
                "estimated_size_mb": round(estimated_mb, 2),
                "rollback_notes": rollback_notes.strip(),
            }
        finally:
            conn.close()

    def vacuum(self, checkpoint_wal: bool = True) -> None:
        """
        Reclaim disk space by rebuilding the SQLite database file.

        Note: This only affects SQLite metadata storage. Usearch indexes
        don't support in-place compaction; use rebuild_index() for that.

        Args:
            checkpoint_wal: If True (default), also truncate the WAL file.
        """
        if checkpoint_wal:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.execute("VACUUM")
        self.conn.execute("PRAGMA optimize")

    def save(self) -> None:
        """Save all collection indexes to disk."""
        for collection in self._collections.values():
            collection.save()

    def close(self) -> None:
        """Close the database connection and save indexes."""
        self.save()
        self.conn.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
