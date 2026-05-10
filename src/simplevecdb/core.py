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
import threading
import time
import numpy as np
import uuid
from collections import defaultdict
from collections.abc import Generator, Iterable, Iterator, Sequence
from typing import Any, TYPE_CHECKING
from pathlib import Path
import platform
import multiprocessing
from .types import (
    Document,
    DistanceStrategy,
    Quantization,
    Edge,
    Event,
    StreamingProgress,
    ProgressCallback,
    ClusterResult,
    ClusterTagCallback,
)
from .utils import _import_optional
from .engine.quantization import QuantizationStrategy
from .engine.search import SearchEngine
from .engine.catalog import CatalogManager, _TxState
from .engine.usearch_index import UsearchIndex
from .engine.clustering import ClusterEngine, ClusterAlgorithm
from . import constants
from .encryption import (
    create_encrypted_connection,
    encrypt_index_file,
    decrypt_index_file,
    get_encrypted_index_path,
    EncryptionError,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from .integrations.langchain import SimpleVecDBVectorStore
    from .integrations.llamaindex import SimpleVecDBLlamaStore

_logger = logging.getLogger("simplevecdb.core")


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
        encryption_key: str | bytes | None = None,
        store_embeddings: bool = False,
        lock: threading.RLock | None = None,
        tx_state: _TxState | None = None,
    ):
        self.conn = conn
        self._db_path = db_path
        self.name = name
        self.distance_strategy = distance_strategy
        self.quantization = quantization
        self._quantizer = QuantizationStrategy(quantization)
        self._encryption_key = encryption_key
        self._store_embeddings = store_embeddings
        # Connection-level lock shared with the parent VectorDB so all
        # collections sharing the same sqlite3.Connection serialize their
        # transactional access from Python.
        self._lock: threading.RLock = lock if lock is not None else threading.RLock()
        # Shared transaction depth; defaults to a per-collection state when
        # the parent VectorDB didn't pass one (e.g. legacy direct ctor use).
        self._tx_state: _TxState = tx_state if tx_state is not None else _TxState()

        # Sanitize name to prevent issues
        if not re.match(constants.COLLECTION_NAME_PATTERN, name):
            raise ValueError(
                f"Invalid collection name '{name}'. Must be alphanumeric + underscores."
            )

        # Table names
        if name == "default":
            self._table_name = "tinyvec_items"
        else:
            self._table_name = f"items_{name}"

        self._fts_table_name = f"{self._table_name}_fts"

        # Usearch index path: {db_path}.{collection}.usearch
        if db_path == ":memory:":
            self._index_path = None  # In-memory index
        else:
            self._index_path = f"{db_path}.{name}.usearch"

        # Initialize components — share the connection lock with the catalog
        # so add_documents / delete_by_ids / etc. all serialize properly.
        # The optional _tx_state is also shared with the parent VectorDB
        # so db.transaction() can suspend per-call commits everywhere.
        self._catalog = CatalogManager(
            conn=self.conn,
            table_name=self._table_name,
            fts_table_name=self._fts_table_name,
            lock=self._lock,
            tx_state=getattr(self, "_tx_state", None),
        )
        self._catalog.create_tables()

        # Handle encrypted index loading
        actual_index_path = self._resolve_index_path()

        # In-memory databases need a temp file for the usearch index. Track
        # it so close() can unlink it; otherwise every in-memory VectorDB
        # leaks a file in $TMPDIR.
        if actual_index_path is None:
            self._ephemeral_index_path: str | None = os.path.join(
                tempfile.gettempdir(), f"simplevecdb_{uuid.uuid4().hex}.usearch"
            )
        else:
            self._ephemeral_index_path = None

        # Create usearch index. Exactly one of actual_index_path /
        # self._ephemeral_index_path is non-None per the if/else above.
        chosen_index_path = actual_index_path or self._ephemeral_index_path
        assert chosen_index_path is not None
        self._index = UsearchIndex(
            index_path=chosen_index_path,
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

    def _resolve_index_path(self) -> str | None:
        """
        Resolve the actual index path, handling encryption.

        If encryption is enabled and an encrypted index exists, decrypt it first.
        Returns the path to use for the usearch index.
        """
        if self._index_path is None:
            return None

        index_path = Path(self._index_path)

        # Check for encrypted index
        encrypted_path = get_encrypted_index_path(index_path)

        if encrypted_path is not None:
            if self._encryption_key is None:
                raise EncryptionError(
                    f"Encrypted index found at {encrypted_path} but no encryption_key provided. "
                    "Pass encryption_key to VectorDB to decrypt."
                )
            # Decrypt to the expected path
            decrypt_index_file(encrypted_path, self._encryption_key)
            _logger.info("Decrypted index file: %s", encrypted_path)

        return self._index_path

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
        embeddings: Sequence[Sequence[float]] | None = None,
        ids: Sequence[int | None] | None = None,
        *,
        parent_ids: Sequence[int | None] | None = None,
        threads: int = 0,
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
            parent_ids: Optional parent document IDs for hierarchical relationships.
            threads: Number of threads for parallel insertion (0=auto).

        Returns:
            List of inserted/updated document IDs.

        Raises:
            ValueError: If embedding dimensions don't match, or if no embeddings
                provided and local embedder not available.
        """
        if not texts:
            return []

        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})"
            )
        if embeddings is not None and len(embeddings) != len(texts):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) must match texts length ({len(texts)})"
            )
        if ids is not None and len(ids) != len(texts):
            raise ValueError(
                f"ids length ({len(ids)}) must match texts length ({len(texts)})"
            )
        if parent_ids is not None and len(parent_ids) != len(texts):
            raise ValueError(
                f"parent_ids length ({len(parent_ids)}) must match texts length ({len(texts)})"
            )

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
            batch_parent_ids = parent_ids[batch_start:batch_end] if parent_ids else None

            # Reject NaN/Inf before any persistence. HNSW graph construction
            # with non-finite vectors silently produces undefined neighbours
            # and can corrupt the graph; rejecting after the SQLite insert
            # would leave orphan rows with no index entry.
            emb_np = np.asarray(batch_embeds, dtype=np.float32)
            if not np.all(np.isfinite(emb_np)):
                raise ValueError(
                    "Input vectors contain NaN or Inf; refusing to add to index"
                )

            # Add to SQLite metadata store
            doc_ids = self._catalog.add_documents(
                batch_texts,
                list(batch_metas),
                batch_ids,
                embeddings=batch_embeds if self._store_embeddings else None,
                parent_ids=batch_parent_ids,
            )

            # Add to usearch index
            self._index.add(np.asarray(doc_ids, dtype=np.uint64), emb_np, threads=threads)

            all_ids.extend(doc_ids)

        return all_ids

    def add_texts_streaming(
        self,
        items: Iterable[tuple[str, dict | None, Sequence[float] | None]],
        *,
        batch_size: int | None = None,
        threads: int = 0,
        on_progress: ProgressCallback | None = None,
    ) -> Generator[StreamingProgress, None, list[int]]:
        """
        Stream documents into the collection with controlled memory usage.

        Processes documents in batches from any iterable (generator, file reader,
        API paginator, etc.) without loading all data into memory. Yields progress
        after each batch for monitoring large ingestions.

        Args:
            items: Iterable of (text, metadata, embedding) tuples.
                - text: Document content (required)
                - metadata: Optional dict, use None for empty
                - embedding: Optional pre-computed vector, use None to auto-embed
            batch_size: Documents per batch (default: config.EMBEDDING_BATCH_SIZE).
            threads: Threads for parallel insertion (0=auto).
            on_progress: Optional callback invoked after each batch.

        Yields:
            StreamingProgress dict after each batch with:
            - batch_num: Current batch number (1-indexed)
            - total_batches: Estimated total (None if unknown)
            - docs_processed: Cumulative documents inserted
            - docs_in_batch: Documents in current batch
            - batch_ids: IDs of documents in current batch

        Returns:
            List of all inserted document IDs (access via generator.send(None)
            or list(generator) after exhaustion).

        Example:
            >>> def load_documents():
            ...     for line in open("large_file.jsonl"):
            ...         doc = json.loads(line)
            ...         yield (doc["text"], doc.get("meta"), None)
            ...
            >>> gen = collection.add_texts_streaming(load_documents())
            >>> for progress in gen:
            ...     print(f"Batch {progress['batch_num']}: {progress['docs_processed']} total")
            >>> # IDs accumulated in progress['ids'] for each batch

        Example with callback:
            >>> def log_progress(p):
            ...     print(f"{p['docs_processed']} docs inserted")
            >>> list(collection.add_texts_streaming(items, on_progress=log_progress))
        """
        from simplevecdb import config

        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE

        all_ids: list[int] = []
        batch_num = 0
        docs_processed = 0

        # Accumulate batch
        batch_texts: list[str] = []
        batch_metas: list[dict] = []
        # None entries are placeholders for items that need auto-embedding;
        # _process_streaming_batch resolves them before persistence.
        batch_embeds: list[Sequence[float] | None] = []
        needs_embedding = False

        for text, metadata, embedding in items:
            batch_texts.append(text)
            batch_metas.append(metadata or {})
            if embedding is not None:
                batch_embeds.append(embedding)
            else:
                needs_embedding = True
                batch_embeds.append(None)  # Placeholder for auto-embedding

            # Process batch when full
            if len(batch_texts) >= batch_size:
                batch_ids = self._process_streaming_batch(
                    batch_texts, batch_metas, batch_embeds, needs_embedding, threads
                )
                all_ids.extend(batch_ids)
                batch_num += 1
                docs_processed += len(batch_ids)

                progress: StreamingProgress = {
                    "batch_num": batch_num,
                    "total_batches": None,
                    "docs_processed": docs_processed,
                    "docs_in_batch": len(batch_ids),
                    "batch_ids": batch_ids,
                }

                if on_progress:
                    on_progress(progress)

                yield progress

                # Reset batch
                batch_texts = []
                batch_metas = []
                batch_embeds = []
                needs_embedding = False

        # Process final partial batch
        if batch_texts:
            batch_ids = self._process_streaming_batch(
                batch_texts, batch_metas, batch_embeds, needs_embedding, threads
            )
            all_ids.extend(batch_ids)
            batch_num += 1
            docs_processed += len(batch_ids)

            progress = {
                "batch_num": batch_num,
                "total_batches": batch_num,  # Now we know total
                "docs_processed": docs_processed,
                "docs_in_batch": len(batch_ids),
                "batch_ids": batch_ids,
            }

            if on_progress:
                on_progress(progress)

            yield progress

        return all_ids

    def _process_streaming_batch(
        self,
        texts: list[str],
        metas: list[dict],
        embeds: list[Sequence[float] | None],
        needs_embedding: bool,
        threads: int,
    ) -> list[int]:
        """Process a single batch for streaming insert."""
        # Generate embeddings if needed
        if needs_embedding:
            try:
                from simplevecdb.embeddings.models import embed_texts as embed_fn

                generated = embed_fn(texts)
                # Replace placeholders with generated embeddings
                for i, emb in enumerate(embeds):
                    if emb is None or (isinstance(emb, list) and len(emb) == 0):
                        embeds[i] = generated[i]
            except Exception as e:
                raise ValueError(
                    "Auto-embedding failed - install with [server] extra or provide embeddings"
                ) from e

        # By this point any None placeholder has been replaced with a real
        # embedding (either auto-generated above or supplied by the caller).
        # Narrow the type so the asarray and add_documents calls type-check.
        if any(e is None for e in embeds):
            raise ValueError(
                "Internal error: streaming batch reached persistence with "
                "unresolved auto-embedding placeholders."
            )
        embeds_resolved: list[Sequence[float]] = [
            e for e in embeds if e is not None
        ]

        # Validate vectors before any persistence — see add_texts for rationale.
        emb_np = np.asarray(embeds_resolved, dtype=np.float32)
        if not np.all(np.isfinite(emb_np)):
            raise ValueError(
                "Input vectors contain NaN or Inf; refusing to add to index"
            )

        # Add to catalog and index
        doc_ids = self._catalog.add_documents(
            texts, metas, None, embeddings=embeds_resolved
        )
        self._index.add(np.asarray(doc_ids, dtype=np.uint64), emb_np, threads=threads)

        return doc_ids

    def similarity_search(
        self,
        query: str | Sequence[float],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
        threads: int = 0,
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
            threads: Number of threads for parallel search (0=auto).

        Returns:
            List of (Document, distance) tuples, sorted by ascending distance.
        """
        if k <= 0:
            return []
        return self._search.similarity_search(
            query, k, filter, exact=exact, threads=threads
        )

    def similarity_search_batch(
        self,
        queries: Sequence[Sequence[float]],
        k: int = 5,
        filter: dict[str, Any] | None = None,
        *,
        exact: bool | None = None,
        threads: int = 0,
    ) -> list[list[tuple[Document, float]]]:
        """
        Search for similar vectors across multiple queries in parallel.

        Automatically batches queries for ~10x throughput compared to
        sequential single-query searches. Uses usearch's native batch
        search optimization.

        Args:
            queries: List of query vectors.
            k: Number of nearest neighbors per query.
            filter: Optional metadata filter (applied to all queries).
            exact: Force search mode. None=adaptive, True=brute-force, False=HNSW.
            threads: Number of threads for parallel search (0=auto).

        Returns:
            List of result lists, one per query. Each result is (Document, distance).

        Example:
            >>> queries = [embedding1, embedding2, embedding3]
            >>> results = collection.similarity_search_batch(queries, k=5)
            >>> for query_results in results:
            ...     print(f"Found {len(query_results)} matches")
        """
        if k <= 0 or not queries:
            return []
        return self._search.similarity_search_batch(
            queries, k, filter, exact=exact, threads=threads
        )

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
        if k <= 0 or not query:
            return []
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
        if k <= 0 or not query:
            return []
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

        # Delete from SQLite first (transactional, can rollback on failure)
        self._catalog.delete_by_ids(ids_list)

        # Then remove from usearch (if this fails, catalog is clean and
        # rebuild_index() can recover the index from stored data)
        self._index.remove(ids_list)

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
        """
        Save the usearch index to disk.

        If encryption is enabled, the index is encrypted after saving.
        """
        self._index.save()

        # Encrypt index if encryption is enabled
        if self._encryption_key is not None and self._index_path is not None:
            index_path = Path(self._index_path)
            if index_path.exists():
                encrypt_index_file(index_path, self._encryption_key)

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

        # Serialize the entire fetch + build + swap on the connection-level
        # lock so concurrent add/delete operations cannot mutate the catalog
        # mid-rebuild and produce a stale or inconsistent snapshot. The lock
        # is reentrant; CatalogManager read methods reacquire it but that
        # is harmless under RLock.
        with self._lock:
            return self._rebuild_index_locked(connectivity, expansion_add, expansion_search)

    def _rebuild_index_locked(
        self,
        connectivity: int | None,
        expansion_add: int | None,
        expansion_search: int | None,
    ) -> int:
        # Precondition: caller must already hold ``self._lock``. ``rebuild_index``
        # is the only public entry point and acquires it before delegating; this
        # private helper relies on RLock re-entrancy so the catalog reads below
        # are serialized against concurrent add/delete on the shared connection.
        # Routing the read through CatalogManager keeps the lock invariant
        # explicit instead of bare ``self.conn.execute(...)``.
        all_ids = self._catalog.list_all_ids()

        if not all_ids:
            _logger.warning("No documents found in collection")
            return 0

        # Fetch embeddings from SQLite
        embeddings_map = self._catalog.get_embeddings_by_ids(all_ids)

        if not embeddings_map and not self._store_embeddings:
            raise RuntimeError(
                "Cannot rebuild index: no embeddings stored in SQLite. "
                "Create the collection with store_embeddings=True to enable "
                "rebuild_index(), or re-add documents with store_embeddings=True."
            )

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

        # Atomic rebuild: build the new index at a sibling path, save it
        # durably, then os.replace() it onto the live path. The old index
        # remains intact and recoverable until the final rename succeeds.
        old_path = self._index._path
        self._index.close()

        rebuild_path = old_path.with_suffix(old_path.suffix + ".rebuild")
        if rebuild_path.exists():
            # Clean up remnant from a prior failed rebuild
            rebuild_path.unlink()

        new_index = UsearchIndex(
            index_path=str(rebuild_path),
            ndim=ndim,
            distance_strategy=self.distance_strategy,
            quantization=self.quantization,
            connectivity=connectivity if connectivity is not None else constants.USEARCH_DEFAULT_CONNECTIVITY,
            expansion_add=expansion_add if expansion_add is not None else constants.USEARCH_DEFAULT_EXPANSION_ADD,
            expansion_search=expansion_search if expansion_search is not None else constants.USEARCH_DEFAULT_EXPANSION_SEARCH,
        )
        new_index.add(keys, vectors)
        new_index.save()

        # Atomically swap the rebuilt index into place. Until this rename,
        # the old index file at old_path is still the canonical copy.
        os.replace(str(rebuild_path), str(old_path))
        try:
            dir_fd = os.open(str(old_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass

        # Repoint the rebuilt index at the canonical path so future saves
        # land at old_path rather than the now-vanished rebuild_path.
        new_index._path = old_path
        self._index = new_index

        # Update search engine reference
        self._search._index = self._index

        _logger.info("Rebuilt index with %d vectors", len(keys))
        return len(keys)

    # ------------------------------------------------------------------ #
    # Hierarchical Relationships
    # ------------------------------------------------------------------ #

    def get_children(self, doc_id: int) -> list[Document]:
        """
        Get all direct children of a document.

        Args:
            doc_id: ID of the parent document

        Returns:
            List of child Documents

        Example:
            >>> # Add parent and children
            >>> parent_id = collection.add_texts(["Parent doc"], embeddings=[emb])[0]
            >>> collection.add_texts(
            ...     ["Child 1", "Child 2"],
            ...     embeddings=[emb1, emb2],
            ...     parent_ids=[parent_id, parent_id]
            ... )
            >>> children = collection.get_children(parent_id)
        """
        rows = self._catalog.get_children(doc_id)
        return [Document(page_content=text, metadata=meta) for _, text, meta in rows]

    def get_parent(self, doc_id: int) -> Document | None:
        """
        Get the parent document of a given document.

        Args:
            doc_id: ID of the child document

        Returns:
            Parent Document, or None if no parent
        """
        result = self._catalog.get_parent(doc_id)
        if result is None:
            return None
        _, text, meta = result
        return Document(page_content=text, metadata=meta)

    def get_descendants(
        self, doc_id: int, max_depth: int | None = None
    ) -> list[tuple[Document, int]]:
        """
        Get all descendants of a document (recursive).

        Args:
            doc_id: ID of the root document
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of (Document, depth) tuples, ordered by depth then ID
        """
        rows = self._catalog.get_descendants(doc_id, max_depth)
        return [
            (Document(page_content=text, metadata=meta), depth)
            for _, text, meta, depth in rows
        ]

    def get_ancestors(
        self, doc_id: int, max_depth: int | None = None
    ) -> list[tuple[Document, int]]:
        """
        Get all ancestors of a document (path to root).

        Args:
            doc_id: ID of the document
            max_depth: Maximum depth to traverse (None for unlimited)

        Returns:
            List of (Document, depth) tuples, from immediate parent to root
        """
        rows = self._catalog.get_ancestors(doc_id, max_depth)
        return [
            (Document(page_content=text, metadata=meta), depth)
            for _, text, meta, depth in rows
        ]

    def set_parent(self, doc_id: int, parent_id: int | None) -> bool:
        """
        Set or update the parent of a document.

        Args:
            doc_id: ID of the document to update
            parent_id: New parent ID (None to remove parent relationship)

        Returns:
            True if document was updated, False if document not found
        """
        return self._catalog.set_parent(doc_id, parent_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Clustering Methods
    # ─────────────────────────────────────────────────────────────────────────

    def cluster(
        self,
        n_clusters: int | None = None,
        algorithm: ClusterAlgorithm = "minibatch_kmeans",
        *,
        filter: dict[str, Any] | None = None,
        sample_size: int | None = None,
        min_cluster_size: int = 5,
        random_state: int | None = None,
    ) -> ClusterResult:
        """
        Cluster documents in the collection by their embeddings.

        Requires scikit-learn and hdbscan (included in the standard install).

        Args:
            n_clusters: Number of clusters (required for kmeans/minibatch_kmeans).
            algorithm: Clustering algorithm - 'kmeans', 'minibatch_kmeans', or 'hdbscan'.
            filter: Optional metadata filter to cluster a subset of documents.
            sample_size: If set, cluster a random sample and assign rest to nearest centroid.
            min_cluster_size: Minimum cluster size (HDBSCAN only).
            random_state: Random seed for reproducibility.

        Returns:
            ClusterResult with labels, centroids, and doc_ids.

        Raises:
            ImportError: If scikit-learn or hdbscan (for HDBSCAN) not installed.
            ValueError: If n_clusters required but not provided.

        Example:
            >>> result = collection.cluster(n_clusters=5)
            >>> print(result.summary())  # {0: 42, 1: 38, 2: 20, ...}
        """
        engine = ClusterEngine()

        doc_ids = list(self._index.keys())
        if not doc_ids:
            return ClusterResult(
                labels=np.array([], dtype=np.int32),
                centroids=None,
                doc_ids=[],
                n_clusters=0,
                algorithm=algorithm,
            )

        if filter:
            filtered_ids = set(
                self._catalog.find_ids_by_filter(
                    filter, self._catalog.build_filter_clause
                )
            )
            doc_ids = [d for d in doc_ids if d in filtered_ids]

        vectors = self._index.get(np.array(doc_ids, dtype=np.uint64))

        effective_n_clusters = n_clusters
        if n_clusters is not None and algorithm in ("kmeans", "minibatch_kmeans"):
            effective_n_clusters = min(n_clusters, len(doc_ids))

        if sample_size and sample_size < len(doc_ids):
            rng = np.random.default_rng(random_state)
            sample_indices = rng.choice(len(doc_ids), sample_size, replace=False)
            sample_ids = [doc_ids[i] for i in sample_indices]
            sample_vectors = vectors[sample_indices]

            result = engine.cluster_vectors(
                sample_vectors,
                sample_ids,
                algorithm=algorithm,
                n_clusters=effective_n_clusters,
                min_cluster_size=min_cluster_size,
                random_state=random_state,
            )

            if result.centroids is not None:
                remaining_mask = np.ones(len(doc_ids), dtype=bool)
                remaining_mask[sample_indices] = False
                remaining_ids = [doc_ids[i] for i, m in enumerate(remaining_mask) if m]
                remaining_vectors = vectors[remaining_mask]

                remaining_labels = engine.assign_to_nearest_centroid(
                    remaining_vectors, result.centroids
                )

                all_ids = sample_ids + remaining_ids
                all_labels = np.concatenate([result.labels, remaining_labels])

                order = np.argsort(all_ids)
                return ClusterResult(
                    labels=all_labels[order],
                    centroids=result.centroids,
                    doc_ids=[all_ids[i] for i in order],
                    n_clusters=result.n_clusters,
                    algorithm=algorithm,
                )
            return result

        return engine.cluster_vectors(
            vectors,
            doc_ids,
            algorithm=algorithm,
            n_clusters=effective_n_clusters,
            min_cluster_size=min_cluster_size,
            random_state=random_state,
        )

    def auto_tag(
        self,
        cluster_result: ClusterResult,
        *,
        method: str = "keywords",
        n_keywords: int = 5,
        custom_callback: ClusterTagCallback | None = None,
    ) -> dict[int, str]:
        """
        Generate descriptive tags for each cluster.

        Args:
            cluster_result: Result from cluster() method.
            method: Tagging method - 'keywords' (TF-IDF) or 'custom'.
            n_keywords: Number of keywords per cluster (for 'keywords' method).
            custom_callback: Custom function (texts: list[str]) -> str for 'custom' method.

        Returns:
            Dict mapping cluster_id -> tag string.

        Example:
            >>> result = collection.cluster(n_clusters=3)
            >>> tags = collection.auto_tag(result)
            >>> print(tags)  # {0: 'machine learning, neural', 1: 'database, sql', ...}
        """
        docs = self._catalog.get_documents_by_ids(cluster_result.doc_ids)

        cluster_texts: dict[int, list[str]] = defaultdict(list)
        for doc_id, label in zip(cluster_result.doc_ids, cluster_result.labels):
            if doc_id in docs:
                cluster_texts[int(label)].append(docs[doc_id][0])

        if method == "custom" and custom_callback:
            return {
                cluster_id: custom_callback(texts)
                for cluster_id, texts in cluster_texts.items()
            }

        engine = ClusterEngine()
        return engine.generate_keywords(cluster_texts, n_keywords)

    def assign_cluster_metadata(
        self,
        cluster_result: ClusterResult,
        tags: dict[int, str] | None = None,
        *,
        metadata_key: str = "cluster",
        tag_key: str = "cluster_tag",
    ) -> int:
        """
        Persist cluster assignments to document metadata.

        After calling this, you can filter by cluster: filter={"cluster": 2}

        Args:
            cluster_result: Result from cluster() method.
            tags: Optional cluster tags from auto_tag(). If provided, also sets tag_key.
            metadata_key: Metadata key for cluster ID (default: "cluster").
            tag_key: Metadata key for cluster tag (default: "cluster_tag").

        Returns:
            Number of documents updated.

        Example:
            >>> result = collection.cluster(n_clusters=5)
            >>> tags = collection.auto_tag(result)
            >>> collection.assign_cluster_metadata(result, tags)
            >>> # Now filter by cluster
            >>> docs = collection.similarity_search(query, filter={"cluster": 2})
        """
        updates: list[tuple[int, dict[str, Any]]] = []
        for doc_id, label in zip(cluster_result.doc_ids, cluster_result.labels):
            meta: dict[str, Any] = {metadata_key: int(label)}
            if tags and int(label) in tags:
                meta[tag_key] = tags[int(label)]
            updates.append((doc_id, meta))

        return self._catalog.update_metadata_batch(updates)

    def get_cluster_members(
        self,
        cluster_id: int,
        *,
        metadata_key: str = "cluster",
    ) -> list[Document]:
        """
        Get all documents in a cluster (requires assign_cluster_metadata first).

        Args:
            cluster_id: Cluster ID to retrieve.
            metadata_key: Metadata key where cluster is stored (default: "cluster").

        Returns:
            List of Documents in the cluster.
        """
        rows = self._catalog.get_all_docs_with_text(
            filter_dict={metadata_key: cluster_id},
            filter_builder=self._catalog.build_filter_clause,
        )
        return [Document(page_content=text, metadata=meta) for _, text, meta in rows]

    def save_cluster(
        self,
        name: str,
        cluster_result: ClusterResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save cluster state for later reuse without re-clustering.

        Persists centroids and algorithm info so new documents can be assigned
        to existing clusters using assign_to_cluster().

        Args:
            name: Unique name for this cluster configuration.
            cluster_result: Result from cluster() method.
            metadata: Optional additional metadata (tags, metrics, etc.).

        Example:
            >>> result = collection.cluster(n_clusters=5)
            >>> tags = collection.auto_tag(result)
            >>> collection.save_cluster("product_categories", result, metadata={"tags": tags})
        """
        centroids_bytes = None
        if cluster_result.centroids is not None:
            centroids_bytes = cluster_result.centroids.tobytes()

        self._catalog.save_cluster_state(
            name=name,
            algorithm=cluster_result.algorithm,
            n_clusters=cluster_result.n_clusters,
            centroids=centroids_bytes,
            metadata=metadata,
        )

    def load_cluster(self, name: str) -> tuple[ClusterResult, dict[str, Any]] | None:
        """
        Load a saved cluster configuration.

        Args:
            name: Name of the saved cluster configuration.

        Returns:
            Tuple of (ClusterResult with centroids, metadata dict) or None if not found.

        Example:
            >>> saved = collection.load_cluster("product_categories")
            >>> if saved:
            ...     result, meta = saved
            ...     print(f"Loaded {result.n_clusters} clusters")
        """
        state = self._catalog.load_cluster_state(name)
        if state is None:
            return None

        algorithm, n_clusters, centroids_bytes, metadata = state

        centroids = None
        if centroids_bytes is not None:
            dim = self.dim
            if dim:
                centroids = np.frombuffer(centroids_bytes, dtype=np.float32).reshape(
                    n_clusters, dim
                )

        result = ClusterResult(
            labels=np.array([], dtype=np.int32),
            centroids=centroids,
            doc_ids=[],
            n_clusters=n_clusters,
            algorithm=algorithm,
        )
        return result, metadata

    def list_clusters(self) -> list[dict[str, Any]]:
        """List all saved cluster configurations."""
        return self._catalog.list_cluster_states()

    def delete_cluster(self, name: str) -> bool:
        """Delete a saved cluster configuration."""
        return self._catalog.delete_cluster_state(name)

    def assign_to_cluster(
        self,
        name: str,
        doc_ids: list[int] | None = None,
        *,
        metadata_key: str = "cluster",
    ) -> int:
        """
        Assign documents to clusters using saved centroids.

        Fast assignment without re-clustering - uses nearest centroid matching.
        Useful for assigning newly added documents to existing cluster structure.

        Args:
            name: Name of saved cluster configuration (from save_cluster).
            doc_ids: Document IDs to assign. If None, assigns all unassigned docs.
            metadata_key: Metadata key to store cluster assignment.

        Returns:
            Number of documents assigned.

        Raises:
            ValueError: If cluster not found or has no centroids (HDBSCAN).

        Example:
            >>> # Add new documents
            >>> new_ids = collection.add_texts(new_texts, embeddings=new_embs)
            >>> # Assign to existing clusters
            >>> collection.assign_to_cluster("product_categories", new_ids)
        """
        saved = self.load_cluster(name)
        if saved is None:
            raise ValueError(f"Cluster '{name}' not found")

        result, _ = saved
        if result.centroids is None:
            raise ValueError(
                f"Cluster '{name}' has no centroids (HDBSCAN clusters cannot be used for assignment)"
            )

        if doc_ids is None:
            all_ids = list(self._index.keys())
            # Get all documents to check for metadata key existence
            all_docs = self._catalog.get_all_docs_with_text()
            assigned_ids = {
                doc_id for doc_id, _, meta in all_docs if metadata_key in meta
            }
            doc_ids = [d for d in all_ids if d not in assigned_ids]

        if not doc_ids:
            return 0

        vectors = self._index.get(np.array(doc_ids, dtype=np.uint64))

        engine = ClusterEngine()
        labels = engine.assign_to_nearest_centroid(vectors, result.centroids)

        updates = [
            (doc_id, {metadata_key: int(label)})
            for doc_id, label in zip(doc_ids, labels)
        ]
        return self._catalog.update_metadata_batch(updates)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._catalog.count()

    def get_documents(
        self,
        filter_dict: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """Get documents with text content and metadata.

        Args:
            filter_dict: Optional metadata filter to narrow results.
            limit: Maximum number of documents to return (None = all).
            offset: Number of documents to skip (None = 0).

        Returns:
            List of (doc_id, text, metadata) tuples, ordered by ID.
        """
        filter_builder = self._catalog.build_filter_clause if filter_dict else None
        return self._catalog.get_all_docs_with_text(
            filter_dict=filter_dict,
            filter_builder=filter_builder,
            limit=limit,
            offset=offset,
        )

    def get_embeddings_by_ids(self, ids: Sequence[int]) -> dict[int, Any]:
        """Fetch stored embeddings by document IDs.

        Args:
            ids: Document IDs to fetch embeddings for.

        Returns:
            Dict mapping doc_id to embedding array (or None).
        """
        return self._catalog.get_embeddings_by_ids(list(ids))

    def update_metadata(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        """Update metadata for multiple documents (shallow merge).

        Args:
            updates: List of (doc_id, metadata_dict) tuples.

        Returns:
            Number of documents updated.
        """
        return self._catalog.update_metadata_batch(updates)

    def update_embedding(
        self,
        doc_id: int,
        vector: Sequence[float] | "np.ndarray",
        *,
        source: str | None = None,
    ) -> None:
        """Buffer a native vector update without HNSW remove+re-add (gap 1).

        Writes the new vector to the per-collection pending buffer in a
        single SQL transaction. The HNSW index is **not** modified until
        `pending.flush()` runs (manually or via threshold-driven flush
        during normal writes), so this method is cheap regardless of how
        many edits a single doc receives between flushes.

        Until flush, similarity searches still rank the doc using its
        previous vector — by design. Use `pending.flush()` to promote
        buffered vectors when their effect on retrieval matters.

        Args:
            doc_id: Existing document id.
            vector: Float vector with the same dim as the index.
            source: Optional free-form tag stored alongside the buffered
                vector (useful for tracing which subsystem queued it).

        Raises:
            ValueError: If vector dimension differs from the index dim.
        """
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"update_embedding expects a 1-D vector, got shape {arr.shape}"
            )
        if self._index.ndim is not None and arr.shape[0] != self._index.ndim:
            raise ValueError(
                f"Vector dim {arr.shape[0]} != index dim {self._index.ndim}"
            )
        self._catalog.upsert_pending_vector(doc_id, arr.tobytes(), source)

    @property
    def pending(self) -> "_PendingNamespace":
        """Sub-namespace for buffered vector updates (gaps 1 & 6)."""
        ns = self.__dict__.get("_pending_ns")
        if ns is None:
            ns = _PendingNamespace(self)
            self.__dict__["_pending_ns"] = ns
        return ns

    def increment_metadata(
        self, doc_id: int, deltas: dict[str, float | int]
    ) -> int:
        """Atomically add numeric deltas to metadata counters (gap 4).

        One UPDATE statement applies all deltas via chained json_set, so
        concurrent writers cannot lose increments — every call is a
        SQLite-atomic read-modify-write. Missing keys are treated as 0.

        Example:
            >>> collection.increment_metadata(
            ...     doc_id, {"retrieval_count": 1, "drift_total": 0.02}
            ... )
            1

        Args:
            doc_id: Target document id.
            deltas: Map of counter name -> numeric delta (int or float).
                Keys must be safe identifiers (alphanumeric + underscore,
                not starting with a digit).

        Returns:
            1 if the row exists and was updated, 0 otherwise.
        """
        return self._catalog.increment_metadata(doc_id, deltas)

    @property
    def counters(self) -> "_CountersNamespace":
        """Sub-namespace for atomic counter operations (gap 4)."""
        ns = self.__dict__.get("_counters_ns")
        if ns is None:
            ns = _CountersNamespace(self)
            self.__dict__["_counters_ns"] = ns
        return ns

    @property
    def edges(self) -> "_EdgesNamespace":
        """Sub-namespace for weighted directed edges (gap 3)."""
        ns = self.__dict__.get("_edges_ns")
        if ns is None:
            ns = _EdgesNamespace(self)
            self.__dict__["_edges_ns"] = ns
        return ns

    @property
    def events(self) -> "_EventsNamespace":
        """Sub-namespace for the change feed (gap 7)."""
        ns = self.__dict__.get("_events_ns")
        if ns is None:
            ns = _EventsNamespace(self)
            self.__dict__["_events_ns"] = ns
        return ns

    @property
    def ttl(self) -> "_TTLNamespace":
        """Sub-namespace for TTL/expiry (gap 8)."""
        ns = self.__dict__.get("_ttl_ns")
        if ns is None:
            ns = _TTLNamespace(self)
            self.__dict__["_ttl_ns"] = ns
        return ns

    def tx(self) -> "_CollectionTransaction":
        """Single-collection transaction convenience (gap 2).

        Equivalent to opening a `db.transaction()` at the parent
        VectorDB, but the context manager yields this collection
        directly instead of a dict-of-collections proxy.
        """
        return _CollectionTransaction(self)

    @property
    def maintenance(self) -> "_MaintenanceNamespace":
        """Sub-namespace for periodic rebuild scheduling (gap 9)."""
        ns = self.__dict__.get("_maint_ns")
        if ns is None:
            ns = _MaintenanceNamespace(self)
            self.__dict__["_maint_ns"] = ns
        return ns

    @property
    def dim(self) -> int | None:
        """Vector dimension (None if no vectors added yet)."""
        return self._index.ndim

    def __repr__(self) -> str:
        # No SQL: debuggers and exception formatters auto-stringify objects,
        # and a repr that runs queries fails with ProgrammingError after
        # close(). dim reads only the in-memory usearch index.
        return (
            f"VectorCollection(name={self.name!r}, dim={self.dim}, "
            f"distance={self.distance_strategy.value})"
        )


class _PendingNamespace:
    """Buffered vector updates exposed as `collection.pending` (gaps 1 & 6).

    The buffer stores new vectors in a SQL table; HNSW only sees them
    after `flush()` promotes the batch in a single locked operation.
    Multiple updates to the same doc_id between flushes coalesce
    (last-write-wins).
    """

    __slots__ = ("_collection",)

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection

    def update(
        self,
        doc_id: int,
        vector: Sequence[float] | "np.ndarray",
        *,
        source: str | None = None,
    ) -> None:
        """Alias for `VectorCollection.update_embedding`."""
        self._collection.update_embedding(doc_id, vector, source=source)

    def update_many(
        self,
        updates: Sequence[tuple[int, Sequence[float]]],
        *,
        source: str | None = None,
    ) -> None:
        """Buffer many vector updates in one SQL transaction."""
        if not updates:
            return
        ndim = self._collection._index.ndim
        rows: list[tuple[int, bytes, str | None]] = []
        for doc_id, vec in updates:
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(
                    f"update_many: vector for {doc_id} must be 1-D, "
                    f"got shape {arr.shape}"
                )
            if ndim is not None and arr.shape[0] != ndim:
                raise ValueError(
                    f"update_many: vector dim {arr.shape[0]} for {doc_id} "
                    f"!= index dim {ndim}"
                )
            rows.append((int(doc_id), arr.tobytes(), source))
        self._collection._catalog.upsert_pending_vectors_many(rows)

    def blend_toward(
        self,
        doc_ids: Sequence[int],
        centroid: Sequence[float],
        alpha: float,
        *,
        source: str | None = "blend_toward",
    ) -> int:
        """Drift each doc's vector toward `centroid` by alpha.

        v' = (1 - alpha) * v + alpha * centroid

        Reads the current vector from the pending buffer first (so
        repeated blends compose) and falls back to the HNSW index. The
        blended vector is written back to the buffer; HNSW is untouched
        until flush.

        Args:
            doc_ids: Documents to drift.
            centroid: Target vector (same dim as the index).
            alpha: Mixing weight in [0, 1]. 0 keeps the original; 1
                replaces with the centroid entirely.
            source: Tag stored on the buffered rows.

        Returns:
            Number of vectors blended and re-buffered.
        """
        if not doc_ids:
            return 0
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        c = np.asarray(centroid, dtype=np.float32)
        if c.ndim != 1:
            raise ValueError(f"centroid must be 1-D, got shape {c.shape}")

        ids_arr = np.asarray(list(doc_ids), dtype=np.uint64)
        # Pull current vectors: pending wins, fallback to HNSW.
        cat = self._collection._catalog
        idx = self._collection._index
        rows: list[tuple[int, bytes, str | None]] = []
        # Bulk fetch from HNSW once; fall through to per-id pending overlay.
        idx_vecs = idx.get(ids_arr)
        for i, doc_id in enumerate(doc_ids):
            pending = cat.get_pending_vector(int(doc_id))
            if pending is not None:
                v = np.frombuffer(pending, dtype=np.float32)
            else:
                v = np.asarray(idx_vecs[i], dtype=np.float32)
            if v.shape[0] != c.shape[0]:
                raise ValueError(
                    f"blend_toward: vector dim {v.shape[0]} for {doc_id} "
                    f"!= centroid dim {c.shape[0]}"
                )
            blended = ((1.0 - alpha) * v + alpha * c).astype(np.float32)
            rows.append((int(doc_id), blended.tobytes(), source))
        cat.upsert_pending_vectors_many(rows)
        return len(rows)

    def size(self) -> int:
        """Number of buffered vector updates not yet flushed."""
        return self._collection._catalog.count_pending_vectors()

    def flush(self, *, max_batch: int | None = None) -> int:
        """Promote buffered vectors to the HNSW index.

        Reads up to `max_batch` rows in enqueue order, calls
        `UsearchIndex.add()` (which already does remove+add atomically
        under a single lock), then deletes the flushed pending rows in
        the same SQL transaction. Returns the count flushed.
        """
        cat = self._collection._catalog
        idx = self._collection._index
        rows = cat.list_pending_vectors(limit=max_batch)
        if not rows:
            return 0
        ids = np.asarray([r[0] for r in rows], dtype=np.uint64)
        # Reconstruct float32 matrix from BLOBs.
        ndim = idx.ndim
        if ndim is None:
            # Empty index — infer from first row.
            ndim = len(np.frombuffer(rows[0][1], dtype=np.float32))
        mat = np.frombuffer(
            b"".join(r[1] for r in rows), dtype=np.float32
        ).reshape(len(rows), ndim).copy()
        # add() takes the write lock and does remove+add per existing key.
        idx.add(ids, mat)
        cat.delete_pending_vectors([int(i) for i in ids])
        # Track flush count for the rebuild scheduler.
        try:
            self._collection.maintenance.record_flush(len(rows))
        except Exception:
            _logger.debug("flush counter record failed", exc_info=True)
        return len(rows)


class _TTLNamespace:
    """Sub-namespace exposed as `collection.ttl` (gap 8).

    Stores per-doc expiry timestamps. `sweep()` is the explicit cleanup
    pass; `start_background()` launches an opt-in daemon thread that
    calls `sweep()` on a fixed cadence. The thread is off by default —
    nothing wakes up unless the caller asks.
    """

    __slots__ = ("_collection", "_thread", "_stop_event")

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def set(
        self,
        doc_id: int,
        expires_at: float | None = None,
        *,
        seconds: float | None = None,
        on_expire: str = "delete",
    ) -> int:
        """Set a TTL by absolute timestamp or relative seconds.

        Either pass `expires_at` (unix seconds) or `seconds`
        (relative to now). Both is an error.
        """
        if expires_at is None and seconds is None:
            raise ValueError("Must pass expires_at or seconds")
        if expires_at is not None and seconds is not None:
            raise ValueError("Pass exactly one of expires_at / seconds")
        if seconds is not None:
            expires_at = time.time() + float(seconds)
        assert expires_at is not None
        return self._collection._catalog.set_ttl(
            doc_id, float(expires_at), on_expire=on_expire,
        )

    def clear(self, doc_id: int) -> int:
        """Remove a TTL entry. Returns 1 if removed, 0 otherwise."""
        return self._collection._catalog.clear_ttl(doc_id)

    def sweep(
        self,
        *,
        now: float | None = None,
        limit: int = 1000,
    ) -> tuple[list[int], list[int]]:
        """Apply expired entries; returns (deleted_ids, callback_ids).

        Also removes the doc from the in-memory HNSW index when an
        entry is deleted. Callers handling on_expire='callback' rows
        are responsible for any further action on those ids.
        """
        deleted, callback_ids = self._collection._catalog.sweep_ttl(
            now=now, limit=limit
        )
        if deleted:
            try:
                self._collection._index.remove(deleted)
            except Exception:
                _logger.debug(
                    "ttl.sweep: failed to remove %d ids from HNSW",
                    len(deleted), exc_info=True,
                )
        return deleted, callback_ids

    def start_background(
        self,
        *,
        interval: float = constants.TTL_SWEEP_DEFAULT_INTERVAL_S,
    ) -> None:
        """Launch a daemon thread that calls sweep() every `interval`s.

        Idempotent: a second call is a no-op. The thread runs until
        `stop_background()` is called or the process exits.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        stop_event = threading.Event()
        coll = self._collection

        def _loop() -> None:
            while not stop_event.is_set():
                try:
                    self.sweep()
                except Exception:
                    _logger.debug("ttl background sweep failed",
                                  exc_info=True)
                stop_event.wait(interval)

        thread = threading.Thread(
            target=_loop,
            name=f"simplevecdb-ttl-{coll.name}",
            daemon=True,
        )
        self._stop_event = stop_event
        self._thread = thread
        thread.start()

    def stop_background(self) -> None:
        """Stop the background sweeper. Idempotent."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._stop_event = None


class _EventsNamespace:
    """Append-only change feed exposed as `collection.events` (gap 7).

    Backed by the per-collection `_events` SQL table. SQLite WAL mode
    (already enabled at connection open) lets cross-process readers see
    committed rows immediately, so multi-process subscribers work
    without an external bus.
    """

    __slots__ = ("_collection",)

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection

    def append(
        self,
        kind: str,
        *,
        doc_id: int | None = None,
        payload: dict | None = None,
    ) -> int:
        """Append an event (caller-driven). Returns the assigned seq."""
        return self._collection._catalog.append_event(
            kind, doc_id=doc_id, payload=payload
        )

    def last_seq(self) -> int:
        """Highest seq currently in the feed (0 if empty)."""
        return self._collection._catalog.last_event_seq()

    def read(
        self,
        *,
        since: int = 0,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list["Event"]:
        """Read events with seq > since."""
        rows = self._collection._catalog.read_events(
            since=since, kind=kind, limit=limit
        )

        return [
            Event(seq=r[0], ts=r[1], kind=r[2], doc_id=r[3], payload=r[4])
            for r in rows
        ]

    def subscribe(
        self,
        *,
        since: int = 0,
        kind: str | None = None,
        poll_interval: float = constants.EVENTS_POLL_INTERVAL_S,
        batch: int = 500,
    ) -> "Iterator[Event]":
        """Generator yielding events as they appear. Caller controls exit.

        Uses simple polling on a background-friendly cadence; WAL mode
        means cross-process commits become visible to this reader on
        the next iteration without explicit synchronization.
        """
        last = int(since)
        while True:
            events = self.read(since=last, kind=kind, limit=batch)
            if events:
                for e in events:
                    yield e
                last = events[-1].seq
                if len(events) == batch:
                    # Drained a full batch; loop again immediately to keep up.
                    continue
            time.sleep(poll_interval)

    def prune(self, *, before_seq: int) -> int:
        """Drop events with seq < before_seq."""
        return self._collection._catalog.prune_events(before_seq=before_seq)


class _MaintenanceNamespace:
    """Threshold-driven rebuild scheduler (gap 9).

    Wraps `VectorCollection.rebuild_index()` with a heuristic gate so
    callers can opportunistically rebuild without rolling their own
    bookkeeping. The triggers compose with OR — any one being true
    fires a rebuild.

    Triggers (overridable per call):
      * max_pending  — total pending flushes since last rebuild
      * max_deleted  — tombstones in usearch (size mismatch with catalog)
      * max_age_s    — wall-clock seconds since last rebuild
    """

    __slots__ = ("_collection", "_pending_flushes", "_last_rebuild_ts")

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection
        self._pending_flushes = 0
        self._last_rebuild_ts = time.time()

    def record_flush(self, count: int = 1) -> None:
        """Bump the pending-flush counter (called by pending.flush())."""
        self._pending_flushes += int(count)

    def suggest_rebuild(
        self,
        *,
        max_pending: int = constants.REBUILD_PENDING_THRESHOLD,
        max_deleted: int = constants.REBUILD_TOMBSTONE_THRESHOLD,
        max_age_s: float = constants.REBUILD_MIN_INTERVAL_S,
    ) -> tuple[bool, str | None]:
        """Return (should_rebuild, reason)."""
        if self._pending_flushes >= max_pending:
            return True, f"pending_flushes={self._pending_flushes}"
        # Tombstones: usearch reports deletions as size shrinkage but
        # the index file may still be larger; approximate as
        # catalog_count vs index.size disagreement scaled by deletions.
        try:
            cat_count = self._collection._catalog.count()
            idx_size = self._collection._index.size
            tombstones = max(0, idx_size - cat_count)
            if tombstones >= max_deleted:
                return True, f"tombstones={tombstones}"
        except Exception:
            pass
        age = time.time() - self._last_rebuild_ts
        if age >= max_age_s and self._pending_flushes > 0:
            return True, f"age_s={age:.0f}"
        return False, None

    def rebuild_if_needed(
        self,
        *,
        max_pending: int = constants.REBUILD_PENDING_THRESHOLD,
        max_deleted: int = constants.REBUILD_TOMBSTONE_THRESHOLD,
        max_age_s: float = constants.REBUILD_MIN_INTERVAL_S,
    ) -> bool:
        """Run rebuild_index() iff a trigger fired. Returns True if it ran."""
        should, reason = self.suggest_rebuild(
            max_pending=max_pending,
            max_deleted=max_deleted,
            max_age_s=max_age_s,
        )
        if not should:
            return False
        _logger.info(
            "Rebuilding %s index (reason=%s)", self._collection.name, reason,
        )
        self._collection.rebuild_index()
        self._pending_flushes = 0
        self._last_rebuild_ts = time.time()
        try:
            self._collection.events.append(
                "rebuild", payload={"reason": reason}
            )
        except Exception:
            _logger.debug("rebuild event append failed", exc_info=True)
        return True


class _DBTransaction:
    """Context manager backing `VectorDB.transaction()` (gap 2).

    Acquires the DB-level RLock, opens a SAVEPOINT, and increments the
    shared transaction-depth counter so every catalog method this DB
    owns skips its per-call commit. On success the SAVEPOINT is
    released; on exception it's rolled back and the depth is reset.

    Yields a mapping-like object so callers can do
    `tx["collection_name"]` to operate on individual collections.
    """

    __slots__ = ("_db", "_savepoint_name", "_entered")

    def __init__(self, db: "VectorDB") -> None:
        self._db = db
        self._savepoint_name: str | None = None
        self._entered = False

    def __enter__(self) -> "_DBTransaction":
        self._db._lock.acquire()
        try:
            depth = self._db._tx_state.depth
            name = f"simplevecdb_tx_{depth + 1}"
            self._db.conn.execute(f"SAVEPOINT {name}")
            self._db._tx_state.depth = depth + 1
            self._savepoint_name = name
            self._entered = True
        except Exception:
            self._db._lock.release()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            name = self._savepoint_name
            assert name is not None
            try:
                if exc_type is None:
                    self._db.conn.execute(f"RELEASE SAVEPOINT {name}")
                else:
                    self._db.conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    self._db.conn.execute(f"RELEASE SAVEPOINT {name}")
            finally:
                self._db._tx_state.depth = max(
                    0, self._db._tx_state.depth - 1
                )
                # Outermost commit: if depth fell to 0, finalize the
                # implicit Python sqlite3 transaction so changes flush.
                if self._db._tx_state.depth == 0 and exc_type is None:
                    try:
                        self._db.conn.commit()
                    except Exception:
                        _logger.error(
                            "outer transaction commit failed", exc_info=True
                        )
                        raise
        finally:
            self._db._lock.release()

    def __getitem__(self, name: str) -> "VectorCollection":
        return self._db.collection(name)

    def collection(self, name: str) -> "VectorCollection":
        return self._db.collection(name)


class _CollectionTransaction(_DBTransaction):
    """Single-collection wrapper that yields the collection directly."""

    __slots__ = ("_collection",)

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection
        # Reuse the shared tx_state and lock from the collection.
        # _DBTransaction expects ._db; we create a shim.
        super().__init__(_CollectionTxShim(collection))  # type: ignore[arg-type]

    def __enter__(self) -> "VectorCollection":  # type: ignore[override]
        super().__enter__()
        return self._collection


class _CollectionTxShim:
    """Minimal proxy emulating the VectorDB attributes _DBTransaction reads."""

    __slots__ = ("_lock", "_tx_state", "conn")

    def __init__(self, collection: "VectorCollection") -> None:
        self._lock = collection._lock
        self._tx_state = collection._tx_state
        self.conn = collection.conn


class _EdgesNamespace:
    """Sub-namespace exposed as `collection.edges` (gap 3).

    Provides the four canonical primitives — `add_edge`, `get_edges`,
    `update_edge`, `delete_edge` — plus `upsert` and `prune` convenience.
    Numeric attributes (weight, bonus, hits, last_touch) are real
    columns and are addressable through the range-filter grammar
    (`filter={"weight": {"$lt": 0.1}}`).
    """

    __slots__ = ("_collection",)

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection

    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        *,
        kind: str = "",
        weight: float = 0.0,
        bonus: float = 0.0,
        hits: int = 0,
        metadata: dict | None = None,
    ) -> int:
        """Insert a new edge. Use upsert() if collisions are expected."""
        return self._collection._catalog.add_edge(
            src_id, dst_id, kind=kind, weight=weight, bonus=bonus,
            hits=hits, metadata=metadata,
        )

    def upsert(
        self,
        src_id: int,
        dst_id: int,
        *,
        kind: str = "",
        weight: float | None = None,
        bonus: float | None = None,
        hits: int | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Create-or-update; preserves existing fields where args are None."""
        return self._collection._catalog.upsert_edge(
            src_id, dst_id, kind=kind, weight=weight, bonus=bonus,
            hits=hits, metadata=metadata,
        )

    def update_edge(
        self,
        src_id: int,
        dst_id: int,
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
        """Set absolutes and/or apply atomic deltas. See catalog.update_edge."""
        return self._collection._catalog.update_edge(
            src_id, dst_id, kind=kind,
            weight=weight, bonus=bonus, hits=hits, metadata=metadata,
            dweight=dweight, dbonus=dbonus, dhits=dhits,
        )

    def delete_edge(
        self, src_id: int, dst_id: int, *, kind: str = ""
    ) -> int:
        """Drop a single edge by (src, dst, kind)."""
        return self._collection._catalog.delete_edge(
            src_id, dst_id, kind=kind
        )

    def get_edges(
        self,
        src: int | None = None,
        dst: int | None = None,
        *,
        kind: str | None = None,
        filter: dict | None = None,
        limit: int | None = None,
    ) -> list["Edge"]:
        """Read edges. `src`/`dst` constrain to outgoing/incoming/specific.

        `filter` accepts the same grammar as similarity_search; numeric
        keys (weight/bonus/hits/last_touch) map to direct column
        comparisons, anything else queries the JSON metadata column.
        """
        rows = self._collection._catalog.get_edges(
            src_id=src, dst_id=dst, kind=kind, filter=filter, limit=limit,
        )

        return [
            Edge(
                src_id=r[0], dst_id=r[1], kind=r[2], weight=r[3],
                hits=r[4], bonus=r[5], last_touch=r[6], metadata=r[7],
            )
            for r in rows
        ]

    def prune(
        self,
        *,
        kind: str | None = None,
        max_weight: float | None = None,
        idle_before: float | None = None,
    ) -> int:
        """Bulk-delete edges by weight ceiling and/or age cutoff."""
        return self._collection._catalog.prune_edges(
            kind=kind, max_weight=max_weight, idle_before=idle_before,
        )


class _CountersNamespace:
    """Sub-namespace exposed as `collection.counters` (gap 4)."""

    __slots__ = ("_collection",)

    def __init__(self, collection: "VectorCollection") -> None:
        self._collection = collection

    def increment(
        self, doc_id: int, deltas: dict[str, float | int]
    ) -> int:
        """Alias for `VectorCollection.increment_metadata`."""
        return self._collection._catalog.increment_metadata(doc_id, deltas)

    def increment_many(
        self, updates: list[tuple[int, dict[str, float | int]]]
    ) -> int:
        """Apply many counter increments in one transaction."""
        return self._collection._catalog.increment_metadata_many(updates)

    def get(
        self, doc_id: int, key: str, default: float | int = 0
    ) -> float | int | None:
        """Read a single numeric counter value (None if row missing)."""
        return self._collection._catalog.get_metadata_counter(
            doc_id, key, default
        )


class VectorDB:
    """
    Dead-simple local vector database powered by usearch HNSW.

    SQLite stores metadata and text; usearch stores vectors in separate
    .usearch files per collection. Provides Chroma-like API with built-in
    quantization for storage efficiency.

    Storage layout:
    - {path} - SQLite database (metadata, text, FTS)
    - {path}.{collection}.usearch - usearch HNSW index per collection

    Encryption (optional):
    - SQLite encrypted via SQLCipher (transparent page-level AES-256)
    - Index files encrypted via AES-256-GCM (at-rest only, zero runtime overhead)
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        distance_strategy: DistanceStrategy = DistanceStrategy(constants.DEFAULT_DISTANCE_STRATEGY),
        quantization: Quantization = Quantization(constants.DEFAULT_QUANTIZATION),
        *,
        encryption_key: str | bytes | None = None,
    ):
        """Initialize the vector database.

        Args:
            path: Database file path or ":memory:" for in-memory database.
            distance_strategy: Default distance metric for similarity search.
            quantization: Default vector compression strategy.
            encryption_key: Optional passphrase or 32-byte key for at-rest encryption.
                Encrypts both SQLite (via SQLCipher) and usearch index files (via AES-256-GCM).

        Raises:
            EncryptionUnavailableError: If encryption_key provided but encryption
                dependencies are missing.
            EncryptionError: If encrypted database cannot be opened (wrong key).
            ValueError: If encryption_key used with ":memory:" database.
        """
        self.path = str(path)
        self.distance_strategy = distance_strategy
        self.quantization = quantization
        self._encryption_key = encryption_key
        self._collections: dict[tuple, VectorCollection] = {}
        # Single RLock serializing both the _collections cache (avoid
        # check-then-insert TOCTOU) and the shared sqlite3.Connection's
        # Python-level transaction context. Shared with every VectorCollection
        # and CatalogManager constructed by this VectorDB.
        self._lock = threading.RLock()
        # Shared transaction-depth counter. Bumped by VectorDB.transaction()
        # so all catalogs in this DB suspend per-call commits.
        self._tx_state: _TxState = _TxState()

        # Create connection (encrypted or plain)
        if encryption_key is not None:
            if self.path == ":memory:":
                raise ValueError(
                    "In-memory databases cannot be encrypted. "
                    "Use a file path for encrypted databases."
                )
            self.conn = create_encrypted_connection(
                self.path,
                encryption_key,
                check_same_thread=False,
                timeout=30.0,
            )
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            # Native lock-wait window so SQLite blocks the caller in C
            # rather than surfacing 'database is locked' immediately
            # under multi-writer load (gap 10).
            self.conn.execute(
                f"PRAGMA busy_timeout={constants.SQLITE_BUSY_TIMEOUT_MS}"
            )
            self.conn.execute("PRAGMA foreign_keys=ON")
            self._encrypted = True
            _logger.info("Opened encrypted database: %s", self.path)
        else:
            self.conn = sqlite3.connect(
                self.path, check_same_thread=False, timeout=30.0
            )
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute(
                f"PRAGMA busy_timeout={constants.SQLITE_BUSY_TIMEOUT_MS}"
            )
            self.conn.execute("PRAGMA foreign_keys=ON")
            self._encrypted = False

        # Verify connection is healthy
        try:
            self.conn.execute("SELECT 1")
        except sqlite3.DatabaseError as e:
            self.conn.close()
            raise RuntimeError(f"Database health check failed: {e}") from e


    def transaction(self) -> "_DBTransaction":
        """Atomic write context spanning all collections (gap 2).

        Wraps the work in a single SQLite SAVEPOINT and bumps the shared
        transaction-depth counter so every catalog method skips its
        per-call commit. SQL-side mutations (metadata, counters, edges,
        events, TTL, and `update_embedding`'s pending-vector overlay) are
        atomic: a raised exception triggers ROLLBACK TO SAVEPOINT and all
        SQL writes are reverted.

        The HNSW index is not rolled back. Coarse vector mutations
        (`add_texts`, `delete`, etc.) call into usearch directly and
        their effects persist even if the surrounding SAVEPOINT rolls
        back. Use `update_embedding` + `pending.flush()` for vector
        changes whose visibility you want gated on commit.

        Example:
            >>> with db.transaction() as tx:
            ...     tx["docs"].update_embedding(id, vec)
            ...     tx["docs"].increment_metadata(id, {"hits": 1})
            ...     tx["docs"].edges.add_edge(src, dst, weight=0.7)
        """
        return _DBTransaction(self)

    def list_collections(self) -> list[str]:
        """
        Return names of all persisted collections in the database.

        Scans the database schema for collection tables, returning both
        collections accessed this session and those created in previous sessions.

        Returns:
            Sorted list of collection names stored in this database.

        Example:
            >>> db = VectorDB("app.db")
            >>> db.collection("users")
            >>> db.close()
            >>> db2 = VectorDB("app.db")
            >>> db2.list_collections()
            ['users']
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND (name = 'tinyvec_items' OR name LIKE 'items_%')"
            ).fetchall()
        # Collect all table names, then filter out FTS/cluster derivatives.
        # FTS5 creates shadow tables: items_<name>_fts, items_<name>_fts_data,
        # items_<name>_fts_idx, items_<name>_fts_content, items_<name>_fts_docsize,
        # items_<name>_fts_config.  Cluster tables: items_<name>_clusters.
        # We identify derivatives by checking if a suffix is <coll>_fts*
        # or <coll>_clusters for some other known collection suffix.
        all_suffixes: set[str] = set()
        has_default = False
        for (table_name,) in rows:
            if table_name == "tinyvec_items":
                has_default = True
            elif table_name.startswith("items_"):
                all_suffixes.add(table_name[6:])

        # A suffix is a real collection if no other suffix is a prefix of it
        # followed by an auxiliary suffix. 2.6.1 added _pending_vectors,
        # _edges, _events, _ttl alongside the existing FTS / cluster ones.
        _fts_suffixes = ("_fts", "_fts_data", "_fts_idx", "_fts_content",
                         "_fts_docsize", "_fts_config")
        _aux_suffixes = ("_clusters", "_pending_vectors", "_edges", "_events",
                         "_ttl")
        derivative_suffixes: set[str] = set()
        for s in all_suffixes:
            for fts in _fts_suffixes:
                derivative_suffixes.add(f"{s}{fts}")
            for aux in _aux_suffixes:
                derivative_suffixes.add(f"{s}{aux}")

        names: list[str] = []
        if has_default:
            names.append("default")
        for s in sorted(all_suffixes - derivative_suffixes):
            names.append(s)
        return names

    def delete_collection(self, name: str) -> None:
        """
        Delete a collection and all its data.

        Drops the SQLite tables (items, FTS, clusters) and deletes
        the usearch index file. Removes the collection from the cache.

        Args:
            name: Collection name to delete.

        Raises:
            ValueError: If the collection name is invalid.
            KeyError: If the collection does not exist.
        """
        if not re.match(constants.COLLECTION_NAME_PATTERN, name):
            raise ValueError(
                f"Invalid collection name '{name}'. Must be alphanumeric + underscores."
            )

        table_name = "tinyvec_items" if name == "default" else f"items_{name}"
        fts_table = f"{table_name}_fts"
        cluster_table = f"{table_name}_clusters"
        pending_table = f"{table_name}_pending_vectors"
        edges_table = f"{table_name}_edges"
        events_table = f"{table_name}_events"
        ttl_table = f"{table_name}_ttl"

        # Hold the lock for the full delete: drop tables, remove files, and
        # evict cached collections atomically. The existence check runs
        # *inside* the lock to close the TOCTOU between checking and the
        # actual DROP — two concurrent delete_collection calls would
        # otherwise both pass the check and the loser would surface a
        # SQLite error instead of the documented KeyError.
        with self._lock:
            # Re-entrant: list_collections() also takes self._lock. The
            # check + drop must run inside the same lock acquisition so a
            # second concurrent delete_collection cannot pass the
            # existence check after our DROP runs.
            if name not in self.list_collections():
                raise KeyError(f"Collection '{name}' does not exist.")
            # Close any cached collection's open index before removing the file
            for cached_key, cached_col in list(self._collections.items()):
                if cached_key[0] == name:
                    try:
                        cached_col._index.close()
                    except Exception:
                        _logger.debug(
                            "Failed to close index for collection %r during delete",
                            name,
                            exc_info=True,
                        )

            # Drop SQLite tables. Auxiliary tables (gap 2.6.1) reference
            # the main table via FK ON DELETE CASCADE, so dropping the
            # main table cleans the children too — but DROP TABLE doesn't
            # cascade in SQLite, so drop them explicitly first.
            with self.conn:
                self.conn.execute(f"DROP TABLE IF EXISTS {fts_table}")
                self.conn.execute(f"DROP TABLE IF EXISTS {cluster_table}")
                self.conn.execute(f"DROP TABLE IF EXISTS {pending_table}")
                self.conn.execute(f"DROP TABLE IF EXISTS {edges_table}")
                self.conn.execute(f"DROP TABLE IF EXISTS {events_table}")
                self.conn.execute(f"DROP TABLE IF EXISTS {ttl_table}")
                self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Delete usearch index file (and encrypted variant if present),
            # plus any per-DB salt sidecars left behind by the encryption
            # layer (".enc.salt").
            if self.path != ":memory:":
                index_path = Path(self.path + f".{name}.usearch")
                encrypted_path = Path(str(index_path) + ".enc")
                salt_path = Path(str(encrypted_path) + ".salt")
                for p in (index_path, encrypted_path, salt_path):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        _logger.debug(
                            "Could not unlink %s during delete_collection", p,
                            exc_info=True,
                        )

            # Remove from cache (match any tuple key with this name)
            keys_to_remove = [k for k in self._collections if k[0] == name]
            for k in keys_to_remove:
                del self._collections[k]

        _logger.info("Deleted collection: %s", name)

    def search_collections(
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

        Performs similarity search on each collection and merges results using
        score normalization for fair comparison across distance metrics.

        Args:
            query: Query vector (must match dimension of all searched collections).
            collections: List of collection names to search. None searches all
                initialized collections (from list_collections()).
            k: Number of top results to return after merging.
            filter: Optional metadata filter applied to all collections.
            normalize_scores: If True, convert distances to similarity scores
                in [0, 1] range using `1 / (1 + distance)`. Enables fair
                comparison across COSINE [0,2] and L2 [0,∞) metrics.
            parallel: If True, search collections concurrently using ThreadPoolExecutor.

        Returns:
            List of (Document, similarity_score, collection_name) tuples,
            sorted by descending similarity score (highest first).

        Raises:
            ValueError: If no collections specified and none initialized,
                or if collections have mismatched dimensions.
            KeyError: If a specified collection name doesn't exist.

        Example:
            >>> db = VectorDB("app.db")
            >>> db.collection("users").add_texts(["alice"], embeddings=[[0.1]*384])
            >>> db.collection("products").add_texts(["widget"], embeddings=[[0.2]*384])
            >>> results = db.search_collections([0.15]*384, k=2)
            >>> for doc, score, coll in results:
            ...     print(f"{coll}: {doc.page_content} ({score:.3f})")
        """
        target_names = (
            collections if collections is not None else self.list_collections()
        )

        if not target_names:
            return []

        # Resolve and validate collections
        targets: list[VectorCollection] = []
        dims: set[int | None] = set()
        # Validate explicit collection names exist in DB
        if collections is not None:
            persisted = set(self.list_collections())
            for name in target_names:
                if name not in persisted:
                    # Check cache too (collection may exist but not yet persisted)
                    if not any(k[0] == name for k in self._collections):
                        raise KeyError(
                            f"Collection '{name}' not initialized. "
                            f"Call db.collection('{name}') first."
                        )

        for name in target_names:
            # Find cached collection by name (may have any strategy/quantization)
            matched = [v for k, v in self._collections.items() if k[0] == name]
            if matched:
                coll = matched[0]
            else:
                # Auto-initialize with defaults for persisted but uncached collections
                coll = self.collection(name)
            targets.append(coll)
            dims.add(coll.dim)

        # Check dimension consistency (ignore None for empty collections)
        dims.discard(None)
        if len(dims) > 1:
            raise ValueError(
                f"Dimension mismatch across collections: {dims}. "
                "All searched collections must have the same embedding dimension."
            )

        # Search function for each collection
        def _search_one(coll: VectorCollection) -> list[tuple[Document, float, str]]:
            results = coll.similarity_search(query, k=k, filter=filter)
            return [(doc, dist, coll.name) for doc, dist in results]

        # Execute searches
        all_results: list[tuple[Document, float, str]] = []
        if parallel and len(targets) > 1:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

            with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as executor:
                futures = [executor.submit(_search_one, coll) for coll in targets]
                for future in futures:
                    try:
                        all_results.extend(
                            future.result(timeout=constants.SEARCH_COLLECTION_TIMEOUT)
                        )
                    except FuturesTimeoutError:
                        future.cancel()
                        _logger.error(
                            "search_collections: collection search timed out after %.0fs",
                            constants.SEARCH_COLLECTION_TIMEOUT,
                        )
                    except Exception:
                        _logger.warning(
                            "search_collections: one collection search failed",
                            exc_info=True,
                        )
        else:
            for coll in targets:
                all_results.extend(_search_one(coll))

        # Normalize scores: similarity = 1 / (1 + distance)
        if normalize_scores:
            all_results = [
                (doc, 1.0 / (1.0 + dist), name) for doc, dist, name in all_results
            ]
        else:
            # Invert for sorting (lower distance = higher rank)
            all_results = [(doc, -dist, name) for doc, dist, name in all_results]

        # Sort by score descending and take top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:k]

    def collection(
        self,
        name: str = "default",
        distance_strategy: DistanceStrategy | None = None,
        quantization: Quantization | None = None,
        store_embeddings: bool = False,
    ) -> VectorCollection:
        """
        Get or create a named collection.

        Collections provide isolated namespaces within a single database.
        Each collection has its own usearch index file.

        Args:
            name: Collection name (alphanumeric + underscore only).
            distance_strategy: Override database-level distance metric.
            quantization: Override database-level quantization.
            store_embeddings: If True, store embeddings as BLOBs in SQLite
                alongside the usearch index. Required for rebuild_index().
                Default False to save ~2x storage.

        Returns:
            VectorCollection instance.

        Raises:
            ValueError: If collection name contains invalid characters.
        """
        cache_key = (name, distance_strategy, quantization, store_embeddings)
        with self._lock:
            if cache_key not in self._collections:
                self._collections[cache_key] = VectorCollection(
                    conn=self.conn,
                    db_path=self.path,
                    name=name,
                    distance_strategy=distance_strategy or self.distance_strategy,
                    quantization=quantization or self.quantization,
                    encryption_key=self._encryption_key,
                    store_embeddings=store_embeddings,
                    lock=self._lock,
                    tx_state=self._tx_state,
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

    def __repr__(self) -> str:
        # Avoid hitting SQL on every repr() call — debuggers and exception
        # formatters that auto-stringify objects shouldn't trigger I/O.
        return f"VectorDB(path={self.path!r})"

    def close(self) -> None:
        """Close the database connection and save indexes."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.save()
        except Exception:
            _logger.warning("Failed to save indexes during close", exc_info=True)
        finally:
            self.conn.close()
            # Clean up ephemeral usearch index files created for in-memory DBs.
            for col in self._collections.values():
                ephemeral = getattr(col, "_ephemeral_index_path", None)
                if ephemeral:
                    for p in (Path(ephemeral), Path(ephemeral + ".tmp"),
                              Path(ephemeral + ".lock")):
                        try:
                            p.unlink()
                        except FileNotFoundError:
                            pass
                        except OSError:
                            pass

    def __enter__(self) -> "VectorDB":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
