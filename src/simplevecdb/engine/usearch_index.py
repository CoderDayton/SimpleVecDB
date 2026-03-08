"""
Usearch HNSW index wrapper for SimpleVecDB.

Provides a thread-safe wrapper around usearch.Index with:
- Automatic persistence (save on close, load on open)
- Distance strategy mapping to usearch MetricKind
- Quantization support (F32, F16, I8)
- Thread-safe writes with lock, lock-free reads
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import DistanceStrategy, Quantization

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger("simplevecdb.engine.usearch_index")

# Default HNSW parameters (tuned for recall/speed balance)
DEFAULT_CONNECTIVITY = 16  # M parameter - edges per node
DEFAULT_EXPANSION_ADD = 128  # efConstruction - build quality
DEFAULT_EXPANSION_SEARCH = 64  # ef - search quality


def _get_metric_kind(distance_strategy: DistanceStrategy) -> Any:
    """Map SimpleVecDB distance strategy to usearch MetricKind."""
    from usearch.index import MetricKind

    mapping = {
        DistanceStrategy.COSINE: MetricKind.Cos,
        DistanceStrategy.L2: MetricKind.L2sq,  # usearch uses squared L2
    }
    return mapping.get(distance_strategy, MetricKind.Cos)


def _get_scalar_kind(quantization: Quantization) -> Any:
    """Map SimpleVecDB quantization to usearch ScalarKind."""
    from usearch.index import ScalarKind

    mapping = {
        Quantization.FLOAT: ScalarKind.F32,
        Quantization.FLOAT16: ScalarKind.F16,
        Quantization.INT8: ScalarKind.I8,
        Quantization.BIT: ScalarKind.B1,
    }
    return mapping.get(quantization, ScalarKind.F32)


def _pack_bits(vectors: np.ndarray) -> np.ndarray:
    """Pack float vectors into binary representation (uint8).

    Converts sign bits to packed bytes for usearch B1 format.
    Positive values -> 1, negative/zero -> 0.
    """
    # Convert to binary: positive = 1, else 0
    binary = (vectors > 0).astype(np.uint8)

    # Pad to multiple of 8 for byte packing
    n, dim = binary.shape
    padded_dim = ((dim + 7) // 8) * 8
    if padded_dim > dim:
        padding = np.zeros((n, padded_dim - dim), dtype=np.uint8)
        binary = np.hstack([binary, padding])

    # Pack bits into bytes (8 bits per byte)
    packed = np.packbits(binary, axis=1)
    return packed


def _unpack_bits(packed: np.ndarray, ndim: int) -> np.ndarray:
    """Unpack binary vectors back to float representation."""
    unpacked = np.unpackbits(packed, axis=1)[:, :ndim]
    # Convert 0/1 to -1/+1 for distance computation
    return unpacked.astype(np.float32) * 2 - 1


class UsearchIndex:
    """
    Thread-safe wrapper around usearch.Index.

    Handles index creation, persistence, and provides a clean interface
    for add/search/remove operations. Writes are serialized with a lock;
    reads are lock-free (usearch is thread-safe for concurrent reads).

    Args:
        index_path: Path to the .usearch index file
        ndim: Vector dimension (required for new index, inferred on load)
        distance_strategy: Distance metric for similarity
        quantization: Vector storage precision
        connectivity: HNSW M parameter (edges per node)
        expansion_add: efConstruction (higher = better recall, slower build)
        expansion_search: ef (higher = better recall, slower search)
    """

    def __init__(
        self,
        index_path: str | Path,
        ndim: int | None = None,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        quantization: Quantization = Quantization.FLOAT,
        connectivity: int = DEFAULT_CONNECTIVITY,
        expansion_add: int = DEFAULT_EXPANSION_ADD,
        expansion_search: int = DEFAULT_EXPANSION_SEARCH,
    ):
        self._path = Path(index_path)
        self._ndim = ndim
        self._distance_strategy = distance_strategy
        self._quantization = quantization
        self._connectivity = connectivity
        self._expansion_add = expansion_add
        self._expansion_search = expansion_search
        self._write_lock = threading.Lock()
        self._dirty = False  # Track if index needs saving
        self._is_view = False  # Track if using memory-mapped view

        self._index: Any = None  # usearch.Index, typed as Any for lazy import
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing index or create new one.

        Automatically uses memory-mapping (view) for large indexes to reduce
        memory footprint and enable instant startup.
        """
        from usearch.index import Index
        from .. import constants

        if self._path.exists():
            # Check file size to decide load vs view
            file_size = self._path.stat().st_size
            # Estimate vector count: file_size / (ndim * dtype_size + overhead)
            # Conservative estimate assuming f32 and ~50 bytes overhead per vector
            estimated_vectors = file_size // 100  # Very rough estimate

            if estimated_vectors > constants.USEARCH_MMAP_THRESHOLD:
                # Use memory-mapped view for large indexes
                _logger.debug(
                    "Using memory-mapped view for large index: %s", self._path
                )
                self._index = Index.restore(str(self._path), view=True)
                self._is_view = True
            else:
                # Load into memory for smaller indexes
                _logger.debug("Loading index into memory: %s", self._path)
                self._index = Index.restore(str(self._path), view=False)
                self._is_view = False

            self._ndim = self._index.ndim
            _logger.info(
                "Loaded index: %d vectors, dim=%d, mmap=%s",
                len(self._index),
                self._ndim,
                self._is_view,
            )
        elif self._ndim is not None:
            self._create_index()
        # else: index will be created lazily on first add when we know the dimension

    def _create_index(self) -> None:
        """Create a new usearch index with configured parameters."""
        from usearch.index import Index, MetricKind

        if self._ndim is None:
            raise ValueError("Cannot create index without dimension")

        # BIT quantization requires Hamming metric
        if self._quantization == Quantization.BIT:
            metric = MetricKind.Hamming
        else:
            metric = _get_metric_kind(self._distance_strategy)

        dtype = _get_scalar_kind(self._quantization)

        _logger.debug(
            "Creating new index: dim=%d, metric=%s, dtype=%s",
            self._ndim,
            metric,
            dtype,
        )

        self._index = Index(
            ndim=self._ndim,
            metric=metric,
            dtype=dtype,
            connectivity=self._connectivity,
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )

    @property
    def ndim(self) -> int | None:
        """Vector dimension (None if index not yet created)."""
        return self._ndim

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self._index is None:
            return 0
        return len(self._index)

    @property
    def is_memory_mapped(self) -> bool:
        """Whether index is using memory-mapped (view) mode.

        Large indexes (>100k vectors) automatically use memory-mapping
        for lower memory footprint and instant startup.
        """
        return self._is_view

    def add(
        self,
        keys: NDArray[np.uint64],
        vectors: NDArray[np.float32],
        threads: int = 0,
    ) -> None:
        """
        Add vectors to the index.

        Supports upsert: if a key already exists, it will be removed first.

        Args:
            keys: Array of uint64 keys (typically SQLite rowids)
            vectors: Array of float32 vectors, shape (n, ndim)
            threads: Number of threads for parallel insertion (0=auto)

        Raises:
            ValueError: If vector dimension doesn't match index dimension
        """
        if len(keys) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        keys = np.asarray(keys, dtype=np.uint64)

        # Lazy index creation on first add
        with self._write_lock:
            # If currently in view mode, need to reload as writable
            if self._is_view and self._index is not None:
                _logger.debug("Upgrading from view to writable mode for add operation")
                from usearch.index import Index

                self._index = Index.restore(str(self._path), view=False)
                self._is_view = False

            if self._index is None:
                self._ndim = vectors.shape[1]
                self._create_index()

            if vectors.shape[1] != self._ndim:
                raise ValueError(
                    f"Vector dimension {vectors.shape[1]} != index dimension {self._ndim}"
                )

            # Preprocess based on quantization/distance
            if self._quantization == Quantization.BIT:
                # Pack float vectors into binary for Hamming distance
                vectors = _pack_bits(vectors)
            elif self._distance_strategy == DistanceStrategy.COSINE:
                # Normalize for cosine similarity
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / np.maximum(norms, 1e-12)

            # Upsert: remove existing keys first (usearch doesn't allow duplicates)
            for key in keys:
                if int(key) in self._index:
                    self._index.remove(int(key))

            self._index.add(keys, vectors, threads=threads)
            self._dirty = True

            _logger.debug("Added %d vectors to index (threads=%d)", len(keys), threads)

    def search(
        self,
        query: NDArray[np.float32],
        k: int,
        expansion_search: int | None = None,
        exact: bool | None = None,
        threads: int = 0,
    ) -> tuple[NDArray[np.uint64], NDArray[np.float32]]:
        """
        Search for k nearest neighbors.

        Uses adaptive search strategy: brute-force for small indexes (faster),
        HNSW approximate search for large indexes (scales better).

        Args:
            query: Query vector(s), shape (ndim,) or (n, ndim)
            k: Number of neighbors to return
            expansion_search: Override default ef parameter for this search
            exact: Force exact (brute-force) search. If None, auto-selects
                   based on index size (brute-force if < 10k vectors).
            threads: Number of threads for parallel search (0=auto)

        Returns:
            Tuple of (keys, distances) arrays. For cosine, distance is in [0, 2].
            Lower distance = more similar.
        """
        from .. import constants

        if self._index is None or self.size == 0:
            # Return empty results for empty index
            empty_keys = np.array([], dtype=np.uint64)
            empty_dists = np.array([], dtype=np.float32)
            return empty_keys, empty_dists

        query = np.asarray(query, dtype=np.float32)

        # Preprocess query based on quantization/distance
        if self._quantization == Quantization.BIT:
            # Pack query to binary for Hamming distance
            if query.ndim == 1:
                query = query.reshape(1, -1)
            query = _pack_bits(query)
        elif self._distance_strategy == DistanceStrategy.COSINE:
            # Normalize query for cosine
            if query.ndim == 1:
                norm = float(np.linalg.norm(query))
                query = query / max(norm, 1e-12)
            else:
                norms = np.linalg.norm(query, axis=1, keepdims=True)
                query = query / np.maximum(norms, 1e-12)

        # Adaptive search: brute-force for small indexes, HNSW for large
        if exact is None:
            use_exact = self.size < constants.USEARCH_BRUTEFORCE_THRESHOLD
        else:
            use_exact = exact

        if use_exact:
            _logger.debug(
                "Using brute-force search (index size %d < threshold %d)",
                self.size,
                constants.USEARCH_BRUTEFORCE_THRESHOLD,
            )

        # usearch search is thread-safe for reads
        matches = self._index.search(query, k, exact=use_exact, threads=threads)

        # Handle single query vs batch
        keys = np.asarray(matches.keys, dtype=np.uint64)
        distances = np.asarray(matches.distances, dtype=np.float32)

        return keys, distances

    def remove(self, keys: NDArray[np.uint64] | list[int]) -> int:
        """
        Remove vectors by their keys.

        Note: usearch HNSW doesn't support true deletion efficiently.
        Keys are marked as deleted but space isn't reclaimed until rebuild.
        For heavy delete workloads, consider periodic rebuild().

        Args:
            keys: Keys to remove

        Returns:
            Number of keys actually removed
        """
        if self._index is None:
            return 0

        keys = np.asarray(keys, dtype=np.uint64)
        if len(keys) == 0:
            return 0

        with self._write_lock:
            removed = 0
            for key in keys:
                try:
                    self._index.remove(int(key))
                    removed += 1
                except KeyError:
                    pass  # Key not in index
            self._dirty = True
            _logger.debug("Removed %d vectors from index", removed)
            return removed

    def contains(self, key: int) -> bool:
        """Check if a key exists in the index."""
        if self._index is None:
            return False
        return key in self._index

    def save(self) -> None:
        """Save index to disk if modified."""
        if self._index is None or not self._dirty:
            return

        with self._write_lock:
            # Ensure parent directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save(str(self._path))
            self._dirty = False
            _logger.debug("Saved index to %s", self._path)

    def close(self) -> None:
        """Save and close the index."""
        self.save()
        self._index = None

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: int) -> bool:
        return self.contains(key)

    def keys(self) -> list[int]:
        """Return all keys in the index."""
        if self._index is None:
            return []
        return [int(k) for k in self._index.keys]

    def get(self, keys: NDArray[np.uint64]) -> NDArray[np.float32]:
        """
        Retrieve vectors by their keys.

        Args:
            keys: Array of keys to retrieve

        Returns:
            Array of vectors, shape (len(keys), ndim). Missing keys return zeros.
        """
        if self._index is None or len(keys) == 0:
            return np.array([], dtype=np.float32).reshape(0, self._ndim or 1)

        keys = np.asarray(keys, dtype=np.uint64)
        vectors = []
        missing_logged = False
        for key in keys:
            try:
                vec = self._index.get(int(key))
                vectors.append(np.asarray(vec, dtype=np.float32))
            except KeyError:
                if not missing_logged:
                    _logger.warning(
                        "Index missing key(s); returning zero vectors. "
                        "Index may be out of sync with catalog."
                    )
                    missing_logged = True
                vectors.append(np.zeros(self._ndim or 1, dtype=np.float32))

        return (
            np.stack(vectors)
            if vectors
            else np.array([], dtype=np.float32).reshape(0, self._ndim or 1)
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
