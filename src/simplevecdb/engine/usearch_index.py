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
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import DistanceStrategy, Quantization
from .. import constants

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger("simplevecdb.engine.usearch_index")


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
        connectivity: int = constants.USEARCH_DEFAULT_CONNECTIVITY,
        expansion_add: int = constants.USEARCH_DEFAULT_EXPANSION_ADD,
        expansion_search: int = constants.USEARCH_DEFAULT_EXPANSION_SEARCH,
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
            from ..utils import file_lock

            with file_lock(self._path):
                # Use file size threshold for mmap decision
                file_size = self._path.stat().st_size

                if file_size > constants.USEARCH_MMAP_THRESHOLD:
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

            # Upsert: remove existing keys first (usearch doesn't allow duplicates).
            # Use batch remove for efficiency.
            if self.size > 0:
                existing_mask = np.array(
                    [int(k) in self._index for k in keys], dtype=bool
                )
                if existing_mask.any():
                    self._index.remove(keys[existing_mask])

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
            # Filter to only keys that exist in the index
            existing_mask = np.array(
                [int(k) in self._index for k in keys], dtype=bool
            )
            existing_keys = keys[existing_mask]
            if len(existing_keys) > 0:
                self._index.remove(existing_keys)
            removed = int(existing_mask.sum())
            self._dirty = True
            _logger.debug("Removed %d vectors from index", removed)
            return removed

    def contains(self, key: int) -> bool:
        """Check if a key exists in the index."""
        if self._index is None:
            return False
        return key in self._index

    def save(self) -> None:
        """Save index to disk atomically if modified.

        Writes to a sibling temp file, fsyncs it, then os.replace()s it onto
        the target path. A crash mid-save leaves either the previous good
        file or the temp file (which can be safely removed); the target
        path is never torn.
        """
        from ..utils import file_lock

        with self._write_lock:
            if self._index is None or not self._dirty:
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")

            with file_lock(self._path):
                try:
                    self._index.save(str(tmp_path))
                    # fsync the temp file so the data hits disk before the rename
                    try:
                        fd = os.open(str(tmp_path), os.O_RDONLY)
                        try:
                            os.fsync(fd)
                        finally:
                            os.close(fd)
                    except OSError as exc:
                        _logger.warning("fsync on %s failed: %s", tmp_path, exc)
                    os.replace(str(tmp_path), str(self._path))
                    # fsync the directory so the rename itself is durable
                    try:
                        dir_fd = os.open(str(self._path.parent), os.O_RDONLY)
                        try:
                            os.fsync(dir_fd)
                        finally:
                            os.close(dir_fd)
                    except OSError as exc:
                        _logger.debug("Directory fsync on %s skipped: %s",
                                      self._path.parent, exc)
                except Exception:
                    # Clean up the temp file on failure; the original index
                    # at self._path is untouched.
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except OSError:
                        pass
                    raise

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
        ndim = self._ndim or 1

        # Filter to existing keys for batch retrieval
        existing_mask = np.array(
            [int(k) in self._index for k in keys], dtype=bool
        )

        if not existing_mask.any():
            _logger.warning(
                "Index missing key(s); returning zero vectors. "
                "Index may be out of sync with catalog."
            )
            return np.zeros((len(keys), ndim), dtype=np.float32)

        if existing_mask.all():
            # Fast path: all keys exist, batch retrieve
            return np.asarray(self._index[keys], dtype=np.float32)

        # Mixed: some keys missing
        _logger.warning(
            "Index missing key(s); returning zero vectors. "
            "Index may be out of sync with catalog."
        )
        result = np.zeros((len(keys), ndim), dtype=np.float32)
        existing_keys = keys[existing_mask]
        result[existing_mask] = np.asarray(
            self._index[existing_keys], dtype=np.float32
        )
        return result

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
