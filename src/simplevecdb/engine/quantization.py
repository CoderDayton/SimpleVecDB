from __future__ import annotations

import logging
import warnings

import numpy as np
from ..types import Quantization

_logger = logging.getLogger("simplevecdb.engine.quantization")
# Module-level latch so we warn once per process, not on every serialize call.
_INT8_RANGE_WARNED = False


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector.

    Args:
        vector: Input vector to normalize

    Returns:
        L2-normalized vector (unit length), or original if effectively zero.
    """
    norm = float(np.linalg.norm(vector))
    # An exact ``norm == 0`` check misses subnormal floats (e.g. 1e-40) which
    # would explode on division. Treat anything below 1e-12 as zero, matching
    # the guard already used in usearch_index.
    return vector if norm < 1e-12 else vector / norm


class QuantizationStrategy:
    """
    Handles vector quantization and serialization.

    Supports FLOAT (32-bit), INT8 (8-bit), and BIT (1-bit) quantization modes.

    Args:
        quantization: Quantization mode to use
    """

    def __init__(self, quantization: Quantization):
        self.quantization = quantization

    def serialize(self, vector: np.ndarray) -> bytes:
        """
        Serialize a normalized float vector according to quantization mode.

        Args:
            vector: Input vector to serialize

        Returns:
            Serialized bytes ready for SQLite storage

        Raises:
            ValueError: If quantization mode is unsupported
        """
        if self.quantization == Quantization.FLOAT:
            return np.asarray(vector, dtype=np.float32).tobytes()

        elif self.quantization == Quantization.INT8:
            # Scalar quantization assumes inputs are in roughly [-1, 1] (e.g.,
            # L2-normalized embeddings). Out-of-range values lose magnitude
            # information when clipped. Pre-2.6.0 silently clipped; 2.6.0
            # initially raised, which broke callers that relied on the
            # silent-clip behavior. Compromise: clip and emit a one-time
            # DeprecationWarning so users have time to normalize.
            arr = np.asarray(vector)
            max_abs = float(np.abs(arr).max()) if arr.size else 0.0
            if max_abs > 1.0 + 1e-5:
                global _INT8_RANGE_WARNED
                if not _INT8_RANGE_WARNED:
                    _INT8_RANGE_WARNED = True
                    warnings.warn(
                        "INT8 quantization received a vector with "
                        f"max(|x|)={max_abs:.4f}, outside the expected "
                        "[-1, 1] range. The value will be clipped, which "
                        "loses magnitude information. Call normalize_l2() "
                        "first; future versions may raise instead of "
                        "warning.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
            scaled = np.clip(np.round(arr * 127), -128, 127).astype(np.int8)
            return scaled.tobytes()

        elif self.quantization == Quantization.FLOAT16:
            return np.asarray(vector, dtype=np.float16).tobytes()

        elif self.quantization == Quantization.BIT:
            # Binary quantization: threshold at 0 → pack bits
            bits = (vector > 0).astype(np.uint8)
            packed = np.packbits(bits)
            return packed.tobytes()

        raise ValueError(f"Unsupported quantization: {self.quantization}")

    def deserialize(self, blob: bytes, dim: int | None) -> np.ndarray:
        """
        Reverse serialization for fallback path.

        Args:
            blob: Serialized bytes from SQLite
            dim: Original vector dimension (required for BIT mode)

        Returns:
            Deserialized float32 vector

        Raises:
            ValueError: If quantization mode unsupported or dim missing for BIT
        """
        if self.quantization == Quantization.FLOAT:
            return np.frombuffer(blob, dtype=np.float32)

        elif self.quantization == Quantization.INT8:
            return np.frombuffer(blob, dtype=np.int8).astype(np.float32) / 127.0

        elif self.quantization == Quantization.FLOAT16:
            return np.frombuffer(blob, dtype=np.float16).astype(np.float32)

        elif self.quantization == Quantization.BIT and dim is not None:
            unpacked = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
            v = unpacked[:dim].astype(np.float32)
            return np.where(v == 1, 1.0, -1.0)

        raise ValueError(
            f"Unsupported quantization: {self.quantization} or unknown dim {dim}"
        )
