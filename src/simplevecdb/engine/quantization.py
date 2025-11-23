from __future__ import annotations

import struct
import numpy as np
from ..types import Quantization


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector.

    Args:
        vector: Input vector to normalize

    Returns:
        L2-normalized vector (unit length), or original if zero norm
    """
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


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
            return struct.pack("<%sf" % len(vector), *(float(x) for x in vector))

        elif self.quantization == Quantization.INT8:
            # Scalar quantization: scale to [-128, 127]
            scaled = np.clip(np.round(vector * 127), -128, 127).astype(np.int8)
            return scaled.tobytes()

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

        elif self.quantization == Quantization.BIT and dim is not None:
            unpacked = np.unpackbits(np.frombuffer(blob, dtype=np.uint8))
            v = unpacked[:dim].astype(np.float32)
            return np.where(v == 1, 1.0, -1.0)

        raise ValueError(
            f"Unsupported quantization: {self.quantization} or unknown dim {dim}"
        )
