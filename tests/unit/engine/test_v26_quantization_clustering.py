"""Quantization range guard + silhouette sample cap (2.6.0).

The 2.5.0 review flagged two correctness/performance issues:
- ``QuantizationStrategy.serialize`` for INT8 silently clipped any vector
  whose components exceeded |1|, destroying magnitude information without
  any signal to the caller. 2.6.0 raises a ValueError above 1+1e-5.
- ``normalize_l2`` returned the unchanged vector only when ``norm == 0``,
  so subnormal-scale inputs (e.g. 1e-40) divided by a tiny norm and
  exploded into Inf. 2.6.0 treats ``norm < 1e-12`` as zero.
- ``silhouette_score`` is O(n²) and OOMs on large collections. 2.6.0 caps
  ``sample_size`` to ``SILHOUETTE_MAX_SAMPLE`` (10k) with a fixed seed
  for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from simplevecdb import constants
from simplevecdb.engine.quantization import QuantizationStrategy, normalize_l2
from simplevecdb.types import Quantization


class TestNormalizeL2SubnormalGuard:
    def test_zero_vector_returned_unchanged(self):
        v = np.zeros(8, dtype=np.float32)
        out = normalize_l2(v)
        # Returned as-is when norm < 1e-12.
        np.testing.assert_array_equal(out, v)

    def test_subnormal_vector_returned_unchanged(self):
        # All components ~1e-40 → norm well below the 1e-12 threshold.
        v = np.full(8, 1e-40, dtype=np.float32)
        out = normalize_l2(v)
        # Must not produce Inf/NaN from divide-by-tiny.
        assert np.all(np.isfinite(out))
        # And must equal the input (not normalized).
        np.testing.assert_array_equal(out, v)

    def test_normal_vector_normalized_to_unit_length(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        out = normalize_l2(v)
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6


class TestINT8RangeGuard:
    def test_unit_norm_vector_accepted(self):
        strat = QuantizationStrategy(Quantization.INT8)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Should not raise.
        out = strat.serialize(v)
        assert isinstance(out, bytes)
        assert len(out) == 4  # int8 == 1 byte each

    def test_just_over_unit_rejected(self):
        strat = QuantizationStrategy(Quantization.INT8)
        v = np.array([1.5, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match=r"INT8 quantization expects vectors in \[-1, 1\]"):
            strat.serialize(v)

    def test_just_under_negative_unit_rejected(self):
        strat = QuantizationStrategy(Quantization.INT8)
        v = np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match=r"max\(\|x\|\)"):
            strat.serialize(v)

    def test_within_tolerance_band_accepted(self):
        # Slightly over 1.0 but within the 1e-5 tolerance — should pass.
        strat = QuantizationStrategy(Quantization.INT8)
        v = np.array([1.0 + 1e-6, 0.0, 0.0, 0.0], dtype=np.float32)
        out = strat.serialize(v)
        assert len(out) == 4

    def test_empty_vector_does_not_raise(self):
        strat = QuantizationStrategy(Quantization.INT8)
        v = np.array([], dtype=np.float32)
        out = strat.serialize(v)
        assert out == b""

    def test_error_message_includes_max_abs(self):
        strat = QuantizationStrategy(Quantization.INT8)
        v = np.array([3.7, -0.5, 0.2], dtype=np.float32)
        with pytest.raises(ValueError, match=r"3\.7"):
            strat.serialize(v)


class TestSilhouetteSampleCap:
    """SILHOUETTE_MAX_SAMPLE is set so silhouette_score doesn't OOM."""

    def test_constant_is_set_to_safe_default(self):
        # The 2.6.0 fix established 10k as a safe upper bound.
        assert constants.SILHOUETTE_MAX_SAMPLE == 10_000

    def test_constant_is_in_a_reasonable_range(self):
        # Defensive: must be small enough that O(n²) memory fits in
        # ~1GB (10k² * 8 bytes ≈ 800MB) but big enough to be useful.
        assert 1_000 <= constants.SILHOUETTE_MAX_SAMPLE <= 50_000
