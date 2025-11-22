"""Batch size detection and hardware optimization tests."""

import sys
from unittest.mock import MagicMock, patch
import multiprocessing
import platform


def test_get_optimal_batch_size_no_psutil():
    """Test get_optimal_batch_size when psutil is missing."""
    from tinyvecdb.core import get_optimal_batch_size

    with patch.dict(sys.modules, {"psutil": None}):
        # Should fallback to a reasonable default
        batch_size = get_optimal_batch_size()
        assert batch_size > 0


def test_get_optimal_batch_size_with_torch_cuda():
    """Test get_optimal_batch_size with torch and CUDA available."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_props = MagicMock()
    mock_props.total_memory = 16 * 1024**3  # 16GB
    mock_torch.cuda.get_device_properties.return_value = mock_props

    with patch.dict(sys.modules, {"torch": mock_torch}):
        # Should be calculated based on VRAM
        batch_size = get_optimal_batch_size()
        assert batch_size > 32


def test_get_optimal_batch_size_cpu_memory():
    """Test get_optimal_batch_size based on system RAM (psutil)."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value.available = 8 * 1024**3  # 8GB
    mock_psutil.cpu_count.return_value = 8

    # Mock torch to be missing or no cuda
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"psutil": mock_psutil, "torch": mock_torch}):
        batch_size = get_optimal_batch_size()
        assert batch_size > 0


def test_get_optimal_batch_size_onnx_cuda():
    """Test batch size detection with ONNX Runtime CUDA provider."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CUDAExecutionProvider"]

    with patch.dict(
        sys.modules, {"onnxruntime": mock_ort, "torch": None, "psutil": None}
    ):
        batch_size = get_optimal_batch_size()
        assert batch_size == 128


def test_get_optimal_batch_size_onnx_tensorrt():
    """Test batch size detection with ONNX Runtime TensorRT provider."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["TensorrtExecutionProvider"]

    with patch.dict(
        sys.modules, {"onnxruntime": mock_ort, "torch": None, "psutil": None}
    ):
        batch_size = get_optimal_batch_size()
        assert batch_size == 128


def test_get_optimal_batch_size_onnx_dml():
    """Test batch size detection with ONNX Runtime DirectML provider."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["DmlExecutionProvider"]

    with patch.dict(
        sys.modules, {"onnxruntime": mock_ort, "torch": None, "psutil": None}
    ):
        batch_size = get_optimal_batch_size()
        assert batch_size == 64


def test_get_optimal_batch_size_onnx_coreml():
    """Test batch size detection with ONNX Runtime CoreML provider."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_ort = MagicMock()
    mock_ort.get_available_providers.return_value = ["CoreMLExecutionProvider"]

    with patch.dict(
        sys.modules, {"onnxruntime": mock_ort, "torch": None, "psutil": None}
    ):
        batch_size = get_optimal_batch_size()
        assert batch_size == 32


def test_get_optimal_batch_size_arm_low_cores():
    """Test batch size detection on ARM with few cores."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 2  # Few cores
    mock_psutil.virtual_memory.return_value.available = 2 * 1024**3

    with patch.dict(
        sys.modules, {"onnxruntime": None, "torch": None, "psutil": mock_psutil}
    ):
        with patch.object(platform, "machine", return_value="aarch64"):
            batch_size = get_optimal_batch_size()
            assert batch_size == 4


def test_get_optimal_batch_size_arm_high_cores():
    """Test batch size detection on ARM with many cores."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = 8  # Many cores
    mock_psutil.virtual_memory.return_value.available = 8 * 1024**3

    with patch.dict(
        sys.modules, {"onnxruntime": None, "torch": None, "psutil": mock_psutil}
    ):
        with patch.object(platform, "machine", return_value="arm64"):
            batch_size = get_optimal_batch_size()
            # ARM caps at min(cpu_count, 8)
            assert batch_size == 8


def test_get_optimal_batch_size_psutil_none_cpu_count():
    """Test get_optimal_batch_size when psutil.cpu_count returns None."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_psutil = MagicMock()
    mock_psutil.cpu_count.return_value = None  # Returns None
    mock_psutil.virtual_memory.return_value.available = 8 * 1024**3

    with patch.dict(
        sys.modules, {"onnxruntime": None, "torch": None, "psutil": mock_psutil}
    ):
        batch_size = get_optimal_batch_size()
        # Should fallback to multiprocessing.cpu_count()
        assert batch_size > 0


def test_get_optimal_batch_size_importerror_fallback():
    """Test get_optimal_batch_size falls back to defaults when psutil import fails."""
    from tinyvecdb.core import get_optimal_batch_size

    # Simulate ImportError by setting psutil to None
    with patch.dict(sys.modules, {"onnxruntime": None, "torch": None, "psutil": None}):
        batch_size = get_optimal_batch_size()
        # Should use multiprocessing.cpu_count() and assume 8GB RAM
        assert batch_size > 0
        # Should be based on CPU count
        assert batch_size >= min(multiprocessing.cpu_count(), 2)


def test_get_optimal_batch_size_all_cuda_branches():
    """Test all CUDA VRAM size branches."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.hip.is_available.return_value = False
    mock_props = MagicMock()
    mock_torch.cuda.get_device_properties.return_value = mock_props

    with patch.dict(sys.modules, {"torch": mock_torch}):
        # >20GB VRAM
        mock_props.total_memory = 24 * (1024**3)
        assert get_optimal_batch_size() == 512

        # >12GB VRAM
        mock_props.total_memory = 16 * (1024**3)
        assert get_optimal_batch_size() == 256

        # >8GB VRAM
        mock_props.total_memory = 10 * (1024**3)
        assert get_optimal_batch_size() == 128

        # <8GB VRAM
        mock_props.total_memory = 4 * (1024**3)
        assert get_optimal_batch_size() == 64


def test_get_optimal_batch_size_mps_branches():
    """Test Apple MPS chip detection branches."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.hip.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True

    with patch.dict(sys.modules, {"torch": mock_torch}):
        with patch.object(platform, "machine", return_value="arm64"):
            # M3/M4 chips
            with patch("subprocess.check_output", return_value="apple m3"):
                assert get_optimal_batch_size() == 64

            # Max chips
            with patch("subprocess.check_output", return_value="apple m2 max"):
                assert get_optimal_batch_size() == 128

            # Ultra chips
            with patch("subprocess.check_output", return_value="apple m2 ultra"):
                assert get_optimal_batch_size() == 128

            # Base M1/M2 (no M3/M4 and no Max/Ultra)
            with patch("subprocess.check_output", return_value="apple m1"):
                assert get_optimal_batch_size() == 32

            # Exception fallback
            with patch("subprocess.check_output", side_effect=Exception("Fail")):
                assert get_optimal_batch_size() == 32


def test_get_optimal_batch_size_cpu_ram_branches():
    """Test CPU cores and RAM constraint branches."""
    from tinyvecdb.core import get_optimal_batch_size

    mock_psutil = MagicMock()
    mock_mem = MagicMock()
    mock_psutil.virtual_memory.return_value = mock_mem

    with patch.dict(
        sys.modules, {"torch": None, "onnxruntime": None, "psutil": mock_psutil}
    ):
        # x86 with different core counts
        with patch.object(platform, "machine", return_value="x86_64"):
            mock_mem.available = 16 * (1024**3)  # Plenty of RAM

            # >=32 cores
            mock_psutil.cpu_count.return_value = 32
            assert get_optimal_batch_size() == 64

            # >=16 cores
            mock_psutil.cpu_count.return_value = 16
            assert get_optimal_batch_size() == 48

            # >=8 cores
            mock_psutil.cpu_count.return_value = 8
            assert get_optimal_batch_size() == 32

            # <8 cores
            mock_psutil.cpu_count.return_value = 4
            assert get_optimal_batch_size() == 16

            # RAM constraints (with 8 cores -> base 32)
            mock_psutil.cpu_count.return_value = 8

            # <2GB available
            mock_mem.available = 1.5 * (1024**3)
            assert get_optimal_batch_size() == 4

            # <4GB available
            mock_mem.available = 3 * (1024**3)
            assert get_optimal_batch_size() == 8

            # <8GB available
            mock_mem.available = 6 * (1024**3)
            assert get_optimal_batch_size() == 16
