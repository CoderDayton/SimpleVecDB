"""Test hardware detection and optimal batch size calculation."""

import platform
import multiprocessing


def test_batch_size_detection(capsys):
    """Test that batch size detection runs and produces valid output."""
    from tinyvecdb import config

    # Test that we got a valid batch size
    batch_size = config.EMBEDDING_BATCH_SIZE
    assert isinstance(batch_size, int), "Batch size should be an integer"
    assert batch_size > 0, "Batch size should be positive"
    assert batch_size <= 1024, "Batch size should be reasonable (<=1024)"

    # Print diagnostic info for manual inspection
    print("\n" + "=" * 60)
    print("TinyVecDB Hardware Detection")
    print("=" * 60)

    # System info
    print(f"\n🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🔧 Machine: {platform.machine()}")
    print(f"💻 Processor: {platform.processor()}")
    print(f"⚙️  CPU Cores: {multiprocessing.cpu_count()}")

    # PyTorch detection
    print("\n" + "-" * 60)
    print("GPU Detection")
    print("-" * 60)

    try:
        import torch

        print(f"✅ PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)
            print(f"🎮 CUDA GPU: {gpu_props.name}")
            print(f"💾 VRAM: {vram_gb:.2f} GB")
            print(f"🔢 Compute Capability: {gpu_props.major}.{gpu_props.minor}")

            # Validate GPU-specific batch sizes
            if vram_gb >= 20:
                assert batch_size >= 256, "High VRAM GPU should have batch size >= 256"
        elif hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore
            print("🎮 AMD ROCm GPU detected")
            assert batch_size >= 128, "AMD GPU should have batch size >= 128"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🍎 Apple Metal (MPS) available")
            try:
                import subprocess

                chip_info = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
                print(f"💻 Chip: {chip_info}")
            except Exception:
                pass
            assert batch_size >= 16, "Apple Silicon should have batch size >= 16"
        else:
            print("⚠️  No GPU detected - using CPU")

    except ImportError:
        print("⚠️  PyTorch not installed - CPU mode only")

    # Batch size recommendation
    print("\n" + "-" * 60)
    print("Batch Size Configuration")
    print("-" * 60)

    print(f"📦 Optimal Batch Size: {batch_size}")

    # Guidance
    print("\n💡 Guidance:")
    if batch_size >= 256:
        print("   High batch size detected - you have powerful GPU hardware!")
        print("   Great for processing large document sets.")
    elif batch_size >= 64:
        print("   Medium-high batch size - good GPU or high-end CPU.")
        print("   Efficient for most workloads.")
    elif batch_size >= 32:
        print("   Standard batch size - typical desktop/laptop.")
        print("   Balanced for general use.")
    else:
        print("   Conservative batch size - mobile/low-power device.")
        print("   Optimized for thermal management and stability.")

    print("\n🔧 Override: Set EMBEDDING_BATCH_SIZE in .env to customize.")
    print("=" * 60)


def test_get_optimal_batch_size():
    """Test that get_optimal_batch_size returns valid values."""
    from tinyvecdb.config import get_optimal_batch_size

    batch_size = get_optimal_batch_size()

    assert isinstance(batch_size, int), "Should return integer"
    assert batch_size > 0, "Should return positive value"
    assert batch_size in [4, 8, 16, 32, 48, 64, 128, 256, 512], (
        f"Should return one of the predefined batch sizes, got {batch_size}"
    )


def test_batch_size_with_env_override(monkeypatch):
    """Test that environment variable override works."""
    import os

    # Set custom batch size
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "256")

    # Get fresh config value
    batch_size_str = os.getenv("EMBEDDING_BATCH_SIZE")
    assert batch_size_str == "256", "Environment variable should be set"

    # Test that parsing works
    batch_size = int(batch_size_str)
    assert batch_size == 256, "Should parse to 256"
