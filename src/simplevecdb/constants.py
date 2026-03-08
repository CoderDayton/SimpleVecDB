"""Configuration constants for SimpleVecDB."""

# Hardware batch size thresholds for GPU VRAM (in GB)
BATCH_SIZE_VRAM_THRESHOLDS = {
    20: 512,  # RTX 4090, A100, H100
    12: 256,  # RTX 4070 Ti, 3090, A10
    8: 128,  # RTX 4060 Ti, 3070
    4: 64,  # GTX 1660, RTX 3050
}

# CPU batch size by core count
CPU_BATCH_SIZE_BY_CORES = {
    32: 64,
    16: 48,
    8: 32,
}

# Default parameters
DEFAULT_DISTANCE_STRATEGY = "cosine"
DEFAULT_QUANTIZATION = "float"

# SQL injection prevention
COLLECTION_NAME_PATTERN = r"^[a-zA-Z0-9_]+$"

# Search defaults
DEFAULT_K = 5
DEFAULT_RRF_K = 60
DEFAULT_FETCH_K = 20

# Hardware-specific defaults
DEFAULT_AMD_ROCM_BATCH_SIZE = 256
DEFAULT_APPLE_M1_M2_BATCH_SIZE = 32
DEFAULT_APPLE_M3_M4_BATCH_SIZE = 64
DEFAULT_APPLE_MAX_ULTRA_BATCH_SIZE = 128
DEFAULT_ARM_MOBILE_BATCH_SIZE = 4
DEFAULT_ARM_PI_BATCH_SIZE = 8
DEFAULT_ARM_SERVER_BATCH_SIZE = 16
DEFAULT_CPU_FALLBACK_BATCH_SIZE = 16

# ============================================================================
# Usearch HNSW Parameters
# ============================================================================
# These parameters control the trade-off between recall, speed, and memory.
# See: https://unum-cloud.github.io/usearch/
#
# connectivity (M): Number of edges per node in the HNSW graph.
#   - Higher = better recall, more memory, slower build
#   - Typical range: 8-64
#   - Default: 16 (good balance)
#
# expansion_add (efConstruction): Search depth during index construction.
#   - Higher = better recall, slower build
#   - Typical range: 64-512
#   - Default: 128 (good recall for most use cases)
#
# expansion_search (ef): Search depth during queries.
#   - Higher = better recall, slower search
#   - Can be tuned per-query for latency-sensitive applications
#   - Typical range: 32-256
#   - Default: 64 (good balance)

USEARCH_DEFAULT_CONNECTIVITY = 16
USEARCH_DEFAULT_EXPANSION_ADD = 128
USEARCH_DEFAULT_EXPANSION_SEARCH = 64

# Adaptive search threshold: use brute-force (exact=True) below this size
# HNSW has overhead that makes it slower than brute-force for small collections.
# Empirically, crossover is around 10k-50k vectors depending on dimension.
# We use a conservative threshold to ensure brute-force is always faster below it.
USEARCH_BRUTEFORCE_THRESHOLD = 10000

# Memory-mapping threshold: use view() instead of load() above this size
# For large indexes (>100k vectors), memory-mapping provides:
# - Instant startup (no full load into RAM)
# - Lower memory footprint (OS manages page cache)
# - Slight latency increase for cold pages (acceptable trade-off)
USEARCH_MMAP_THRESHOLD = 100000

# Batch search threshold: auto-batch queries when > this count
# usearch batch search provides ~10x throughput for multi-query workloads
USEARCH_BATCH_THRESHOLD = 10

# Over-fetch multiplier for filtered searches
# When filtering, we fetch k * FILTER_OVERFETCH_MULTIPLIER candidates
# and filter down to k results
USEARCH_FILTER_OVERFETCH_MULTIPLIER = 3
