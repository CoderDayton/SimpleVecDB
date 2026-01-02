from __future__ import annotations

from .types import (
    Document,
    DistanceStrategy,
    Quantization,
    MigrationRequiredError,
    StreamingProgress,
    ProgressCallback,
)
from .core import VectorDB, VectorCollection, get_optimal_batch_size
from .async_core import AsyncVectorDB, AsyncVectorCollection
from .config import config
from .integrations.langchain import SimpleVecDBVectorStore
from .integrations.llamaindex import SimpleVecDBLlamaStore
from .logging import get_logger, configure_logging, log_operation
from .utils import DatabaseLockedError, retry_on_lock, validate_filter
from .encryption import EncryptionError, EncryptionUnavailableError

__version__ = "2.1.0"
__all__ = [
    # Core classes
    "VectorDB",
    "VectorCollection",
    "AsyncVectorDB",
    "AsyncVectorCollection",
    # Types
    "Quantization",
    "Document",
    "DistanceStrategy",
    "StreamingProgress",
    "ProgressCallback",
    # Integrations
    "SimpleVecDBVectorStore",
    "SimpleVecDBLlamaStore",
    # Configuration
    "config",
    "get_optimal_batch_size",
    # Logging
    "get_logger",
    "configure_logging",
    "log_operation",
    # Error handling
    "DatabaseLockedError",
    "MigrationRequiredError",
    "EncryptionError",
    "EncryptionUnavailableError",
    "retry_on_lock",
    "validate_filter",
]
