from __future__ import annotations

from .types import Document, DistanceStrategy, Quantization
from .core import VectorDB, VectorCollection, get_optimal_batch_size
from .async_core import AsyncVectorDB, AsyncVectorCollection
from .config import config
from .integrations.langchain import SimpleVecDBVectorStore
from .integrations.llamaindex import SimpleVecDBLlamaStore

__version__ = "1.2.0"
__all__ = [
    "VectorDB",
    "VectorCollection",
    "AsyncVectorDB",
    "AsyncVectorCollection",
    "Quantization",
    "Document",
    "DistanceStrategy",
    "SimpleVecDBVectorStore",
    "SimpleVecDBLlamaStore",
    "config",
    "get_optimal_batch_size",
]
