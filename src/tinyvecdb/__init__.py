from __future__ import annotations

from .types import Document, DistanceStrategy
from .core import VectorDB, Quantization
from .config import config
from .integrations.langchain import TinyVecDBVectorStore
from .integrations.llamaindex import TinyVecDBLlamaStore

__version__ = "0.0.1"
__all__ = [
    "VectorDB",
    "Quantization",
    "Document",
    "DistanceStrategy",
    "TinyVecDBVectorStore",
    "TinyVecDBLlamaStore",
    "config",
]
