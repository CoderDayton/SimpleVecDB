"""Embeddings module — local embedding models and OpenAI-compatible server."""

from .models import embed_texts, get_embedder, load_model
from .server import app, run_server

__all__ = [
    "app",
    "embed_texts",
    "get_embedder",
    "load_model",
    "run_server",
]
