from __future__ import annotations

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from ..config import config

DEFAULT_MODEL = config.EMBEDDING_MODEL
CACHE_DIR = Path(os.path.expanduser(config.EMBEDDING_CACHE_DIR))


def load_default_model() -> SentenceTransformer:
    """
    Load the embedding model specified in the config.

    Returns:
        Loaded SentenceTransformer model.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        repo_id=DEFAULT_MODEL,
        cache_dir=CACHE_DIR,
        local_files_only=False,  # auto-download first time
    )

    # Quantized + CPU-friendly
    model = SentenceTransformer(
        model_path,
        model_kwargs={"dtype": "auto", "file_name": "model.onnx"},
        tokenizer_kwargs={"padding": True, "truncation": True, "max_length": 512},
        backend="onnx",
    )
    # Optional: enable memory-efficient attention if flash-attn available (no-op otherwise)
    # Modern PyTorch (2.0+) uses SDPA by default, so explicit BetterTransformer conversion
    # is often unnecessary or deprecated in newer transformers versions.

    return model


# Global singleton
_default_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """
    Get the global singleton embedding model.

    Returns:
        SentenceTransformer instance.
    """
    global _default_model
    if _default_model is None:
        _default_model = load_default_model()
    return _default_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the default model.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    model = get_embedder()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
    )
    return embeddings.tolist()
