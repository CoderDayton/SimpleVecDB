from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
import threading

from ..config import config

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
else:  # Fallback to Any to keep runtime import optional
    SentenceTransformerType = Any

DEFAULT_MODEL = config.EMBEDDING_MODEL
CACHE_DIR = Path(os.path.expanduser(config.EMBEDDING_CACHE_DIR))
_model_lock = threading.Lock()
_loaded_models: dict[str, SentenceTransformerType] = {}


def _load_sentence_transformer_cls() -> type[SentenceTransformerType]:
    """Import SentenceTransformer lazily to avoid heavy deps at module import."""
    try:
        from sentence_transformers import SentenceTransformer as cls
    except Exception as exc:  # pragma: no cover - exercised when deps missing
        raise ImportError(
            "Embeddings support requires the 'simplevecdb[server]' extra."
        ) from exc
    return cls


def _load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - exercised when deps missing
        raise ImportError(
            "Embeddings support requires the 'simplevecdb[server]' extra."
        ) from exc
    return snapshot_download


def load_model(repo_id: str) -> SentenceTransformerType:
    """Load (and cache on disk) a SentenceTransformer for the given repo id."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = _load_snapshot_download()
    st_cls = _load_sentence_transformer_cls()

    model_path = snapshot(
        repo_id=repo_id,
        cache_dir=CACHE_DIR,
        local_files_only=False,  # auto-download first time
    )

    # Use PyTorch backend by default (most compatible)
    # ONNX backend has compatibility issues with optimum>=2.0
    model = st_cls(
        model_path,
        tokenizer_kwargs={"padding": True, "truncation": True, "max_length": 512},
        backend="torch",
    )

    return model


def get_embedder(model_id: str | None = None) -> SentenceTransformerType:
    """Return a cached embedder for the requested model (defaults to config value)."""
    repo_id = model_id or DEFAULT_MODEL
    with _model_lock:
        model = _loaded_models.get(repo_id)
        if model is None:
            model = load_model(repo_id)
            _loaded_models[repo_id] = model
    return model


def embed_texts(
    texts: list[str], *, model_id: str | None = None, batch_size: int | None = None
) -> list[list[float]]:
    """
    Embed a list of texts using the default model.

    Args:
        texts: List of strings to embed.
        model_id: Optional repo id / alias override.
        batch_size: Optional override for encode batch size.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    model = get_embedder(model_id)
    effective_batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=effective_batch_size,
        show_progress_bar=False,
    )
    return embeddings.tolist()
