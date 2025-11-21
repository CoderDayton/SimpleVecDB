from __future__ import annotations

import os
from torch.cuda import is_available
from pathlib import Path

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

DEFAULT_MODEL = (
    "TaylorAI/bge-micro-v2"  # 384-dim, state-of-the-art tiny model (Nov 2025)
)
CACHE_DIR = Path(
    os.getenv("TINYVECDB_CACHE", str(Path.home() / ".cache" / "tinyvecdb"))
)


def load_default_model() -> SentenceTransformer:
    """Load quantized BGE-Micro-v2 with smart caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        repo_id=DEFAULT_MODEL,
        cache_dir=CACHE_DIR,
        local_files_only=False,  # auto-download first time
    )

    # Quantized + CPU-friendly
    model = SentenceTransformer(
        model_path,
        device="cpu" if not is_available() else "cuda",  # works perfectly on CPU
        model_kwargs={"dtype": "auto"},
        tokenizer_kwargs={"padding": True, "truncation": True, "max_length": 512},
    )
    # Optional: enable memory-efficient attention if flash-attn available (no-op otherwise)
    # Modern PyTorch (2.0+) uses SDPA by default, so explicit BetterTransformer conversion
    # is often unnecessary or deprecated in newer transformers versions.

    return model


# Global singleton
_default_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _default_model
    if _default_model is None:
        _default_model = load_default_model()
    return _default_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32 if is_available() else 64,
        show_progress_bar=False,
    )
    return embeddings.tolist()
