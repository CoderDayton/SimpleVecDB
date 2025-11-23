"""Environment configuration for TinyVecDB."""

import os
from pathlib import Path
from dotenv import load_dotenv

from .core import get_optimal_batch_size

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


def _parse_registry(raw: str | None, default_model: str) -> dict[str, str]:
    """Convert comma-separated alias=repo entries into a registry dict."""
    registry: dict[str, str] = {}
    if raw:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                alias, repo = entry.split("=", 1)
                registry[alias.strip()] = repo.strip()
            else:
                registry[entry] = entry
    registry.setdefault("default", default_model)
    return registry


def _parse_api_keys(raw: str | None) -> set[str]:
    """Return a sanitized set of API keys from comma-separated env values."""
    if not raw:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def _parse_bool_env(raw: str | None, default: bool) -> bool:
    """Handle common truthy/falsey env strings with a fallback default."""
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class Config:
    """Environment configuration."""

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "TaylorAI/bge-micro-v2")
    EMBEDDING_CACHE_DIR: str = os.getenv(
        "EMBEDDING_CACHE_DIR", str(Path.home() / ".cache" / "tinyvecdb")
    )
    _registry_env = os.getenv("EMBEDDING_MODEL_REGISTRY")
    EMBEDDING_MODEL_REGISTRY: dict[str, str] = _parse_registry(
        _registry_env, EMBEDDING_MODEL
    )
    EMBEDDING_MODEL_REGISTRY_LOCKED: bool = _parse_bool_env(
        os.getenv("EMBEDDING_MODEL_REGISTRY_LOCKED"), True
    )
    # Auto-detect optimal batch size if not explicitly set
    _batch_size_env = os.getenv("EMBEDDING_BATCH_SIZE")
    EMBEDDING_BATCH_SIZE: int = (
        int(_batch_size_env)
        if _batch_size_env is not None
        else get_optimal_batch_size()
    )
    _request_limit_env = os.getenv("EMBEDDING_SERVER_MAX_REQUEST_ITEMS") or os.getenv(
        "EMBEDDING_SERVER_MAX_BATCH"
    )
    EMBEDDING_SERVER_MAX_REQUEST_ITEMS: int = (
        int(_request_limit_env) if _request_limit_env else max(32, EMBEDDING_BATCH_SIZE)
    )
    EMBEDDING_SERVER_API_KEYS: set[str] = _parse_api_keys(
        os.getenv("EMBEDDING_SERVER_API_KEYS")
    )

    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", ":memory:")

    # Server
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls()


# Singleton instance
config = Config.from_env()
