"""Environment configuration for TinyVecDB."""

import os
from pathlib import Path
from dotenv import load_dotenv

from .core import get_optimal_batch_size

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Environment configuration."""

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "TaylorAI/bge-micro-v2")
    EMBEDDING_CACHE_DIR: str = os.getenv(
        "EMBEDDING_CACHE_DIR", str(Path.home() / ".cache" / "tinyvecdb")
    )
    # Auto-detect optimal batch size if not explicitly set
    _batch_size_env = os.getenv("EMBEDDING_BATCH_SIZE")
    EMBEDDING_BATCH_SIZE: int = (
        int(_batch_size_env)
        if _batch_size_env is not None
        else get_optimal_batch_size()
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
