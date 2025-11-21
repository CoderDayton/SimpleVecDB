"""Environment configuration for TinyVecDB."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Environment configuration."""

    # Embedding
    EMBEDDING_API_BASE: str = os.getenv(
        "EMBEDDING_API_BASE", "http://127.0.0.1:53287/v1"
    )
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "unsloth/gemma-3-4b-it")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://llm.chutes.ai/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

    # Quantization
    QUANTIZATION_DIM: int = int(os.getenv("QUANTIZATION_DIM", "4"))

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
