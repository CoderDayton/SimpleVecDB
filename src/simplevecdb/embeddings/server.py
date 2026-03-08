from __future__ import annotations

import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .models import DEFAULT_MODEL, embed_texts
from simplevecdb.config import config

_logger = logging.getLogger("simplevecdb.embeddings.server")


# Simple in-memory rate limiter
class RateLimiter:
    """Token bucket rate limiter per IP/identity with TTL cleanup."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int = 10,
        ttl_seconds: int = 3600,
        max_buckets: int = 10000,
    ):
        self._lock = Lock()
        self._buckets: dict[str, dict[str, float]] = {}
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst = burst
        self._ttl = ttl_seconds
        self._max_buckets = max_buckets
        self._last_cleanup = time.time()

    def _cleanup_stale(self, now: float) -> None:
        """Remove buckets not accessed within TTL. Called under lock."""
        stale_keys = [
            k for k, v in self._buckets.items() if now - v["last"] > self._ttl
        ]
        for k in stale_keys:
            del self._buckets[k]

    def is_allowed(self, identity: str) -> bool:
        """Check if request is allowed and consume a token."""
        now = time.time()
        with self._lock:
            # Periodic cleanup: every TTL/4 seconds or if bucket count exceeds limit
            if (
                now - self._last_cleanup > self._ttl / 4
                or len(self._buckets) > self._max_buckets
            ):
                self._cleanup_stale(now)
                self._last_cleanup = now

            if identity not in self._buckets:
                self._buckets[identity] = {"tokens": self._burst, "last": now}

            bucket = self._buckets[identity]
            elapsed = now - bucket["last"]
            bucket["tokens"] = min(self._burst, bucket["tokens"] + elapsed * self._rate)
            bucket["last"] = now

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            return False


rate_limiter = RateLimiter(requests_per_minute=100, burst=20)

app = FastAPI(
    title="SimpleVecDB Embeddings",
    description="OpenAI-compatible /v1/embeddings endpoint – 100% local",
    version="0.0.1",
    openapi_url="/openapi.json",
    docs_url="/docs",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


class ModelRegistry:
    """In-memory mapping of allowed embedding models."""

    def __init__(self, mapping: dict[str, str], allow_unlisted: bool = True):
        self._mapping = mapping or {"default": DEFAULT_MODEL}
        self._default_alias = "default"
        if self._default_alias not in self._mapping:
            self._mapping[self._default_alias] = DEFAULT_MODEL
        self._repo_ids = set(self._mapping.values())
        self._allow_unlisted = allow_unlisted

    def resolve(self, requested: str | None) -> tuple[str, str]:
        """Return (display_id, repo_id) for a requested alias/model name."""
        if not requested:
            return self._default_alias, self._mapping[self._default_alias]
        if requested in self._mapping:
            return requested, self._mapping[requested]
        if requested in self._repo_ids:
            return requested, requested
        if self._allow_unlisted:
            return requested, requested

        allowed = sorted(set(self._mapping.keys()) | self._repo_ids)
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Model '{requested}' is not allowed.",
                "allowed": allowed,
            },
        )

    def list_models(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible model listings."""
        models = []
        seen: set[str] = set()
        for alias, repo in self._mapping.items():
            models.append(
                {
                    "id": alias,
                    "object": "model",
                    "created": 0,
                    "owned_by": "simplevecdb",
                    "metadata": {"repo_id": repo},
                }
            )
            seen.add(alias)
        for repo in self._repo_ids:
            if repo in seen:
                continue
            models.append(
                {
                    "id": repo,
                    "object": "model",
                    "created": 0,
                    "owned_by": "simplevecdb",
                    "metadata": {"repo_id": repo},
                }
            )
        return models


class UsageMeter:
    """Minimal in-memory tracker for request usage statistics."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "prompt_tokens": 0, "last_request_ts": 0.0}
        )

    def record(self, identity: str, prompt_tokens: int) -> None:
        now = time.time()
        with self._lock:
            bucket = self._stats[identity]
            bucket["requests"] += 1
            bucket["prompt_tokens"] += prompt_tokens
            bucket["last_request_ts"] = now

    def snapshot(self, identity: str | None = None) -> dict[str, dict[str, float]]:
        with self._lock:
            if identity:
                data = self._stats.get(
                    identity,
                    {"requests": 0, "prompt_tokens": 0, "last_request_ts": 0.0},
                )
                return {identity: dict(data)}
            return {key: dict(value) for key, value in self._stats.items()}


auth_scheme = HTTPBearer(auto_error=False)
registry = ModelRegistry(
    config.EMBEDDING_MODEL_REGISTRY,
    allow_unlisted=not config.EMBEDDING_MODEL_REGISTRY_LOCKED,
)
usage_meter = UsageMeter()


def authenticate_request(
    credentials: HTTPAuthorizationCredentials | None = Security(auth_scheme),
    api_key_header: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Validate API key if auth is enabled; otherwise return anonymous identity."""
    allowed_keys = config.EMBEDDING_SERVER_API_KEYS
    if not allowed_keys:
        return "anonymous"

    token = api_key_header or (credentials.credentials if credentials else None)
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    if token not in allowed_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token


class EmbeddingRequest(BaseModel):
    input: str | list[str] | list[int] | list[list[int]]
    model: str | None = None
    encoding_format: Literal["float", "base64"] | None = "float"
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})


@app.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    raw_request: Request,
    api_identity: str = Depends(authenticate_request),
) -> EmbeddingResponse:
    """
    Create embeddings for the input text(s).

    Args:
        request: EmbeddingRequest containing input text and model.

    Returns:
        EmbeddingResponse with vector data.
    """
    # Rate limit by IP or API key
    rate_key = (
        api_identity
        if api_identity != "anonymous"
        else (raw_request.client.host if raw_request.client else "unknown")
    )
    if not rate_limiter.is_allowed(rate_key):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Try again later."
        )
    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list) and all(
        isinstance(i, int) for i in request.input
    ):
        texts = [str(i) for i in request.input]  # token arrays – just stringify
    else:
        texts = [str(item) for item in request.input]

    if len(texts) > config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Batch size "
                f"{len(texts)} exceeds EMBEDDING_SERVER_MAX_REQUEST_ITEMS="
                f"{config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS}"
            ),
        )

    resolved_model_name, repo_id = registry.resolve(request.model)

    if not texts:
        embeddings = []
    else:
        try:
            effective_batch = min(
                config.EMBEDDING_BATCH_SIZE,
                config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS,
            )
            embeddings = embed_texts(
                texts, model_id=repo_id, batch_size=effective_batch
            )
        except Exception as e:
            # Log the full error internally but return generic message
            _logger.exception("Embedding failed: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Embedding operation failed. Check server logs for details.",
            )

    # Fake token usage (optional – some tools expect it)
    total_tokens = sum(len(t.split()) for t in texts)
    usage_meter.record(api_identity, total_tokens)

    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ],
        model=resolved_model_name or repo_id,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


@app.get("/v1/models")
async def list_models(
    api_identity: str = Depends(authenticate_request),
) -> dict[str, Any]:
    """List available embedding models (requires auth when configured)."""
    _ = api_identity  # dependency enforces auth when enabled
    return {"data": registry.list_models(), "object": "list"}


@app.get("/v1/usage")
async def usage(api_identity: str = Depends(authenticate_request)) -> dict[str, Any]:
    """Return aggregate or per-key usage statistics."""
    # If auth is enabled, only return the caller's stats; otherwise expose all.
    scope = api_identity if config.EMBEDDING_SERVER_API_KEYS else None
    return {"object": "usage", "data": usage_meter.snapshot(scope)}


def run_server(host: str | None = None, port: int | None = None) -> None:
    """Run the embedding server.

    Can be called programmatically or via the ``simplevecdb-server`` CLI.

    Examples
    --------
    Run with default settings:
    $ simplevecdb-server

    Override port:
    $ simplevecdb-server --port 8000

    Args:
        host: Server host (defaults to config.SERVER_HOST).
        port: Server port (defaults to config.SERVER_PORT).
    """
    # Minimal CLI-style override when invoked as a script/entry point
    # Allows commands like: simplevecdb-server --host 0.0.0.0 --port 8000
    import sys

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg in {"--host", "-h"} and i + 1 < len(argv):
            host = argv[i + 1]
        if arg in {"--port", "-p"} and i + 1 < len(argv):
            try:
                port = int(argv[i + 1])
            except ValueError:
                pass

    host = host or config.SERVER_HOST
    port = port or config.SERVER_PORT

    # Security warnings
    if not config.EMBEDDING_SERVER_API_KEYS:
        _logger.warning(
            "⚠️  No API keys configured (EMBEDDING_SERVER_API_KEYS is empty). "
            "Server is running without authentication. "
            "Set EMBEDDING_SERVER_API_KEYS for production use."
        )
    if host == "0.0.0.0":
        _logger.warning(
            "⚠️  Server binding to all interfaces (0.0.0.0). "
            "This exposes the server to the network. "
            "Use 127.0.0.1 for local-only access."
        )

    uvicorn.run(app, host=host, port=port, log_level="info")
