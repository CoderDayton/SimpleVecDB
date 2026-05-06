from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import signal
import time
from collections import defaultdict
from threading import Lock
from typing import Any, Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from .models import DEFAULT_MODEL, embed_texts, get_embedder
from simplevecdb.config import config

_logger = logging.getLogger("simplevecdb.embeddings.server")

# Maximum length (chars) for a single input text to prevent OOM in the encoder.
_MAX_TEXT_LENGTH = 100_000


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

    _CLEANUP_BATCH = 500  # Max stale keys to evict per call to bound lock time

    def _cleanup_stale(self, now: float) -> None:
        """Remove up to _CLEANUP_BATCH stale buckets. Called under lock."""
        removed = 0
        to_delete: list[str] = []
        for k, v in self._buckets.items():
            if now - v["last"] > self._ttl:
                to_delete.append(k)
                removed += 1
                if removed >= self._CLEANUP_BATCH:
                    break
        for k in to_delete:
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

# (#9) Pull version from package metadata instead of hardcoding
try:
    from importlib.metadata import version as _pkg_version

    _server_version = _pkg_version("simplevecdb")
except Exception:
    _server_version = "0.0.0"

app = FastAPI(
    title="SimpleVecDB Embeddings",
    description="OpenAI-compatible /v1/embeddings endpoint – 100% local",
    version=_server_version,
    openapi_url="/openapi.json",
    docs_url="/docs",
)

# (#4) CORS middleware — configurable via EMBEDDING_SERVER_CORS_ORIGINS env var.
# Default is no CORS (no allow_origins, no credentials) so the server is safe
# to deploy without explicit CORS configuration. Operators that want CORS set
# EMBEDDING_SERVER_CORS_ORIGINS to an explicit list of allowed origins, which
# enables credentials. The wildcard ("*") + allow_credentials=True combo is
# rejected by browsers per the CORS spec and is never produced here.
_cors_origins = getattr(config, "EMBEDDING_SERVER_CORS_ORIGINS", None)
if _cors_origins:
    if "*" in _cors_origins:
        # Wildcard origin must not pair with credentials. Strip credentials
        # in that case so the server stays compliant; if you actually need
        # credentialed CORS, set explicit origins instead.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


class ModelRegistry:
    """In-memory mapping of allowed embedding models."""

    def __init__(self, mapping: dict[str, str], allow_unlisted: bool = False):
        # Default to locked: programmatic ModelRegistry instances (e.g., in
        # tests) get the same secure default as the configured server. Until
        # callers explicitly opt in, unlisted models cannot be served.
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

    def snapshot(
        self,
        identity: str | None = None,
        *,
        aggregate: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Return per-identity usage stats, or an aggregate total.

        When ``identity`` is given, return only that bucket. Otherwise:
        - ``aggregate=False`` (default): the full per-identity map.
        - ``aggregate=True``: a single ``{"_total": {...}}`` bucket
          summed across identities. The aggregate mode is what the
          ``/v1/usage`` endpoint exposes when auth is disabled, so the
          server doesn't leak the list of client IPs that have used it.
        """
        with self._lock:
            if identity:
                data = self._stats.get(
                    identity,
                    {"requests": 0, "prompt_tokens": 0, "last_request_ts": 0.0},
                )
                return {identity: dict(data)}
            if aggregate:
                total = {"requests": 0.0, "prompt_tokens": 0.0, "last_request_ts": 0.0}
                for value in self._stats.values():
                    total["requests"] += value["requests"]
                    total["prompt_tokens"] += value["prompt_tokens"]
                    total["last_request_ts"] = max(
                        total["last_request_ts"], value["last_request_ts"]
                    )
                return {"_total": total}
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
    # Constant-time comparison so the response time does not leak prefixes of
    # valid API keys to an attacker probing the endpoint. ``in`` on a set
    # short-circuits on the first differing character.
    if not any(hmac.compare_digest(token, k) for k in allowed_keys):
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


def _normalize_input(raw_input: str | list[str] | list[int] | list[list[int]]) -> list[str]:
    """Convert any valid OpenAI-compatible input format to a flat list of strings.

    Handles:
    - str → ["str"]
    - list[str] → as-is
    - list[int] → token IDs stringified individually
    - list[list[int]] → each sub-list decoded as a token sequence string
    """
    if isinstance(raw_input, str):
        return [raw_input]

    if not raw_input:
        return []

    first = raw_input[0]

    # list[int] — flat token array (single input per OpenAI spec)
    if isinstance(first, int):
        return [" ".join(str(i) for i in raw_input)]

    # list[list[int]] — nested token arrays (#8)
    if isinstance(first, list):
        return [" ".join(str(tok) for tok in sub) for sub in raw_input]

    # list[str]
    return [str(item) for item in raw_input]


def _validate_texts(texts: list[str]) -> None:
    """Reject empty strings and texts exceeding the per-item length cap (#5)."""
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"Input text at index {i} is empty or whitespace-only.",
            )
        if len(text) > _MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Input text at index {i} is {len(text)} chars, "
                    f"exceeding the {_MAX_TEXT_LENGTH} char limit."
                ),
            )


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

    # (#8) Properly normalize all input formats including nested token arrays
    texts = _normalize_input(request.input)

    # (#5) Validate individual texts
    if texts:
        _validate_texts(texts)

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
        embeddings: list[list[float]] = []
    else:
        try:
            effective_batch = min(
                config.EMBEDDING_BATCH_SIZE,
                config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS,
            )
            # (#2) Run embedding in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: embed_texts(
                    texts, model_id=repo_id, batch_size=effective_batch
                ),
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
    """Return aggregate or per-key usage statistics.

    When auth is configured, return only the caller's bucket. When auth is
    disabled, return a single aggregated total — the per-identity buckets
    are keyed by client IP and exposing the full list to anyone is an
    information leak.
    """
    if config.EMBEDDING_SERVER_API_KEYS:
        return {"object": "usage", "data": usage_meter.snapshot(api_identity)}
    return {"object": "usage", "data": usage_meter.snapshot(aggregate=True)}


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (#6)."""
    parser = argparse.ArgumentParser(
        prog="simplevecdb-server",
        description="Run the SimpleVecDB embeddings server (OpenAI-compatible).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Bind address (default: {config.SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Listen port (default: {config.SERVER_PORT})",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip model warm-up on startup",
    )
    return parser


def run_server(host: str | None = None, port: int | None = None) -> None:
    """Run the embedding server.

    Can be called programmatically or via the ``simplevecdb-server`` CLI.

    Examples
    --------
    Run with default settings::

        $ simplevecdb-server

    Override host and port::

        $ simplevecdb-server --host 0.0.0.0 --port 9000

    Skip model warm-up::

        $ simplevecdb-server --no-warmup

    Args:
        host: Server host (defaults to config.SERVER_HOST).
        port: Server port (defaults to config.SERVER_PORT).
    """
    # (#6) Only parse CLI args when invoked as entry point (not programmatically)
    skip_warmup = False
    if host is None and port is None:
        parser = _build_cli_parser()
        args = parser.parse_args()
        host = args.host
        port = args.port
        skip_warmup = args.no_warmup

    host = host or config.SERVER_HOST
    port = port or config.SERVER_PORT

    # (#7) Startup banner with config summary
    auth_status = (
        f"{len(config.EMBEDDING_SERVER_API_KEYS)} key(s)"
        if config.EMBEDDING_SERVER_API_KEYS
        else "DISABLED"
    )
    _logger.info(
        "\n"
        "┌─────────────────────────────────────────────┐\n"
        "│       SimpleVecDB Embeddings Server         │\n"
        "├─────────────────────────────────────────────┤\n"
        "│  Host:       %-30s│\n"
        "│  Port:       %-30s│\n"
        "│  Model:      %-30s│\n"
        "│  Auth:       %-30s│\n"
        "│  Rate limit: %-30s│\n"
        "│  Version:    %-30s│\n"
        "└─────────────────────────────────────────────┘",
        host,
        port,
        config.EMBEDDING_MODEL,
        auth_status,
        "100 req/min, burst 20",
        _server_version,
    )

    # Security warnings
    if not config.EMBEDDING_SERVER_API_KEYS:
        _logger.warning(
            "No API keys configured (EMBEDDING_SERVER_API_KEYS is empty). "
            "Server is running without authentication. "
            "Set EMBEDDING_SERVER_API_KEYS for production use."
        )
    # Coordinate the per-request item cap with the encode-call cap so the
    # first never exceeds the second (otherwise embed_texts raises after the
    # request has already been validated and accepted).
    from .models import _MAX_ENCODE_BATCH

    try:
        request_cap = int(config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS)
    except (TypeError, ValueError):
        request_cap = None
    if request_cap is not None and request_cap > _MAX_ENCODE_BATCH:
        raise RuntimeError(
            f"EMBEDDING_SERVER_MAX_REQUEST_ITEMS="
            f"{request_cap} exceeds the embed_texts cap of "
            f"{_MAX_ENCODE_BATCH}. Lower the env var or raise "
            "_MAX_ENCODE_BATCH in embeddings/models.py."
        )

    if host == "0.0.0.0":
        _logger.warning(
            "Server binding to all interfaces (0.0.0.0). "
            "This exposes the server to the network. "
            "Use 127.0.0.1 for local-only access."
        )

    # (#3) Model warm-up — pre-load default model before accepting traffic
    if not skip_warmup:
        _logger.info("Warming up default model: %s ...", config.EMBEDDING_MODEL)
        try:
            get_embedder(config.EMBEDDING_MODEL)
            _logger.info("Model warm-up complete.")
        except Exception:
            _logger.warning(
                "Model warm-up failed (will retry on first request).", exc_info=True
            )

    # (#1) Graceful shutdown with in-flight request draining
    uvi_config = uvicorn.Config(
        app, host=host, port=port, log_level="info", timeout_graceful_shutdown=10
    )
    server = uvicorn.Server(uvi_config)

    # Install signal handlers that tell uvicorn to drain gracefully
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _graceful_shutdown(signum: int, frame: Any) -> None:
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        _logger.info("Received %s — draining in-flight requests...", sig_name)
        server.should_exit = True

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        server.run()
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
