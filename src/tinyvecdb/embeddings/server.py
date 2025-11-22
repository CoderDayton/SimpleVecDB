from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn

from .models import embed_texts, DEFAULT_MODEL
from tinyvecdb.config import config

app = FastAPI(
    title="TinyVecDB Embeddings",
    description="OpenAI-compatible /v1/embeddings endpoint – 100% local",
    version="0.0.1",
    openapi_url="/openapi.json",
    docs_url="/docs",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


class EmbeddingRequest(BaseModel):
    input: str | list[str] | list[int] | list[list[int]]
    model: str | None = DEFAULT_MODEL
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
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for the input text(s).

    Args:
        request: EmbeddingRequest containing input text and model.

    Returns:
        EmbeddingResponse with vector data.
    """
    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list) and all(
        isinstance(i, int) for i in request.input
    ):
        texts = [str(i) for i in request.input]  # token arrays – just stringify
    else:
        texts = [str(item) for item in request.input]

    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Fake token usage (optional – some tools expect it)
    total_tokens = sum(len(t.split()) for t in texts)

    return EmbeddingResponse(
        data=[
            EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(embeddings)
        ],
        model=request.model or DEFAULT_MODEL,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    )


@app.get("/v1/models")
async def list_models():
    """
    List available embedding models.

    Returns:
        JSON response with model list.
    """
    return {
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "tinyvecdb",
            }
        ],
        "object": "list",
    }


def run_server(host: str | None = None, port: int | None = None):
    """Run the embedding server.

    Can be called programmatically or via the ``tinyvecdb-server`` CLI.

    Examples
    --------
    Run with default settings:
    $ tinyvecdb-server

    Override port:
    $ tinyvecdb-server --port 8000

    Args:
        host: Server host (defaults to config.SERVER_HOST).
        port: Server port (defaults to config.SERVER_PORT).
    """
    # Minimal CLI-style override when invoked as a script/entry point
    # Allows commands like: tinyvecdb-server --host 0.0.0.0 --port 8000
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
    uvicorn.run(app, host=host, port=port, log_level="info")
