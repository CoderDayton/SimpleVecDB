# Environment Setup

SimpleVecDB uses environment variables for configuration, particularly for the optional embeddings server and RAG examples.

## Quick Setup

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your preferred settings.

## Configuration Variables

### Embedding Model (Local)

Used for local embeddings when running SimpleVecDB without an external API.

| Variable                          | Description                                                                                       | Default                     |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------- |
| `EMBEDDING_MODEL`                 | HuggingFace model ID for local embeddings.                                                        | `TaylorAI/bge-micro-v2`     |
| `EMBEDDING_CACHE_DIR`             | Directory to cache downloaded models.                                                             | `~/.cache/simplevecdb`        |
| `EMBEDDING_MODEL_REGISTRY`        | Comma-separated `alias=repo_id` entries to allowlist multiple local models.                       | `default=<EMBEDDING_MODEL>` |
| `EMBEDDING_MODEL_REGISTRY_LOCKED` | Keep the registry allowlist enforced (`1`). Set to `0` to let clients request arbitrary repo IDs. | `1`                         |
| `EMBEDDING_BATCH_SIZE`            | Batch size for inference (auto-detected if not set).                                              | _Auto_                      |

#### Batch Size Auto-Detection

SimpleVecDB automatically detects the optimal batch size based on your hardware:

- **NVIDIA GPUs**: 64-512 based on VRAM (4GB-24GB+)
- **AMD GPUs**: 256 (ROCm)
- **Apple Silicon**: 32-128 based on chip (M1/M2 vs M3/M4, base vs Max/Ultra)
- **ARM CPUs**: 4-16 based on core count (mobile, Pi, servers)
- **x86 CPUs**: 8-64 based on core count

Only override `EMBEDDING_BATCH_SIZE` if you need to tune for specific workloads or troubleshoot memory issues.

### Database

| Variable        | Description                       | Default    |
| --------------- | --------------------------------- | ---------- |
| `DATABASE_PATH` | Path to the SQLite database file. | `:memory:` |

### Server

Configuration for `simplevecdb-server`.

| Variable                             | Description                                                                    | Default                                   |
| ------------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------- |
| `SERVER_HOST`                        | Host to bind the server to.                                                    | `0.0.0.0`                                 |
| `SERVER_PORT`                        | Port to bind the server to.                                                    | `53287` (Code default) / `8000` (Example) |
| `EMBEDDING_SERVER_MAX_REQUEST_ITEMS` | Max number of prompts allowed per `/v1/embeddings` request (protects latency). | `max(32, EMBEDDING_BATCH_SIZE)`           |
| `EMBEDDING_SERVER_API_KEYS`          | Comma-separated API keys to require `Authorization: Bearer`/`X-API-Key`.       | _Disabled (unauthenticated)_              |

When `EMBEDDING_SERVER_API_KEYS` is set, SimpleVecDB also tracks request counts and token usage per key. Call `GET /v1/usage` with the same key to retrieve your stats.

### Keyword & Hybrid Search Requirements

The `keyword_search` and `hybrid_search` helpers rely on SQLite's FTS5 module. Python's built-in SQLite and most distro packages already enable it. If you're compiling SQLite manually (e.g., embedding SimpleVecDB inside another binary), be sure to pass `-DSQLITE_ENABLE_FTS5` so the FTS virtual table is available. Without FTS5, these methods will raise a descriptive runtime error.

## Using with Custom Embedding Models

To use a different HuggingFace model for local embeddings:

```bash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_CACHE_DIR=~/.cache/my_embeddings
```

Popular embedding models:

- `Snowflake/snowflake-arctic-embed-xs` - 384 dims, 22M params (default, best balance, fast)
- `TaylorAI/bge-micro-v2` - 384 dims, 17M params (fast, slightly lower quality)
- `BAAI/bge-small-en-v1.5` - 384 dims, 33M params
- `BAAI/bge-m3` - 1024 dims, ~568M params (best quality, multilingual, slower)
