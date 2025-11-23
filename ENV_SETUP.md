# Environment Setup

TinyVecDB uses environment variables for configuration, particularly for the optional embeddings server and RAG examples.

## Quick Setup

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your preferred settings.

## Configuration Variables

### Embedding Model (Local)

Used for local embeddings when running TinyVecDB without an external API.

| Variable               | Description                                          | Default                 |
| ---------------------- | ---------------------------------------------------- | ----------------------- |
| `EMBEDDING_MODEL`      | HuggingFace model ID for local embeddings.           | `TaylorAI/bge-micro-v2` |
| `EMBEDDING_CACHE_DIR`  | Directory to cache downloaded models.                | `~/.cache/tinyvecdb`    |
| `EMBEDDING_BATCH_SIZE` | Batch size for inference (auto-detected if not set). | _Auto_                  |

#### Batch Size Auto-Detection

TinyVecDB automatically detects the optimal batch size based on your hardware:

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

Configuration for `tinyvecdb-server`.

| Variable      | Description                 | Default                                   |
| ------------- | --------------------------- | ----------------------------------------- |
| `SERVER_HOST` | Host to bind the server to. | `0.0.0.0`                                 |
| `SERVER_PORT` | Port to bind the server to. | `53287` (Code default) / `8000` (Example) |

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
