# Environment Setup Guide

## Quick Start

1. **Copy the example env file:**

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your values:**

   ```bash
   nano .env  # or your preferred editor
   ```

3. **Fill in your API keys and endpoints:**
   ```env
   EMBEDDING_API_KEY=sk-xxx...
   LLM_API_KEY=cpk_xxx...
   ```

## Environment Variables

### Embeddings

- `EMBEDDING_API_BASE` - OpenAI-compatible embeddings endpoint (default: `http://127.0.0.1:53287/v1`)
- `EMBEDDING_API_KEY` - API key for embeddings service

### LLM (for RAG)

- `LLM_MODEL` - Model name (default: `unsloth/gemma-3-4b-it`)
- `LLM_API_BASE` - LLM API endpoint
- `LLM_API_KEY` - API key for LLM service

### Database

- `DATABASE_PATH` - Path to SQLite database file (default: `:memory:`)

### Server

- `SERVER_HOST` - Server host (default: `0.0.0.0`)
- `SERVER_PORT` - Server port (default: `8000`)

## In Your Code

```python
from tinyvecdb.config import config

# Access configuration
print(config.EMBEDDING_API_BASE)
print(config.LLM_API_KEY)
```

Or load environment variables directly:

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("EMBEDDING_API_KEY")
```

## Security Notes

- **Never commit `.env`** to version control (it's in `.gitignore`)
- Use `.env.example` to document required variables
- Keep API keys in `.env` only, never in code
- Use different keys for development vs production
