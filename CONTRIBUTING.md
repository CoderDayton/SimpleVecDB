# Contributing to TinyVecDB

Thanks for considering a contribution to TinyVecDB; your help steadily improves this local-first vector database.

## Getting Started

### Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`
- Git

### Local Setup

```bash
git clone https://github.com/coderdayton/tinyvecdb.git
cd tinyvecdb

# Install dependencies with development tools
uv sync

# Or with pip
pip install -e ".[dev]"
```

### Project Structure

```
tinyvecdb/
├── src/tinyvecdb/
│   ├── core.py              # Main VectorDB class
│   ├── types.py             # Document, DistanceStrategy types
│   ├── config.py            # Configuration management
│   ├── embeddings/
│   │   ├── models.py        # Local embedding models
│   │   └── server.py        # FastAPI embedding server
│   └── integrations/
│       ├── langchain.py     # LangChain VectorStore wrapper
│       └── llamaindex.py    # LlamaIndex VectorStore wrapper
├── tests/                   # Unitm integration and performance tests
├── examples/                # RAG notebooks, demos
└── docs/                    # Documentation
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=tinyvecdb

# Specific test file
pytest tests/unit/test_search.py
```

### Code Style

- Follow PEP 8 standards
- Use type hints wherever possible (Python 3.10+ syntax: `list[str]` instead of `List[str]`)
- Run a linter (consider using `ruff` or `black`)

### Making Changes

1. **Create a feature branch**

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** and commit with clear messages

   ```bash
   git commit -m "feat: add cool feature" # or fix:, docs:, etc.
   ```

3. **Add/update tests** for any new functionality

4. **Run tests locally** to ensure nothing breaks

5. **Submit a pull request** with a clear description

## Areas for Contribution

### High Priority

- **Hybrid search**: BM25 + vector combined search (FTS5 + vec0)
- **HNSW indexing**: Faster approximate nearest neighbor search (waiting on sqlite-vec)
- **Advanced Metadata filtering**: Complex WHERE clause support (AND, OR, IN)
- **Documentation**: Docstrings, guides, API docs

### Medium Priority

- **Multi-collection support**: Handle multiple vector collections per database
- **Custom Quantization**: Support for custom quantization tables/centroids
- **Performance benchmarks**: Add more comprehensive benchmarks (1M+ vectors)
- **Integration tests**: Expand test coverage for LangChain/LlamaIndex

### Lower Priority

- **GUI**: Desktop app (Tauri-based)
- **Encryption**: SQLCipher integration
- **Analytics**: Query performance monitoring

## Testing Guidelines

- Write tests for all new features
- Ensure tests pass locally before submitting PR
- Aim for >80% code coverage
- Test edge cases (empty vectors, large datasets, etc.)

Example test structure:

```python
def test_similarity_search_with_k():
    db = VectorDB(":memory:")
    db.add_texts(["doc1", "doc2", "doc3"])
    results = db.similarity_search("query", k=2)
    assert len(results) == 2
    assert all(isinstance(score, float) for _, score in results)
```

## Documentation

- Update docstrings for any API changes
- Add examples in the `examples/` directory for new features
- Update README.md if adding major features
- Use type hints to make APIs self-documenting

## Performance Considerations

- TinyVecDB prioritizes simplicity over maximum performance
- Benchmark large-scale operations (10k+ vectors)
- Use NumPy efficiently for vector operations
- Minimize database round-trips

## Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Run the embedding server locally for testing:

```bash
tinyvecdb-server
# Server runs at http://localhost:53287 by default
```

## Submitting a Pull Request

1. Ensure all tests pass: `pytest`
2. Keep commits clean and focused
3. Write a clear PR description explaining:
   - What problem does it solve?
   - How does it work?
   - Any breaking changes?
4. Link any related issues
5. Be patient — we'll review as soon as we can!

## Questions?

- Open a GitHub issue for bugs or feature requests
- Reach out to [@coderdayton](https://github.com/coderdayton) on GitHub
- Check existing issues before filing a duplicate

---

**Thank you for contributing!** Every bit helps make TinyVecDB better for everyone. 🚀
