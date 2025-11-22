# tests/unit/test_core.py
import pytest
import numpy as np
import json
import sqlite3
from tinyvecdb import VectorDB
from tinyvecdb.types import Document, DistanceStrategy


def test_init(empty_db):
    """Verify that the database initializes with correct default values."""
    assert empty_db._dim is None
    assert (
        empty_db.quantization == "float"
    )  # Ensure default configuration values are set correctly.
    assert empty_db.distance_strategy == "cosine"


def test_add_texts_basic(empty_db):
    """Test adding texts with embeddings and verify storage integrity."""
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2], [0.3, 0.4]]
    ids = empty_db.add_texts(texts, embeddings=embs)
    assert len(ids) == 2
    assert empty_db._dim == 2

    # Verify that the text content is persisted in the main table.
    rows = empty_db.conn.execute(
        "SELECT text FROM tinyvec_items ORDER BY id"
    ).fetchall()
    assert rows[0][0] == "test1"

    # Verify that the embedding vector is stored in the virtual table.
    vec_row = empty_db.conn.execute(
        "SELECT embedding FROM vec_index WHERE rowid = ?", (ids[0],)
    ).fetchone()

    # Ensure vectors are normalized when using Cosine distance strategy.
    expected = np.array([0.1, 0.2], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(np.frombuffer(vec_row[0], dtype=np.float32), expected)


def test_add_with_metadata(populated_db):
    """Verify that metadata is correctly stored and retrievable."""
    row = populated_db.conn.execute(
        "SELECT metadata FROM tinyvec_items WHERE id=1"
    ).fetchone()[0]
    meta = json.loads(row)
    assert meta["color"] == "red"
    assert meta["likes"] == 10


def test_upsert(populated_db):
    """Test the upsert functionality (update existing records)."""
    new_emb = [0.5, 0.5, 0.5, 0.5]
    populated_db.add_texts(
        ["updated apple"], embeddings=[new_emb], ids=[1], metadatas=[{"color": "green"}]
    )

    updated = populated_db.conn.execute(
        "SELECT text, metadata FROM tinyvec_items WHERE id=1"
    ).fetchone()
    assert updated[0] == "updated apple"
    assert json.loads(updated[1])["color"] == "green"


def test_delete_by_ids(populated_db):
    """Test deletion of records by their IDs."""
    populated_db.delete_by_ids([1, 2])
    remaining = populated_db.conn.execute(
        "SELECT COUNT(*) FROM tinyvec_items"
    ).fetchone()[0]
    assert remaining == 2
    vec_count = populated_db.conn.execute("SELECT COUNT(*) FROM vec_index").fetchone()[
        0
    ]
    assert vec_count == 2  # Ensure the virtual table row count matches the main table.


def test_add_no_embeddings_raises(empty_db, monkeypatch):
    """Ensure ValueError is raised when no embeddings are provided and local embedder fails."""
    import sys

    # Simulate module missing by setting it to None in sys.modules
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "tinyvecdb.embeddings.models", None)

        with pytest.raises(ValueError, match="No embeddings provided"):
            empty_db.add_texts(["test"])


def test_close_and_del():
    """Test explicit closing of the database connection and resource cleanup."""
    db = VectorDB(":memory:")
    conn = db.conn
    db.close()

    # Verify connection is closed by attempting an operation
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_recover_dim(tmp_path):
    """Test dimension recovery from existing DB."""
    db_path = str(tmp_path / "recover.db")

    # Create DB and add data
    db1 = VectorDB(db_path)
    db1.add_texts(["test"], embeddings=[[0.1] * 10])
    db1.close()

    # Reopen
    db2 = VectorDB(db_path)
    assert db2._dim == 10
    db2.close()


def test_dequantize_fallback():
    """Test dequantization logic directly."""
    from tinyvecdb.core import _serialize_vector, _dequantize_vector, Quantization
    import numpy as np

    vec = np.array([0.1, 0.5, -0.5], dtype=np.float32)

    # Float
    blob = _serialize_vector(vec, Quantization.FLOAT)
    out = _dequantize_vector(blob, 3, Quantization.FLOAT)
    assert np.allclose(vec, out)

    # Int8
    blob = _serialize_vector(vec, Quantization.INT8)
    out = _dequantize_vector(blob, 3, Quantization.INT8)
    # Precision loss expected
    assert np.allclose(vec, out, atol=0.01)

    # Bit
    blob = _serialize_vector(vec, Quantization.BIT)
    out = _dequantize_vector(blob, 3, Quantization.BIT)
    # Binary: >0 -> 1, <=0 -> -1
    expected = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    assert np.allclose(out, expected)


def test_similarity_search_basic(populated_db):
    """Test basic similarity search functionality."""
    query = [0.1, 0.0, 0.0, 0.0]  # Close to "apple"
    results = populated_db.similarity_search(query, k=2)

    assert len(results) == 2
    # Should match "apple is red" first (closest to query)
    assert results[0][0].page_content == "apple is red"
    assert results[0][1] < results[1][1]  # First result should have lower distance


def test_similarity_search_with_filter(populated_db):
    """Test similarity search with metadata filtering."""
    query = [0.1, 0.0, 0.0, 0.0]
    results = populated_db.similarity_search(query, k=5, filter={"color": "yellow"})

    assert len(results) == 1
    assert results[0][0].page_content == "banana is yellow"
    assert results[0][0].metadata["color"] == "yellow"


def test_similarity_search_empty_db(empty_db):
    """Test similarity search on empty database."""
    results = empty_db.similarity_search([0.1, 0.2], k=5)
    assert len(results) == 0


def test_similarity_search_text_query(populated_db, monkeypatch):
    """Test similarity search with text query (requires embeddings)."""

    # Mock the embed_texts function
    def mock_embed_texts(texts):
        return [[0.1, 0.0, 0.0, 0.0]]  # Close to apple

    import tinyvecdb.embeddings.models

    monkeypatch.setattr(tinyvecdb.embeddings.models, "embed_texts", mock_embed_texts)

    results = populated_db.similarity_search("apple fruit", k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "apple is red"


def test_similarity_search_dimension_mismatch(populated_db):
    """Test that querying with wrong dimension raises error."""
    with pytest.raises(ValueError, match="Query dim"):
        populated_db.similarity_search([0.1, 0.2], k=1)  # 2D instead of 4D


def test_max_marginal_relevance_search(populated_db):
    """Test MMR search for diversity."""
    query = [0.9, 0.9, 0.9, 0.9]
    results = populated_db.max_marginal_relevance_search(query, k=2, fetch_k=4)

    assert len(results) == 2
    # MMR should select diverse documents
    assert isinstance(results[0], Document)
    assert isinstance(results[1], Document)


def test_quantization_int8(quant_db):
    """Test INT8 quantization storage and retrieval."""
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ids = quant_db.add_texts(texts, embeddings=embs)

    assert len(ids) == 2
    assert quant_db._dim == 3

    # Search should work with quantized vectors
    results = quant_db.similarity_search([0.1, 0.2, 0.3], k=1)
    assert len(results) == 1


def test_quantization_bit(bit_db):
    """Test BIT quantization storage and retrieval."""
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]]
    ids = bit_db.add_texts(texts, embeddings=embs)

    assert len(ids) == 2
    # BIT quantization rounds up to byte boundary
    assert bit_db._dim == 3

    # Search should work with binary vectors
    results = bit_db.similarity_search([0.1, 0.2, 0.3], k=1)
    assert len(results) == 1


def test_distance_strategy_l2():
    """Test L2 (Euclidean) distance strategy."""
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L2)

    texts = ["a", "b", "c"]
    embs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    db.add_texts(texts, embeddings=embs)

    # Query closest to "a"
    results = db.similarity_search([1.0, 0.0], k=1)
    assert results[0][0].page_content == "a"

    db.close()


@pytest.mark.skip(reason="sqlite-vec may not support 'inner' distance_metric yet")
def test_distance_strategy_ip():
    """Test Inner Product distance strategy."""
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.IP)

    texts = ["a", "b"]
    embs = [[1.0, 0.0], [0.0, 1.0]]
    db.add_texts(texts, embeddings=embs)

    results = db.similarity_search([1.0, 0.5], k=2)
    assert len(results) == 2

    db.close()


def test_add_texts_batching(empty_db, monkeypatch):
    """Test that large inserts are batched correctly."""
    from tinyvecdb import config

    # Set small batch size for testing
    original_batch_size = config.EMBEDDING_BATCH_SIZE
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 2)

    texts = ["text1", "text2", "text3", "text4", "text5"]
    embs = [[0.1, 0.2]] * 5

    ids = empty_db.add_texts(texts, embeddings=embs)
    assert len(ids) == 5

    # Restore
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", original_batch_size)


def test_metadata_json_filtering(populated_db):
    """Test advanced JSON metadata filtering."""
    # Test exact match
    results = populated_db.similarity_search(
        [0.1, 0.0, 0.0, 0.0], k=5, filter={"likes": 10}
    )
    assert len(results) == 1
    assert results[0][0].metadata["likes"] == 10

    # Test list membership (IN clause)
    results = populated_db.similarity_search(
        [0.1, 0.0, 0.0, 0.0], k=5, filter={"color": ["red", "yellow"]}
    )
    assert len(results) == 2


def test_normalize_l2():
    """Test L2 normalization helper function."""
    from tinyvecdb.core import _normalize_l2

    vec = np.array([3.0, 4.0])
    normalized = _normalize_l2(vec)

    # Should have unit length
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.allclose(normalized, [0.6, 0.8])

    # Zero vector should remain zero
    zero_vec = np.array([0.0, 0.0])
    assert np.allclose(_normalize_l2(zero_vec), zero_vec)


def test_as_langchain(empty_db):
    """Test LangChain integration factory method."""
    lc_store = empty_db.as_langchain()

    from tinyvecdb.integrations.langchain import TinyVecDBVectorStore

    assert isinstance(lc_store, TinyVecDBVectorStore)


def test_as_llama_index(empty_db):
    """Test LlamaIndex integration factory method."""
    li_store = empty_db.as_llama_index()

    from tinyvecdb.integrations.llamaindex import TinyVecDBLlamaStore

    assert isinstance(li_store, TinyVecDBLlamaStore)


def test_dimension_mismatch_on_add(populated_db):
    """Test that adding vectors with different dimensions raises error."""
    with pytest.raises(ValueError, match="Dimension mismatch"):
        populated_db.add_texts(
            ["new text"], embeddings=[[0.1, 0.2]]
        )  # 2D instead of 4D


def test_unsupported_quantization():
    """Test that invalid quantization mode raises error."""
    from tinyvecdb.core import _serialize_vector

    with pytest.raises(ValueError, match="Unsupported quantization"):
        _serialize_vector(np.array([0.1, 0.2]), ("invalid"))  # type: ignore


def test_delete_empty_list(populated_db):
    """Test that deleting empty list is a no-op."""
    original_count = populated_db.conn.execute(
        "SELECT COUNT(*) FROM tinyvec_items"
    ).fetchone()[0]
    populated_db.delete_by_ids([])
    new_count = populated_db.conn.execute(
        "SELECT COUNT(*) FROM tinyvec_items"
    ).fetchone()[0]
    assert original_count == new_count


def test_persist_to_file(tmp_path):
    """Test that data persists to disk."""
    db_path = str(tmp_path / "persist.db")

    # Create and populate
    db1 = VectorDB(db_path)
    db1.add_texts(["test"], embeddings=[[0.1, 0.2]])
    db1.close()

    # Reopen and verify
    db2 = VectorDB(db_path)
    results = db2.similarity_search([0.1, 0.2], k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "test"
    db2.close()


def test_wal_mode(tmp_path):
    """Test that WAL mode is enabled."""
    db_path = str(tmp_path / "wal.db")
    db = VectorDB(db_path)

    journal_mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert journal_mode.lower() == "wal"
    db.close()
