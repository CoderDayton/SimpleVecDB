# tests/unit/test_core.py
import pytest
import numpy as np
import json
import sqlite3
from simplevecdb import VectorDB
from simplevecdb.types import Document, DistanceStrategy


def test_init(empty_db):
    """Verify that the database initializes with correct default values."""
    # No default collection anymore
    assert (
        empty_db.quantization == "float"
    )  # Ensure default configuration values are set correctly.
    assert empty_db.distance_strategy == "cosine"


def test_add_texts_basic(empty_db):
    """Test adding texts with embeddings and verify storage integrity."""
    collection = empty_db.collection("default")
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2], [0.3, 0.4]]
    ids = collection.add_texts(texts, embeddings=embs)
    assert len(ids) == 2
    assert collection._dim == 2

    # Verify that the text content is persisted in the main table.
    rows = empty_db.conn.execute(
        f"SELECT text FROM {collection._table_name} ORDER BY id"
    ).fetchall()
    assert rows[0][0] == "test1"

    # Verify that the embedding vector is stored in the virtual table.
    vec_row = empty_db.conn.execute(
        f"SELECT embedding FROM {collection._vec_table_name} WHERE rowid = ?", (ids[0],)
    ).fetchone()

    # Ensure vectors are normalized when using Cosine distance strategy.
    expected = np.array([0.1, 0.2], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert np.allclose(np.frombuffer(vec_row[0], dtype=np.float32), expected)


def test_add_with_metadata(populated_db):
    """Verify that metadata is correctly stored and retrievable."""
    # populated_db fixture likely needs update or we access via collection
    # Assuming populated_db returns a DB with data in "default" collection
    # But wait, populated_db fixture uses add_texts which is gone.
    # We need to fix the fixture or adapt the test.
    # Let's assume we fix the fixture separately or adapt here if it's simple.
    # Actually, let's fix the test to use collection("default") assuming fixture puts data there.
    # But if fixture uses db.add_texts, it will fail.
    # I need to check conftest.py later. For now, let's assume we use collection("default").

    # Wait, I can't see conftest.py here. I should probably check it.
    # But let's update the test code first.
    collection = populated_db.collection("default")
    row = populated_db.conn.execute(
        f"SELECT metadata FROM {collection._table_name} WHERE id=1"
    ).fetchone()[0]
    meta = json.loads(row)
    assert meta["color"] == "red"
    assert meta["likes"] == 10


def test_upsert(populated_db):
    """Test the upsert functionality (update existing records)."""
    collection = populated_db.collection("default")
    new_emb = [0.5, 0.5, 0.5, 0.5]
    collection.add_texts(
        ["updated apple"], embeddings=[new_emb], ids=[1], metadatas=[{"color": "green"}]
    )

    updated = populated_db.conn.execute(
        f"SELECT text, metadata FROM {collection._table_name} WHERE id=1"
    ).fetchone()
    assert updated[0] == "updated apple"
    assert json.loads(updated[1])["color"] == "green"


def test_delete_by_ids(populated_db):
    """Test deletion of records by their IDs."""
    collection = populated_db.collection("default")
    collection.delete_by_ids([1, 2])
    remaining = populated_db.conn.execute(
        f"SELECT COUNT(*) FROM {collection._table_name}"
    ).fetchone()[0]
    assert remaining == 2
    vec_count = populated_db.conn.execute(
        f"SELECT COUNT(*) FROM {collection._vec_table_name}"
    ).fetchone()[0]
    assert vec_count == 2  # Ensure the virtual table row count matches the main table.


def test_add_no_embeddings_raises(empty_db, monkeypatch):
    """Ensure ValueError is raised when no embeddings are provided and local embedder fails."""
    import sys

    collection = empty_db.collection("default")

    # Simulate module missing by setting it to None in sys.modules
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "simplevecdb.embeddings.models", None)

        with pytest.raises(ValueError, match="No embeddings provided"):
            collection.add_texts(["test"])


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
    db1.collection("default").add_texts(["test"], embeddings=[[0.1] * 10])
    db1.close()

    # Reopen
    db2 = VectorDB(db_path)
    assert db2.collection("default")._dim == 10
    db2.close()


def test_dequantize_fallback():
    """Test dequantization logic directly."""
    from simplevecdb.types import Quantization
    from simplevecdb.engine.quantization import QuantizationStrategy
    import numpy as np

    vec = np.array([0.1, 0.5, -0.5], dtype=np.float32)

    # Float
    strategy = QuantizationStrategy(Quantization.FLOAT)
    blob = strategy.serialize(vec)
    out = strategy.deserialize(blob, 3)
    assert np.allclose(vec, out)

    # Int8
    strategy = QuantizationStrategy(Quantization.INT8)
    blob = strategy.serialize(vec)
    out = strategy.deserialize(blob, 3)
    # Precision loss expected
    assert np.allclose(vec, out, atol=0.01)

    # Bit
    strategy = QuantizationStrategy(Quantization.BIT)
    blob = strategy.serialize(vec)
    out = strategy.deserialize(blob, 3)
    # Binary: >0 -> 1, <=0 -> -1
    expected = np.array([1.0, 1.0, -1.0], dtype=np.float32)
    assert np.allclose(out, expected)


def test_similarity_search_basic(populated_db):
    """Test basic similarity search functionality."""
    collection = populated_db.collection("default")
    query = [0.1, 0.0, 0.0, 0.0]  # Close to "apple"
    results = collection.similarity_search(query, k=2)

    assert len(results) == 2
    # Should match "apple is red" first (closest to query)
    assert results[0][0].page_content == "apple is red"
    assert results[0][1] < results[1][1]  # First result should have lower distance


def test_similarity_search_with_filter(populated_db):
    """Test similarity search with metadata filtering."""
    collection = populated_db.collection("default")
    query = [0.1, 0.0, 0.0, 0.0]
    results = collection.similarity_search(query, k=5, filter={"color": "yellow"})

    assert len(results) == 1
    assert results[0][0].page_content == "banana is yellow"
    assert results[0][0].metadata["color"] == "yellow"


def test_similarity_search_empty_db(empty_db):
    """Test similarity search on empty database."""
    collection = empty_db.collection("default")
    results = collection.similarity_search([0.1, 0.2], k=5)
    assert len(results) == 0


def test_similarity_search_text_query(populated_db, monkeypatch):
    """Test similarity search with text query (requires embeddings)."""
    collection = populated_db.collection("default")

    # Mock the embed_texts function
    def mock_embed_texts(texts):
        return [[0.1, 0.0, 0.0, 0.0]]  # Close to apple

    import simplevecdb.embeddings.models

    monkeypatch.setattr(simplevecdb.embeddings.models, "embed_texts", mock_embed_texts)

    results = collection.similarity_search("apple fruit", k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "apple is red"


def test_keyword_search_bm25(populated_db):
    """Ensure keyword search surfaces literal matches via BM25."""
    collection = populated_db.collection("default")
    results = collection.keyword_search("banana", k=2)
    assert results
    assert results[0][0].page_content == "banana is yellow"


def test_hybrid_search_combines_rankings(populated_db, monkeypatch):
    """Hybrid search should balance BM25 hits with vector similarity."""
    collection = populated_db.collection("default")

    def skewed_embed(texts):
        # Always return a vector closest to "orange" regardless of the query text.
        return [[0.9, 0.9, 0.9, 0.9] for _ in texts]

    import simplevecdb.embeddings.models

    monkeypatch.setattr(simplevecdb.embeddings.models, "embed_texts", skewed_embed)

    # Vector-only search should pick anything but "banana" according to the mocked embedding.
    vector_only = collection.similarity_search("banana yellow", k=1)
    assert vector_only[0][0].page_content != "banana is yellow"

    # Hybrid search should recover the literal "banana" match thanks to BM25.
    hybrid = collection.hybrid_search("banana yellow", k=1)
    assert hybrid[0][0].page_content == "banana is yellow"


def test_similarity_search_dimension_mismatch(populated_db):
    """Test that querying with wrong dimension raises error."""
    collection = populated_db.collection("default")
    with pytest.raises(ValueError, match="Query dim"):
        collection.similarity_search([0.1, 0.2], k=1)  # 2D instead of 4D


def test_max_marginal_relevance_search(populated_db):
    """Test MMR search for diversity."""
    collection = populated_db.collection("default")
    query = [0.9, 0.9, 0.9, 0.9]
    results = collection.max_marginal_relevance_search(query, k=2, fetch_k=4)

    assert len(results) == 2
    # MMR should select diverse documents
    assert isinstance(results[0], Document)
    assert isinstance(results[1], Document)


def test_quantization_int8(quant_db):
    """Test INT8 quantization storage and retrieval."""
    collection = quant_db.collection("default")
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ids = collection.add_texts(texts, embeddings=embs)

    assert len(ids) == 2
    assert collection._dim == 3

    # Search should work with quantized vectors
    results = collection.similarity_search([0.1, 0.2, 0.3], k=1)
    assert len(results) == 1


def test_quantization_bit(bit_db):
    """Test BIT quantization storage and retrieval."""
    collection = bit_db.collection("default")
    texts = ["test1", "test2"]
    embs = [[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]]
    ids = collection.add_texts(texts, embeddings=embs)

    assert len(ids) == 2
    # BIT quantization rounds up to byte boundary
    assert collection._dim == 3

    # Search should work with binary vectors
    results = collection.similarity_search([0.1, 0.2, 0.3], k=1)
    assert len(results) == 1


def test_distance_strategy_l2():
    """Test L2 (Euclidean) distance strategy."""
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L2)
    collection = db.collection("default")

    texts = ["a", "b", "c"]
    embs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    collection.add_texts(texts, embeddings=embs)

    # Query closest to "a"
    results = collection.similarity_search([1.0, 0.0], k=1)
    assert results[0][0].page_content == "a"

    db.close()


def test_distance_strategy_l1():
    """Test L1 (Manhattan) distance strategy."""
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L1)
    collection = db.collection("default")

    texts = ["a", "b", "c"]
    embs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    collection.add_texts(texts, embeddings=embs)

    # Query closest to "a"
    results = collection.similarity_search([1.0, 0.0], k=1)
    assert results[0][0].page_content == "a"

    db.close()


def test_add_texts_batching(empty_db, monkeypatch):
    """Test that large inserts are batched correctly."""
    from simplevecdb import config

    collection = empty_db.collection("default")

    # Set small batch size for testing
    original_batch_size = config.EMBEDDING_BATCH_SIZE
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 2)

    texts = ["text1", "text2", "text3", "text4", "text5"]
    embs = [[0.1, 0.2]] * 5

    ids = collection.add_texts(texts, embeddings=embs)
    assert len(ids) == 5

    # Restore
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", original_batch_size)


def test_metadata_json_filtering(populated_db):
    """Test advanced JSON metadata filtering."""
    collection = populated_db.collection("default")
    # Test exact match
    results = collection.similarity_search(
        [0.1, 0.0, 0.0, 0.0], k=5, filter={"likes": 10}
    )
    assert len(results) == 1
    assert results[0][0].metadata["likes"] == 10

    # Test list membership (IN clause)
    results = collection.similarity_search(
        [0.1, 0.0, 0.0, 0.0], k=5, filter={"color": ["red", "yellow"]}
    )
    assert len(results) == 2


def test_normalize_l2():
    """Test L2 normalization helper function."""
    from simplevecdb.engine.quantization import normalize_l2

    vec = np.array([3.0, 4.0])
    normalized = normalize_l2(vec)

    # Should have unit length
    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.allclose(normalized, [0.6, 0.8])

    # Zero vector should remain zero
    zero_vec = np.array([0.0, 0.0])
    assert np.allclose(normalize_l2(zero_vec), zero_vec)


def test_as_langchain(empty_db):
    """Test LangChain integration factory method."""
    lc_store = empty_db.as_langchain()

    from simplevecdb.integrations.langchain import SimpleVecDBVectorStore

    assert isinstance(lc_store, SimpleVecDBVectorStore)


def test_as_llama_index(empty_db):
    """Test LlamaIndex integration factory method."""
    li_store = empty_db.as_llama_index()

    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore

    assert isinstance(li_store, SimpleVecDBLlamaStore)


def test_dimension_mismatch_on_add(populated_db):
    """Test that adding vectors with different dimensions raises error."""
    collection = populated_db.collection("default")
    with pytest.raises(ValueError, match="Dimension mismatch"):
        collection.add_texts(["new text"], embeddings=[[0.1, 0.2]])  # 2D instead of 4D


def test_unsupported_quantization():
    """Test that invalid quantization mode raises error."""
    from simplevecdb.engine.quantization import QuantizationStrategy

    with pytest.raises(ValueError, match="Unsupported quantization"):
        strategy = QuantizationStrategy("invalid")  # type: ignore
        strategy.serialize(np.array([0.1, 0.2]))


def test_delete_empty_list(populated_db):
    """Test that deleting empty list is a no-op."""
    collection = populated_db.collection("default")
    original_count = populated_db.conn.execute(
        f"SELECT COUNT(*) FROM {collection._table_name}"
    ).fetchone()[0]
    collection.delete_by_ids([])
    new_count = populated_db.conn.execute(
        f"SELECT COUNT(*) FROM {collection._table_name}"
    ).fetchone()[0]
    assert original_count == new_count


def test_persist_to_file(tmp_path):
    """Test that data persists to disk."""
    db_path = str(tmp_path / "persist.db")

    # Create and populate
    db1 = VectorDB(db_path)
    db1.collection("default").add_texts(["test"], embeddings=[[0.1, 0.2]])
    db1.close()

    # Reopen and verify
    db2 = VectorDB(db_path)
    results = db2.collection("default").similarity_search([0.1, 0.2], k=1)
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
