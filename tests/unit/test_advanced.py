import pytest
import numpy as np
from tinyvecdb import VectorDB, Quantization


@pytest.fixture
def db_quant():
    return VectorDB(":memory:", quantization=Quantization.INT8)


@pytest.fixture
def db():
    return VectorDB(":memory:")


def test_quantization_storage(db_quant):
    emb = np.random.randn(100, 128).tolist()
    db_quant.add_texts(["t"] * 100, embeddings=emb)
    # Manual check serialized is int8
    blob = db_quant.conn.execute(
        "SELECT embedding FROM vec_index LIMIT 1"
    ).fetchone()[0]
    assert len(blob) == 128  # 1 byte/dim


def test_filter_advanced(db):
    db.add_texts(["apple", "banana"], metadatas=[{"likes": 10}, {"likes": 20}])
    results = db.similarity_search([0.1] * 384, k=2, filter={"likes": [10, 20]})
    assert len(results) == 2
    results_like = db.similarity_search([0.1] * 384, filter={"likes": 15})  # no match
    assert len(results_like) == 0
