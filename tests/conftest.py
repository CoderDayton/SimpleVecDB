# tests/conftest.py
import pytest
import numpy as np
from tinyvecdb import VectorDB, Quantization, DistanceStrategy


@pytest.fixture
def empty_db():
    return VectorDB(":memory:")


@pytest.fixture
def populated_db():
    db = VectorDB(
        ":memory:",
        quantization=Quantization.FLOAT,
        distance_strategy=DistanceStrategy.COSINE,
    )
    texts = ["apple is red", "banana is yellow", "orange is orange", "grape is purple"]
    embeddings = np.array(
        [
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.9, 0.9, 0.9, 0.9],
            [0.85, 0.85, 0.85, 0.85],
        ],
        dtype=np.float32,
    )
    metadatas = [
        {"color": "red", "likes": 10},
        {"color": "yellow", "likes": 20},
        {"color": "orange", "likes": 15},
        {"color": "purple", "likes": 5},
    ]
    db.add_texts(texts, embeddings=embeddings.tolist(), metadatas=metadatas)
    return db


@pytest.fixture
def quant_db():
    return VectorDB(":memory:", quantization=Quantization.INT8)


@pytest.fixture
def bit_db():
    return VectorDB(":memory:", quantization=Quantization.BIT)
