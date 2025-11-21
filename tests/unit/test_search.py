# tests/unit/test_search.py
import pytest
import numpy as np
from tinyvecdb import VectorDB, DistanceStrategy, Quantization


@pytest.fixture
def db():
    """Fixture providing a populated database with 3D vectors for testing."""
    db = VectorDB(":memory:")
    texts = ["apple", "banana", "orange", "grape"]
    embeddings = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.9, 0.9, 0.9],
            [0.85, 0.85, 0.85],
        ],
        dtype=np.float32,
    )
    metadatas = [
        {"type": "fruit", "likes": 10},
        {"type": "fruit", "likes": 20},
        {"type": "fruit", "likes": 15},
        {"type": "fruit", "likes": 5},
    ]
    db.add_texts(texts, embeddings=embeddings.tolist(), metadatas=metadatas)
    return db


def test_similarity_search_basic(db):
    """Test basic similarity search functionality with cosine distance."""
    # Query vector must match the dimension of stored vectors (3D).
    query = [0.95, 0.95, 0.95]
    results = db.similarity_search(query, k=2)
    
    assert len(results) == 2
    # "grape" ([0.85, 0.85, 0.85]) and "orange" ([0.9, 0.9, 0.9]) are closest to [0.95, 0.95, 0.95]
    # Note: The order depends on exact distance calculations.
    # Both are very close to the query direction (1,1,1).
    
    # Verify that results are returned with scores.
    assert 0 <= results[0][1] < 0.1
    assert results[0][1] <= results[1][1]


def test_similarity_search_filter(db):
    """Test similarity search with metadata filtering."""
    # Query with 3D vector matching the database dimension.
    results = db.similarity_search([0.95] * 3, k=4, filter={"likes": [10, 15]})
    
    assert len(results) == 2  # Should match "apple" (10) and "orange" (15)
    found_texts = {r[0].page_content for r in results}
    assert found_texts == {"apple", "orange"}


def test_recall_gold_standard(populated_db):
    """
    Verify search recall against a brute-force numpy calculation.
    Uses 'populated_db' fixture from conftest.py which has 4D vectors.
    """
    query = np.array([0.95, 0.95, 0.95, 0.95])
    
    # Reconstruct embeddings from the fixture logic for ground truth calculation
    all_embs = np.array(
        [
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.9, 0.9, 0.9, 0.9],
            [0.85, 0.85, 0.85, 0.85],
        ]
    )
    
    # Normalize for cosine similarity comparison
    all_embs = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query)
    
    # Compute ground truth similarities
    sims = np.dot(all_embs, query_norm)
    
    # Dynamically determine expected top-k based on numpy calculation
    # This integrates 'sims' to ensure the test validates against the actual math
    top_k_indices = np.argsort(-sims)[:2]
    all_texts = ["apple is red", "banana is yellow", "orange is orange", "grape is purple"]
    expected = [all_texts[i] for i in top_k_indices]
    
    results = populated_db.similarity_search(query, k=2)
    result_texts = [r[0].page_content for r in results]
    
    # Calculate recall
    intersection = set(result_texts) & set(expected)
    recall = len(intersection) / 2
    assert recall >= 0.9


def test_quantization_search(quant_db):
    """Test search functionality with INT8 quantization."""
    # Generate random 128D vectors
    embs = np.random.randn(10, 128).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    
    quant_db.add_texts(["t"] * 10, embeddings=embs.tolist())
    
    # Search with one of the inserted vectors
    results = quant_db.similarity_search(embs[0], k=1)
    
    # Expect the vector to find itself with very low distance
    assert results[0][1] < 0.05


def test_mmr_diversity():
    """Test Maximal Marginal Relevance (MMR) search for result diversity."""
    db = VectorDB(":memory:")
    # A and B are very similar. C is orthogonal.
    # Query is close to A and B.
    
    texts = ["A", "B", "C"]
    embeddings = [
        [1.0, 0.0, 0.0],    # A
        [0.99, 0.01, 0.0],  # B (very close to A)
        [0.0, 1.0, 0.0],    # C (orthogonal)
    ]
    db.add_texts(texts, embeddings=embeddings)

    query = [1.0, 0.0, 0.0]

    # Standard search should return A and B (most similar)
    res_std = db.similarity_search(query, k=2)
    assert res_std[0][0].page_content == "A"
    assert res_std[1][0].page_content == "B"

    # MMR search should prefer C over B for diversity
    # fetch_k=3 ensures all candidates are considered
    res_mmr = db.max_marginal_relevance_search(query, k=2, fetch_k=3)
    assert res_mmr[0].page_content == "A"
    assert res_mmr[1].page_content == "C"  # B skipped due to redundancy


def test_delete_by_ids():
    """Test that deleting items removes them from search results."""
    db = VectorDB(":memory:")
    ids = db.add_texts(["a", "b"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert len(ids) == 2

    # Delete the first item
    db.delete_by_ids([ids[0]])

    # Search should only find the remaining item
    res = db.similarity_search([1.0, 0.0, 0.0], k=2)
    assert len(res) == 1
    assert res[0][0].page_content == "b"


def test_quantization_int8():
    """Test INT8 quantization accuracy for simple vectors."""
    db = VectorDB(":memory:", quantization=Quantization.INT8)
    texts = ["a", "b"]
    embeddings = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]
    db.add_texts(texts, embeddings=embeddings)

    res = db.similarity_search([0.9, 0.1, 0.0], k=1)
    assert res[0][0].page_content == "a"


def test_quantization_bit():
    """Test BIT quantization (binary) functionality."""
    db = VectorDB(":memory:", quantization=Quantization.BIT)
    texts = ["a", "b"]
    # BIT quantization: >0 is 1, <=0 is 0
    # a: [1, 1, -1] -> 110
    # b: [-1, -1, 1] -> 001
    embeddings = [[0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]
    db.add_texts(texts, embeddings=embeddings)

    # Query matching 'a'
    res = db.similarity_search([0.5, 0.5, -0.5], k=1)
    assert res[0][0].page_content == "a"


def test_distance_l2():
    """Test Euclidean (L2) distance strategy."""
    db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L2)
    texts = ["origin", "far"]
    embeddings = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]
    db.add_texts(texts, embeddings=embeddings)

    # Query close to origin
    res = db.similarity_search([0.1, 0.1, 0.1], k=1)
    assert res[0][0].page_content == "origin"
    # Distance should be small L2 distance
    assert res[0][1] < 1.0
