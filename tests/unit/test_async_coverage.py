"""Additional coverage tests for async_core.

Targets missing lines: 131-132, 153-154, 262-263, 286-287, 308-309, 508-509.
"""

from __future__ import annotations

import numpy as np
import pytest

from simplevecdb import AsyncVectorDB


@pytest.fixture
def sample_embeddings():
    """Normalized embeddings for testing."""
    np.random.seed(123)
    emb = np.random.randn(10, 384).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.tolist()


@pytest.fixture
def sample_texts():
    return [f"Document {i} content here" for i in range(10)]


@pytest.mark.asyncio
async def test_async_keyword_search(sample_texts, sample_embeddings):
    """Cover lines 131-132: keyword_search async wrapper."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )

        results = await collection.keyword_search("Document 0", k=5)
        assert len(results) >= 1
        # Results are (Document, score) tuples
        doc, score = results[0]
        assert hasattr(doc, "page_content")


@pytest.mark.asyncio
async def test_async_hybrid_search(sample_texts, sample_embeddings):
    """Cover lines 153-154: hybrid_search async wrapper."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )

        results = await collection.hybrid_search(
            "Document 0",
            k=3,
            query_vector=sample_embeddings[0],
        )
        assert len(results) >= 1
        doc, score = results[0]
        assert hasattr(doc, "page_content")


@pytest.mark.asyncio
async def test_async_auto_tag(sample_texts, sample_embeddings):
    """Cover lines 262-263: auto_tag async wrapper."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")

        np.random.seed(42)
        emb = np.random.randn(20, 384).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        texts = [f"Document about topic {i % 3} content {i}" for i in range(20)]

        await collection.add_texts(texts=texts, embeddings=emb.tolist())

        cluster_result = await collection.cluster(n_clusters=3, random_state=42)
        tags = await collection.auto_tag(cluster_result, method="keywords", n_keywords=3)

        assert isinstance(tags, dict)
        assert len(tags) > 0


@pytest.mark.asyncio
async def test_async_assign_cluster_metadata(sample_texts, sample_embeddings):
    """Cover lines 286-287: assign_cluster_metadata async wrapper."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")

        np.random.seed(42)
        emb = np.random.randn(15, 384).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        texts = [f"doc {i}" for i in range(15)]

        await collection.add_texts(texts=texts, embeddings=emb.tolist())

        cluster_result = await collection.cluster(n_clusters=2, random_state=42)
        count = await collection.assign_cluster_metadata(cluster_result)

        assert count > 0


@pytest.mark.asyncio
async def test_async_get_cluster_members():
    """Cover lines 308-309: get_cluster_members async wrapper."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")

        np.random.seed(42)
        emb = np.random.randn(10, 384).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        texts = [f"doc {i}" for i in range(10)]

        await collection.add_texts(texts=texts, embeddings=emb.tolist())

        cluster_result = await collection.cluster(n_clusters=2, random_state=42)
        await collection.assign_cluster_metadata(cluster_result)

        members = await collection.get_cluster_members(0)
        assert isinstance(members, list)


@pytest.mark.asyncio
async def test_async_vacuum(tmp_path):
    """Cover lines 508-509: vacuum async wrapper."""
    db_path = str(tmp_path / "vacuum_test.db")
    async with AsyncVectorDB(db_path) as db:
        collection = db.collection("test")

        emb = np.random.randn(5, 384).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        await collection.add_texts(
            texts=[f"doc {i}" for i in range(5)],
            embeddings=emb.tolist(),
        )

        # Should not raise
        await db.vacuum(checkpoint_wal=True)
