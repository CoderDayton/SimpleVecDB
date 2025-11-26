"""Tests for async API wrappers."""

import pytest
import numpy as np

from simplevecdb import AsyncVectorDB, Quantization


@pytest.fixture
def sample_embeddings():
    """Generate sample normalized embeddings."""
    np.random.seed(42)
    emb = np.random.randn(10, 384).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb.tolist()


@pytest.fixture
def sample_texts():
    """Sample text documents."""
    return [f"Document number {i} with some content" for i in range(10)]


@pytest.mark.asyncio
async def test_async_db_creation():
    """Test creating an async database."""
    db = AsyncVectorDB(":memory:")
    collection = db.collection("test")
    assert collection.name == "test"
    await db.close()


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        assert collection.name == "test"


@pytest.mark.asyncio
async def test_async_add_texts(sample_texts, sample_embeddings):
    """Test adding texts asynchronously."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        ids = await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )
        assert len(ids) == 10
        assert all(isinstance(id_, int) for id_ in ids)


@pytest.mark.asyncio
async def test_async_similarity_search(sample_texts, sample_embeddings):
    """Test similarity search asynchronously."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )

        # Search with the first embedding
        results = await collection.similarity_search(
            sample_embeddings[0],
            k=3,
        )

        assert len(results) == 3
        # First result should be exact match (lowest distance)
        doc, score = results[0]
        assert doc.page_content == sample_texts[0]
        assert score < 0.01  # Very close to 0 for exact match


@pytest.mark.asyncio
async def test_async_similarity_search_with_filter(sample_texts, sample_embeddings):
    """Test similarity search with metadata filter."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")

        # Add with metadata
        metadatas = [{"category": "even" if i % 2 == 0 else "odd"} for i in range(10)]
        await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
            metadatas=metadatas,
        )

        # Search only even category
        results = await collection.similarity_search(
            sample_embeddings[0],
            k=10,
            filter={"category": "even"},
        )

        assert len(results) == 5
        for doc, _ in results:
            assert doc.metadata["category"] == "even"


@pytest.mark.asyncio
async def test_async_delete_by_ids(sample_texts, sample_embeddings):
    """Test deleting documents by ID."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        ids = await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )

        # Delete first 3 documents
        await collection.delete_by_ids(ids[:3])

        # Search should not return deleted docs
        results = await collection.similarity_search(
            sample_embeddings[0],
            k=10,
        )

        assert len(results) == 7
        for doc, _ in results:
            assert doc.page_content not in sample_texts[:3]


@pytest.mark.asyncio
async def test_async_remove_texts(sample_texts, sample_embeddings):
    """Test removing texts by content."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )

        # Remove specific texts
        removed = await collection.remove_texts(texts=sample_texts[:2])
        assert removed == 2


@pytest.mark.asyncio
async def test_async_mmr_search(sample_texts, sample_embeddings):
    """Test max marginal relevance search."""
    async with AsyncVectorDB(":memory:") as db:
        collection = db.collection("test")
        await collection.add_texts(
            texts=sample_texts,
            embeddings=sample_embeddings,
        )

        results = await collection.max_marginal_relevance_search(
            sample_embeddings[0],
            k=3,
            fetch_k=10,
        )

        assert len(results) == 3
        # Results are Documents, not tuples
        assert all(hasattr(doc, "page_content") for doc in results)


@pytest.mark.asyncio
async def test_async_concurrent_inserts(tmp_path):
    """Test concurrent insertions don't cause issues."""
    db_path = str(tmp_path / "concurrent.db")
    async with AsyncVectorDB(db_path, max_workers=4) as db:
        collection = db.collection("test")

        # Insert initial doc to set dimension
        init_emb = np.random.randn(384).astype(np.float32)
        init_emb /= np.linalg.norm(init_emb)
        await collection.add_texts(
            texts=["init_doc"],
            embeddings=[init_emb.tolist()],
        )

        # Create tasks for concurrent inserts (sequential to avoid SQLite locks)
        results = []
        for i in range(20):
            emb = np.random.randn(384).astype(np.float32)
            emb /= np.linalg.norm(emb)
            ids = await collection.add_texts(
                texts=[f"concurrent_doc_{i}"],
                embeddings=[emb.tolist()],
            )
            results.append(ids)

        # All should succeed
        assert len(results) == 20
        assert all(len(ids) == 1 for ids in results)

        # Verify all docs searchable
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query)
        search_results = await collection.similarity_search(query.tolist(), k=100)
        assert len(search_results) == 21  # 20 + 1 init doc


@pytest.mark.asyncio
async def test_async_multiple_collections():
    """Test multiple collections in same database."""
    async with AsyncVectorDB(":memory:") as db:
        col1 = db.collection("collection1")
        col2 = db.collection("collection2")

        emb = [0.1] * 384

        await col1.add_texts(["doc in col1"], embeddings=[emb])
        await col2.add_texts(["doc in col2"], embeddings=[emb])

        results1 = await col1.similarity_search(emb, k=10)
        results2 = await col2.similarity_search(emb, k=10)

        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0][0].page_content == "doc in col1"
        assert results2[0][0].page_content == "doc in col2"


@pytest.mark.asyncio
async def test_async_quantization():
    """Test async with different quantization levels."""
    for quant in [Quantization.FLOAT, Quantization.INT8, Quantization.BIT]:
        async with AsyncVectorDB(":memory:", quantization=quant) as db:
            collection = db.collection("test")

            emb = np.random.randn(384).astype(np.float32)
            emb /= np.linalg.norm(emb)

            ids = await collection.add_texts(
                texts=["test doc"],
                embeddings=[emb.tolist()],
            )
            assert len(ids) == 1

            results = await collection.similarity_search(emb.tolist(), k=1)
            assert len(results) == 1
