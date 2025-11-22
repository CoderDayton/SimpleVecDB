
import pytest
from unittest.mock import MagicMock
from tinyvecdb.integrations.langchain import TinyVecDBVectorStore
from langchain_core.documents import Document

@pytest.mark.integration
def test_langchain_add_texts(tmp_path):
    db_path = str(tmp_path / "lc_test.db")
    # Create store wrapper around new db
    # We need to mock the embedding object passed to __init__
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1] * 384]
    mock_embedding.embed_query.return_value = [0.1] * 384
    
    store = TinyVecDBVectorStore(db_path=db_path, embedding=mock_embedding)
    
    texts = ["LangChain test"]
    metadatas = [{"source": "lc"}]
    
    ids = store.add_texts(texts, metadatas)
    assert len(ids) == 1
    
    # Verify it's in the DB
    # Access internal DB to verify
    results = store._db.similarity_search([0.1] * 384, k=1)
    assert results[0][0].page_content == "LangChain test"

@pytest.mark.integration
def test_langchain_similarity_search(tmp_path):
    db_path = str(tmp_path / "lc_search.db")
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 384
    mock_embedding.embed_documents.return_value = [[0.1] * 384]
    
    store = TinyVecDBVectorStore(db_path=db_path, embedding=mock_embedding)
    
    # Add data
    store.add_texts(["Query target"])
    
    docs = store.similarity_search("query", k=1)
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Query target"

@pytest.mark.integration
def test_langchain_from_texts(tmp_path):
    db_path = str(tmp_path / "langchain.db")
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1] * 10]
    
    store = TinyVecDBVectorStore.from_texts(
        texts=["Hello"],
        embedding=mock_embedding,
        metadatas=[{"id": 1}],
        db_path=db_path
    )
    
    assert isinstance(store, TinyVecDBVectorStore)
    # Verify DB was created
    assert store._db._dim == 10

@pytest.mark.integration
def test_langchain_delete(tmp_path):
    db_path = str(tmp_path / "lc_del.db")
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1] * 384]
    store = TinyVecDBVectorStore(db_path=db_path, embedding=mock_embedding)
    
    ids = store.add_texts(["To delete"])
    assert len(ids) == 1
    
    store.delete(ids)
    
    # Verify deletion
    results = store._db.similarity_search([0.1] * 384, k=1)
    # Should be empty or not find the doc
    assert len(results) == 0

@pytest.mark.integration
def test_langchain_async_methods(tmp_path):
    """Test async wrappers."""
    import asyncio
    
    async def _run_test():
        db_path = str(tmp_path / "lc_async.db")
        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1] * 384]
        mock_embedding.embed_query.return_value = [0.1] * 384
        
        store = TinyVecDBVectorStore(db_path=db_path, embedding=mock_embedding)
        
        # aadd_texts
        ids = await store.aadd_texts(["Async test"])
        assert len(ids) == 1
        
        # asimilarity_search
        docs = await store.asimilarity_search("query", k=1)
        assert len(docs) == 1
        
        # amax_marginal_relevance_search
        docs_mmr = await store.amax_marginal_relevance_search("query", k=1)
        assert len(docs_mmr) == 1

    asyncio.run(_run_test())

@pytest.mark.integration
def test_langchain_mmr(tmp_path):
    """Test MMR wrapper."""
    db_path = str(tmp_path / "lc_mmr.db")
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = [0.1] * 384
    mock_embedding.embed_documents.return_value = [[0.1] * 384]
    
    store = TinyVecDBVectorStore(db_path=db_path, embedding=mock_embedding)
    store.add_texts(["MMR test"])
    
    docs = store.max_marginal_relevance_search("query", k=1)
    assert len(docs) == 1 
