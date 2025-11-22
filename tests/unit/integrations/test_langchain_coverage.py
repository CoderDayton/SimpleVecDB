"""LangChain integration tests."""

from unittest.mock import MagicMock
import numpy as np


def test_langchain_delete(tmp_path):
    """Test LangChain integration delete method."""
    from tinyvecdb.integrations.langchain import TinyVecDBVectorStore

    db_path = tmp_path / "lc_del.db"

    # Mock the embedding function to return valid embeddings
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [np.random.rand(384).tolist()]

    store = TinyVecDBVectorStore(str(db_path), embedding=mock_embedding)

    # Add some dummy data to delete with pre-computed embeddings
    ids = store.add_texts(["text1"])

    # Delete - LangChain's delete returns None
    store.delete(ids)
    # No assertion needed, just verify it doesn't crash
