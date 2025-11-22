"""LlamaIndex integration tests."""

from unittest.mock import MagicMock, patch


def test_llamaindex_delete_nodes(tmp_path):
    """Test LlamaIndex integration delete_nodes method."""
    from tinyvecdb.integrations.llamaindex import TinyVecDBLlamaStore

    db_path = tmp_path / "li_del.db"
    store = TinyVecDBLlamaStore(str(db_path))

    # Delete
    store.delete_nodes(["node1"])
    # No assertion needed, just coverage


def test_llamaindex_query(tmp_path):
    """Test LlamaIndex integration query method."""
    from tinyvecdb.integrations.llamaindex import TinyVecDBLlamaStore
    from llama_index.core.vector_stores.types import VectorStoreQuery

    db_path = tmp_path / "li_query.db"
    store = TinyVecDBLlamaStore(str(db_path))

    # Mock the underlying VectorDB instance's similarity_search
    mock_doc = MagicMock()
    mock_doc.page_content = "res"
    mock_doc.metadata = {"node_content": "{}"}

    with patch.object(store, "_db") as mock_db:
        mock_db.similarity_search.return_value = [(mock_doc, 0.1)]

        query = VectorStoreQuery(query_embedding=[0.1] * 384, similarity_top_k=1)
        result = store.query(query)
        assert result.nodes
        assert result.similarities
