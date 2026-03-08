"""LlamaIndex integration coverage helpers."""

from unittest.mock import MagicMock, patch

import pytest


def test_llamaindex_delete_nodes(tmp_path):
    """Test LlamaIndex integration delete_nodes method."""
    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore

    db_path = tmp_path / "li_del.db"
    store = SimpleVecDBLlamaStore(str(db_path))

    # Delete
    store.delete_nodes(["node1"])
    # No assertion needed, just coverage


def test_llamaindex_query(tmp_path):
    """Test LlamaIndex integration query method."""
    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore
    from llama_index.core.vector_stores.types import VectorStoreQuery

    db_path = tmp_path / "li_query.db"
    store = SimpleVecDBLlamaStore(str(db_path))

    # Mock the underlying VectorDB instance's similarity_search
    mock_doc = MagicMock()
    mock_doc.page_content = "res"
    mock_doc.metadata = {"node_content": "{}"}

    with patch.object(store, "_collection") as mock_col:
        mock_col.similarity_search.return_value = [(mock_doc, 0.1)]

        query = VectorStoreQuery(query_embedding=[0.1] * 384, similarity_top_k=1)
        result = store.query(query)
        assert result.nodes
        assert result.similarities


def test_llamaindex_store_text_property(tmp_path):
    """store_text property mirrors stores_text attribute."""
    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore

    store = SimpleVecDBLlamaStore(str(tmp_path / "li_prop.db"))
    assert store.store_text is True


def test_llamaindex_add_handles_missing_embeddings(tmp_path):
    """Ensure add() gracefully handles nodes with partial embeddings."""
    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore

    store = SimpleVecDBLlamaStore(str(tmp_path / "li_add.db"))

    node_with_embedding = MagicMock()
    node_with_embedding.get_content.return_value = "text1"
    node_with_embedding.metadata = {"idx": 1}
    node_with_embedding.embedding = [0.1, 0.2]
    node_with_embedding.node_id = "node-1"

    node_without_embedding = MagicMock()
    node_without_embedding.get_content.return_value = "text2"
    node_without_embedding.metadata = {"idx": 2}
    node_without_embedding.embedding = None
    node_without_embedding.node_id = "node-2"

    with patch.object(store, "_collection") as mock_col:
        mock_col.add_texts.return_value = [1, 2]
        node_ids = store.add([node_with_embedding, node_without_embedding])

    assert node_ids == ["node-1", "node-2"]
    args, _ = mock_col.add_texts.call_args
    assert args[2] is None  # embeddings should be None when any node lacks embeddings


def test_llamaindex_delete_nodes_invokes_delete(tmp_path):
    """delete_nodes should iterate over provided IDs and call delete()."""
    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore

    store = SimpleVecDBLlamaStore(str(tmp_path / "li_delete_nodes.db"))

    with patch.object(SimpleVecDBLlamaStore, "delete", autospec=True) as mock_delete:
        store.delete_nodes(["node-a", "node-b"])

    assert mock_delete.call_count == 2
    mock_delete.assert_any_call(store, "node-a")
    mock_delete.assert_any_call(store, "node-b")


def test_llamaindex_query_requires_input(tmp_path):
    """query() should raise when both embedding and string inputs are missing."""
    from simplevecdb.integrations.llamaindex import SimpleVecDBLlamaStore
    from llama_index.core.vector_stores.types import VectorStoreQuery

    store = SimpleVecDBLlamaStore(str(tmp_path / "li_query_err.db"))
    query = VectorStoreQuery(query_embedding=None, query_str=None, similarity_top_k=1)

    with pytest.raises(ValueError):
        store.query(query)
