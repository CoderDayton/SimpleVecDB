
import pytest
from tinyvecdb.integrations.llamaindex import TinyVecDBLlamaStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
from llama_index.core.schema import TextNode

@pytest.mark.integration
def test_llamaindex_add(tmp_path):
    db_path = str(tmp_path / "llama_add.db")
    store = TinyVecDBLlamaStore(db_path=db_path)
    
    node = TextNode(text="LlamaIndex test", embedding=[0.1] * 384)
    store.add([node])
    
    # Verify via DB
    results = store._db.similarity_search([0.1] * 384, k=1)
    assert results[0][0].page_content == "LlamaIndex test"

@pytest.mark.integration
def test_llamaindex_query(tmp_path):
    db_path = str(tmp_path / "llama_query.db")
    store = TinyVecDBLlamaStore(db_path=db_path)
    
    # Add a node to ensure we have something to query
    node = TextNode(text="Query target", embedding=[0.2] * 384)
    store.add([node])
    
    query = VectorStoreQuery(
        query_embedding=[0.2] * 384,
        similarity_top_k=1,
        mode=VectorStoreQueryMode.DEFAULT
    )
    
    result = store.query(query)
    assert len(result.nodes) == 1
    assert result.nodes[0].text == "Query target"
    assert len(result.similarities) == 1
    assert len(result.ids) == 1

@pytest.mark.integration
def test_llamaindex_delete(tmp_path):
    db_path = str(tmp_path / "llama_del.db")
    store = TinyVecDBLlamaStore(db_path=db_path)
    
    node = TextNode(text="To delete", embedding=[0.3] * 384)
    store.add([node])
    # Use node_id, not ref_doc_id, as that's what we map in add()
    node_id = node.node_id
    
    store.delete(node_id)
    
    # Verify deletion
    query = VectorStoreQuery(
        query_embedding=[0.3] * 384,
        similarity_top_k=1
    )
    result = store.query(query)
    # Should not find the deleted node
    if result.nodes:
        assert result.nodes[0].text != "To delete"

@pytest.mark.integration
def test_llamaindex_client(tmp_path):
    db_path = str(tmp_path / "llama_client.db")
    store = TinyVecDBLlamaStore(db_path=db_path)
    assert store.client is not None

@pytest.mark.integration
def test_llamaindex_query_with_filter(tmp_path):
    """Test query with metadata filters."""
    db_path = str(tmp_path / "llama_filter.db")
    store = TinyVecDBLlamaStore(db_path=db_path)
    
    node1 = TextNode(text="Apple", embedding=[0.1] * 384, metadata={"category": "fruit"})
    node2 = TextNode(text="Carrot", embedding=[0.1] * 384, metadata={"category": "veg"})
    store.add([node1, node2])
    
    # Filter for fruit
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
    filters = MetadataFilters(filters=[MetadataFilter(key="category", value="fruit")])
    
    query = VectorStoreQuery(
        query_embedding=[0.1] * 384,
        similarity_top_k=2,
        filters=filters
    )
    
    result = store.query(query)
    assert len(result.nodes) == 1
    assert result.nodes[0].text == "Apple"

@pytest.mark.integration
def test_llamaindex_delete_nonexistent(tmp_path):
    """Test deleting a non-existent node."""
    db_path = str(tmp_path / "llama_del_none.db")
    store = TinyVecDBLlamaStore(db_path=db_path)
    store.delete("non-existent-id")
    # Should not raise error
