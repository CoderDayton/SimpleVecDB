
import pytest
from simplevecdb import VectorDB, Quantization

def test_multi_collection_basic():
    db = VectorDB(":memory:")
    
    # Create two collections
    c1 = db.collection("c1", quantization=Quantization.FLOAT)
    c2 = db.collection("c2", quantization=Quantization.FLOAT)
    
    # Add data
    c1.add_texts(["doc1"], embeddings=[[0.1, 0.2]])
    c2.add_texts(["doc2"], embeddings=[[0.9, 0.8]])
    
    # Search c1
    res1 = c1.similarity_search([0.1, 0.2], k=1)
    assert len(res1) == 1
    assert res1[0][0].page_content == "doc1"
    
    # Search c2
    res2 = c2.similarity_search([0.9, 0.8], k=1)
    assert len(res2) == 1
    assert res2[0][0].page_content == "doc2"
    
    # Ensure isolation
    res1_cross = c1.similarity_search([0.9, 0.8], k=1)
    # Should still find doc1 because it's the only doc in c1, but distance should be large
    assert res1_cross[0][0].page_content == "doc1"
    
    # Check isolation by count (if we had more docs)
    # Let's add another doc to c1
    c1.add_texts(["doc1b"], embeddings=[[0.11, 0.21]])
    assert len(c1.similarity_search([0.1, 0.2], k=10)) == 2
    assert len(c2.similarity_search([0.9, 0.8], k=10)) == 1

def test_collection_persistence(tmp_path):
    db_path = tmp_path / "test.db"
    db = VectorDB(db_path)
    
    c1 = db.collection("users")
    c1.add_texts(["alice"], embeddings=[[1.0, 0.0]])
    
    db.close()
    
    # Reopen
    db2 = VectorDB(db_path)
    c1_reopened = db2.collection("users")
    res = c1_reopened.similarity_search([1.0, 0.0], k=1)
    assert res[0][0].page_content == "alice"

def test_invalid_collection_name():
    db = VectorDB(":memory:")
    with pytest.raises(ValueError):
        db.collection("invalid name with spaces")
    with pytest.raises(ValueError):
        db.collection("drop table users;")

def test_default_collection_compat():
    db = VectorDB(":memory:")
    # Explicitly use collection("default") as add_texts is removed from VectorDB
    db.collection("default").add_texts(["default_doc"], embeddings=[[0.5, 0.5]])
    
    # Access via default collection explicitly
    def_col = db.collection("default")
    res = def_col.similarity_search([0.5, 0.5], k=1)
    assert res[0][0].page_content == "default_doc"
