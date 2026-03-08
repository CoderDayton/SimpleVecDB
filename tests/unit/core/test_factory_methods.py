"""Factory method tests."""

from simplevecdb import VectorDB


def test_as_langchain_factory(tmp_path):
    """Test as_langchain factory method."""
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))

    lc = db.as_langchain()
    assert lc is not None
    assert hasattr(lc, "add_texts")
    assert hasattr(lc, "similarity_search")
    db.close()


def test_as_llama_index_factory(tmp_path):
    """Test as_llama_index factory method."""
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))

    li = db.as_llama_index()
    assert li is not None
    assert hasattr(li, "add")
    assert hasattr(li, "query")
    db.close()
