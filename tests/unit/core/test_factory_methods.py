"""Factory method tests."""

import pytest

from simplevecdb import VectorDB

try:
    import langchain_core  # noqa: F401

    _has_langchain = True
except ImportError:
    _has_langchain = False

try:
    import llama_index  # noqa: F401

    _has_llamaindex = True
except ImportError:
    _has_llamaindex = False


@pytest.mark.skipif(not _has_langchain, reason="langchain-core not installed")
def test_as_langchain_factory(tmp_path):
    """Test as_langchain factory method."""
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))

    lc = db.as_langchain()
    assert lc is not None
    assert hasattr(lc, "add_texts")
    assert hasattr(lc, "similarity_search")
    db.close()


def test_as_langchain_factory_missing_dep(tmp_path):
    """Test as_langchain raises ImportError when langchain not installed."""
    if _has_langchain:
        pytest.skip("langchain-core is installed")
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))
    with pytest.raises(ImportError, match="integrations"):
        db.as_langchain()
    db.close()


@pytest.mark.skipif(not _has_llamaindex, reason="llama-index not installed")
def test_as_llama_index_factory(tmp_path):
    """Test as_llama_index factory method."""
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))

    li = db.as_llama_index()
    assert li is not None
    assert hasattr(li, "add")
    assert hasattr(li, "query")
    db.close()


def test_as_llama_index_factory_missing_dep(tmp_path):
    """Test as_llama_index raises ImportError when llama-index not installed."""
    if _has_llamaindex:
        pytest.skip("llama-index is installed")
    db_path = tmp_path / "factory.db"
    db = VectorDB(str(db_path))
    with pytest.raises(ImportError, match="integrations"):
        db.as_llama_index()
    db.close()
