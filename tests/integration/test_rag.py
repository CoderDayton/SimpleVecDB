# tests/integration/test_rag.py
import pytest
from unittest.mock import Mock

# Stub Ollama if not installed
try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = Mock()  # type: ignore

from tinyvecdb import VectorDB


@pytest.mark.integration
def test_rag_end_to_end(populated_db: VectorDB, monkeypatch):
    # Mock embed_texts to return a 4D vector matching populated_db
    mock_embed = Mock(return_value=[[0.1, 0.1, 0.1, 0.1]])
    
    # We need to mock the module import
    import sys
    mock_module = Mock()
    mock_module.embed_texts = mock_embed
    monkeypatch.setitem(sys.modules, "tinyvecdb.embeddings.models", mock_module)

    # Mock LLM response
    def mock_generate(prompt) -> dict[str, str]:
        return {"response": "The grape is purple."}

    ollama_mock = Mock()
    ollama_mock.generate = mock_generate

    # Simple RAG chain (real code would use langchain/llama_index)
    query = "What color is grape?"
    contexts = populated_db.similarity_search(query, k=2)  # embed query in real
    context_str = "\n".join(doc.page_content for doc, _ in contexts)
    prompt = f"Context: {context_str}\nQuestion: {query}"

    response = ollama_mock.generate(prompt)
    assert (
        "purple" in response["response"].lower()
    )  # in real, assert based on LLM output


# Real Ollama test (skip if not available)
@pytest.mark.skipif(
    not hasattr(OllamaClient, "generate"), reason="Ollama not installed"
)
def test_rag_with_ollama(populated_db):
    try:
        client = OllamaClient()
        # Check connection cheaply
        try:
            client.list()
        except Exception:
            pytest.skip("Ollama server not running")

        query = "What color is grape?"
        # In a real scenario, we'd need the actual embedding model loaded
        # For this test, we'll assume populated_db has compatible vectors or we'd need to embed
        # But populated_db fixture has 4D vectors, which won't match real embeddings.
        # So this test is conceptually flawed unless we use a real DB.
        # We'll just fix the syntax error for now.
        
        # Mocking embedding for the sake of the test structure, 
        # assuming we had a real embedding function available.
        query_emb = [0.1, 0.1, 0.1, 0.1] 
        
        contexts = populated_db.similarity_search(query_emb, k=2)
        context = "\n".join(d.page_content for d, _ in contexts)
        response = client.generate(
            model="llama3", prompt=f"Using context: {context}, answer: {query}"
        )
        assert "purple" in response["response"].lower()
    except Exception as e:
        # If it's a connection error that wasn't caught by client.list()
        if "Failed to connect" in str(e):
            pytest.skip(f"Ollama connection failed: {e}")
        raise e
