
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from tinyvecdb.embeddings.server import app

client = TestClient(app)

@pytest.mark.integration
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.integration
def test_embeddings_endpoint():
    # Mock the embedding model to avoid loading heavy models during tests
    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        # Setup mock return value
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        payload = {
            "input": ["Hello world", "Another sentence"],
            "model": "test-model"
        }
        
        response = client.post("/v1/embeddings", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["data"][0]["index"] == 0
        assert data["data"][1]["index"] == 1
        assert data["model"] == "test-model"
        # Simple token counting (whitespace split): "Hello world" (2) + "Another sentence" (2) = 4
        assert data["usage"]["total_tokens"] == 4

@pytest.mark.integration
def test_embeddings_invalid_input():
    # Empty input should return empty list with 200 OK
    response = client.post("/v1/embeddings", json={"input": []})
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 0

@pytest.mark.integration
def test_run_server():
    """Test run_server function calls uvicorn."""
    from tinyvecdb.embeddings.server import run_server
    with patch("uvicorn.run") as mock_run:
        run_server(host="1.2.3.4", port=9999)
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert kwargs["host"] == "1.2.3.4"
        assert kwargs["port"] == 9999
