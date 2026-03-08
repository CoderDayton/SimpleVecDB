import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from simplevecdb.config import config
from simplevecdb.embeddings import server
from simplevecdb.embeddings.server import UsageMeter, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_usage_meter():
    """Ensure tests start from a clean slate for usage tracking."""
    server.usage_meter = UsageMeter()
    yield


@pytest.mark.integration
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_embeddings_endpoint():
    # Mock the embedding model to avoid loading heavy models during tests
    with patch("simplevecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        payload = {
            "input": ["Hello world", "Another sentence"],
            "model": config.EMBEDDING_MODEL,
        }

        response = client.post("/v1/embeddings", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["data"][0]["index"] == 0
        assert data["data"][1]["index"] == 1
        assert data["model"] == payload["model"]
        # Simple token counting (whitespace split): "Hello world" (2) + "Another sentence" (2) = 4
        assert data["usage"]["total_tokens"] == 4
        _, kwargs = mock_embed.call_args
        assert kwargs["model_id"] == config.EMBEDDING_MODEL


@pytest.mark.integration
def test_embeddings_invalid_input():
    # Empty input should return empty list with 200 OK
    response = client.post("/v1/embeddings", json={"input": []})
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 0


@pytest.mark.integration
def test_embeddings_invalid_model_rejected():
    response = client.post(
        "/v1/embeddings",
        json={"input": "hi", "model": "non-existent-model"},
    )
    assert response.status_code == 400
    assert "allowed" in response.json()["detail"]


@pytest.mark.integration
def test_usage_endpoint_reports_stats():
    with patch("simplevecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1]]
        client.post("/v1/embeddings", json={"input": "hello"})

    usage_response = client.get("/v1/usage")
    assert usage_response.status_code == 200
    data = usage_response.json()["data"]
    assert "anonymous" in data
    assert data["anonymous"]["requests"] == 1


@pytest.mark.integration
def test_run_server():
    """Test run_server function calls uvicorn."""
    from simplevecdb.embeddings.server import run_server

    with patch("uvicorn.run") as mock_run:
        run_server(host="1.2.3.4", port=9999)
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert kwargs["host"] == "1.2.3.4"
        assert kwargs["port"] == 9999
