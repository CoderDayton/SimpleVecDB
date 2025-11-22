"""Embeddings server API tests."""

from unittest.mock import patch
from fastapi.testclient import TestClient

from tinyvecdb.embeddings.server import app


client = TestClient(app)


def test_server_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embeddings_endpoint_single_string():
    """Test /v1/embeddings with single string input."""
    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        response = client.post(
            "/v1/embeddings", json={"input": "hello world", "model": "test-model"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["data"][0]["index"] == 0
        assert data["model"] == "test-model"
        mock_embed.assert_called_once_with(["hello world"])


def test_embeddings_endpoint_list_of_strings():
    """Test /v1/embeddings with list of strings."""
    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        response = client.post("/v1/embeddings", json={"input": ["first", "second"]})

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["data"][0]["embedding"] == [0.1, 0.2]
        assert data["data"][1]["embedding"] == [0.3, 0.4]
        assert data["data"][1]["index"] == 1


def test_embeddings_endpoint_token_array():
    """Test /v1/embeddings with token array (list of ints)."""
    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.5, 0.6]]

        response = client.post("/v1/embeddings", json={"input": [1, 2, 3, 4]})

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        # Should stringify tokens
        mock_embed.assert_called_once()


def test_embeddings_endpoint_error_handling():
    """Test /v1/embeddings error handling."""
    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.side_effect = Exception("Model error")

        response = client.post("/v1/embeddings", json={"input": "test"})

        assert response.status_code == 500
        assert "Embedding failed" in response.json()["detail"]


def test_embeddings_endpoint_usage_tokens():
    """Test /v1/embeddings returns usage tokens."""
    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2]]

        response = client.post("/v1/embeddings", json={"input": "hello world test"})

        assert response.status_code == 200
        data = response.json()
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 3  # 3 words
        assert data["usage"]["total_tokens"] == 3


def test_list_models_endpoint():
    """Test /v1/models endpoint."""
    response = client.get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert "id" in data["data"][0]
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "tinyvecdb"


def test_server_run_with_args():
    """Test server.run_server with custom host/port."""
    from tinyvecdb.embeddings.server import run_server

    with patch("uvicorn.run") as mock_run:
        run_server(host="127.0.0.1", port=9000)
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "127.0.0.1"
        assert call_kwargs["port"] == 9000


def test_server_run_default_config():
    """Test server.run_server uses config defaults."""
    from tinyvecdb.embeddings.server import run_server

    with patch("uvicorn.run") as mock_run:
        with patch("tinyvecdb.embeddings.server.config") as mock_config:
            mock_config.SERVER_HOST = "0.0.0.0"
            mock_config.SERVER_PORT = 8080

            run_server()

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080


def test_server_run_cli_args():
    """Test server.run_server parses CLI arguments."""
    from tinyvecdb.embeddings.server import run_server
    import sys

    with patch("uvicorn.run") as mock_run:
        with patch.object(
            sys, "argv", ["script", "--host", "192.168.1.1", "--port", "7000"]
        ):
            run_server()

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "192.168.1.1"
            assert call_kwargs["port"] == 7000


def test_embeddings_default_model():
    """Test /v1/embeddings uses default model when not specified."""
    from tinyvecdb.embeddings.server import DEFAULT_MODEL

    with patch("tinyvecdb.embeddings.server.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2]]

        response = client.post("/v1/embeddings", json={"input": "test"})

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == DEFAULT_MODEL
