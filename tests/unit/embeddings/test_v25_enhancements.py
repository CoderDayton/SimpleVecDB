"""Tests for simplevecdb 2.5.0 embeddings server enhancements (#1-#10)."""

import argparse
import signal
from typing import Any

import pytest
from unittest.mock import patch, MagicMock, ANY

from fastapi.testclient import TestClient

import simplevecdb
from simplevecdb.embeddings.server import (
    app,
    _normalize_input,
    _build_cli_parser,
    _server_version,
    ModelRegistry,
)
from simplevecdb.embeddings import server

client = TestClient(app)


@pytest.fixture(autouse=True)
def _unlocked_registry():
    """Allow arbitrary test models in unit tests."""
    original = server.registry
    server.registry = ModelRegistry({"default": "test-default"}, allow_unlisted=True)
    yield
    server.registry = original


# ---------------------------------------------------------------------------
# 1. Graceful shutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """run_server() creates uvicorn.Server with timeout_graceful_shutdown=10."""

    @patch("simplevecdb.embeddings.server.get_embedder")
    @patch("simplevecdb.embeddings.server.uvicorn.Server")
    @patch("simplevecdb.embeddings.server.uvicorn.Config")
    def test_timeout_graceful_shutdown_is_set(
        self, mock_config_cls, mock_server_cls, mock_get_embedder
    ):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        from simplevecdb.embeddings.server import run_server

        run_server(host="127.0.0.1", port=9000)

        mock_config_cls.assert_called_once_with(
            app,
            host="127.0.0.1",
            port=9000,
            log_level="info",
            timeout_graceful_shutdown=10,
        )

    @patch("simplevecdb.embeddings.server.get_embedder")
    @patch("simplevecdb.embeddings.server.uvicorn.Server")
    @patch("simplevecdb.embeddings.server.uvicorn.Config")
    def test_signal_handlers_installed(
        self, mock_config_cls, mock_server_cls, mock_get_embedder
    ):
        mock_server = MagicMock()
        mock_server_cls.return_value = mock_server

        captured_handlers: dict[int, Any] = {}
        original_signal = signal.signal

        def fake_signal(signum, handler):
            captured_handlers[signum] = handler
            return original_signal(signum, signal.SIG_DFL)

        with patch("simplevecdb.embeddings.server.signal.signal", side_effect=fake_signal):
            from simplevecdb.embeddings.server import run_server

            run_server(host="127.0.0.1", port=9000)

        assert signal.SIGINT in captured_handlers
        assert signal.SIGTERM in captured_handlers


# ---------------------------------------------------------------------------
# 2. Async executor offload
# ---------------------------------------------------------------------------


class TestAsyncExecutorOffload:
    """create_embeddings runs embed_texts via loop.run_in_executor."""

    @patch("simplevecdb.embeddings.server.embed_texts")
    def test_embed_texts_called_via_endpoint(self, mock_embed):
        mock_embed.return_value = [[0.1, 0.2, 0.3]]

        response = client.post(
            "/v1/embeddings", json={"input": "test text", "model": "test-model"}
        )

        assert response.status_code == 200
        mock_embed.assert_called_once_with(
            ["test text"], model_id="test-model", batch_size=ANY
        )
        data = response.json()
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# 3. Model warm-up
# ---------------------------------------------------------------------------


class TestModelWarmUp:
    """run_server() calls get_embedder before starting unless --no-warmup."""

    @patch("simplevecdb.embeddings.server.get_embedder")
    @patch("simplevecdb.embeddings.server.uvicorn.Server")
    @patch("simplevecdb.embeddings.server.uvicorn.Config")
    def test_warmup_calls_get_embedder(
        self, mock_config_cls, mock_server_cls, mock_get_embedder
    ):
        mock_server_cls.return_value = MagicMock()

        from simplevecdb.embeddings.server import run_server
        from simplevecdb.config import config

        run_server(host="127.0.0.1", port=9000)

        mock_get_embedder.assert_called_once_with(config.EMBEDDING_MODEL)

    @patch("simplevecdb.embeddings.server.get_embedder")
    @patch("simplevecdb.embeddings.server.uvicorn.Server")
    @patch("simplevecdb.embeddings.server.uvicorn.Config")
    def test_no_warmup_skips_get_embedder(
        self, mock_config_cls, mock_server_cls, mock_get_embedder
    ):
        """Simulate --no-warmup by calling run_server via CLI path."""
        mock_server_cls.return_value = MagicMock()

        with patch(
            "simplevecdb.embeddings.server._build_cli_parser"
        ) as mock_parser_fn:
            mock_args = argparse.Namespace(host=None, port=None, no_warmup=True)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_parser_fn.return_value = mock_parser

            from simplevecdb.embeddings.server import run_server

            # host=None, port=None triggers CLI parsing path
            run_server(host=None, port=None)

        mock_get_embedder.assert_not_called()


# ---------------------------------------------------------------------------
# 4. CORS middleware
# ---------------------------------------------------------------------------


class TestCORSMiddleware:
    """CORS is opt-in via EMBEDDING_SERVER_CORS_ORIGINS.

    The 2.6.0 default is no CORS — operators that need it must set the
    env var explicitly. This test class verifies the safe default rather
    than the prior behavior where CORS was always enabled with allow_credentials.
    """

    def test_options_preflight_no_cors_by_default(self):
        response = client.options(
            "/v1/embeddings",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        # With CORS disabled (default), the access-control-allow-origin
        # header is absent. Configure EMBEDDING_SERVER_CORS_ORIGINS to opt
        # in; the wildcard form drops allow_credentials automatically.
        assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """_validate_texts rejects empty strings (422) and texts > 100k chars (413)."""

    @patch("simplevecdb.embeddings.server.embed_texts")
    def test_empty_string_returns_422(self, mock_embed):
        response = client.post(
            "/v1/embeddings", json={"input": "", "model": "test-model"}
        )
        assert response.status_code == 422

    @patch("simplevecdb.embeddings.server.embed_texts")
    def test_whitespace_only_returns_422(self, mock_embed):
        response = client.post(
            "/v1/embeddings", json={"input": "   ", "model": "test-model"}
        )
        assert response.status_code == 422

    @patch("simplevecdb.embeddings.server.embed_texts")
    def test_list_with_empty_string_returns_422(self, mock_embed):
        response = client.post(
            "/v1/embeddings",
            json={"input": ["hello", ""], "model": "test-model"},
        )
        assert response.status_code == 422

    @patch("simplevecdb.embeddings.server.embed_texts")
    def test_text_exceeding_100k_chars_returns_413(self, mock_embed):
        long_text = "a" * 100_001
        response = client.post(
            "/v1/embeddings", json={"input": long_text, "model": "test-model"}
        )
        assert response.status_code == 413
        assert "100000" in response.json()["detail"]

    @patch("simplevecdb.embeddings.server.embed_texts")
    def test_text_at_100k_chars_succeeds(self, mock_embed):
        mock_embed.return_value = [[0.1]]
        text = "a" * 100_000
        response = client.post(
            "/v1/embeddings", json={"input": text, "model": "test-model"}
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 6. argparse CLI
# ---------------------------------------------------------------------------


class TestArgparseCLI:
    """_build_cli_parser() returns a parser with --host, --port, --no-warmup."""

    def test_parse_known_args_defaults(self):
        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.host is None
        assert args.port is None
        assert args.no_warmup is False

    def test_parse_explicit_values(self):
        parser = _build_cli_parser()
        args = parser.parse_args(["--host", "0.0.0.0", "--port", "8080", "--no-warmup"])
        assert args.host == "0.0.0.0"
        assert args.port == 8080
        assert args.no_warmup is True

    def test_help_does_not_crash(self):
        parser = _build_cli_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# 7. Startup banner
# ---------------------------------------------------------------------------


class TestStartupBanner:
    """run_server() logs a banner with host/port/model/auth/version."""

    @patch("simplevecdb.embeddings.server.get_embedder")
    @patch("simplevecdb.embeddings.server.uvicorn.Server")
    @patch("simplevecdb.embeddings.server.uvicorn.Config")
    @patch("simplevecdb.embeddings.server._logger")
    def test_banner_logged_with_host_and_port(
        self, mock_logger, mock_config_cls, mock_server_cls, mock_get_embedder
    ):
        mock_server_cls.return_value = MagicMock()

        from simplevecdb.embeddings.server import run_server

        run_server(host="127.0.0.1", port=9000)

        info_calls = mock_logger.info.call_args_list
        # The first info call should be the banner
        banner_call = info_calls[0]
        banner_args = banner_call[0]  # positional args
        banner_template = banner_args[0]

        assert "SimpleVecDB" in banner_template
        # host and port are passed as format args
        assert "127.0.0.1" in banner_args
        assert 9000 in banner_args


# ---------------------------------------------------------------------------
# 8. Nested token arrays
# ---------------------------------------------------------------------------


class TestNormalizeInput:
    """_normalize_input handles all OpenAI-compatible input formats."""

    def test_string_input(self):
        assert _normalize_input("hello world") == ["hello world"]

    def test_list_of_strings(self):
        assert _normalize_input(["a", "b", "c"]) == ["a", "b", "c"]

    def test_list_of_ints(self):
        # Flat token array is a single input per OpenAI spec
        assert _normalize_input([1, 2, 3]) == ["1 2 3"]

    def test_nested_token_arrays(self):
        result = _normalize_input([[1, 2, 3], [4, 5]])
        assert result == ["1 2 3", "4 5"]

    def test_empty_list(self):
        assert _normalize_input([]) == []

    def test_single_nested_token_array(self):
        result = _normalize_input([[10, 20, 30]])
        assert result == ["10 20 30"]


# ---------------------------------------------------------------------------
# 9. OpenAPI version
# ---------------------------------------------------------------------------


class TestOpenAPIVersion:
    """app.version should match the package version."""

    def test_app_version_matches_package(self):
        assert app.version == simplevecdb.__version__

    def test_server_version_matches_package(self):
        assert _server_version == simplevecdb.__version__


# ---------------------------------------------------------------------------
# 10. Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """simplevecdb.embeddings exposes all expected public symbols."""

    def test_embed_texts_importable(self):
        from simplevecdb.embeddings import embed_texts

        assert callable(embed_texts)

    def test_get_embedder_importable(self):
        from simplevecdb.embeddings import get_embedder

        assert callable(get_embedder)

    def test_load_model_importable(self):
        from simplevecdb.embeddings import load_model

        assert callable(load_model)

    def test_app_importable(self):
        from simplevecdb.embeddings import app as imported_app

        assert imported_app is app

    def test_run_server_importable(self):
        from simplevecdb.embeddings import run_server

        assert callable(run_server)

    def test_all_exports_listed(self):
        from simplevecdb.embeddings import __all__

        expected = {"app", "embed_texts", "get_embedder", "load_model", "run_server"}
        assert set(__all__) == expected
