"""Additional coverage tests for embeddings server.

Targets missing lines: 41-45, 56-57, 70, 97, 106, 138,
171-175, 196-201, 246, 259.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from simplevecdb.embeddings.server import (
    RateLimiter,
    ModelRegistry,
    UsageMeter,
    app,
)
from simplevecdb.embeddings import server


client = TestClient(app)


@pytest.fixture(autouse=True)
def unlocked_registry():
    """Allow arbitrary test models in unit tests."""
    original = server.registry
    server.registry = ModelRegistry({"default": "test-default"}, allow_unlisted=True)
    yield
    server.registry = original


class TestRateLimiter:
    """Cover lines 41-45, 56-57, 70."""

    def test_cleanup_stale_removes_old_buckets(self):
        """Lines 41-45: _cleanup_stale removes expired entries."""
        rl = RateLimiter(requests_per_minute=60, burst=10, ttl_seconds=1)
        now = time.time()

        # Manually insert a stale bucket
        rl._buckets["stale_ip"] = {"tokens": 5.0, "last": now - 100}
        rl._buckets["fresh_ip"] = {"tokens": 5.0, "last": now}

        rl._cleanup_stale(now)

        assert "stale_ip" not in rl._buckets
        assert "fresh_ip" in rl._buckets

    def test_cleanup_triggered_by_bucket_count(self):
        """Lines 56-57: cleanup triggered when buckets exceed max."""
        rl = RateLimiter(
            requests_per_minute=60, burst=10, ttl_seconds=3600, max_buckets=2
        )

        # Fill buckets beyond max
        rl._buckets["ip1"] = {"tokens": 5.0, "last": time.time()}
        rl._buckets["ip2"] = {"tokens": 5.0, "last": time.time()}
        rl._buckets["ip3"] = {"tokens": 5.0, "last": time.time()}

        # Next is_allowed should trigger cleanup
        rl.is_allowed("ip4")
        # ip4 should exist after check
        assert "ip4" in rl._buckets

    def test_cleanup_triggered_by_ttl_interval(self):
        """Lines 56-57: cleanup triggered by TTL/4 interval."""
        rl = RateLimiter(
            requests_per_minute=60, burst=10, ttl_seconds=4, max_buckets=10000
        )
        # Set last cleanup far in the past
        rl._last_cleanup = time.time() - 10  # > ttl/4 = 1 second ago

        # Add a stale bucket
        rl._buckets["old"] = {"tokens": 5.0, "last": time.time() - 100}

        rl.is_allowed("new_ip")

        # Stale bucket should have been cleaned
        assert "old" not in rl._buckets

    def test_rate_limit_denied(self):
        """Line 70: returns False when tokens exhausted."""
        rl = RateLimiter(requests_per_minute=1, burst=1)

        # First request allowed
        assert rl.is_allowed("test_ip") is True
        # Second should be denied (burst=1, very slow refill)
        assert rl.is_allowed("test_ip") is False


class TestModelRegistry:
    """Cover lines 97, 106, 138."""

    def test_default_alias_added_when_missing(self):
        """Line 97: 'default' alias auto-added if not in mapping."""
        registry = ModelRegistry({"custom": "repo/custom-model"})
        display, repo = registry.resolve(None)
        assert display == "default"

    def test_resolve_alias_match(self):
        """Line 106: resolve returns alias mapping."""
        registry = ModelRegistry({"my_alias": "repo/my-model"})
        display, repo = registry.resolve("my_alias")
        assert display == "my_alias"
        assert repo == "repo/my-model"

    def test_resolve_repo_id_match(self):
        """Resolve by direct repo_id."""
        registry = ModelRegistry({"alias": "repo/model"})
        display, repo = registry.resolve("repo/model")
        assert display == "repo/model"
        assert repo == "repo/model"

    def test_resolve_unlisted_allowed(self):
        """Unlisted model allowed when allow_unlisted=True."""
        registry = ModelRegistry({"alias": "repo/model"}, allow_unlisted=True)
        display, repo = registry.resolve("unknown/model")
        assert display == "unknown/model"
        assert repo == "unknown/model"

    def test_resolve_unlisted_denied(self):
        """Unlisted model raises HTTPException when allow_unlisted=False."""
        from fastapi import HTTPException

        registry = ModelRegistry({"alias": "repo/model"}, allow_unlisted=False)
        with pytest.raises(HTTPException) as exc_info:
            registry.resolve("unknown/model")
        assert exc_info.value.status_code == 400

    def test_list_models_dedup(self):
        """Line 138: list_models deduplicates aliases and repo IDs."""
        # If alias name equals repo_id, don't list twice
        registry = ModelRegistry({"my_model": "my_model"})
        models = registry.list_models()
        ids = [m["id"] for m in models]
        # "my_model" appears as alias, "default" is auto-added
        # repo "my_model" should not be listed again since it matches the alias
        assert ids.count("my_model") == 1


class TestUsageMeter:
    """Cover lines 171-175."""

    def test_snapshot_specific_identity(self):
        """Lines 171-175: snapshot with identity returns single bucket."""
        meter = UsageMeter()
        meter.record("user_a", 10)
        meter.record("user_b", 20)

        snap = meter.snapshot("user_a")
        assert "user_a" in snap
        assert "user_b" not in snap
        assert snap["user_a"]["requests"] == 1
        assert snap["user_a"]["prompt_tokens"] == 10

    def test_snapshot_unknown_identity(self):
        """Lines 171-175: snapshot for unknown identity returns zeros."""
        meter = UsageMeter()
        snap = meter.snapshot("unknown")
        assert snap["unknown"]["requests"] == 0
        assert snap["unknown"]["prompt_tokens"] == 0

    def test_snapshot_all(self):
        """Snapshot without identity returns all."""
        meter = UsageMeter()
        meter.record("a", 5)
        meter.record("b", 10)

        snap = meter.snapshot()
        assert "a" in snap
        assert "b" in snap


class TestAuthentication:
    """Cover lines 196-201."""

    def test_auth_required_no_token(self):
        """Lines 197-198: missing API key -> 401."""
        with patch.object(server, "config") as mock_config:
            mock_config.EMBEDDING_SERVER_API_KEYS = {"valid-key"}
            mock_config.EMBEDDING_BATCH_SIZE = 32
            mock_config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS = 100

            response = client.post(
                "/v1/embeddings",
                json={"input": "test"},
            )
            assert response.status_code == 401

    def test_auth_required_invalid_token(self):
        """Lines 199-200: invalid API key -> 403."""
        with patch.object(server, "config") as mock_config:
            mock_config.EMBEDDING_SERVER_API_KEYS = {"valid-key"}
            mock_config.EMBEDDING_BATCH_SIZE = 32
            mock_config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS = 100

            response = client.post(
                "/v1/embeddings",
                json={"input": "test"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 403

    def test_auth_valid_bearer_token(self):
        """Line 196: valid Bearer token accepted."""
        with patch.object(server, "config") as mock_config:
            mock_config.EMBEDDING_SERVER_API_KEYS = {"valid-key"}
            mock_config.EMBEDDING_BATCH_SIZE = 32
            mock_config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS = 100

            with patch("simplevecdb.embeddings.server.embed_texts") as mock_embed:
                mock_embed.return_value = [[0.1, 0.2]]

                response = client.post(
                    "/v1/embeddings",
                    json={"input": "test"},
                    headers={"Authorization": "Bearer valid-key"},
                )
                assert response.status_code == 200


class TestRateLimitEndpoint:
    """Cover line 246."""

    def test_rate_limit_exceeded_returns_429(self):
        """Line 246: rate limit exceeded -> 429."""
        with patch.object(server.rate_limiter, "is_allowed", return_value=False):
            with patch("simplevecdb.embeddings.server.embed_texts") as mock_embed:
                mock_embed.return_value = [[0.1]]

                response = client.post(
                    "/v1/embeddings",
                    json={"input": "test"},
                )
                assert response.status_code == 429
                assert "Rate limit" in response.json()["detail"]


class TestBatchSizeLimit:
    """Cover line 259."""

    def test_batch_size_exceeded_returns_413(self):
        """Line 259: batch too large -> 413."""
        with patch.object(server, "config") as mock_config:
            mock_config.EMBEDDING_SERVER_API_KEYS = set()
            mock_config.EMBEDDING_BATCH_SIZE = 32
            mock_config.EMBEDDING_SERVER_MAX_REQUEST_ITEMS = 2

            with patch.object(server.rate_limiter, "is_allowed", return_value=True):
                response = client.post(
                    "/v1/embeddings",
                    json={"input": ["a", "b", "c"]},
                )
                assert response.status_code == 413
                assert "exceeds" in response.json()["detail"]
