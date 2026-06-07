"""Tier-3 security fixes — TDD regression tests."""

from __future__ import annotations

import asyncio

import pytest


class TestServerBodyCap:
    def test_oversize_content_length_rejected(self) -> None:
        server = pytest.importorskip("simplevecdb.embeddings.server")
        captured: list[dict] = []

        async def downstream(scope, receive, send):  # noqa: ANN001
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(msg):  # noqa: ANN001
            captured.append(msg)

        mw = server._MaxBodySizeMiddleware(downstream, max_bytes=100)
        scope = {"type": "http", "headers": [(b"content-length", b"100000")]}
        asyncio.run(mw(scope, receive, send))
        assert captured and captured[0]["status"] == 413

    def test_small_body_passes_through(self) -> None:
        server = pytest.importorskip("simplevecdb.embeddings.server")
        captured: list[dict] = []

        async def downstream(scope, receive, send):  # noqa: ANN001
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def receive():
            return {"type": "http.request", "body": b"hi", "more_body": False}

        async def send(msg):  # noqa: ANN001
            captured.append(msg)

        mw = server._MaxBodySizeMiddleware(downstream, max_bytes=1000)
        scope = {"type": "http", "headers": [(b"content-length", b"2")]}
        asyncio.run(mw(scope, receive, send))
        assert captured[0]["status"] == 200


class TestBodyCapEnvGuard:
    def test_bad_or_zero_env_does_not_crash_and_is_floored(self, monkeypatch) -> None:
        """A malformed or zero EMBEDDING_SERVER_MAX_BODY_BYTES must not crash the
        server import and must never drop the cap below the 1 MiB floor."""
        import importlib

        server = pytest.importorskip("simplevecdb.embeddings.server")
        try:
            monkeypatch.setenv("EMBEDDING_SERVER_MAX_BODY_BYTES", "not-an-int")
            importlib.reload(server)
            assert server._MAX_BODY_BYTES >= (1 << 20)

            monkeypatch.setenv("EMBEDDING_SERVER_MAX_BODY_BYTES", "0")
            importlib.reload(server)
            assert server._MAX_BODY_BYTES >= (1 << 20)
        finally:
            monkeypatch.delenv("EMBEDDING_SERVER_MAX_BODY_BYTES", raising=False)
            importlib.reload(server)
