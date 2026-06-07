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
