"""Tests for the HF model ``repo_id`` allowlist (security fix in 2.6.0).

Before 2.6.0, ``load_model`` accepted any string and forwarded it to
``snapshot_download``/``SentenceTransformer``. A caller could supply
absolute paths or traversal patterns to point the loader at arbitrary
on-disk directories. 2.6.0 enforces a strict ``namespace/name`` regex
matching the canonical HuggingFace repo-id format and forces
``trust_remote_code=False`` on the SentenceTransformer constructor so a
malicious model card cannot execute downloaded Python at load time.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from simplevecdb.embeddings.models import _validate_repo_id, load_model


class TestRepoIdValidation:
    """``_validate_repo_id`` accepts canonical HF IDs and rejects everything else."""

    @pytest.mark.parametrize(
        "valid_id",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-large-en-v1.5",
            "intfloat/e5-base-v2",
            "Alibaba-NLP/gte-large-en-v1.5",
            "nomic-ai/nomic-embed-text-v1",
            "user1/model.with.dots",
            "a/b",
            "Org_With-Mixed.Chars/model_v2",
        ],
    )
    def test_accepts_canonical_repo_ids(self, valid_id: str):
        # Should not raise.
        _validate_repo_id(valid_id)

    @pytest.mark.parametrize(
        "bad_id",
        [
            # Path traversal attempts.
            "../etc/passwd",
            "../../some/dir",
            "/etc/passwd",
            "/absolute/path",
            # Missing namespace/name structure.
            "no-slash",
            "",
            "/",
            "trailing/",
            "/leading",
            # Three segments — not allowed.
            "a/b/c",
            # Disallowed characters.
            "user/model$$",
            "user/model space",
            "user/model;rm",
            "user/model\nattack",
            "user/model\x00null",
            # Disallowed leading character (must start [A-Za-z0-9]).
            ".dotfile/model",
            "_underscore/model",
            "user/.dotmodel",
            # Hash-fragment attempts.
            "user/model#branch",
            # URL-style.
            "https://hf.co/user/model",
            "user@host/model",
        ],
    )
    def test_rejects_invalid_repo_ids(self, bad_id: str):
        with pytest.raises(ValueError, match="Invalid model repo_id"):
            _validate_repo_id(bad_id)


class TestLoadModelEnforcesTrustRemoteCodeFalse:
    """``load_model`` must always pass trust_remote_code=False."""

    def test_trust_remote_code_forced_off(self):
        with patch("simplevecdb.embeddings.models._load_snapshot_download") as snap, \
             patch(
                 "simplevecdb.embeddings.models._load_sentence_transformer_cls"
             ) as st_cls:
            snap.return_value = lambda **kw: "/tmp/fake-model-path"  # noqa: ARG005
            st_cls.return_value = lambda *args, **kwargs: kwargs

            kwargs = load_model("user/model")
            assert kwargs["trust_remote_code"] is False

    def test_load_model_rejects_traversal_before_calling_snapshot(self):
        # Even with mocks in place, validation must run *before* snapshot is
        # called; otherwise an attacker could point the loader at /etc.
        with patch("simplevecdb.embeddings.models._load_snapshot_download") as snap:
            snap.return_value = lambda **kw: "/tmp"  # noqa: ARG005
            with pytest.raises(ValueError):
                load_model("../etc/passwd")
            snap.assert_not_called()
