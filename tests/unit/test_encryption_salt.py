"""Tests for the per-DB random salt sidecar (C3 fix in 2.6.0).

The 2.5.0 review flagged the fixed PBKDF2 salt as a design weakness: the
same passphrase produced the same SQLCipher key across every simplevecdb
installation, so a single rainbow table broke every database. 2.6.0
generates a random salt per encrypted resource and stores it in a
``<resource>.salt`` sidecar. Pre-2.6.0 databases (no sidecar) keep
working via fallback to the legacy fixed salt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simplevecdb.encryption import (
    SALT_SIZE,
    _NORMALIZE_KEY_SALT,
    _resolve_salt,
    EncryptionUnavailableError,
    create_encrypted_connection,
)


class TestSaltSidecar:
    def test_resolve_salt_creates_sidecar_for_new_resource(self, tmp_path: Path):
        target = tmp_path / "fresh.db"
        salt = _resolve_salt(target, create_if_missing=True)
        sidecar = tmp_path / "fresh.db.salt"
        assert sidecar.exists()
        assert sidecar.read_bytes() == salt
        assert len(salt) == SALT_SIZE
        # Generated salt must not match the legacy fixed salt.
        assert salt != _NORMALIZE_KEY_SALT

    def test_resolve_salt_reads_existing_sidecar(self, tmp_path: Path):
        target = tmp_path / "x.db"
        first = _resolve_salt(target, create_if_missing=True)
        second = _resolve_salt(target, create_if_missing=True)
        assert first == second  # idempotent — sidecar is reused

    def test_resolve_salt_legacy_fallback(self, tmp_path: Path):
        target = tmp_path / "legacy.db"
        # No sidecar; create_if_missing=False → legacy fixed salt for
        # backwards compatibility with pre-2.6.0 encrypted resources.
        assert _resolve_salt(target, create_if_missing=False) == _NORMALIZE_KEY_SALT

    def test_resolve_salt_invalid_size_falls_back(self, tmp_path: Path):
        target = tmp_path / "bad.db"
        sidecar = tmp_path / "bad.db.salt"
        sidecar.write_bytes(b"too-short")
        salt = _resolve_salt(target, create_if_missing=False)
        assert salt == _NORMALIZE_KEY_SALT

    def test_distinct_dbs_get_distinct_salts(self, tmp_path: Path):
        a = _resolve_salt(tmp_path / "a.db", create_if_missing=True)
        b = _resolve_salt(tmp_path / "b.db", create_if_missing=True)
        assert a != b


class TestSaltSidecarWithSQLCipher:
    """End-to-end: a new encrypted DB picks up a random salt sidecar."""

    def test_new_encrypted_db_writes_sidecar(self, tmp_path: Path):
        try:
            db_path = tmp_path / "encrypted.db"
            conn = create_encrypted_connection(db_path, "secret")
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

            sidecar = tmp_path / "encrypted.db.salt"
            assert sidecar.exists()
            assert len(sidecar.read_bytes()) == SALT_SIZE
            assert sidecar.read_bytes() != _NORMALIZE_KEY_SALT
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")

    def test_reopen_uses_existing_sidecar(self, tmp_path: Path):
        try:
            db_path = tmp_path / "stable.db"
            conn = create_encrypted_connection(db_path, "passphrase")
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

            sidecar = tmp_path / "stable.db.salt"
            salt_before = sidecar.read_bytes()

            # Reopen with the same passphrase — must use the existing sidecar.
            conn2 = create_encrypted_connection(db_path, "passphrase")
            conn2.execute("SELECT count(*) FROM t").fetchone()
            conn2.close()

            assert sidecar.read_bytes() == salt_before
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")
