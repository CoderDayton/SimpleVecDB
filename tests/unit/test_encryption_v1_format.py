"""Tests for the v0/v1 encrypted file format and atomic write helper (2.6.0).

The 2.5.0 review flagged two encryption-layer issues:
- Encrypted files were written with a single ``write_bytes`` call. A crash
  mid-write left a half-written file on disk; on next decrypt that file
  was indistinguishable from a corrupt blob.
- The format had no version byte, so future format evolution would have
  to be inferred heuristically.

2.6.0 introduces a 3-byte header (``'SV' + version``) and routes every
file write through ``_atomic_write_bytes`` (tmp + fsync + os.replace +
chmod 0o600 + dir fsync). ``decrypt_file`` accepts both v0 (legacy) and
v1 to keep pre-2.6.0 encrypted indexes openable.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from simplevecdb.encryption import (  # noqa: E402
    AES_KEY_SIZE,
    AES_NONCE_SIZE,
    _ENC_MAGIC,
    _ENC_VERSION,
    _NORMALIZE_KEY_CACHE,
    _NORMALIZE_KEY_SALT,
    _atomic_write_bytes,
    _normalize_key,
    decrypt_file,
    decrypt_index_file,
    encrypt_file,
    encrypt_index_file,
)

TEST_KEY = b"\x00" * AES_KEY_SIZE


class TestAtomicWriteBytes:
    """``_atomic_write_bytes`` must be durable, atomic, and 0o600 by default."""

    def test_writes_data_to_target(self, tmp_path: Path):
        target = tmp_path / "out.bin"
        _atomic_write_bytes(target, b"hello world")
        assert target.read_bytes() == b"hello world"

    def test_default_mode_is_owner_only(self, tmp_path: Path):
        target = tmp_path / "secret.bin"
        _atomic_write_bytes(target, b"private")
        # Lower 9 bits == 0o600 (owner read+write only).
        assert (target.stat().st_mode & 0o777) == 0o600

    def test_explicit_mode_honored(self, tmp_path: Path):
        target = tmp_path / "shared.bin"
        _atomic_write_bytes(target, b"x", mode=0o644)
        assert (target.stat().st_mode & 0o777) == 0o644

    def test_overwrites_existing_target(self, tmp_path: Path):
        target = tmp_path / "out.bin"
        target.write_bytes(b"old")
        _atomic_write_bytes(target, b"new")
        assert target.read_bytes() == b"new"

    def test_creates_missing_parent_directory(self, tmp_path: Path):
        target = tmp_path / "nested" / "dir" / "out.bin"
        _atomic_write_bytes(target, b"x")
        assert target.read_bytes() == b"x"

    def test_no_temp_file_left_behind(self, tmp_path: Path):
        target = tmp_path / "out.bin"
        _atomic_write_bytes(target, b"x")
        # After a clean write, the .tmp sibling must be gone.
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []


class TestEncryptedFileFormatV1:
    """encrypt_file writes the v1 format (magic + version + body)."""

    def test_v1_header_written(self, tmp_path: Path):
        plaintext = tmp_path / "in.txt"
        plaintext.write_bytes(b"top secret payload")
        enc = tmp_path / "out.enc"
        encrypt_file(plaintext, enc, TEST_KEY)

        data = enc.read_bytes()
        assert data[: len(_ENC_MAGIC)] == _ENC_MAGIC
        assert data[len(_ENC_MAGIC)] == _ENC_VERSION

    def test_encrypted_file_has_owner_only_mode(self, tmp_path: Path):
        plaintext = tmp_path / "in.txt"
        plaintext.write_bytes(b"data")
        enc = tmp_path / "out.enc"
        encrypt_file(plaintext, enc, TEST_KEY)
        assert (enc.stat().st_mode & 0o777) == 0o600

    def test_v1_roundtrip(self, tmp_path: Path):
        plaintext_path = tmp_path / "in.txt"
        original = b"Hello, encrypted world! " * 20
        plaintext_path.write_bytes(original)

        enc = tmp_path / "out.enc"
        dec = tmp_path / "out.dec"
        encrypt_file(plaintext_path, enc, TEST_KEY)
        decrypt_file(enc, dec, TEST_KEY)
        assert dec.read_bytes() == original


class TestV0BackwardsCompatibility:
    """decrypt_file must still open files written by pre-2.6.0 simplevecdb."""

    def _build_v0_blob(self, plaintext: bytes, key: bytes) -> bytes:
        """Build a v0 (no header) encrypted blob: nonce + ciphertext+tag."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = secrets.token_bytes(AES_NONCE_SIZE)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data=None)
        return nonce + ciphertext

    def test_decrypt_v0_legacy_blob(self, tmp_path: Path):
        original = b"legacy payload from 2.5.0"
        blob = self._build_v0_blob(original, TEST_KEY)
        enc = tmp_path / "legacy.enc"
        enc.write_bytes(blob)

        out = tmp_path / "out.bin"
        decrypt_file(enc, out, TEST_KEY)
        assert out.read_bytes() == original

    def test_decrypt_v0_blob_starting_with_sv_bytes_still_works(
        self, tmp_path: Path
    ):
        # A v0 nonce that *happens* to start with 'SV' but whose 3rd byte is
        # not the version sentinel must still decrypt as v0. Decrypt logic
        # only strips the header when *both* the magic bytes AND the version
        # byte match. Build a nonce starting with 'SV' followed by 0xFF
        # (clearly not _ENC_VERSION).
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = b"SV" + b"\xff" + secrets.token_bytes(AES_NONCE_SIZE - 3)
        original = b"adversarial nonce path"
        ciphertext = AESGCM(TEST_KEY).encrypt(nonce, original, associated_data=None)
        enc = tmp_path / "tricky.enc"
        enc.write_bytes(nonce + ciphertext)

        out = tmp_path / "out.bin"
        decrypt_file(enc, out, TEST_KEY)
        assert out.read_bytes() == original


class TestNormalizeKeyCache:
    """``_normalize_key`` caches PBKDF2 results by (key, salt)."""

    def setup_method(self):
        # Clear the module-level cache so each test starts fresh.
        _NORMALIZE_KEY_CACHE.clear()

    def test_cache_populates_on_first_call(self):
        derived = _normalize_key("passphrase")
        assert _NORMALIZE_KEY_CACHE[(b"passphrase", _NORMALIZE_KEY_SALT)] == derived

    def test_cache_hit_returns_same_bytes(self):
        first = _normalize_key("passphrase")
        # Second call must return identical bytes from cache (not re-derive).
        second = _normalize_key("passphrase")
        assert first == second

    def test_different_salts_produce_distinct_cache_entries(self):
        salt_a = b"\x00" * 16
        salt_b = b"\x01" * 16
        a = _normalize_key("passphrase", salt=salt_a)
        b = _normalize_key("passphrase", salt=salt_b)
        # Same passphrase + different salt -> different derived key.
        assert a != b
        assert (b"passphrase", salt_a) in _NORMALIZE_KEY_CACHE
        assert (b"passphrase", salt_b) in _NORMALIZE_KEY_CACHE

    def test_raw_32_byte_key_skips_pbkdf2(self):
        raw = os.urandom(AES_KEY_SIZE)
        result = _normalize_key(raw)
        assert result == raw
        # Raw keys must not be cached — they ARE the key.
        assert _NORMALIZE_KEY_CACHE == {}


class TestEncryptIndexFileRoundtrip:
    """encrypt_index_file writes a sidecar and decrypt_index_file consumes it."""

    def test_roundtrip_creates_sidecar_and_recovers_data(self, tmp_path: Path):
        idx = tmp_path / "test.usearch"
        idx.write_bytes(b"fake usearch index data " * 50)
        original = idx.read_bytes()

        encrypt_index_file(idx, "passphrase")

        enc_path = tmp_path / "test.usearch.enc"
        sidecar = tmp_path / "test.usearch.enc.salt"
        assert enc_path.exists()
        assert not idx.exists(), "plaintext must be removed after encrypt"
        assert sidecar.exists(), "salt sidecar must accompany new encrypted file"

        # Decrypt restores the plaintext.
        decrypted = decrypt_index_file(enc_path, "passphrase")
        assert decrypted.read_bytes() == original

    def test_decrypt_with_wrong_passphrase_raises(self, tmp_path: Path):
        from simplevecdb.encryption import EncryptionError

        idx = tmp_path / "x.usearch"
        idx.write_bytes(b"data")
        encrypt_index_file(idx, "right")
        enc_path = tmp_path / "x.usearch.enc"

        with pytest.raises(EncryptionError):
            decrypt_index_file(enc_path, "wrong")
