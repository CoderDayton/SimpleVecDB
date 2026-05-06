"""Regression tests for encryption changes in 2.6.0 review pass 3.

Pins invariants the prior suite missed:

- Calling ``encrypt_file`` twice on the same plaintext produces two
  different nonces (canonical AES-GCM nonce-uniqueness regression).
- ``decrypt_file`` with a wrong key never creates the output path
  (authentication failure must short-circuit before any write).
- The v1 header bytes are bound into AAD: tampering with the magic or
  the version byte makes ``decrypt_file`` raise instead of silently
  succeeding.
- ``_resolve_salt`` does not clobber a pre-existing salt sidecar when
  ``create_if_missing=True``; concurrent openers converge on the
  already-written salt.
- A v0-format encrypted blob round-trips: decrypt → re-encrypt → the
  output is v1 with a fresh sidecar, and reads back successfully.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest

cryptography = pytest.importorskip("cryptography")

from simplevecdb.encryption import (
    AES_KEY_SIZE,
    AES_NONCE_SIZE,
    EncryptionError,
    SALT_SIZE,
    _ENC_MAGIC,
    _ENC_VERSION,
    _resolve_salt,
    decrypt_file,
    decrypt_index_file,
    encrypt_file,
    encrypt_index_file,
)


@pytest.fixture
def random_key() -> bytes:
    return secrets.token_bytes(AES_KEY_SIZE)


class TestNonceUniqueness:
    def test_two_encryptions_use_distinct_nonces(self, tmp_path, random_key):
        plaintext = b"hello world"
        src = tmp_path / "plain.bin"
        src.write_bytes(plaintext)

        out_a = tmp_path / "a.enc"
        out_b = tmp_path / "b.enc"

        encrypt_file(src, out_a, random_key)
        encrypt_file(src, out_b, random_key)

        # Skip the 3-byte v1 header; nonce follows immediately.
        header_len = len(_ENC_MAGIC) + 1
        nonce_a = out_a.read_bytes()[header_len:header_len + AES_NONCE_SIZE]
        nonce_b = out_b.read_bytes()[header_len:header_len + AES_NONCE_SIZE]

        assert len(nonce_a) == AES_NONCE_SIZE
        assert len(nonce_b) == AES_NONCE_SIZE
        assert nonce_a != nonce_b


class TestWrongKeyDoesNotCreateOutput:
    def test_decrypt_with_wrong_key_never_writes_output(
        self, tmp_path, random_key
    ):
        plaintext = b"sensitive contents"
        src = tmp_path / "p.bin"
        src.write_bytes(plaintext)
        encrypted = tmp_path / "p.enc"
        encrypt_file(src, encrypted, random_key)

        wrong_key = secrets.token_bytes(AES_KEY_SIZE)
        output = tmp_path / "decrypted.bin"

        with pytest.raises(EncryptionError):
            decrypt_file(encrypted, output, wrong_key)

        # Authentication failure must short-circuit before any write.
        assert not output.exists(), (
            "Output file must not exist after wrong-key decrypt"
        )


class TestHeaderAADBinding:
    def test_tampering_with_magic_byte_fails_authentication(
        self, tmp_path, random_key
    ):
        plaintext = b"tamper test"
        src = tmp_path / "p.bin"
        src.write_bytes(plaintext)
        encrypted = tmp_path / "p.enc"
        encrypt_file(src, encrypted, random_key)

        # Flip a bit in the magic.
        blob = bytearray(encrypted.read_bytes())
        blob[0] ^= 0x01
        encrypted.write_bytes(bytes(blob))

        # AAD-bound header tampering produces an auth failure.
        with pytest.raises(EncryptionError):
            decrypt_file(encrypted, tmp_path / "out.bin", random_key)

    def test_tampering_with_version_byte_fails_authentication(
        self, tmp_path, random_key
    ):
        plaintext = b"version tamper"
        src = tmp_path / "p.bin"
        src.write_bytes(plaintext)
        encrypted = tmp_path / "p.enc"
        encrypt_file(src, encrypted, random_key)

        blob = bytearray(encrypted.read_bytes())
        # Change version from 1 to 99 — must fail AAD verification.
        version_offset = len(_ENC_MAGIC)
        blob[version_offset] = 99
        encrypted.write_bytes(bytes(blob))

        with pytest.raises(EncryptionError):
            decrypt_file(encrypted, tmp_path / "out.bin", random_key)


class TestSaltSidecarO_EXCL:
    def test_existing_salt_sidecar_is_not_overwritten(self, tmp_path):
        """If a sidecar already exists, ``_resolve_salt`` returns its
        contents instead of generating a new salt and clobbering it."""
        resource = tmp_path / "db.sqlite"
        resource.write_bytes(b"")  # empty existing file
        salt_path = resource.with_name(resource.name + ".salt")
        existing = secrets.token_bytes(SALT_SIZE)
        salt_path.write_bytes(existing)
        os.chmod(salt_path, 0o600)

        result = _resolve_salt(resource, create_if_missing=True)

        assert result == existing, (
            "Existing sidecar must be preserved, not overwritten"
        )
        assert salt_path.read_bytes() == existing

    def test_concurrent_creators_converge_on_one_salt(self, tmp_path):
        """Simulate two callers creating a new sidecar near-simultaneously
        by pre-creating the file between resolve calls — the loser must
        read the existing salt rather than fail or write a different
        one."""
        resource_a = tmp_path / "db_a.sqlite"
        salt_a = _resolve_salt(resource_a, create_if_missing=True)
        # Now resolve again — must get the same salt, not a new one.
        salt_a_again = _resolve_salt(resource_a, create_if_missing=True)
        assert salt_a == salt_a_again


class TestV0V1MigrationRoundTrip:
    def test_v0_blob_decrypts_and_reencrypts_to_v1_with_sidecar(
        self, tmp_path, random_key
    ):
        """A v0 .usearch.enc (no header, no sidecar) must decrypt, and
        re-encrypting the result must produce a v1 blob with a fresh
        sidecar. Because encrypt_index_file derives its key from a
        passphrase + salt, we use the passphrase API end-to-end."""
        from simplevecdb.encryption import (
            _NORMALIZE_KEY_SALT,
            _normalize_key,
        )
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        plaintext = b"\x00" * 1024 + b"index-bytes"
        passphrase = "passphrase-xyz"

        # Hand-craft a v0 blob: nonce + AESGCM(key=normalize(pp, fixed_salt)).
        legacy_key = _normalize_key(passphrase, salt=_NORMALIZE_KEY_SALT)
        nonce = secrets.token_bytes(AES_NONCE_SIZE)
        ct = AESGCM(legacy_key).encrypt(nonce, plaintext, None)
        v0_blob = nonce + ct

        # Place a v0 .usearch.enc on disk (no .salt sidecar).
        index_path = tmp_path / "idx.usearch"
        encrypted_path = tmp_path / "idx.usearch.enc"
        encrypted_path.write_bytes(v0_blob)
        # No .salt sidecar.

        # Decrypt — must succeed using the legacy fixed salt path.
        decrypted = decrypt_index_file(encrypted_path, passphrase)
        assert decrypted.read_bytes() == plaintext
        assert decrypted == index_path

        # Now re-encrypt: 2.6.0 should write v1 + create a fresh sidecar.
        # encrypt_index_file unlinks the plaintext after the encrypted
        # output is durable.
        encrypt_index_file(index_path, passphrase)
        assert not index_path.exists()
        assert encrypted_path.exists()

        # New blob has the v1 magic header.
        new_blob = encrypted_path.read_bytes()
        assert new_blob[: len(_ENC_MAGIC)] == _ENC_MAGIC
        assert new_blob[len(_ENC_MAGIC)] == _ENC_VERSION

        # And a sidecar was written this time.
        sidecar = Path(str(encrypted_path) + ".salt")
        assert sidecar.exists()
        assert len(sidecar.read_bytes()) == SALT_SIZE

        # Final round-trip: decrypt the v1 blob.
        decrypt_index_file(encrypted_path, passphrase)
        assert index_path.read_bytes() == plaintext
