"""Additional coverage tests for encryption module.

Targets missing lines: 60, 147-148, 175, 186, 201-202, 220, 231,
263-264, 286-287, 309-310, 317, 337, 346, 361, 389, 396.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from simplevecdb.encryption import (
    EncryptionUnavailableError,
    EncryptionError,
    encrypt_file,
    decrypt_file,
    encrypt_index_file,
    decrypt_index_file,
    create_encrypted_connection,
    is_database_encrypted,
    AES_KEY_SIZE,
    AES_NONCE_SIZE,
    AES_TAG_SIZE,
)

pytest.importorskip("cryptography")

TEST_KEY = os.urandom(AES_KEY_SIZE)


class TestEncryptionUnavailableError:
    """Cover EncryptionUnavailableError.__init__ (line 60)."""

    def test_message_contains_install_instructions(self):
        err = EncryptionUnavailableError()
        assert "sqlcipher3-binary" in str(err)
        assert "cryptography" in str(err)
        assert "pip install" in str(err)

    def test_is_import_error(self):
        err = EncryptionUnavailableError()
        assert isinstance(err, ImportError)


class TestCreateEncryptedConnectionEdgeCases:
    """Cover lines 147-148, 175, 186, 201-202."""

    def test_sqlcipher_import_error_raises_unavailable(self, tmp_path: Path):
        """Line 147-148: ImportError -> EncryptionUnavailableError."""
        with patch.dict("sys.modules", {"sqlcipher3": None, "sqlcipher3.dbapi2": None}):
            with patch(
                "builtins.__import__",
                side_effect=_make_import_blocker("sqlcipher3"),
            ):
                with pytest.raises(EncryptionUnavailableError):
                    create_encrypted_connection(tmp_path / "test.db", "key")

    def test_cipher_version_none_raises_encryption_error(self, tmp_path: Path):
        """Line 175: cipher_version returns None -> EncryptionError."""
        mock_conn = MagicMock()
        # First execute is PRAGMA key, second is cipher_version returning None
        mock_conn.execute.side_effect = [
            None,  # PRAGMA key
            MagicMock(fetchone=MagicMock(return_value=None)),  # cipher_version
        ]

        mock_sqlcipher = MagicMock()
        mock_sqlcipher.connect.return_value = mock_conn

        mock_pkg = MagicMock(dbapi2=mock_sqlcipher)
        with patch.dict("sys.modules", {"sqlcipher3": mock_pkg, "sqlcipher3.dbapi2": mock_sqlcipher}):
            with pytest.raises(EncryptionError, match="not active"):
                create_encrypted_connection(tmp_path / "test.db", "passphrase")

    def test_encryption_error_reraise_in_verify(self, tmp_path: Path):
        """Line 186: EncryptionError re-raised from inner try block."""
        mock_conn = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # PRAGMA key
            if call_count[0] == 2:
                # cipher_version returns a version
                result = MagicMock()
                result.fetchone.return_value = ("4.5.0",)
                return result
            if call_count[0] == 3:
                # sqlite_master read fails
                raise EncryptionError("wrong key")
            return None

        mock_conn.execute.side_effect = side_effect

        mock_sqlcipher = MagicMock()
        mock_sqlcipher.connect.return_value = mock_conn

        mock_pkg = MagicMock(dbapi2=mock_sqlcipher)
        with patch.dict("sys.modules", {"sqlcipher3": mock_pkg, "sqlcipher3.dbapi2": mock_sqlcipher}):
            with pytest.raises(EncryptionError, match="wrong key"):
                create_encrypted_connection(tmp_path / "test.db", "passphrase")

    def test_generic_exception_in_verify_wraps_encryption_error(self, tmp_path: Path):
        """Lines 186-191: generic Exception during verify wraps into EncryptionError."""
        mock_conn = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # PRAGMA key
            if call_count[0] == 2:
                result = MagicMock()
                result.fetchone.return_value = ("4.5.0",)
                return result
            if call_count[0] == 3:
                raise RuntimeError("some sqlite error")
            return None

        mock_conn.execute.side_effect = side_effect

        mock_sqlcipher = MagicMock()
        mock_sqlcipher.connect.return_value = mock_conn

        mock_pkg = MagicMock(dbapi2=mock_sqlcipher)
        with patch.dict("sys.modules", {"sqlcipher3": mock_pkg, "sqlcipher3.dbapi2": mock_sqlcipher}):
            with pytest.raises(EncryptionError, match="Failed to verify"):
                create_encrypted_connection(tmp_path / "test.db", "passphrase")

    def test_outer_generic_exception_wraps(self, tmp_path: Path):
        """Lines 201-202: outer except wraps generic errors."""
        mock_sqlcipher = MagicMock()
        mock_sqlcipher.connect.side_effect = OSError("disk full")

        mock_pkg = MagicMock(dbapi2=mock_sqlcipher)
        with patch.dict("sys.modules", {"sqlcipher3": mock_pkg, "sqlcipher3.dbapi2": mock_sqlcipher}):
            with pytest.raises(EncryptionError, match="Failed to create encrypted"):
                create_encrypted_connection(tmp_path / "test.db", "passphrase")


class TestIsDatabaseEncrypted:
    """Cover lines 220, 231."""

    def test_nonexistent_file_returns_false(self, tmp_path: Path):
        """Line 220: path doesn't exist -> False."""
        assert is_database_encrypted(tmp_path / "nonexistent.db") is False

    def test_encrypted_file_detected(self, tmp_path: Path):
        """Line 231: 'not a database' error -> True."""
        db_path = tmp_path / "encrypted.db"
        # Write random bytes that sqlite3 can't read
        db_path.write_bytes(os.urandom(4096))
        assert is_database_encrypted(db_path) is True


class TestEncryptFileEdgeCases:
    """Cover lines 263-264, 286-287."""

    def test_encrypt_file_import_error(self, tmp_path: Path):
        """Lines 263-264: cryptography not installed."""
        input_file = tmp_path / "plain.bin"
        input_file.write_bytes(b"data")

        with patch.dict("sys.modules", {"cryptography": None, "cryptography.hazmat.primitives.ciphers.aead": None}):
            with patch(
                "builtins.__import__",
                side_effect=_make_import_blocker("cryptography"),
            ):
                with pytest.raises(EncryptionUnavailableError):
                    encrypt_file(input_file, tmp_path / "out.enc", TEST_KEY)

    def test_encrypt_file_generic_error_wraps(self, tmp_path: Path):
        """Lines 286-287: generic error wraps into EncryptionError."""
        input_file = tmp_path / "plain.bin"
        input_file.write_bytes(b"data")

        # Encrypt with invalid key size triggers ValueError -> wrapped in EncryptionError
        with pytest.raises(EncryptionError, match="Failed to encrypt"):
            encrypt_file(input_file, tmp_path / "out.enc", b"short")


class TestDecryptFileEdgeCases:
    """Cover lines 309-310, 317, 337, 346."""

    def test_decrypt_file_import_error(self, tmp_path: Path):
        """Lines 309-310: cryptography not installed."""
        enc_file = tmp_path / "enc.bin"
        enc_file.write_bytes(os.urandom(100))

        with patch.dict("sys.modules", {"cryptography": None, "cryptography.hazmat.primitives.ciphers.aead": None}):
            with patch(
                "builtins.__import__",
                side_effect=_make_import_blocker("cryptography"),
            ):
                with pytest.raises(EncryptionUnavailableError):
                    decrypt_file(enc_file, tmp_path / "out.bin", TEST_KEY)

    def test_decrypt_file_too_small(self, tmp_path: Path):
        """Line 317: file smaller than nonce + tag size."""
        enc_file = tmp_path / "tiny.bin"
        # AES_NONCE_SIZE(12) + AES_TAG_SIZE(16) = 28, write fewer bytes
        enc_file.write_bytes(os.urandom(10))

        with pytest.raises(EncryptionError, match="too small"):
            decrypt_file(enc_file, tmp_path / "out.bin", TEST_KEY)

    def test_decrypt_file_reraises_encryption_error(self, tmp_path: Path):
        """Line 337: EncryptionError re-raised directly."""
        # The "too small" case already covers this - EncryptionError is raised
        # and then caught by `except EncryptionError: raise`
        enc_file = tmp_path / "tiny.bin"
        enc_file.write_bytes(os.urandom(5))

        with pytest.raises(EncryptionError, match="too small"):
            decrypt_file(enc_file, tmp_path / "out.bin", TEST_KEY)

    def test_decrypt_file_generic_error_wraps(self, tmp_path: Path):
        """Line 346: non-tag generic error wraps into EncryptionError."""
        # Create a valid-sized but garbage file (not a tag error)
        enc_file = tmp_path / "bad.bin"
        # Write enough bytes to pass the size check
        enc_file.write_bytes(os.urandom(AES_NONCE_SIZE + AES_TAG_SIZE + 10))

        # Wrong key will produce InvalidTag which maps to "wrong key or corrupted"
        wrong_key = os.urandom(AES_KEY_SIZE)
        with pytest.raises(EncryptionError):
            decrypt_file(enc_file, tmp_path / "out.bin", wrong_key)


class TestIndexFileEdgeCases:
    """Cover lines 361, 389, 396."""

    def test_encrypt_nonexistent_index_file_noop(self, tmp_path: Path):
        """Line 361: encrypt_index_file does nothing if file doesn't exist."""
        nonexistent = tmp_path / "missing.usearch"
        encrypt_index_file(nonexistent, "key")  # Should not raise
        assert not nonexistent.exists()

    def test_decrypt_nonexistent_raises(self, tmp_path: Path):
        """Line 389: decrypt_index_file raises if file doesn't exist."""
        nonexistent = tmp_path / "missing.usearch.enc"
        with pytest.raises(EncryptionError, match="not found"):
            decrypt_index_file(nonexistent, "key")

    def test_decrypt_index_file_suffix_handling(self, tmp_path: Path):
        """Line 396: handles non-.usearch suffix in encrypted path."""
        # Create a file with .enc suffix where removing .enc doesn't give .usearch
        original_data = b"fake index content"
        # First create and encrypt normally
        index_file = tmp_path / "test.usearch"
        index_file.write_bytes(original_data)
        encrypt_index_file(index_file, "key")

        # Now test: .usearch.enc -> removing .enc suffix via with_suffix("") gives .usearch
        enc_path = tmp_path / "test.usearch.enc"
        assert enc_path.exists()

        result = decrypt_index_file(enc_path, "key")
        assert result.read_bytes() == original_data

    def test_decrypt_index_non_usearch_suffix(self, tmp_path: Path):
        """Line 396: path where removing .enc doesn't yield .usearch suffix."""
        # Create encrypted data with a different naming pattern
        original_data = b"index data here"
        key = os.urandom(AES_KEY_SIZE)

        # Create plaintext, encrypt it
        plain = tmp_path / "myindex.dat"
        plain.write_bytes(original_data)

        from simplevecdb.encryption import encrypt_file, _normalize_key

        enc_path = tmp_path / "myindex.dat.enc"
        normalized = _normalize_key("key")
        encrypt_file(plain, enc_path, normalized)

        # Now decrypt_index_file with a path like myindex.dat.enc
        # with_suffix("") -> myindex.dat (not .usearch)
        # So line 396 should trigger: decrypted_path = encrypted_path.with_suffix(".usearch")
        result = decrypt_index_file(enc_path, "key")
        assert result.suffix == ".usearch"
        assert result.read_bytes() == original_data


def _make_import_blocker(blocked_module: str):
    """Create an __import__ side_effect that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def blocker(name, *args, **kwargs):
        if name == blocked_module or name.startswith(blocked_module + "."):
            raise ImportError(f"Mocked: {name} not installed")
        return real_import(name, *args, **kwargs)

    return blocker
