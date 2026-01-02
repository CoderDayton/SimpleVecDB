"""
Tests for encryption functionality.

These tests cover:
- SQLCipher encrypted database connections
- Usearch index file encryption/decryption
- Full encrypted VectorDB workflow
- Error handling for missing keys and wrong keys
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from simplevecdb import VectorDB, EncryptionError, EncryptionUnavailableError
from simplevecdb.encryption import (
    _normalize_key,
    _derive_key,
    encrypt_file,
    decrypt_file,
    encrypt_index_file,
    decrypt_index_file,
    get_encrypted_index_path,
    is_database_encrypted,
    create_encrypted_connection,
    AES_KEY_SIZE,
    SALT_SIZE,
)

# Test-only encryption keys - NOT for production use
# nosec: B105 - hardcoded passwords are intentional for testing
TEST_ENCRYPTION_KEY = "test-only-key-do-not-use-in-production"  # noqa: S105
TEST_ENCRYPTION_KEY_ALT = "alternate-test-key-for-wrong-key-tests"  # noqa: S105


# Skip all tests if encryption dependencies are not available
pytest.importorskip("cryptography")


class TestKeyDerivation:
    """Test key derivation functions."""

    def test_normalize_key_raw_32_bytes(self):
        """32-byte raw key should be used directly."""
        raw_key = os.urandom(32)
        result = _normalize_key(raw_key)
        assert result == raw_key

    def test_normalize_key_passphrase(self):
        """String passphrase should derive consistent key."""
        key1 = _normalize_key(TEST_ENCRYPTION_KEY)
        key2 = _normalize_key(TEST_ENCRYPTION_KEY)
        assert key1 == key2
        assert len(key1) == AES_KEY_SIZE

    def test_normalize_key_short_bytes(self):
        """Short bytes should be hashed to 32 bytes."""
        short_key = b"short"
        result = _normalize_key(short_key)
        assert len(result) == AES_KEY_SIZE
        assert result != short_key

    def test_derive_key_with_salt(self):
        """Key derivation with salt should be deterministic."""
        salt = os.urandom(SALT_SIZE)
        key1 = _derive_key(TEST_ENCRYPTION_KEY, salt)
        key2 = _derive_key(TEST_ENCRYPTION_KEY, salt)
        assert key1 == key2
        assert len(key1) == AES_KEY_SIZE

    def test_derive_key_different_salt(self):
        """Different salts should produce different keys."""
        salt1 = os.urandom(SALT_SIZE)
        salt2 = os.urandom(SALT_SIZE)
        key1 = _derive_key(TEST_ENCRYPTION_KEY, salt1)
        key2 = _derive_key(TEST_ENCRYPTION_KEY_ALT, salt2)
        assert key1 != key2


class TestFileEncryption:
    """Test file encryption/decryption functions."""

    def test_encrypt_decrypt_roundtrip(self, tmp_path: Path):
        """Encrypt and decrypt should preserve content."""
        plaintext = b"Hello, World! " * 1000  # ~14KB
        key = os.urandom(AES_KEY_SIZE)

        input_file = tmp_path / "plaintext.bin"
        encrypted_file = tmp_path / "encrypted.bin"
        decrypted_file = tmp_path / "decrypted.bin"

        input_file.write_bytes(plaintext)

        encrypt_file(input_file, encrypted_file, key)
        decrypt_file(encrypted_file, decrypted_file, key)

        assert decrypted_file.read_bytes() == plaintext

    def test_encrypted_file_is_different(self, tmp_path: Path):
        """Encrypted content should differ from plaintext."""
        plaintext = b"Secret data"
        key = os.urandom(AES_KEY_SIZE)

        input_file = tmp_path / "plaintext.bin"
        encrypted_file = tmp_path / "encrypted.bin"

        input_file.write_bytes(plaintext)
        encrypt_file(input_file, encrypted_file, key)

        encrypted_content = encrypted_file.read_bytes()
        assert encrypted_content != plaintext
        assert len(encrypted_content) > len(plaintext)  # nonce + tag overhead

    def test_decrypt_wrong_key_fails(self, tmp_path: Path):
        """Decryption with wrong key should fail."""
        plaintext = b"Secret data"
        key1 = os.urandom(AES_KEY_SIZE)
        key2 = os.urandom(AES_KEY_SIZE)

        input_file = tmp_path / "plaintext.bin"
        encrypted_file = tmp_path / "encrypted.bin"
        output_file = tmp_path / "decrypted.bin"

        input_file.write_bytes(plaintext)
        encrypt_file(input_file, encrypted_file, key1)

        with pytest.raises(EncryptionError, match="wrong key or corrupted"):
            decrypt_file(encrypted_file, output_file, key2)

    def test_decrypt_corrupted_file_fails(self, tmp_path: Path):
        """Decryption of corrupted file should fail."""
        plaintext = b"Secret data"
        key = os.urandom(AES_KEY_SIZE)

        input_file = tmp_path / "plaintext.bin"
        encrypted_file = tmp_path / "encrypted.bin"
        output_file = tmp_path / "decrypted.bin"

        input_file.write_bytes(plaintext)
        encrypt_file(input_file, encrypted_file, key)

        # Corrupt the encrypted file
        corrupted = bytearray(encrypted_file.read_bytes())
        corrupted[20] ^= 0xFF
        encrypted_file.write_bytes(bytes(corrupted))

        with pytest.raises(EncryptionError):
            decrypt_file(encrypted_file, output_file, key)


class TestIndexFileEncryption:
    """Test usearch index file encryption."""

    def test_encrypt_index_file_creates_enc(self, tmp_path: Path):
        """encrypt_index_file should create .enc file and remove original."""
        index_file = tmp_path / "test.usearch"
        index_file.write_bytes(b"fake index data")

        encrypt_index_file(index_file, TEST_ENCRYPTION_KEY)

        assert not index_file.exists()
        assert (tmp_path / "test.usearch.enc").exists()

    def test_decrypt_index_file_restores(self, tmp_path: Path):
        """decrypt_index_file should restore original content."""
        original_data = b"fake index data " * 100
        index_file = tmp_path / "test.usearch"
        index_file.write_bytes(original_data)

        encrypt_index_file(index_file, TEST_ENCRYPTION_KEY)
        encrypted_path = tmp_path / "test.usearch.enc"

        decrypted_path = decrypt_index_file(encrypted_path, TEST_ENCRYPTION_KEY)

        assert decrypted_path.read_bytes() == original_data

    def test_get_encrypted_index_path(self, tmp_path: Path):
        """get_encrypted_index_path should find .enc files."""
        index_path = tmp_path / "test.usearch"
        encrypted_path = tmp_path / "test.usearch.enc"

        # No encrypted file
        assert get_encrypted_index_path(index_path) is None

        # Create encrypted file
        encrypted_path.write_bytes(b"encrypted data")
        assert get_encrypted_index_path(index_path) == encrypted_path


class TestEncryptedVectorDB:
    """Test full VectorDB encryption workflow."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "encrypted_test.db"

    def test_encrypted_db_roundtrip(self, db_path: Path):
        """Full workflow: create encrypted DB, add data, reopen, search."""
        dim = 128
        texts = ["hello world", "foo bar", "test document"]
        embeddings = np.random.randn(len(texts), dim).astype(np.float32).tolist()

        # Create and populate
        db = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        collection = db.collection("test")
        collection.add_texts(texts, embeddings=embeddings)
        db.save()
        db.close()

        # Verify files are encrypted
        assert is_database_encrypted(db_path)
        # Index should be encrypted after save
        encrypted_index = get_encrypted_index_path(Path(f"{db_path}.test.usearch"))
        assert encrypted_index is not None

        # Reopen with same key
        db2 = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        collection2 = db2.collection("test")

        # Should find the documents
        results = collection2.similarity_search(embeddings[0], k=1)
        assert len(results) == 1
        assert results[0][0].page_content == texts[0]

        db2.close()

    def test_encrypted_db_wrong_key_fails(self, db_path: Path):
        """Opening encrypted DB with wrong key should fail."""
        # Create with one key
        db = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        db.collection("test")
        db.close()

        # Try to open with wrong key
        with pytest.raises(EncryptionError):
            VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY_ALT)

    def test_encrypted_db_no_key_fails(self, db_path: Path):
        """Opening encrypted DB without key should fail."""
        # Create encrypted
        db = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        db.collection("test")
        db.close()

        # Try to open without key - should fail as SQLite can't read it
        with pytest.raises(sqlite3.DatabaseError):
            VectorDB(db_path)

    def test_memory_db_encryption_not_allowed(self):
        """In-memory databases cannot be encrypted."""
        with pytest.raises(ValueError, match="In-memory databases cannot be encrypted"):
            VectorDB(":memory:", encryption_key=TEST_ENCRYPTION_KEY)

    def test_encrypted_db_search_performance(self, db_path: Path):
        """Search performance should not degrade with encryption enabled."""
        import time

        dim = 128
        n_vectors = 1000

        texts = [f"document {i}" for i in range(n_vectors)]
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32).tolist()

        # Create and populate
        db = VectorDB(db_path, encryption_key=TEST_ENCRYPTION_KEY)
        collection = db.collection("perf")
        collection.add_texts(texts, embeddings=embeddings)

        # Search (should not involve crypto - index is in memory)
        query = np.random.randn(dim).astype(np.float32).tolist()

        times = []
        for _ in range(10):
            start = time.perf_counter()
            collection.similarity_search(query, k=10)
            times.append(time.perf_counter() - start)

        avg_time_ms = (sum(times) / len(times)) * 1000
        # Should be fast - no crypto overhead during search
        assert avg_time_ms < 50, f"Search too slow: {avg_time_ms:.2f}ms"

        db.close()


class TestSQLCipherConnection:
    """Test SQLCipher connection factory."""

    def test_create_encrypted_connection(self, tmp_path: Path):
        """Should create working encrypted connection."""
        try:
            conn = create_encrypted_connection(
                tmp_path / "test.db",
                TEST_ENCRYPTION_KEY,
            )
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
            conn.execute("INSERT INTO test (data) VALUES (?)", ("hello",))
            conn.commit()

            result = conn.execute("SELECT data FROM test").fetchone()
            assert result[0] == "hello"
            conn.close()
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")

    def test_encrypted_connection_memory_not_allowed(self):
        """In-memory encrypted connections should fail."""
        try:
            with pytest.raises(ValueError, match="In-memory"):
                create_encrypted_connection(":memory:", TEST_ENCRYPTION_KEY)
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")

    def test_is_database_encrypted(self, tmp_path: Path):
        """is_database_encrypted should detect encrypted files."""
        import sqlite3

        # Create unencrypted database
        unencrypted_path = tmp_path / "plain.db"
        conn = sqlite3.connect(str(unencrypted_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        assert not is_database_encrypted(unencrypted_path)

        # Create encrypted database
        try:
            encrypted_path = tmp_path / "encrypted.db"
            conn = create_encrypted_connection(encrypted_path, TEST_ENCRYPTION_KEY)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            assert is_database_encrypted(encrypted_path)
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")

    def test_create_encrypted_connection_raw_key(self, tmp_path: Path):
        """Should work with raw 32-byte key."""
        try:
            raw_key = os.urandom(AES_KEY_SIZE)  # 32 bytes
            conn = create_encrypted_connection(
                tmp_path / "test_raw.db",
                raw_key,
            )
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            # Reopen with same key
            conn2 = create_encrypted_connection(tmp_path / "test_raw.db", raw_key)
            result = conn2.execute("SELECT count(*) FROM test").fetchone()
            assert result[0] == 0
            conn2.close()
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")

    def test_create_encrypted_connection_bytes_passphrase(self, tmp_path: Path):
        """Should work with bytes passphrase (non-32-byte)."""
        try:
            bytes_key = (
                b"short-bytes-key"  # Not 32 bytes, will be treated as passphrase
            )
            conn = create_encrypted_connection(
                tmp_path / "test_bytes.db",
                bytes_key,
            )
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

            # Reopen with same key
            conn2 = create_encrypted_connection(tmp_path / "test_bytes.db", bytes_key)
            result = conn2.execute("SELECT count(*) FROM test").fetchone()
            assert result[0] == 0
            conn2.close()
        except EncryptionUnavailableError:
            pytest.skip("sqlcipher3 not installed")
