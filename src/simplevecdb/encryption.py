"""
Encryption support for SimpleVecDB.

Provides at-rest encryption for both SQLite metadata (via SQLCipher)
and usearch index files (via AES-256-GCM).

Design principles:
- Zero performance overhead during search operations
- Index files are encrypted only at rest (decrypt on load, encrypt on save)
- SQLCipher provides transparent page-level encryption with AES-NI acceleration

Usage:
    from simplevecdb import VectorDB

    # Enable encryption with a passphrase
    db = VectorDB("secure.db", encryption_key="my-secret-passphrase")

    # Or with raw bytes (32 bytes for AES-256)
    db = VectorDB("secure.db", encryption_key=os.urandom(32))

Requirements:
    pip install simplevecdb[encryption]
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_logger = logging.getLogger("simplevecdb.encryption")

# Constants
AES_KEY_SIZE = 32  # 256 bits
AES_NONCE_SIZE = 12  # 96 bits for GCM
AES_TAG_SIZE = 16  # 128 bits
SALT_SIZE = 16
PBKDF2_ITERATIONS = 480000  # OWASP 2023 recommendation for SHA-256
# Fixed salt for deterministic key normalization (SQLCipher/index compatibility)
_NORMALIZE_KEY_SALT = b"simplevecdb-sqlcipher-key"


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""

    pass


class EncryptionUnavailableError(ImportError):
    """Raised when encryption dependencies are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Encryption requires additional dependencies. "
            "Install with: pip install simplevecdb[encryption]"
        )


def _derive_key(passphrase: str | bytes, salt: bytes) -> bytes:
    """
    Derive a 256-bit AES key from a passphrase using PBKDF2-SHA256.

    Args:
        passphrase: User-provided passphrase (string or bytes)
        salt: Random salt for key derivation

    Returns:
        32-byte derived key suitable for AES-256
    """
    if isinstance(passphrase, str):
        passphrase = passphrase.encode("utf-8")

    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase,
        salt,
        PBKDF2_ITERATIONS,
        dklen=AES_KEY_SIZE,
    )


def _normalize_key(key: str | bytes) -> bytes:
    """
    Normalize encryption key to 32 bytes.

    If key is already 32 bytes, use directly.
    Otherwise, derive using PBKDF2 with a fixed salt (for SQLCipher compatibility).
    """
    if isinstance(key, bytes) and len(key) == AES_KEY_SIZE:
        return key

    # Use PBKDF2 with a deterministic salt for consistency
    # This allows the same passphrase to always produce the same key
    if isinstance(key, str):
        key_bytes = key.encode("utf-8")
    else:
        key_bytes = key

    # Derive a 32-byte key using PBKDF2-HMAC-SHA256 with a fixed salt
    # This is intentionally deterministic (same input -> same key) while being
    # computationally expensive enough for password-like passphrases.
    return hashlib.pbkdf2_hmac(
        "sha256",
        key_bytes,
        _NORMALIZE_KEY_SALT,
        PBKDF2_ITERATIONS,
        dklen=AES_KEY_SIZE,
    )


# ============================================================================
# SQLCipher Connection Factory
# ============================================================================


def create_encrypted_connection(
    path: str | Path,
    key: str | bytes,
    *,
    check_same_thread: bool = False,
    timeout: float = 30.0,
) -> sqlite3.Connection:
    """
    Create an encrypted SQLite connection using SQLCipher.

    SQLCipher provides transparent AES-256 encryption at the page level,
    with hardware acceleration on CPUs supporting AES-NI.

    Args:
        path: Database file path (cannot be ":memory:" for encryption)
        key: Encryption passphrase or 32-byte raw key
        check_same_thread: SQLite thread-safety setting
        timeout: Connection timeout in seconds

    Returns:
        sqlite3.Connection with encryption enabled

    Raises:
        EncryptionUnavailableError: If sqlcipher3 is not installed
        EncryptionError: If encryption setup fails
        ValueError: If trying to encrypt an in-memory database
    """
    path_str = str(path)

    if path_str == ":memory:":
        raise ValueError(
            "In-memory databases cannot be encrypted. "
            "Use a file path for encrypted databases."
        )

    try:
        from sqlcipher3 import dbapi2 as sqlcipher  # type: ignore
    except ImportError:
        raise EncryptionUnavailableError()

    try:
        conn = sqlcipher.connect(  # type: ignore[attr-defined]
            path_str,
            check_same_thread=check_same_thread,
            timeout=timeout,
        )

        # Set the encryption key using PRAGMA
        # SQLCipher accepts both raw keys (x'hex') and passphrases
        if isinstance(key, bytes) and len(key) == AES_KEY_SIZE:
            # Use raw key format
            hex_key = key.hex()
            conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
        else:
            # Use passphrase (SQLCipher will derive key internally)
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            # Escape single quotes in passphrase
            escaped_key = key.replace("'", "''")
            conn.execute(f"PRAGMA key = '{escaped_key}'")

        # Verify encryption is working by querying cipher_version
        try:
            result = conn.execute("PRAGMA cipher_version").fetchone()
            if result is None:
                raise EncryptionError(
                    "SQLCipher encryption not active. Database may be corrupted "
                    "or key is incorrect."
                )
            _logger.debug("SQLCipher version: %s", result[0])

            # Validate key by actually reading from the database
            # cipher_version only confirms SQLCipher is loaded, not that the key is correct
            # Reading sqlite_master forces decryption and will fail with wrong key
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except EncryptionError:
            raise
        except Exception as e:
            conn.close()
            raise EncryptionError(
                f"Failed to verify encryption (wrong key?): {e}"
            ) from e

        # Set performance optimizations (same as non-encrypted)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        return conn  # type: ignore[return-value]

    except EncryptionError:
        raise
    except Exception as e:
        raise EncryptionError(f"Failed to create encrypted connection: {e}") from e


def is_database_encrypted(path: str | Path) -> bool:
    """
    Check if a database file is encrypted.

    Attempts to open with standard sqlite3. If it fails with "not a database",
    the file is likely encrypted.

    Args:
        path: Path to database file

    Returns:
        True if database appears to be encrypted
    """
    path = Path(path)
    if not path.exists():
        return False

    try:
        conn = sqlite3.connect(str(path))
        # Try to read the schema - this will fail if encrypted
        conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        conn.close()
        return False
    except sqlite3.DatabaseError as e:
        if "not a database" in str(e).lower() or "encrypted" in str(e).lower():
            return True
        raise


# ============================================================================
# Index File Encryption (AES-256-GCM)
# ============================================================================


def encrypt_file(
    input_path: Path,
    output_path: Path,
    key: bytes,
) -> None:
    """
    Encrypt a file using AES-256-GCM.

    File format:
    - 12 bytes: nonce
    - N bytes: ciphertext
    - 16 bytes: GCM auth tag (appended by cryptography)

    Args:
        input_path: Path to plaintext file
        output_path: Path for encrypted output
        key: 32-byte encryption key

    Raises:
        EncryptionUnavailableError: If cryptography is not installed
        EncryptionError: If encryption fails
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise EncryptionUnavailableError()

    try:
        # Read plaintext
        plaintext = input_path.read_bytes()

        # Generate random nonce (MUST be unique per encryption)
        nonce = secrets.token_bytes(AES_NONCE_SIZE)

        # Encrypt with AES-GCM
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

        # Write: nonce + ciphertext (tag is appended by cryptography)
        output_path.write_bytes(nonce + ciphertext)

        _logger.debug(
            "Encrypted %d bytes -> %d bytes",
            len(plaintext),
            len(nonce) + len(ciphertext),
        )

    except Exception as e:
        raise EncryptionError(f"Failed to encrypt file: {e}") from e


def decrypt_file(
    input_path: Path,
    output_path: Path,
    key: bytes,
) -> None:
    """
    Decrypt a file encrypted with encrypt_file().

    Args:
        input_path: Path to encrypted file
        output_path: Path for decrypted output
        key: 32-byte encryption key

    Raises:
        EncryptionUnavailableError: If cryptography is not installed
        EncryptionError: If decryption fails (wrong key, corrupted data)
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise EncryptionUnavailableError()

    try:
        # Read encrypted data
        data = input_path.read_bytes()

        if len(data) < AES_NONCE_SIZE + AES_TAG_SIZE:
            raise EncryptionError("Encrypted file too small to be valid")

        # Extract nonce and ciphertext
        nonce = data[:AES_NONCE_SIZE]
        ciphertext = data[AES_NONCE_SIZE:]

        # Decrypt with AES-GCM
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

        # Write decrypted data
        output_path.write_bytes(plaintext)

        _logger.debug(
            "Decrypted %d bytes -> %d bytes",
            len(data),
            len(plaintext),
        )

    except EncryptionError:
        raise
    except Exception as e:
        # InvalidTag is raised by cryptography when decryption fails (wrong key/corrupted)
        exc_type = type(e).__name__
        exc_msg = str(e).lower()
        if "tag" in exc_type.lower() or "tag" in exc_msg or "authentication" in exc_msg:
            raise EncryptionError(
                "Decryption failed: wrong key or corrupted data"
            ) from e
        raise EncryptionError(f"Failed to decrypt file: {e}") from e


def encrypt_index_file(index_path: Path, key: str | bytes) -> None:
    """
    Encrypt a usearch index file in-place.

    The original file is replaced with the encrypted version.
    File extension changes from .usearch to .usearch.enc

    Args:
        index_path: Path to .usearch index file
        key: Encryption passphrase or 32-byte raw key
    """
    if not index_path.exists():
        return

    normalized_key = _normalize_key(key)
    encrypted_path = index_path.with_suffix(".usearch.enc")

    encrypt_file(index_path, encrypted_path, normalized_key)

    # Remove original unencrypted file
    index_path.unlink()

    _logger.info("Encrypted index: %s -> %s", index_path, encrypted_path)


def decrypt_index_file(encrypted_path: Path, key: str | bytes) -> Path:
    """
    Decrypt a usearch index file.

    Decrypts to a temporary location for use during runtime.
    Returns the path to the decrypted file.

    Args:
        encrypted_path: Path to .usearch.enc file
        key: Encryption passphrase or 32-byte raw key

    Returns:
        Path to decrypted .usearch file
    """
    if not encrypted_path.exists():
        raise EncryptionError(f"Encrypted index not found: {encrypted_path}")

    normalized_key = _normalize_key(key)

    # Decrypt to same location without .enc suffix
    decrypted_path = encrypted_path.with_suffix("")
    if decrypted_path.suffix != ".usearch":
        decrypted_path = encrypted_path.with_suffix(".usearch")

    decrypt_file(encrypted_path, decrypted_path, normalized_key)

    _logger.info("Decrypted index: %s -> %s", encrypted_path, decrypted_path)
    return decrypted_path


def get_encrypted_index_path(index_path: Path) -> Path | None:
    """
    Check if an encrypted version of the index exists.

    Args:
        index_path: Expected path to .usearch index

    Returns:
        Path to .usearch.enc if it exists, None otherwise
    """
    encrypted_path = Path(str(index_path) + ".enc")
    if encrypted_path.exists():
        return encrypted_path
    return None
