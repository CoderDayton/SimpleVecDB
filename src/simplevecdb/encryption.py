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
    Included in the standard install. If missing, reinstall:
    pip install --force-reinstall simplevecdb
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import secrets
import sqlite3
from collections import OrderedDict
from pathlib import Path
from threading import Lock

_logger = logging.getLogger("simplevecdb.encryption")

# Constants
AES_KEY_SIZE = 32  # 256 bits
AES_NONCE_SIZE = 12  # 96 bits for GCM
AES_TAG_SIZE = 16  # 128 bits
SALT_SIZE = 16
PBKDF2_ITERATIONS = 600000  # OWASP 2024 recommendation for SHA-256
# Fixed salt for deterministic key normalization (SQLCipher/index compatibility)
_NORMALIZE_KEY_SALT = b"simplevecdb-sqlcipher-key"

# Encrypted file format
#   v0 (pre-2.6.0): nonce(12) | ciphertext+tag(N+16)
#   v1 (2.6.0+):    magic(2)='SV' | version(1)=1 | nonce(12) | ciphertext+tag
# v1 is written by encrypt_file; decrypt_file accepts both formats so
# existing encrypted indexes keep working.
_ENC_MAGIC = b"SV"
_ENC_VERSION = 1
_ENC_HEADER_LEN = len(_ENC_MAGIC) + 1  # magic + version byte


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""

    pass


class EncryptionUnavailableError(ImportError):
    """Raised when encryption dependencies are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Encryption requires sqlcipher3-binary and cryptography. "
            "Reinstall simplevecdb: pip install --force-reinstall simplevecdb"
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


# Bounded LRU of derived keys, so repeated encrypt/decrypt round-trips
# avoid the 600k-iteration PBKDF2 cost. Keyed by ``(passphrase_bytes,
# salt_bytes)``. Long-running multi-tenant processes that open many DBs with
# distinct passphrases must not retain key material indefinitely, hence the
# cap. Access is serialized by ``_NORMALIZE_KEY_CACHE_LOCK`` so concurrent
# normalization from multiple threads stays consistent.
_NORMALIZE_KEY_CACHE_MAX = 64
_NORMALIZE_KEY_CACHE: "OrderedDict[tuple[bytes, bytes], bytes]" = OrderedDict()
_NORMALIZE_KEY_CACHE_LOCK = Lock()


def _normalize_key(key: str | bytes, salt: bytes | None = None) -> bytes:
    """
    Normalize encryption key to 32 bytes.

    If key is already 32 bytes, use directly. Otherwise, derive using
    PBKDF2 with the supplied salt, falling back to the legacy fixed salt
    when none is provided (preserves backwards compatibility for any
    encrypted database/index created before per-DB salts were introduced).
    """
    if isinstance(key, bytes) and len(key) == AES_KEY_SIZE:
        return key

    salt_to_use = salt if salt is not None else _NORMALIZE_KEY_SALT

    # Cache key includes both the raw passphrase bytes and the salt so the
    # same passphrase yields different cache entries for different DBs.
    key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    cache_key = (key_bytes, salt_to_use)

    with _NORMALIZE_KEY_CACHE_LOCK:
        cached = _NORMALIZE_KEY_CACHE.get(cache_key)
        if cached is not None:
            _NORMALIZE_KEY_CACHE.move_to_end(cache_key)
            return cached

    derived = _derive_key(key, salt_to_use)

    with _NORMALIZE_KEY_CACHE_LOCK:
        _NORMALIZE_KEY_CACHE[cache_key] = derived
        _NORMALIZE_KEY_CACHE.move_to_end(cache_key)
        while len(_NORMALIZE_KEY_CACHE) > _NORMALIZE_KEY_CACHE_MAX:
            _NORMALIZE_KEY_CACHE.popitem(last=False)
    return derived


_SALT_SIDECAR_SUFFIX = ".salt"


def _resolve_salt(
    resource_path: Path,
    *,
    create_if_missing: bool,
) -> bytes:
    """Return the salt for a given encrypted resource (DB or index file).

    Looks for a sibling ``<resource>.salt`` file:
    - If it exists: read it and return (per-DB random salt).
    - If absent and ``create_if_missing=True``: generate a random salt,
      write it atomically with mode 0o600, and return.
    - If absent and ``create_if_missing=False``: return the legacy fixed
      salt so encrypted resources created before per-DB salts (i.e.,
      pre-2.6.0) keep working unchanged.

    The sidecar salt is not secret on its own — it is only the input to
    PBKDF2 — but restricting its mode prevents accidental world-read in
    container environments with a permissive umask.
    """
    salt_path = resource_path.with_name(resource_path.name + _SALT_SIDECAR_SUFFIX)
    try:
        if salt_path.exists():
            data = salt_path.read_bytes()
            if len(data) == SALT_SIZE:
                return data
            _logger.warning(
                "Salt sidecar %s has unexpected size %d (expected %d); "
                "falling back to legacy fixed salt.",
                salt_path,
                len(data),
                SALT_SIZE,
            )
            return _NORMALIZE_KEY_SALT
    except OSError as exc:
        _logger.warning(
            "Could not read salt sidecar %s: %s. Falling back to legacy salt.",
            salt_path,
            exc,
        )
        return _NORMALIZE_KEY_SALT

    if not create_if_missing:
        # Legacy resource — created before per-DB salts existed.
        return _NORMALIZE_KEY_SALT

    salt = secrets.token_bytes(SALT_SIZE)
    try:
        # O_EXCL guards against two concurrent openers racing to write
        # different random salts (whichever lost would silently derive a
        # wrong key). It also prevents clobbering a sidecar that was
        # already created out-of-band — for an existing DB whose data
        # file was zero-bytes for some reason, overwriting the sidecar
        # would render the DB permanently unreadable.
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(salt_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(fd, salt)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            dir_fd = os.open(str(salt_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            # Lost a race against another writer; read whichever salt
            # they wrote and use that instead so both processes agree.
            try:
                data = salt_path.read_bytes()
                if len(data) == SALT_SIZE:
                    return data
            except OSError:
                pass
        _logger.warning(
            "Could not write salt sidecar %s (%s); using legacy fixed salt. "
            "Per-DB salt protection is disabled until the path is writable.",
            salt_path,
            exc,
        )
        return _NORMALIZE_KEY_SALT
    return salt


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

    # Resolve the per-DB salt. For a brand-new database, generate and write
    # a random sidecar; for an existing one, prefer the sidecar but fall
    # back to the legacy fixed salt so pre-2.6.0 databases keep opening.
    db_path_obj = Path(path_str)
    is_new_db = not db_path_obj.exists() or db_path_obj.stat().st_size == 0
    salt = _resolve_salt(db_path_obj, create_if_missing=is_new_db)

    try:
        conn = sqlcipher.connect(  # type: ignore[attr-defined]
            path_str,
            check_same_thread=check_same_thread,
            timeout=timeout,
        )

        # Set the encryption key using PRAGMA. PRAGMA arguments cannot be
        # parameterized via ``?`` placeholders, so we must interpolate. To
        # avoid ever interpolating user-supplied passphrase characters into a
        # quoted SQL string (where escape rules are subtle and SQLCipher
        # parsing has surprising edge cases), normalize *every* key path to
        # 32 bytes via ``_normalize_key`` and feed it as ``x'hex'`` raw-key
        # form. The hex output is restricted to ``[0-9a-f]`` so no quoting
        # or escaping is needed.
        normalized_key = _normalize_key(key, salt=salt)
        conn.execute(f"PRAGMA key = \"x'{normalized_key.hex()}'\"")

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


def _is_zero_byte(path: Path) -> bool:
    """Treat a missing or empty file as 'definitely not encrypted'."""
    try:
        return path.stat().st_size == 0
    except OSError:
        return True


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
    # A zero-byte file would cause sqlite3 to create a fresh DB and return
    # False, masking a missing/corrupt database as unencrypted. Treat empty
    # files as not encrypted but also not a real DB; callers that need to
    # distinguish should check existence and size themselves.
    if _is_zero_byte(path):
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


def _atomic_write_bytes(target: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Write ``data`` to ``target`` atomically with restricted permissions.

    Writes to a sibling ``.tmp`` file, fsyncs it, sets the file mode, then
    ``os.replace()`` onto the target. A crash leaves at most an orphan temp
    file; the target path is never partially written. The directory is also
    fsynced so the rename itself is durable on POSIX.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    fd = os.open(
        str(tmp_path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        mode,
    )
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(str(tmp_path), mode)
    except OSError:
        pass
    os.replace(str(tmp_path), str(target))
    try:
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


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

        # Encrypt with AES-GCM. Bind the v1 header (magic + version) into
        # the GCM AAD so any tampering with the header bytes — including a
        # downgrade attempt that strips the version byte — fails
        # authentication on decrypt.
        header = _ENC_MAGIC + bytes([_ENC_VERSION])
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=header)

        # Atomically write: header + nonce + ciphertext to a sibling temp
        # file, fsync, restrict permissions, then os.replace() onto the
        # target path. A crash mid-write leaves only the temp file; the
        # target is never torn.
        _atomic_write_bytes(
            output_path, header + nonce + ciphertext, mode=0o600
        )

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

        # Detect format. v1 starts with magic 'SV' + version byte; anything
        # else is treated as v0 (pre-2.6.0) for backwards compatibility.
        # The v1 header is bound into the GCM AAD on encrypt, so any
        # tampering with the magic/version bytes — including a downgrade
        # attempt that strips them — fails authentication here.
        associated_data: bytes | None = None
        if (
            len(data) >= _ENC_HEADER_LEN
            and data[: len(_ENC_MAGIC)] == _ENC_MAGIC
            and data[len(_ENC_MAGIC)] == _ENC_VERSION
        ):
            associated_data = bytes(data[:_ENC_HEADER_LEN])
            data = data[_ENC_HEADER_LEN:]

        if len(data) < AES_NONCE_SIZE + AES_TAG_SIZE:
            raise EncryptionError("Encrypted file too small to be valid")

        # Extract nonce and ciphertext
        nonce = data[:AES_NONCE_SIZE]
        ciphertext = data[AES_NONCE_SIZE:]

        # Decrypt with AES-GCM. associated_data is bound to the v1 header
        # (or None for legacy v0 files where no header was present).
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=associated_data)

        # Write decrypted data atomically. Plaintext is sensitive — restrict
        # permissions to owner only.
        _atomic_write_bytes(output_path, plaintext, mode=0o600)

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

    encrypted_path = index_path.with_suffix(".usearch.enc")
    # Generate a random salt sidecar if one doesn't already exist. This
    # covers two cases:
    #  - First-ever encryption (no .enc, no .salt) → create sidecar.
    #  - Re-encryption of a legacy v0 blob (.enc present, no .salt) →
    #    create sidecar so the next read uses per-DB salts. This is the
    #    migration path from v0 to v1.
    # If a sidecar already exists, _resolve_salt returns it unchanged
    # (the O_EXCL guard inside also prevents accidental clobbering).
    salt_path = encrypted_path.with_name(encrypted_path.name + _SALT_SIDECAR_SUFFIX)
    needs_sidecar = not salt_path.exists()
    salt = _resolve_salt(encrypted_path, create_if_missing=needs_sidecar)
    normalized_key = _normalize_key(key, salt=salt)

    # encrypt_file is atomic (tmp + fsync + os.replace + chmod 0o600), so by
    # the time it returns the encrypted blob is durably on disk. Only then is
    # it safe to remove the plaintext copy. A crash inside encrypt_file leaves
    # the plaintext intact; a crash between the call and the unlink leaves
    # both files (the encrypted side wins on next open).
    encrypt_file(index_path, encrypted_path, normalized_key)
    try:
        index_path.unlink()
    except FileNotFoundError:
        pass

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

    # Existing encrypted file: prefer the sidecar salt (if present) and
    # fall back to the legacy fixed salt for files written before per-DB
    # salts existed. We never *create* a sidecar during decryption.
    salt = _resolve_salt(encrypted_path, create_if_missing=False)
    normalized_key = _normalize_key(key, salt=salt)

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
