# Encryption

SimpleVecDB supports **at-rest encryption** for both SQLite metadata and usearch index files.

## Overview

- **SQLite encryption**: Uses SQLCipher for transparent page-level AES-256 encryption with hardware acceleration (AES-NI)
- **Index file encryption**: Uses AES-256-GCM to encrypt usearch HNSW index files
- **Zero runtime overhead**: Index files are decrypted on load and encrypted on save; search operations have no crypto overhead

## Installation

```bash
pip install simplevecdb[encryption]
```

This installs:
- `sqlcipher3-binary` - SQLCipher Python bindings
- `cryptography` - AES-GCM implementation

## Basic Usage

```python
from simplevecdb import VectorDB

# Create encrypted database with a passphrase
db = VectorDB("secure.db", encryption_key="my-secret-passphrase")

# Use normally - encryption is transparent
collection = db.collection("documents")
collection.add_texts(
    ["Confidential document content"],
    embeddings=[[0.1] * 384]
)

# Save encrypts the index file
db.save()
db.close()

# Reopen with the same key
db = VectorDB("secure.db", encryption_key="my-secret-passphrase")
results = db.collection("documents").similarity_search([0.1] * 384, k=5)
```

## Key Management

### Passphrase vs Raw Key

```python
# Option 1: Passphrase (string) - internally derived to 32-byte key
db = VectorDB("secure.db", encryption_key="my-secret-passphrase")

# Option 2: Raw 32-byte key (more secure, use with a key management system)
import os
raw_key = os.urandom(32)  # Generate once, store securely
db = VectorDB("secure.db", encryption_key=raw_key)
```

### Best Practices

1. **Never hardcode keys** - Use environment variables or a secrets manager
2. **Use strong passphrases** - At least 20 characters with mixed case/numbers/symbols
3. **Backup your key** - If you lose the key, the data is unrecoverable
4. **Consider key rotation** - Re-encrypt periodically for compliance

```python
import os
from simplevecdb import VectorDB

# Load key from environment
encryption_key = os.environ.get("SIMPLEVECDB_KEY")
if not encryption_key:
    raise ValueError("SIMPLEVECDB_KEY environment variable not set")

db = VectorDB("secure.db", encryption_key=encryption_key)
```

## Storage Layout

With encryption enabled, files are stored as:

```
mydb.db                    # SQLCipher encrypted SQLite database
mydb.db.default.usearch.enc  # AES-256-GCM encrypted usearch index
```

When opened, the index is decrypted to memory (or a temp file). On `save()` or `close()`, the index is re-encrypted.

## Performance

### Search Operations

Encryption has **zero overhead** during search because:
- SQLCipher uses page-level encryption with AES-NI hardware acceleration
- Index files are decrypted once on load, then used directly from memory

### Load/Save Operations

| Operation | Overhead |
|-----------|----------|
| Database open | ~10-50ms for key derivation |
| Index load (10k vectors, 384 dim) | ~50-100ms decrypt |
| Index save | ~50-100ms encrypt |
| Search | **0ms** (no crypto) |

## Error Handling

```python
from simplevecdb import VectorDB, EncryptionError, EncryptionUnavailableError

try:
    db = VectorDB("secure.db", encryption_key="wrong-key")
except EncryptionError as e:
    print(f"Failed to open encrypted database: {e}")

try:
    db = VectorDB("secure.db", encryption_key="secret")
except EncryptionUnavailableError:
    print("Install with: pip install simplevecdb[encryption]")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `EncryptionUnavailableError` | Missing dependencies | `pip install simplevecdb[encryption]` |
| `EncryptionError: wrong key` | Incorrect passphrase | Use the correct key |
| `ValueError: In-memory` | Used `:memory:` with encryption | Use a file path |

## Limitations

1. **In-memory databases cannot be encrypted** - Encryption is for at-rest data
2. **Key cannot be changed** - To change keys, export data and re-import
3. **Performance on large indexes** - Decryption on load may take several seconds for 100k+ vectors

## Security Notes

- **SQLCipher** uses AES-256-CBC with HMAC-SHA512 for authentication
- **Index encryption** uses AES-256-GCM with random 96-bit nonces
- **Key derivation** uses PBKDF2-SHA256 with 480,000 iterations (OWASP 2023 recommendation)
- **The encryption key is held in memory** during database usage

## API Reference

::: simplevecdb.encryption
    options:
      members:
        - EncryptionError
        - EncryptionUnavailableError
        - create_encrypted_connection
        - is_database_encrypted
        - encrypt_file
        - decrypt_file
