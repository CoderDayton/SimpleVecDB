import numpy as np
from simplevecdb import VectorDB, Quantization
import time
from pathlib import Path

N, DIM = 10000, 384
vectors = np.random.randn(N, DIM).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)


def get_db_size(path):
    """Get total storage size including SQLite and usearch files."""
    total = 0
    base = Path(path)

    # SQLite files
    for suffix in ["", "-wal", "-shm"]:
        p = Path(str(path) + suffix)
        if p.exists():
            total += p.stat().st_size

    # Usearch index files (.usearch)
    for usearch_file in base.parent.glob(f"{base.name}.*.usearch"):
        total += usearch_file.stat().st_size

    return total / (1024 * 1024)


def bench(quant):
    db_path = f"bench_{quant.value}.db"
    db = VectorDB(db_path, quantization=quant)
    collection = db.collection("default")
    collection.add_texts([f"text_{i}" for i in range(N)], embeddings=vectors.tolist())

    # Measure query time before close
    t0 = time.time()
    for _ in range(100):
        collection.similarity_search(vectors[0], k=10)
    ms = (time.time() - t0) / 100 * 1000

    # Force checkpoint and close to flush usearch index to disk
    db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()

    # Measure size AFTER close (usearch saves on close)
    size_mb = get_db_size(db_path)

    print(f"{quant.value}: {size_mb:.1f} MB, {ms:.2f} ms/query")

    # Cleanup all files
    for f in Path(".").glob(f"bench_{quant.value}*"):
        f.unlink()


bench(Quantization.FLOAT)  # ~36 MB (baseline)
bench(Quantization.FLOAT16)  # ~29 MB (2x vector compression)
bench(Quantization.INT8)  # ~25 MB (4x vector compression)
bench(Quantization.BIT)  # ~22 MB (32x vector compression)
