import numpy as np
from tinyvecdb import VectorDB, Quantization
import os
import time

N, DIM = 10000, 384
vectors = np.random.randn(N, DIM).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)


def get_db_size(path):
    total = os.path.getsize(path)
    for ext in ["-wal", "-shm"]:
        if os.path.exists(path + ext):
            total += os.path.getsize(path + ext)
    return total / (1024 * 1024)


def bench(quant):
    db = VectorDB(f"bench_{quant}.db", quantization=quant)
    db.add_texts([f"text_{i}" for i in range(N)], embeddings=vectors.tolist())
    
    # Force checkpoint to clean up WAL and get true size
    db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    
    size_mb = get_db_size(db.path)

    t0 = time.time()
    for _ in range(100):
        db.similarity_search(vectors[0], k=10)
    ms = (time.time() - t0) / 100 * 1000

    print(f"{quant}: {size_mb:.1f} MB, {ms:.2f} ms/query")
    os.remove(db.path)


bench(Quantization.FLOAT)  # ~15 MB
bench(Quantization.INT8)  # ~4 MB (4x)
bench(Quantization.BIT)  # ~0.5 MB (32x)
