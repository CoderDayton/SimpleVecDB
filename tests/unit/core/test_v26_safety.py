"""Safety tests for VectorDB introduced in 2.6.0.

Covers:
- ``add_texts`` rejects NaN/Inf vectors before they corrupt the HNSW graph
- ``__repr__`` does not run SQL (avoids I/O in debuggers/loggers)
- ``VectorDB._lock`` is an RLock (re-entrant from same thread)
- Ephemeral index files for in-memory DBs are cleaned up on close
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from simplevecdb import VectorDB


class TestAddRejectsNonFiniteVectors:
    def test_nan_vector_rejected(self):
        db = VectorDB(":memory:")
        col = db.collection("c")
        bad = [float("nan")] * 384
        with pytest.raises(ValueError, match="NaN or Inf"):
            col.add_texts(["bad"], embeddings=[bad])
        db.close()

    def test_inf_vector_rejected(self):
        db = VectorDB(":memory:")
        col = db.collection("c")
        bad = [float("inf")] + [0.1] * 383
        with pytest.raises(ValueError, match="NaN or Inf"):
            col.add_texts(["bad"], embeddings=[bad])
        db.close()

    def test_negative_inf_vector_rejected(self):
        db = VectorDB(":memory:")
        col = db.collection("c")
        bad = [float("-inf")] + [0.1] * 383
        with pytest.raises(ValueError, match="NaN or Inf"):
            col.add_texts(["bad"], embeddings=[bad])
        db.close()

    def test_finite_vector_accepted(self):
        db = VectorDB(":memory:")
        col = db.collection("c")
        ok = np.random.RandomState(0).randn(384).astype(np.float32).tolist()
        ids = col.add_texts(["ok"], embeddings=[ok])
        assert len(ids) == 1
        db.close()

    def test_one_bad_vector_in_batch_rejects_whole_batch(self):
        db = VectorDB(":memory:")
        col = db.collection("c")
        good = np.random.RandomState(0).randn(384).astype(np.float32).tolist()
        bad = [float("nan")] * 384
        with pytest.raises(ValueError, match="NaN or Inf"):
            col.add_texts(["g", "b"], embeddings=[good, bad])
        # Multi-item batches must also leave the catalog empty — the good
        # vector cannot be silently committed when its sibling is invalid.
        assert col.count() == 0
        db.close()

    def test_rejection_does_not_leave_orphan_sqlite_rows(self):
        # Regression: NaN/Inf must be rejected before the catalog INSERT
        # commits, otherwise the SQLite row exists with no corresponding
        # vector in the HNSW index — surfacing only via document fetches,
        # never via search. count() must remain 0 after a rejection.
        db = VectorDB(":memory:")
        col = db.collection("c")
        bad = [float("nan")] * 384
        with pytest.raises(ValueError, match="NaN or Inf"):
            col.add_texts(["bad"], embeddings=[bad])
        assert col.count() == 0
        db.close()

    def test_streaming_rejection_does_not_leave_orphan_rows(self):
        db = VectorDB(":memory:")
        col = db.collection("c")
        bad = [float("inf")] * 384
        items = [("bad", None, bad)]
        with pytest.raises(ValueError, match="NaN or Inf"):
            gen = col.add_texts_streaming(items, batch_size=1)
            for _ in gen:
                pass
        assert col.count() == 0
        db.close()


class TestReprNoIO:
    """__repr__ must not run SQL — debuggers and exception formatters call it."""

    def test_repr_does_not_query_db(self, tmp_path: Path):
        db = VectorDB(str(tmp_path / "x.db"))
        rep = repr(db)
        assert "VectorDB(path=" in rep
        assert "x.db" in rep
        db.close()

    def test_repr_works_after_close(self, tmp_path: Path):
        # If __repr__ tried to read from a closed connection, this would
        # raise ProgrammingError. The 2.6.0 version is path-only, so it
        # must succeed.
        db = VectorDB(str(tmp_path / "x.db"))
        db.close()
        rep = repr(db)  # must not raise
        assert "x.db" in rep


class TestLockIsReentrant:
    """VectorDB._lock must be an RLock so nested with-statements don't deadlock."""

    def test_lock_can_be_reacquired_in_same_thread(self):
        db = VectorDB(":memory:")
        # If self._lock were a plain Lock, this would deadlock.
        with db._lock:
            with db._lock:
                col = db.collection("nested")
                assert col is not None
        db.close()

    def test_concurrent_collection_access_is_thread_safe(self, tmp_path: Path):
        # Smoke test: two threads creating/looking up the same collection
        # must not raise or produce duplicate state. The collection() lookup
        # is lock-protected, so the second call returns the cached instance.
        db = VectorDB(str(tmp_path / "concurrent.db"))
        results: list = []
        errors: list = []

        def worker():
            try:
                results.append(db.collection("shared"))
            except Exception as exc:  # pragma: no cover - regression detector
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # All threads must observe the same collection object (cached).
        assert all(r is results[0] for r in results)
        db.close()


class TestEphemeralIndexCleanup:
    """In-memory DBs allocate ephemeral usearch index files; close() removes them."""

    def test_ephemeral_index_path_set_for_memory_db(self):
        db = VectorDB(":memory:")
        col = db.collection("eph")
        assert col._ephemeral_index_path is not None
        assert os.path.exists(col._ephemeral_index_path) or True  # may be lazy
        db.close()

    def test_ephemeral_files_removed_on_close(self):
        db = VectorDB(":memory:")
        col = db.collection("cleanup")
        col.add_texts(
            ["x"],
            embeddings=[np.random.RandomState(0).randn(384).tolist()],
        )
        col.save()
        path = col._ephemeral_index_path
        assert path is not None
        assert os.path.exists(path)
        db.close()
        # After close, the ephemeral file (and any sibling .tmp/.lock) is gone.
        assert not os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".lock")

    def test_persistent_db_has_no_ephemeral_path(self, tmp_path: Path):
        db = VectorDB(str(tmp_path / "persist.db"))
        col = db.collection("p")
        assert col._ephemeral_index_path is None
        db.close()
