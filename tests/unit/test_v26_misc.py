"""Misc 2.6.0 changes: hybrid search dedup, file_lock cleanup, NullHandler.

- ``hybrid_search`` previously deduped by ``page_content``, so two
  distinct documents with identical text were merged into a single
  inflated-score result. 2.6.0 dedupes by document ID instead.
- ``utils.file_lock`` previously left stale ``.lock`` siblings around in
  busy data directories. 2.6.0 unlinks the lock file on context exit.
- ``simplevecdb.logging`` now attaches a ``NullHandler`` at import so
  library users that have not configured logging don't see "No handlers
  could be found" warnings.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import pytest

from simplevecdb import VectorDB
from simplevecdb.utils import file_lock


@pytest.fixture
def db_with_dup_text(tmp_path: Path):
    db = VectorDB(str(tmp_path / "dup.db"))
    col = db.collection("c")
    rng = np.random.RandomState(0)
    # Two documents with the SAME text but different metadata and slightly
    # different embeddings — pre-2.6.0 RRF would merge them by content.
    text = "the quick brown fox jumps over the lazy dog"
    emb_a = rng.randn(384).astype(np.float32).tolist()
    emb_b = rng.randn(384).astype(np.float32).tolist()
    ids = col.add_texts(
        [text, text, "a totally different sentence about cats"],
        metadatas=[
            {"source": "A"},
            {"source": "B"},
            {"source": "C"},
        ],
        embeddings=[emb_a, emb_b, rng.randn(384).astype(np.float32).tolist()],
    )
    yield db, col, ids
    db.close()


class TestHybridSearchDedupesByDocId:
    def test_distinct_docs_with_same_text_kept_separate(self, db_with_dup_text):
        db, col, ids = db_with_dup_text
        # Hybrid search needs FTS — skip cleanly if unavailable.
        try:
            results = col.hybrid_search("fox", k=5)
        except RuntimeError as exc:
            pytest.skip(f"FTS5 unavailable: {exc}")

        # The two duplicate-text rows must still be reported as separate
        # results — pre-2.6.0 they collapsed into one, with whichever
        # metadata happened to come last winning.
        sources = sorted(
            r[0].metadata.get("source") for r in results
            if r[0].page_content.startswith("the quick brown fox")
        )
        assert sources == ["A", "B"], (
            f"hybrid_search must dedup by ID, not by page_content; got {sources!r}"
        )


class TestFileLockCleanup:
    """The ``.lock`` sidecar is intentionally kept on disk after the
    context exits — flock/LK_LOCK are inode-bound, and unlinking the
    path while another process is queued on the old inode would let a
    third process acquire a different lock concurrently. See review
    pass 4 (codex P1)."""

    def test_lock_file_persists_on_release(self, tmp_path: Path):
        target = tmp_path / "target.bin"
        lock_path = tmp_path / "target.bin.lock"
        with file_lock(target):
            assert lock_path.exists()
        # After release the .lock sidecar MUST still exist so future
        # acquisitions reuse the same inode.
        assert lock_path.exists()

    def test_lock_file_persists_after_exception(self, tmp_path: Path):
        target = tmp_path / "target.bin"
        lock_path = tmp_path / "target.bin.lock"
        with pytest.raises(RuntimeError, match="boom"):
            with file_lock(target):
                raise RuntimeError("boom")
        # Exception path must still preserve the lock sidecar.
        assert lock_path.exists()

    def test_serial_acquire_release_reuses_one_lock_file(self, tmp_path: Path):
        target = tmp_path / "loop.bin"
        for _ in range(5):
            with file_lock(target):
                pass
        # Exactly one .lock file — repeated acquisitions reuse the same
        # path/inode rather than each creating a fresh one.
        leftovers = list(tmp_path.glob("*.lock"))
        assert leftovers == [tmp_path / "loop.bin.lock"]

    def test_concurrent_acquire_serializes(self, tmp_path: Path):
        # Sanity: file_lock is mutually exclusive within the same process.
        target = tmp_path / "race.bin"
        order: list[str] = []
        barrier = threading.Barrier(2)

        def worker(name: str, hold_seconds: float):
            barrier.wait()
            with file_lock(target):
                order.append(f"{name}:enter")
                # busy-wait briefly so the second thread is blocked
                end = __import__("time").time() + hold_seconds
                while __import__("time").time() < end:
                    pass
                order.append(f"{name}:leave")

        t1 = threading.Thread(target=worker, args=("a", 0.05))
        t2 = threading.Thread(target=worker, args=("b", 0.05))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Order must be enter-leave-enter-leave (no interleaving), regardless
        # of which thread won the race.
        assert order[0].endswith(":enter")
        assert order[1].endswith(":leave")
        assert order[2].endswith(":enter")
        assert order[3].endswith(":leave")


class TestLoggingNullHandler:
    """Importing simplevecdb.logging must attach exactly one NullHandler.

    These tests are insulated from cross-test pollution by clearing the
    simplevecdb root logger's handlers up front and reloading the module —
    other tests in the suite may add or remove handlers at runtime, so we
    cannot rely on the import-time state surviving until our test runs.
    """

    def setup_method(self):
        # Strip any handlers other tests may have left on the simplevecdb
        # root logger so we can observe a clean reload.
        root = logging.getLogger("simplevecdb")
        for h in list(root.handlers):
            root.removeHandler(h)

    def test_reload_attaches_null_handler(self):
        import importlib

        import simplevecdb.logging as svc_logging

        importlib.reload(svc_logging)

        root = logging.getLogger("simplevecdb")
        null_handlers = [
            h for h in root.handlers if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers) == 1, (
            "simplevecdb root logger must have exactly one NullHandler "
            "after the logging module is imported"
        )

    def test_repeated_reload_is_idempotent(self):
        import importlib

        import simplevecdb.logging as svc_logging

        importlib.reload(svc_logging)
        importlib.reload(svc_logging)
        importlib.reload(svc_logging)

        null_handlers = [
            h
            for h in logging.getLogger("simplevecdb").handlers
            if isinstance(h, logging.NullHandler)
        ]
        # Module guards against duplicate attaches — exactly one even after
        # multiple reloads.
        assert len(null_handlers) == 1
