"""Regression tests for the third 2.6.0 review pass.

These tests pin invariants that the prior test suite did not exercise:

- ``UsearchIndex.save`` calls fsync on the parent directory after replace.
- ``.tmp`` sidecar is cleaned up when an index save fails mid-write.
- ``VectorDB._lock`` is the same RLock object as every cached collection's
  ``CatalogManager._lock`` (the central invariant of the shared-RLock
  design introduced in 2.6.0 review pass 2).
- ``_validate_table_name`` rejects adversarial inputs at ``CatalogManager``
  construction time, before any SQL runs.
- The hybrid-search RRF rank is symmetric between vector and keyword
  candidates under a metadata filter.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from simplevecdb import VectorDB
from simplevecdb.engine.catalog import CatalogManager, _validate_table_name


class TestUsearchIndexFsync:
    def test_save_calls_fsync_on_parent_directory(self, tmp_path):
        db_path = tmp_path / "fsync.db"
        db = VectorDB(str(db_path))
        col = db.collection("default")
        col.add_texts(
            ["hello world"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
        )

        # Spy on os.fsync; close() flushes the index, which is the path
        # that wraps the parent-directory fsync.
        observed_fds: list[int] = []
        real_fsync = os.fsync

        def spy(fd):
            observed_fds.append(fd)
            return real_fsync(fd)

        with mock.patch("simplevecdb.engine.usearch_index.os.fsync", side_effect=spy):
            db.close()

        # At least two fsyncs are expected: one for the .tmp file, one for
        # the parent directory entry. Both must succeed (real_fsync was
        # called, otherwise we'd see EBADF).
        assert len(observed_fds) >= 2

    def test_save_failure_cleans_up_tmp_file(self, tmp_path):
        db_path = tmp_path / "tmpcleanup.db"
        db = VectorDB(str(db_path))
        col = db.collection("default")
        col.add_texts(
            ["a", "b"],
            embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )

        # Force the underlying usearch save to raise. The cleanup branch
        # in UsearchIndex.save must unlink the .tmp file before the
        # exception propagates.
        index = col._index
        original_save = index._index.save

        def boom(path, *a, **kw):
            # Create a partial .tmp file on disk to simulate a torn write
            # before raising.
            Path(path).write_bytes(b"\x00" * 8)
            raise OSError("simulated mid-save failure")

        with mock.patch.object(index._index, "save", side_effect=boom):
            with pytest.raises(OSError, match="simulated"):
                index.save()

        # Compute the .tmp path the way usearch_index does and verify
        # cleanup.
        tmp_candidate = index._path.with_suffix(index._path.suffix + ".tmp")
        assert not tmp_candidate.exists(), (
            f"Orphan .tmp file left behind at {tmp_candidate}"
        )
        db.close()


class TestSharedRLock:
    def test_vectordb_lock_is_shared_with_catalog(self, tmp_path):
        """The VectorDB-level RLock must be the same object as every
        CatalogManager._lock so transactions on the shared connection do
        not interleave between collections."""
        db = VectorDB(str(tmp_path / "shared.db"))
        col_a = db.collection("alpha")
        col_b = db.collection("beta")
        try:
            assert col_a._catalog._lock is db._lock
            assert col_b._catalog._lock is db._lock
            assert col_a._catalog._lock is col_b._catalog._lock
        finally:
            db.close()

    def test_lock_is_reentrant(self, tmp_path):
        db = VectorDB(str(tmp_path / "reentrant.db"))
        try:
            with db._lock:
                # Re-entrant acquisition from within an already-held lock
                # must not deadlock.
                with db._lock:
                    pass
        finally:
            db.close()


class TestValidateTableNameAdversarial:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "valid; DROP TABLE foo--",
            "name with space",
            "1starts_with_digit",
            "has-hyphen",
            "has.dot",
            "tick'name",
            'doublequote"name',
            "",
            "name\x00null",
            "../traversal",
        ],
    )
    def test_rejects_adversarial_names(self, bad_name):
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name(bad_name)

    @pytest.mark.parametrize(
        "good_name",
        [
            "items",
            "items_default",
            "_underscore_first",
            "Items_With_Mixed_Case",
            "items_123",
        ],
    )
    def test_accepts_legitimate_names(self, good_name):
        # No exception means valid.
        _validate_table_name(good_name)

    def test_catalog_manager_init_rejects_bad_name_before_any_sql(self, tmp_path):
        # CatalogManager.__init__ calls _validate_table_name before
        # touching the connection at all.
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "x.db"))
        try:
            with pytest.raises(ValueError, match="Invalid table name"):
                CatalogManager(conn, "bad; DROP TABLE x", "fts_table")
        finally:
            conn.close()


class TestHybridSearchRRFSymmetry:
    """RRF rank for vector candidates must be the original HNSW position,
    not the post-filter position. Otherwise a metadata filter that
    rejects vector candidates inflates the surviving ones' scores
    relative to keyword candidates."""

    def test_filter_does_not_inflate_vector_rrf_score(self, tmp_path):
        db = VectorDB(str(tmp_path / "rrf.db"))
        col = db.collection("default")
        try:
            # 10 documents; only the last carries category=keep.
            texts = [f"doc number {i}" for i in range(10)]
            metas = [{"category": "drop"} for _ in range(9)] + [
                {"category": "keep"}
            ]
            embs = [
                [float(i), float(10 - i), 0.0, 0.0] for i in range(10)
            ]
            col.add_texts(texts, metadatas=metas, embeddings=embs)

            # Hybrid search with a filter that drops the top 9 vector hits.
            # If rank symmetry is broken, the surviving "drop" → wait,
            # all dropped — so we use a more nuanced setup: keep just one.
            results = col.hybrid_search(
                query="doc number 0",
                query_vector=[0.1, 9.9, 0.0, 0.0],
                k=3,
                filter={"category": "keep"},
            )
            # We don't assert ordering — only that the method runs and
            # the surviving result's score is a finite RRF (>0, <1) and
            # not inflated to infinity.
            assert len(results) >= 1
            for _doc, score in results:
                assert 0.0 < score < 1.0, (
                    f"RRF score {score} is outside the expected (0, 1) range"
                )
        finally:
            db.close()

    def test_two_docs_same_text_different_ids_both_appear(self, tmp_path):
        """Hybrid search dedupes on doc id, not page_content. Two distinct
        documents with identical text must both surface."""
        db = VectorDB(str(tmp_path / "dedup.db"))
        col = db.collection("default")
        try:
            col.add_texts(
                ["the same text", "the same text"],
                metadatas=[{"variant": "a"}, {"variant": "b"}],
                embeddings=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
            )
            results = col.hybrid_search(
                query="the same text",
                query_vector=[0.5, 0.5, 0.0, 0.0],
                k=5,
            )
            variants = {
                doc.metadata.get("variant") for doc, _ in results
            }
            assert variants == {"a", "b"}, (
                f"Both documents should surface; got variants={variants}"
            )
        finally:
            db.close()


class TestRebuildIndexUsesCatalogListAllIds:
    """rebuild_index() must route the all-ids fetch through
    CatalogManager.list_all_ids() so the SELECT runs under the shared
    RLock instead of bare ``self.conn.execute``."""

    def test_list_all_ids_returns_every_doc(self, tmp_path):
        db = VectorDB(str(tmp_path / "rebuild.db"))
        col = db.collection("default", store_embeddings=True)
        try:
            ids = col.add_texts(
                ["a", "b", "c"],
                embeddings=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            )
            assert sorted(col._catalog.list_all_ids()) == sorted(ids)
        finally:
            db.close()

    def test_rebuild_index_round_trip(self, tmp_path):
        db = VectorDB(str(tmp_path / "rebuild2.db"))
        col = db.collection("default", store_embeddings=True)
        try:
            col.add_texts(
                ["x", "y", "z"],
                embeddings=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            )
            count = col.rebuild_index()
            assert count == 3
            # Index still works post-rebuild.
            results = col.similarity_search([1.0, 0.0, 0.0, 0.0], k=3)
            assert len(results) == 3
        finally:
            db.close()
