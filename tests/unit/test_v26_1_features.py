"""Tests for SimpleVecDB 2.6.1 catalog extensions (gaps 1-10)."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from simplevecdb import VectorDB


@pytest.fixture
def db_with_docs(tmp_path):
    db = VectorDB(str(tmp_path / "v261.db"))
    c = db.collection("default", store_embeddings=True)
    embs = np.random.rand(5, 4).astype(np.float32)
    c.add_texts(
        ["a", "b", "c", "d", "e"],
        embeddings=embs,
        metadatas=[{"i": i, "score": float(i), "tag": "x" if i % 2 == 0 else "y"}
                   for i in range(5)],
    )
    yield db, c, embs
    db.close()


# ---------------------------- gap 1: update_embedding ---------------------

class TestUpdateEmbedding:
    def test_buffers_without_hnsw_change(self, db_with_docs):
        db, c, embs = db_with_docs
        before_size = c._index.size
        c.update_embedding(1, np.zeros(4, dtype=np.float32))
        assert c.pending.size() == 1
        # HNSW size unchanged until flush.
        assert c._index.size == before_size

    def test_flush_promotes_to_hnsw(self, db_with_docs):
        db, c, embs = db_with_docs
        new = np.ones(4, dtype=np.float32)
        c.update_embedding(1, new)
        assert c.pending.flush() == 1
        assert c.pending.size() == 0
        # The doc closest to all-ones is now id=1.
        results = c.similarity_search(new, k=1)
        assert results[0][0].metadata["i"] == 0  # may match by direction

    def test_dim_mismatch_raises(self, db_with_docs):
        db, c, _ = db_with_docs
        with pytest.raises(ValueError, match="!= index dim"):
            c.update_embedding(1, np.zeros(7, dtype=np.float32))

    def test_blend_toward_combines_with_pending(self, db_with_docs):
        db, c, _ = db_with_docs
        centroid = np.full(4, 0.5, dtype=np.float32)
        n = c.pending.blend_toward([1, 2], centroid, alpha=0.5)
        assert n == 2
        assert c.pending.size() == 2


# ---------------------------- gap 2: transaction --------------------------

class TestTransaction:
    def test_db_transaction_commits_on_success(self, db_with_docs):
        db, c, _ = db_with_docs
        with db.transaction() as tx:
            tx["default"].increment_metadata(1, {"hits": 1})
            tx["default"].edges.add_edge(1, 2, weight=0.7)
        assert c.counters.get(1, "hits") == 1
        assert len(c.edges.get_edges(src=1)) == 1

    def test_db_transaction_rolls_back_on_error(self, db_with_docs):
        db, c, _ = db_with_docs
        c.increment_metadata(1, {"hits": 1})
        baseline = c.counters.get(1, "hits")
        with pytest.raises(RuntimeError):
            with db.transaction() as tx:
                tx["default"].increment_metadata(1, {"hits": 5})
                tx["default"].edges.add_edge(2, 3, weight=0.3)
                raise RuntimeError("boom")
        # SQL state preserved.
        assert c.counters.get(1, "hits") == baseline
        assert len(c.edges.get_edges(src=2)) == 0

    def test_collection_tx_yields_collection(self, db_with_docs):
        db, c, _ = db_with_docs
        with c.tx() as ctx:
            assert ctx is c
            ctx.increment_metadata(1, {"hits": 3})
        assert c.counters.get(1, "hits") == 3

    def test_outer_commit_failure_propagates(self, db_with_docs):
        """Outer commit failures must surface, not be swallowed."""
        db, _c, _ = db_with_docs
        real_conn = db.conn
        calls = {"n": 0}

        class FlakyConn:
            def __getattr__(self, name):
                return getattr(real_conn, name)

            def commit(self):
                calls["n"] += 1
                raise sqlite3.OperationalError("disk full")

        db.conn = FlakyConn()  # type: ignore[assignment]
        try:
            with pytest.raises(sqlite3.OperationalError, match="disk full"):
                with db.transaction() as tx:
                    tx["default"].increment_metadata(1, {"hits": 1})
        finally:
            db.conn = real_conn
        assert calls["n"] == 1


# ---------------------------- gap 3: edges --------------------------------

class TestEdges:
    def test_crud(self, db_with_docs):
        db, c, _ = db_with_docs
        assert c.edges.add_edge(1, 2, kind="similar", weight=0.7) == 1
        edges = c.edges.get_edges(src=1)
        assert len(edges) == 1
        assert edges[0].weight == pytest.approx(0.7)
        # Update with absolute set.
        c.edges.update_edge(1, 2, kind="similar", weight=0.9)
        assert c.edges.get_edges(src=1)[0].weight == pytest.approx(0.9)
        # Delete.
        assert c.edges.delete_edge(1, 2, kind="similar") == 1
        assert c.edges.get_edges(src=1) == []

    def test_atomic_delta_concurrency(self, db_with_docs):
        db, c, _ = db_with_docs
        c.edges.add_edge(1, 2, kind="x", weight=0.0)

        def bump(_):
            c.edges.update_edge(1, 2, kind="x", dweight=0.01, dhits=1)

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(bump, range(200)))
        e = c.edges.get_edges(src=1, dst=2, kind="x")[0]
        assert e.hits == 200
        assert e.weight == pytest.approx(2.0, abs=1e-6)

    def test_range_filter_on_weight(self, db_with_docs):
        db, c, _ = db_with_docs
        c.edges.add_edge(1, 2, weight=0.05)
        c.edges.add_edge(1, 3, weight=0.5)
        c.edges.add_edge(1, 4, weight=0.95)
        low = c.edges.get_edges(filter={"weight": {"$lt": 0.1}})
        assert {e.dst_id for e in low} == {2}
        between = c.edges.get_edges(filter={"weight": {"$between": [0.1, 0.9]}})
        assert {e.dst_id for e in between} == {3}

    def test_prune(self, db_with_docs):
        db, c, _ = db_with_docs
        c.edges.add_edge(1, 2, weight=0.05)
        c.edges.add_edge(1, 3, weight=0.5)
        n = c.edges.prune(max_weight=0.1)
        assert n == 1


# ---------------------------- gap 4: counters -----------------------------

class TestCounters:
    def test_dict_increment(self, db_with_docs):
        db, c, _ = db_with_docs
        c.increment_metadata(1, {"retrieval_count": 1, "drift_total": 0.02})
        c.increment_metadata(1, {"retrieval_count": 1, "drift_total": 0.05})
        assert c.counters.get(1, "retrieval_count") == 2
        assert c.counters.get(1, "drift_total") == pytest.approx(0.07)

    def test_concurrent_increments_are_atomic(self, db_with_docs):
        db, c, _ = db_with_docs

        def bump(_):
            c.increment_metadata(2, {"hits": 1})

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(bump, range(800)))
        assert c.counters.get(2, "hits") == 800

    def test_invalid_key_rejected(self, db_with_docs):
        db, c, _ = db_with_docs
        with pytest.raises(ValueError, match="metadata counter key"):
            c.increment_metadata(1, {"no spaces": 1})

    def test_non_numeric_rejected(self, db_with_docs):
        db, c, _ = db_with_docs
        with pytest.raises(TypeError):
            c.increment_metadata(1, {"x": "string"})


# ---------------------------- gap 5: range filters ------------------------

class TestRangeFilters:
    def test_mongo_operator_dict(self, db_with_docs):
        db, c, embs = db_with_docs
        r = c.similarity_search(embs[0], k=10,
                                filter={"score": {"$gt": 1.5, "$lt": 4.0}})
        scores = sorted(d.metadata["score"] for d, _ in r)
        assert scores == [2.0, 3.0]

    def test_tuple_shorthand(self, db_with_docs):
        db, c, embs = db_with_docs
        r = c.similarity_search(embs[0], k=10,
                                filter={"score": ("range", 1.5, 4.0)})
        scores = sorted(d.metadata["score"] for d, _ in r)
        assert scores == [2.0, 3.0, 4.0]

    def test_in_and_nin(self, db_with_docs):
        db, c, embs = db_with_docs
        r = c.similarity_search(embs[0], k=10, filter={"tag": {"$in": ["x"]}})
        assert all(d.metadata["tag"] == "x" for d, _ in r)
        r = c.similarity_search(embs[0], k=10, filter={"tag": {"$nin": ["x"]}})
        assert all(d.metadata["tag"] != "x" for d, _ in r)

    def test_unknown_operator_raises(self, db_with_docs):
        db, c, embs = db_with_docs
        with pytest.raises(ValueError, match="Unknown operator"):
            c.similarity_search(embs[0], k=1, filter={"score": {"$bogus": 1}})


# ---------------------------- gap 7: change feed --------------------------

class TestEvents:
    def test_mutation_appends_event(self, db_with_docs):
        db, c, _ = db_with_docs
        before = c.events.last_seq()
        c.increment_metadata(1, {"hits": 1})
        c.edges.add_edge(1, 2, weight=0.5)
        after = c.events.last_seq()
        assert after >= before + 2

    def test_read_filters_by_kind(self, db_with_docs):
        db, c, _ = db_with_docs
        c.edges.add_edge(1, 2, weight=0.5)
        c.edges.delete_edge(1, 2)
        adds = c.events.read(kind="edge_add")
        dels = c.events.read(kind="edge_delete")
        assert len(adds) >= 1 and len(dels) >= 1


# ---------------------------- gap 8: TTL ----------------------------------

class TestTTL:
    def test_sweep_deletes_expired(self, db_with_docs):
        db, c, _ = db_with_docs
        c.ttl.set(1, seconds=-10)
        deleted, callback_ids = c.ttl.sweep()
        assert 1 in deleted
        assert callback_ids == []
        # Doc removed.
        assert c._catalog.count() == 4

    def test_callback_keeps_doc(self, db_with_docs):
        db, c, _ = db_with_docs
        c.ttl.set(2, seconds=-5, on_expire="callback")
        deleted, callback_ids = c.ttl.sweep()
        assert deleted == []
        assert 2 in callback_ids

    def test_background_sweep(self, db_with_docs):
        db, c, _ = db_with_docs
        c.ttl.set(3, seconds=0.2)
        c.ttl.start_background(interval=0.1)
        try:
            time.sleep(0.6)
        finally:
            c.ttl.stop_background()
        # 3 should have been swept.
        ids_left = {row[0] for row in
                    c._catalog.get_all_docs_with_text()}
        assert 3 not in ids_left


# ---------------------------- gap 9: maintenance --------------------------

class TestMaintenance:
    def test_threshold_triggers_rebuild(self, db_with_docs):
        db, c, _ = db_with_docs
        c.update_embedding(1, np.zeros(4, dtype=np.float32))
        c.pending.flush()
        ran = c.maintenance.rebuild_if_needed(max_pending=1)
        assert ran is True
        # Subsequent call doesn't rebuild again until threshold re-passed.
        assert c.maintenance.rebuild_if_needed(max_pending=1) is False
