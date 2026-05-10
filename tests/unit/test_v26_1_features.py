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
        metadatas=[
            {"i": i, "score": float(i), "tag": "x" if i % 2 == 0 else "y"}
            for i in range(5)
        ],
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
        r = c.similarity_search(
            embs[0], k=10, filter={"score": {"$gt": 1.5, "$lt": 4.0}}
        )
        scores = sorted(d.metadata["score"] for d, _ in r)
        assert scores == [2.0, 3.0]

    def test_tuple_shorthand(self, db_with_docs):
        db, c, embs = db_with_docs
        r = c.similarity_search(embs[0], k=10, filter={"score": ("range", 1.5, 4.0)})
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
        ids_left = {row[0] for row in c._catalog.get_all_docs_with_text()}
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


# ---------------------------- coverage gap fillers ------------------------


class TestCountersGet:
    def test_get_returns_stored_value(self, db_with_docs):
        _, c, _ = db_with_docs
        c.counters.increment(1, {"hits": 3})
        assert c.counters.get(1, "hits") == 3

    def test_get_returns_default_for_missing_key(self, db_with_docs):
        _, c, _ = db_with_docs
        assert c.counters.get(1, "never_set", default=42) == 42

    def test_get_returns_none_for_missing_row(self, db_with_docs):
        _, c, _ = db_with_docs
        assert c.counters.get(99999, "hits") is None


class TestExistsOperator:
    def test_exists_true_matches_present_key(self, db_with_docs):
        _, c, embs = db_with_docs
        results = c.similarity_search(
            embs[0], k=10, filter={"score": {"$exists": True}}
        )
        # Every seeded doc has a "score" metadata field.
        assert len(results) == 5

    def test_exists_false_matches_absent_key(self, db_with_docs):
        _, c, embs = db_with_docs
        results = c.similarity_search(
            embs[0], k=10, filter={"missing": {"$exists": False}}
        )
        assert len(results) == 5


class TestEventsObservability:
    def test_last_seq_grows_with_appends(self, db_with_docs):
        _, c, _ = db_with_docs
        before = c.events.last_seq()
        c.events.append("manual", payload={"k": 1})
        c.events.append("manual", payload={"k": 2})
        assert c.events.last_seq() == before + 2

    def test_prune_drops_old_events(self, db_with_docs):
        _, c, _ = db_with_docs
        for i in range(5):
            c.events.append("noise", payload={"i": i})
        cutoff = c.events.last_seq()
        c.events.append("keep", payload={"i": "kept"})
        # prune deletes seq < before_seq, so cutoff+1 covers all noise rows.
        removed = c.events.prune(before_seq=cutoff + 1)
        assert removed >= 5
        kinds = [e.kind for e in c.events.read()]
        assert "keep" in kinds
        assert "noise" not in kinds

    def test_subscribe_yields_new_events(self, db_with_docs):
        _, c, _ = db_with_docs
        start = c.events.last_seq()
        c.events.append("first", payload={"n": 1})
        c.events.append("second", payload={"n": 2})

        gen = c.events.subscribe(since=start, poll_interval=0.001, batch=10)
        try:
            seen = [next(gen), next(gen)]
        finally:
            gen.close()
        kinds = [e.kind for e in seen]
        assert "first" in kinds and "second" in kinds


# ---------------------------- fortification guards ------------------------


class TestUpdateEmbeddingFortification:
    def test_nan_vector_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        bad = np.array([1.0, float("nan"), 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            c.update_embedding(1, bad)

    def test_inf_vector_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        bad = np.array([float("inf"), 0.0, 0.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="finite"):
            c.update_embedding(1, bad)

    def test_2d_vector_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        bad = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            c.update_embedding(1, bad)


class TestBetweenOperator:
    def test_between_inclusive_range_matches(self, db_with_docs):
        _, c, embs = db_with_docs
        results = c.similarity_search(
            embs[0], k=10, filter={"score": {"$between": [1.0, 3.0]}}
        )
        scores = sorted(r[0].metadata["score"] for r in results)
        assert scores == [1.0, 2.0, 3.0]

    def test_between_tuple_shorthand(self, db_with_docs):
        _, c, embs = db_with_docs
        results = c.similarity_search(
            embs[0], k=10, filter={"score": ("range", 0.0, 1.0)}
        )
        scores = sorted(r[0].metadata["score"] for r in results)
        assert scores == [0.0, 1.0]

    def test_between_lo_greater_than_hi_rejected(self, db_with_docs):
        _, c, embs = db_with_docs
        with pytest.raises(ValueError, match="lo <= hi"):
            c.similarity_search(
                embs[0], k=10, filter={"score": {"$between": [5.0, 1.0]}}
            )

    def test_between_non_finite_rejected(self, db_with_docs):
        _, c, embs = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.similarity_search(
                embs[0], k=10, filter={"score": {"$between": [0.0, float("inf")]}}
            )

    def test_between_wrong_arity_rejected(self, db_with_docs):
        _, c, embs = db_with_docs
        with pytest.raises(ValueError):
            c.similarity_search(embs[0], k=10, filter={"score": {"$between": [1.0]}})


class TestEdgeFortification:
    def test_add_edge_nan_weight_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.edges.add_edge(1, 2, weight=float("nan"))

    def test_add_edge_inf_bonus_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.edges.add_edge(1, 2, bonus=float("inf"))

    def test_upsert_edge_nan_weight_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.edges.upsert(1, 2, weight=float("nan"))

    def test_update_edge_nan_dweight_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        c.edges.add_edge(1, 2, weight=0.5)
        with pytest.raises(ValueError, match="finite"):
            c.edges.update_edge(1, 2, dweight=float("nan"))

    def test_update_edge_inf_dbonus_rejected(self, db_with_docs):
        _, c, _ = db_with_docs
        c.edges.add_edge(1, 2, weight=0.5)
        with pytest.raises(ValueError, match="finite"):
            c.edges.update_edge(1, 2, dbonus=float("inf"))


class TestTTLFortification:
    def test_set_requires_one_of_seconds_or_expires_at(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="expires_at or seconds"):
            c.ttl.set(1)

    def test_set_rejects_both_seconds_and_expires_at(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="exactly one"):
            c.ttl.set(1, expires_at=time.time() + 5, seconds=10)

    def test_set_rejects_invalid_on_expire(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="on_expire"):
            c.ttl.set(1, seconds=5, on_expire="bogus")

    def test_set_rejects_nan_seconds(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.ttl.set(1, seconds=float("nan"))

    def test_set_rejects_inf_seconds(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.ttl.set(1, seconds=float("inf"))

    def test_set_rejects_nan_expires_at(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="finite"):
            c.ttl.set(1, expires_at=float("nan"))

    def test_clear_returns_zero_for_missing(self, db_with_docs):
        _, c, _ = db_with_docs
        assert c.ttl.clear(99999) == 0

    def test_clear_returns_one_after_set(self, db_with_docs):
        _, c, _ = db_with_docs
        c.ttl.set(1, seconds=60)
        assert c.ttl.clear(1) == 1

    def test_start_background_rejects_zero_interval(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="positive finite"):
            c.ttl.start_background(interval=0)

    def test_start_background_rejects_negative_interval(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="positive finite"):
            c.ttl.start_background(interval=-1.0)

    def test_start_background_rejects_nan_interval(self, db_with_docs):
        _, c, _ = db_with_docs
        with pytest.raises(ValueError, match="positive finite"):
            c.ttl.start_background(interval=float("nan"))

    def test_start_background_idempotent_and_stops_cleanly(self, db_with_docs):
        _, c, _ = db_with_docs
        c.ttl.start_background(interval=60.0)
        # Idempotent — second call is a no-op.
        c.ttl.start_background(interval=60.0)
        c.ttl.stop_background()
        # After clean stop the next start spawns fresh.
        c.ttl.start_background(interval=60.0)
        c.ttl.stop_background()


# ----- regression: numeric filter type guard --------------------------------


class TestNumericFilterTypeGuard:
    """SQL numeric ops must reject non-numeric JSON values.

    Without ``json_type`` gating, ``CAST(json_extract(...) AS REAL)``
    coerces strings/null/objects to 0.0, so ``{"score": "oops"}`` would
    spuriously match ``{"score": {"$lt": 1}}``. The python-side
    ``_matches_filter`` already filters these out, so SQL and Python
    were disagreeing.
    """

    def _build_collection(self, tmp_path):
        db = VectorDB(str(tmp_path / "guard.db"))
        c = db.collection("default")
        c.add_texts(
            ["numeric-half", "numeric-five", "string-oops", "missing-key"],
            embeddings=np.random.rand(4, 4).astype(np.float32),
            metadatas=[
                {"score": 0.5},
                {"score": 5.0},
                {"score": "oops"},  # would CAST to 0.0 without the guard
                {"other": "no score key"},
            ],
        )
        return db, c

    def test_lt_skips_string_value(self, tmp_path):
        db, c = self._build_collection(tmp_path)
        try:
            docs = c.get_documents(filter_dict={"score": {"$lt": 1.0}})
            texts = {t for _, t, _ in docs}
            # Only the numeric 0.5 row passes; "oops" must NOT coerce to 0.0.
            assert texts == {"numeric-half"}
        finally:
            db.close()

    def test_between_skips_string_value(self, tmp_path):
        db, c = self._build_collection(tmp_path)
        try:
            docs = c.get_documents(filter_dict={"score": {"$between": (-1, 1)}})
            texts = {t for _, t, _ in docs}
            assert texts == {"numeric-half"}
        finally:
            db.close()

    def test_gt_skips_string_and_missing(self, tmp_path):
        db, c = self._build_collection(tmp_path)
        try:
            docs = c.get_documents(filter_dict={"score": {"$gt": 0.0}})
            texts = {t for _, t, _ in docs}
            # Both numeric rows pass; string and missing are excluded.
            assert texts == {"numeric-half", "numeric-five"}
        finally:
            db.close()

    def test_sql_and_python_agree_on_string_value(self, tmp_path):
        """SQL pre-filter (get_documents) and Python post-filter
        (similarity_search) must produce the same set of rows."""
        db, c = self._build_collection(tmp_path)
        try:
            sql_texts = {
                text
                for _, text, _ in c.get_documents(filter_dict={"score": {"$lt": 1.0}})
            }
            # similarity_search applies _matches_filter post-fetch
            hits = c.similarity_search([0.0] * 4, k=10, filter={"score": {"$lt": 1.0}})
            py_texts = {doc.page_content for doc, _ in hits}
            assert sql_texts == py_texts == {"numeric-half"}
        finally:
            db.close()


# ----- regression: TTL sweep atomicity --------------------------------------


class TestTTLSweepAtomic:
    """sweep_ttl must claim expired rows atomically.

    Before the fix, ``sweep_ttl`` ran a SELECT and then a DELETE in two
    separate steps; a concurrent ``set_ttl`` extending the deadline (or
    ``clear_ttl``) between those two steps would still see the doc
    deleted off the stale read. Post-fix it uses
    ``DELETE … RETURNING`` inside one write transaction.
    """

    def test_basic_sweep_still_works(self, db_with_docs):
        _, c, _ = db_with_docs
        c.ttl.set(1, expires_at=time.time() - 1, on_expire="delete")
        c.ttl.set(2, expires_at=time.time() - 1, on_expire="callback")
        deleted, callbacks = c.ttl.sweep()
        assert deleted == [1]
        assert callbacks == [2]
        # TTL rows for both must be cleared (RETURNING removed them).
        assert c.ttl.sweep() == ([], [])

    def test_extension_between_logical_check_and_delete(self, db_with_docs):
        """If a TTL is renewed before the sweep transaction runs, the
        atomic ``DELETE … RETURNING`` must observe the new value and
        skip the row — even when the test races by snapshotting the
        cutoff first.
        """
        _, c, _ = db_with_docs
        c.ttl.set(3, expires_at=time.time() - 5, on_expire="delete")
        # Capture a "what would expire as of cutoff_t0" snapshot, then
        # extend the TTL before sweeping with that same cutoff.
        cutoff_t0 = time.time()
        c.ttl.set(3, expires_at=time.time() + 3600, on_expire="delete")
        deleted, callbacks = c.ttl.sweep(now=cutoff_t0)
        # The renewed TTL is now > cutoff_t0 → must NOT be swept.
        assert deleted == []
        assert callbacks == []
        # Doc itself still present.
        rows = c.get_documents()
        assert any(doc_id == 3 for doc_id, _, _ in rows)

    def test_clear_between_logical_check_and_delete(self, db_with_docs):
        _, c, _ = db_with_docs
        c.ttl.set(4, expires_at=time.time() - 1, on_expire="delete")
        cutoff_t0 = time.time()
        c.ttl.clear(4)
        # Cleared TTL means there's no row to claim → no delete.
        deleted, callbacks = c.ttl.sweep(now=cutoff_t0)
        assert deleted == []
        assert callbacks == []
        rows = c.get_documents()
        assert any(doc_id == 4 for doc_id, _, _ in rows)
