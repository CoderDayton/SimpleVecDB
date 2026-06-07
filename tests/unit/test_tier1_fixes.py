"""Tier-1 review fixes — TDD regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simplevecdb import VectorDB


def _emb(n: int, dim: int, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32).tolist()


class TestClusteringGuard:
    def test_cluster_vectors_n_clusters_exceeds_raises_clear_error(self) -> None:
        """cluster_vectors with n_clusters > n_vectors gives a descriptive error."""
        pytest.importorskip("sklearn")
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        vectors = np.asarray(_emb(5, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="cannot exceed"):
            engine.cluster_vectors(
                vectors, doc_ids=[0, 1, 2, 3, 4], algorithm="kmeans", n_clusters=10
            )


class TestAsyncIncrementReturnsInt:
    @pytest.mark.asyncio
    async def test_returns_counter_update_result(self) -> None:
        """async increment_metadata returns the int result (1 updated / 0 missing)."""
        from simplevecdb import AsyncVectorDB

        async with AsyncVectorDB(":memory:") as db:
            col = db.collection("t")
            ids = await col.add_texts(["doc"], embeddings=_emb(1, 16))
            updated = await col.increment_metadata(ids[0], {"hits": 1})
            assert updated == 1
            missing = await col.increment_metadata(999999, {"hits": 1})
            assert missing == 0


class TestCatalogWritableLockRelease:
    def test_lock_released_when_conn_enter_fails(self) -> None:
        """If conn.__enter__ raises, the writable lock must be released, not leaked."""
        import sqlite3
        import threading
        import types

        from simplevecdb.engine.catalog import _CatalogWritable

        lock = threading.RLock()

        class BadConn:
            def __enter__(self):
                raise sqlite3.OperationalError("simulated busy")

            def __exit__(self, *exc):
                return False

        tx = types.SimpleNamespace(depth=0)
        writable = _CatalogWritable(lock, BadConn(), tx)  # type: ignore[arg-type]

        with pytest.raises(sqlite3.OperationalError):
            writable.__enter__()

        # Probe from a *different* thread (RLock is reentrant; same thread would
        # falsely re-acquire and mask a leak).
        probe: dict[str, bool] = {}

        def _probe() -> None:
            got = lock.acquire(blocking=False)
            probe["got"] = got
            if got:
                lock.release()

        t = threading.Thread(target=_probe)
        t.start()
        t.join()
        assert probe["got"], "lock leaked (still held) after __enter__ failure"


class TestMMRMetricAware:
    def test_l2_mmr_respects_diversity(self) -> None:
        """On an L2 collection MMR must still apply diversity (metric-aware relevance)."""
        from simplevecdb.types import DistanceStrategy

        db = VectorDB(":memory:", distance_strategy=DistanceStrategy.L2)
        col = db.collection("t")

        def v(x: float, y: float) -> list[float]:
            a = np.zeros(4, dtype=np.float32)
            a[0], a[1] = x, y
            return a.tolist()

        # C1a/C1b are a redundant near-duplicate pair; D is the diverse option,
        # slightly farther from the query.
        col.add_texts(
            ["C1a", "C1b", "D"], embeddings=[v(1.0, 0.0), v(1.2, 0.0), v(0.0, 0.5)]
        )
        res = col.max_marginal_relevance_search(
            v(10.0, 0.0), k=2, fetch_k=10, lambda_mult=0.3
        )
        got = [d.page_content for d in res]
        assert "D" in got, f"MMR ignored diversity on L2 (picked redundant pair): {got}"


class TestBatchSearchFilterFallback:
    def _populate(self, col, dim: int = 8) -> None:
        texts, embs, metas = [], [], []
        # 18 docs aligned with the query (near), 2 orthogonal "keep" docs (far).
        for i in range(18):
            e = np.zeros(dim, dtype=np.float32)
            e[0] = 1.0
            e[1] = 0.001 * i
            texts.append(f"near{i}")
            embs.append(e.tolist())
            metas.append({"keep": False})
        for j in range(2):
            e = np.zeros(dim, dtype=np.float32)
            e[1] = 1.0
            e[2] = 0.001 * j
            texts.append(f"keep{j}")
            embs.append(e.tolist())
            metas.append({"keep": True})
        col.add_texts(texts, embeddings=embs, metadatas=metas)

    def test_large_filtered_batch_returns_full_k(self) -> None:
        """A >threshold batch with a selective filter must still return k per query."""
        db = VectorDB(":memory:")
        col = db.collection("t")
        self._populate(col)
        q = np.zeros(8, dtype=np.float32)
        q[0] = 1.0
        queries = [q.tolist()] * 11  # > USEARCH_BATCH_THRESHOLD (10) -> native path
        results = col.similarity_search_batch(queries, k=2, filter={"keep": True})
        assert len(results) == 11
        for r in results:
            assert len(r) == 2, f"filtered batch under-delivered: {len(r)}"
            assert all(doc.metadata.get("keep") is True for doc, _ in r)

    def test_text_query_outcome_consistent_across_batch_size(self) -> None:
        """A text query must behave the same in a small (routed) and a large batch,
        not crash only in the large (native) path."""
        db = VectorDB(":memory:")
        col = db.collection("t")
        self._populate(col)

        def outcome(n: int) -> str:
            try:
                col.similarity_search_batch(["some text query"] * n, k=2)  # type: ignore[list-item]
                return "ok"
            except Exception as exc:  # noqa: BLE001 - classifying the failure mode
                m = str(exc).lower()
                if any(
                    s in m
                    for s in (
                        "inhomogeneous",
                        "could not convert",
                        "setting an array element",
                    )
                ):
                    return "numpy-crash"
                return "routed-error"

        small = outcome(3)  # <= USEARCH_BATCH_THRESHOLD -> always per-query routed
        large = outcome(11)  # > threshold
        assert small == large, (
            f"text query differs by batch size: small={small} large={large}"
        )


class TestRebuildIndexRecovery:
    def test_failed_rebuild_keeps_collection_usable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """If the rebuild fails after closing the live index, the collection must
        re-open the intact on-disk index and stay usable, not brick."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        db = VectorDB(tmp_path / "r.db")
        col = db.collection("t", store_embeddings=True)
        embs = _emb(20, 16)
        col.add_texts([f"d{i}" for i in range(20)], embeddings=embs)
        assert len(col.similarity_search(embs[0], k=3)) >= 1

        orig_add = UsearchIndex.add

        def add_hook(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
            # Fail only while building the new (.rebuild) index, after the live
            # index has already been closed.
            if ".rebuild" in str(self._path):
                raise RuntimeError("simulated rebuild add failure")
            return orig_add(self, *a, **k)

        monkeypatch.setattr(UsearchIndex, "add", add_hook)
        with pytest.raises(RuntimeError):
            col.rebuild_index()

        after = col.similarity_search(embs[0], k=3)
        assert len(after) >= 1, "collection bricked after failed rebuild"


class TestFilterLiteralKey:
    def test_sql_filter_matches_literal_top_level_key(self) -> None:
        """build_filter_clause must match a literal key 'a.b', not the nested path a->b."""
        db = VectorDB(":memory:")
        col = db.collection("t")
        col.add_texts(
            ["lit", "nested"],
            embeddings=_emb(2, 8),
            metadatas=[{"a.b": "X"}, {"a": {"b": "X"}}],
        )
        cat = col._catalog
        ids = cat.find_ids_by_filter({"a.b": "X"}, cat.build_filter_clause)
        id_to_text = {doc_id: text for doc_id, text, _ in col.get_documents(limit=100)}
        matched = {id_to_text[i] for i in ids}
        assert matched == {"lit"}, f"expected literal-key match only, got {matched}"

    def test_quote_in_key_rejected(self) -> None:
        db = VectorDB(":memory:")
        col = db.collection("t")
        col.add_texts(["d"], embeddings=_emb(1, 8))
        cat = col._catalog
        with pytest.raises(ValueError):
            cat.find_ids_by_filter({'bad"key': 1}, cat.build_filter_clause)


class TestClusterSampleSizeCap:
    def test_n_clusters_capped_to_sample_size(self, tmp_path: Path) -> None:
        """With sampling, n_clusters caps to the sampled count instead of erroring."""
        pytest.importorskip("sklearn")
        db = VectorDB(tmp_path / "cs.db")
        col = db.collection("t")
        col.add_texts([f"d{i}" for i in range(30)], embeddings=_emb(30, 16))
        result = col.cluster(n_clusters=20, sample_size=10, random_state=0)
        assert result.n_clusters <= 10
