"""Tier-2 review fixes — TDD regression tests."""

from __future__ import annotations

import numpy as np

from simplevecdb import Quantization, VectorDB


def _emb(n: int, dim: int, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32).tolist()


class TestBitGetUnpacks:
    def test_bit_index_get_returns_unpacked_float_vectors(self) -> None:
        """UsearchIndex.get() on a BIT index must unpack bytes to ±1 floats."""
        db = VectorDB(":memory:")
        col = db.collection("t", quantization=Quantization.BIT)
        dim = 16
        ids = col.add_texts([f"d{i}" for i in range(4)], embeddings=_emb(4, dim))
        got = col._index.get(np.array(ids, dtype=np.uint64))
        assert got.shape == (4, dim), f"expected (4, {dim}), got {got.shape}"
        assert set(np.unique(got)).issubset({-1.0, 1.0}), (
            f"non ±1 values: {np.unique(got)}"
        )


class TestRebuildConcurrentCatchup:
    def test_rebuild_catches_up_writes_during_build(
        self, tmp_path, monkeypatch
    ) -> None:
        """Writes that land during the (unlocked) rebuild build are folded into
        the new index before the swap, so none are lost."""
        from simplevecdb.engine.usearch_index import UsearchIndex

        db = VectorDB(tmp_path / "r.db")
        col = db.collection("t", store_embeddings=True)
        ids = col.add_texts([f"d{i}" for i in range(10)], embeddings=_emb(10, 8))

        orig_add = UsearchIndex.add
        fired = {"v": False}
        new_id: dict[str, int] = {}

        def add_hook(self, keys, vectors, **kwargs):  # noqa: ANN001, ANN002, ANN003
            result = orig_add(self, keys, vectors, **kwargs)
            # Simulate a writer landing during the build window (once).
            if ".rebuild" in str(self._path) and not fired["v"]:
                fired["v"] = True
                new_id["v"] = col.add_texts(
                    ["concurrent"], embeddings=_emb(1, 8, seed=99)
                )[0]
                col.delete_by_ids([ids[0]])
            return result

        monkeypatch.setattr(UsearchIndex, "add", add_hook)
        col.rebuild_index()

        keys = set(col._index.keys())
        assert new_id["v"] in keys, "concurrent add lost during rebuild"
        assert ids[0] not in keys, "concurrent delete not reflected after rebuild"


class TestLangChainRelevanceScoreFn:
    def test_cosine_relevance_higher_for_closer(self) -> None:
        import pytest

        pytest.importorskip("langchain_core")
        from simplevecdb.integrations.langchain import SimpleVecDBVectorStore

        store = SimpleVecDBVectorStore(db_path=":memory:")  # default cosine
        fn = store._select_relevance_score_fn()
        assert fn(0.0) == 1.0  # closest -> max relevance
        assert fn(2.0) == 0.0  # farthest cosine distance -> min relevance
        assert fn(0.5) > fn(1.5)  # monotonically decreasing

    def test_l2_relevance_bounded_and_decreasing(self) -> None:
        import pytest

        pytest.importorskip("langchain_core")
        from simplevecdb.integrations.langchain import SimpleVecDBVectorStore
        from simplevecdb.types import DistanceStrategy

        store = SimpleVecDBVectorStore(
            db_path=":memory:", distance_strategy=DistanceStrategy.L2
        )
        fn = store._select_relevance_score_fn()
        assert fn(0.0) == 1.0
        assert 0.0 < fn(100.0) < fn(1.0) <= 1.0
