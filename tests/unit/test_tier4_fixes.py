"""Tier-4 polish/robustness fixes — regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simplevecdb import VectorDB


def _emb(n: int, dim: int, seed: int = 0) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32).tolist()


class TestFTS5SyntaxError:
    def test_malformed_query_raises_valueerror(self) -> None:
        """A malformed FTS5 MATCH query surfaces as ValueError, not OperationalError."""
        db = VectorDB(":memory:")
        col = db.collection("t")
        col.add_texts(["hello world", "foo bar"], embeddings=_emb(2, 8))
        if not col._catalog.fts_enabled:
            pytest.skip("FTS5 not available")
        with pytest.raises(ValueError):
            col.keyword_search('"unbalanced', k=5)


class TestClusterTableRollback:
    def test_save_cluster_works_after_rolled_back_transaction(
        self, tmp_path: Path
    ) -> None:
        """The cluster table is created eagerly, so a rolled-back first use cannot
        leave the ready-flag set without the table existing."""
        pytest.importorskip("sklearn")
        db = VectorDB(tmp_path / "c.db")
        col = db.collection("t")
        col.add_texts([f"d{i}" for i in range(6)], embeddings=_emb(6, 8))
        result = col.cluster(n_clusters=2, random_state=0)

        try:
            with db.transaction():
                col.save_cluster("inside_tx", result)
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass

        # Table still exists -> this save succeeds and round-trips.
        col.save_cluster("after", result)
        assert col.load_cluster("after") is not None
