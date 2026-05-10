"""Exercise every async collection function in simplevecdb.

Walks AsyncVectorCollection's full surface — CRUD, search, hierarchy,
edges, counters, pending vectors, TTL, events, maintenance, clustering —
plus the AsyncVectorDB-level multi-collection helpers. Designed to
double as a smoke test for the 2.6.1 "advanced memory" APIs.

Run:
    uv run python /tmp/exercise_async_collection.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import traceback
from typing import Any, Awaitable, Callable

import numpy as np

from simplevecdb.async_core import AsyncVectorCollection, AsyncVectorDB

DIM = 32
SEED = 0


def _vec(rng: np.random.Generator) -> list[float]:
    """Return a unit-ish random vector of length DIM."""
    v = rng.standard_normal(DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def _seed_corpus(
    n: int = 12,
) -> tuple[list[str], list[dict[str, Any]], list[list[float]]]:
    rng = np.random.default_rng(SEED)
    # Two semantic clusters separated by metadata "topic".
    docs = [
        ("python async tasks scheduler", "code"),
        ("event loops and coroutines", "code"),
        ("threadpool executor patterns", "code"),
        ("vector databases for retrieval", "code"),
        ("hnsw graph approximate nearest neighbor", "code"),
        ("embedding models sentence transformers", "code"),
        ("baking sourdough bread overnight", "food"),
        ("croissant lamination butter technique", "food"),
        ("ramen broth tonkotsu collagen", "food"),
        ("pizza neapolitan flour hydration", "food"),
        ("knife sharpening whetstone angles", "food"),
        ("fermentation kimchi gochugaru salt", "food"),
    ][:n]
    texts = [t for t, _ in docs]
    metas = [{"topic": tag, "hits": 0, "drift": 0.0} for _, tag in docs]
    vectors = [_vec(rng) for _ in range(n)]
    return texts, metas, vectors


class Reporter:
    """Tiny pass/fail accumulator with consistent formatting."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    async def run(
        self,
        label: str,
        coro: Callable[[], Awaitable[Any]],
    ) -> Any:
        try:
            value = await coro()
        except Exception as exc:  # noqa: BLE001 — we report and keep going
            tb = traceback.format_exc(limit=2).strip().splitlines()[-1]
            self.results.append((label, False, f"{type(exc).__name__}: {exc} ({tb})"))
            print(f"  ✗ {label}: {type(exc).__name__}: {exc}")
            return None
        summary = self._summarize(value)
        self.results.append((label, True, summary))
        print(f"  ✓ {label} → {summary}")
        return value

    @staticmethod
    def _summarize(value: Any) -> str:
        if value is None:
            return "ok"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}[{len(value)}]"
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
        if isinstance(value, (int, float, bool, str)):
            return repr(value)
        return type(value).__name__

    def report(self) -> int:
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        failed = total - passed
        print()
        print("=" * 64)
        print(f"  RESULT: {passed}/{total} passed, {failed} failed")
        print("=" * 64)
        if failed:
            for label, ok, detail in self.results:
                if not ok:
                    print(f"  FAIL {label}: {detail}")
        return failed


async def exercise_crud_and_search(
    r: Reporter,
    col: AsyncVectorCollection,
    texts: list[str],
    metas: list[dict[str, Any]],
    vectors: list[list[float]],
) -> list[int]:
    print("\n[1] CRUD + search")
    ids = await r.run(
        "add_texts",
        lambda: col.add_texts(texts, metadatas=metas, embeddings=vectors),
    )
    assert ids is not None and len(ids) == len(texts)
    await r.run("count", col.count)
    await r.run("dim (property)", lambda: asyncio.sleep(0, result=col.dim))
    await r.run(
        "get_documents (filter)",
        lambda: col.get_documents(filter_dict={"topic": "code"}, limit=5),
    )
    await r.run(
        "get_embeddings_by_ids",
        lambda: col.get_embeddings_by_ids(ids[:3]),
    )
    await r.run(
        "update_metadata",
        lambda: col.update_metadata([(ids[0], {"starred": True})]),
    )
    await r.run(
        "similarity_search (vector)",
        lambda: col.similarity_search(vectors[0], k=3),
    )
    await r.run(
        "similarity_search_batch",
        lambda: col.similarity_search_batch(vectors[:2], k=3),
    )
    await r.run(
        "keyword_search",
        lambda: col.keyword_search("python async", k=3),
    )
    await r.run(
        "hybrid_search",
        lambda: col.hybrid_search("python async", k=3, query_vector=vectors[0]),
    )
    await r.run(
        "max_marginal_relevance_search",
        lambda: col.max_marginal_relevance_search(vectors[0], k=3, fetch_k=8),
    )
    return ids


async def exercise_hierarchy(
    r: Reporter, col: AsyncVectorCollection, ids: list[int]
) -> None:
    print("\n[2] Hierarchy")
    root, mid, leaf = ids[0], ids[1], ids[2]
    await r.run("set_parent (mid→root)", lambda: col.set_parent(mid, root))
    await r.run("set_parent (leaf→mid)", lambda: col.set_parent(leaf, mid))
    await r.run("get_parent", lambda: col.get_parent(leaf))
    await r.run("get_children", lambda: col.get_children(root))
    await r.run("get_descendants", lambda: col.get_descendants(root))
    await r.run("get_ancestors", lambda: col.get_ancestors(leaf))
    await r.run("set_parent (clear)", lambda: col.set_parent(leaf, None))


async def exercise_edges(
    r: Reporter, col: AsyncVectorCollection, ids: list[int]
) -> None:
    print("\n[3] Edges")
    a, b, c = ids[0], ids[1], ids[2]
    await r.run(
        "add_edge (related)",
        lambda: col.add_edge(a, b, kind="related", weight=0.8, hits=1),
    )
    await r.run(
        "add_edge (cites)",
        lambda: col.add_edge(a, c, kind="cites", weight=0.4),
    )
    await r.run(
        "update_edge (delta)",
        lambda: col.update_edge(a, b, kind="related", dweight=0.1, dhits=2),
    )
    await r.run(
        "get_edges (by src)",
        lambda: col.get_edges(src=a),
    )
    await r.run(
        "delete_edge",
        lambda: col.delete_edge(a, c, kind="cites"),
    )


async def exercise_counters_pending_ttl_events(
    r: Reporter, col: AsyncVectorCollection, ids: list[int]
) -> None:
    print("\n[4] Counters / pending / TTL / events")
    target = ids[0]

    await r.run(
        "increment_metadata",
        lambda: col.increment_metadata(target, {"hits": 3, "drift": 0.05}),
    )

    rng = np.random.default_rng(SEED + 1)
    new_vec = _vec(rng)
    await r.run(
        "update_embedding (buffer)",
        lambda: col.update_embedding(target, new_vec, source="exerciser"),
    )
    await r.run("flush_pending", lambda: col.flush_pending(max_batch=128))

    seq_before = await r.run("last_event_seq (before)", col.last_event_seq)

    # Already-expired TTL → sweep should harvest it.
    await r.run(
        "set_ttl (expires_at past)",
        lambda: col.set_ttl(ids[-1], expires_at=time.time() - 1, on_expire="callback"),
    )
    await r.run(
        "set_ttl (seconds future)",
        lambda: col.set_ttl(ids[-2], seconds=3600, on_expire="delete"),
    )
    await r.run("sweep_ttl", lambda: col.sweep_ttl(limit=100))
    await r.run("clear_ttl", lambda: col.clear_ttl(ids[-2]))

    await r.run(
        "read_events (since)",
        lambda: col.read_events(since=int(seq_before or 0), limit=50),
    )


async def exercise_maintenance(r: Reporter, col: AsyncVectorCollection) -> None:
    print("\n[5] Maintenance")
    await r.run(
        "rebuild_if_needed",
        lambda: col.rebuild_if_needed(max_pending=10_000, max_deleted=10_000),
    )
    await r.run("rebuild_index", col.rebuild_index)
    await r.run("save", col.save)


async def exercise_clustering(r: Reporter, col: AsyncVectorCollection) -> None:
    print("\n[6] Clustering")
    cluster_result = await r.run(
        "cluster (kmeans, n=2)",
        lambda: col.cluster(n_clusters=2, algorithm="kmeans", min_cluster_size=2),
    )
    if cluster_result is None:
        return
    tags = await r.run(
        "auto_tag",
        lambda: col.auto_tag(cluster_result, method="keywords", n_keywords=3),
    )
    await r.run(
        "assign_cluster_metadata",
        lambda: col.assign_cluster_metadata(cluster_result, tags),
    )
    await r.run("get_cluster_members(0)", lambda: col.get_cluster_members(0))
    await r.run(
        "save_cluster",
        lambda: col.save_cluster("snapshot", cluster_result, metadata={"by": "test"}),
    )
    await r.run("list_clusters", col.list_clusters)
    await r.run("load_cluster", lambda: col.load_cluster("snapshot"))
    await r.run(
        "assign_to_cluster",
        lambda: col.assign_to_cluster("snapshot", [1, 2]),
    )
    await r.run("delete_cluster", lambda: col.delete_cluster("snapshot"))

    # Validation path also lives in async wrapper.
    async def _bad():
        try:
            await col.cluster(algorithm="bogus")
        except ValueError as exc:
            msg = str(exc)[:40]
            return f"ValueError({msg}…)"
        return "no error"

    await r.run("cluster (invalid algo → ValueError)", _bad)


async def exercise_db_level(
    r: Reporter, db: AsyncVectorDB, query_vec: list[float]
) -> None:
    print("\n[7] DB-level helpers")
    # Add a second collection so search_collections has somewhere to fan out.
    other = db.collection("other", store_embeddings=True)
    rng = np.random.default_rng(SEED + 2)
    await other.add_texts(
        ["another world", "second collection"],
        embeddings=[_vec(rng), _vec(rng)],
    )
    await r.run(
        "list_collections",
        lambda: asyncio.sleep(0, result=db.list_collections()),
    )
    await r.run(
        "search_collections",
        lambda: db.search_collections(query_vec, k=3),
    )
    await r.run("vacuum", db.vacuum)
    await r.run(
        "delete_collection (other)",
        lambda: db.delete_collection("other"),
    )


async def exercise_deletion(
    r: Reporter, col: AsyncVectorCollection, ids: list[int]
) -> None:
    print("\n[8] Deletion paths")
    await r.run(
        "delete_by_ids",
        lambda: col.delete_by_ids([ids[-1]]),
    )
    await r.run(
        "remove_texts (by filter)",
        lambda: col.remove_texts(filter={"topic": "food"}),
    )
    await r.run("count (post-delete)", col.count)


async def main() -> int:
    tmp = tempfile.mkdtemp(prefix="async_exerciser_")
    db_path = os.path.join(tmp, "advanced_memory.db")
    print(f"DB: {db_path}")

    r = Reporter()
    texts, metas, vectors = _seed_corpus()

    async with AsyncVectorDB(db_path, max_workers=4) as db:
        col = db.collection("memory", store_embeddings=True)
        ids = await exercise_crud_and_search(r, col, texts, metas, vectors)
        if ids:
            await exercise_hierarchy(r, col, ids)
            await exercise_edges(r, col, ids)
            await exercise_counters_pending_ttl_events(r, col, ids)
        await exercise_maintenance(r, col)
        await exercise_clustering(r, col)
        await exercise_db_level(r, db, vectors[0])
        if ids:
            await exercise_deletion(r, col, ids)

    return r.report()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
