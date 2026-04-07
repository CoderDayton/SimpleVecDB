"""Tests for simplevecdb 2.5.0 features: iterative deepening, store_embeddings,
FLOAT16 quantization, and pagination on get_documents / catalog lookups."""

import numpy as np
import pytest

from simplevecdb import VectorDB, Quantization
from simplevecdb.engine.quantization import QuantizationStrategy


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

DIM = 8


def _rand_embedding(seed: int) -> list[float]:
    """Deterministic random embedding for reproducibility."""
    rng = np.random.RandomState(seed)
    return rng.randn(DIM).tolist()


def _make_collection(db: VectorDB, name: str = "default", **kwargs):
    return db.collection(name, **kwargs)


# ------------------------------------------------------------------ #
#  1. Iterative deepening for filtered search
# ------------------------------------------------------------------ #


class TestIterativeDeepeningFilteredSearch:
    """The old k*3 approach fetches 15 candidates from 100 documents.
    If only 5 out of 100 match the filter and they are scattered throughout
    the index, the old approach would miss some.  Iterative deepening widens
    the fetch window until k results are found or the index is exhausted."""

    def test_sparse_filter_returns_all_matching(self):
        db = VectorDB(":memory:")
        col = db.collection("deep")

        n_total = 100
        n_rare = 5
        rare_indices = {10, 30, 50, 70, 90}

        texts = [f"doc_{i}" for i in range(n_total)]
        metadatas = [
            {"rare": True} if i in rare_indices else {"rare": False}
            for i in range(n_total)
        ]
        embeddings = [_rand_embedding(i) for i in range(n_total)]

        col.add_texts(texts, metadatas=metadatas, embeddings=embeddings)
        assert col.count() == n_total

        query = _rand_embedding(999)
        results = col.similarity_search(query, k=5, filter={"rare": True})

        assert len(results) == n_rare, (
            f"Expected {n_rare} results with filter={{rare: True}}, got {len(results)}"
        )
        for doc, _dist in results:
            assert doc.metadata["rare"] is True

    def test_filter_returns_fewer_than_k_when_insufficient(self):
        """When fewer documents match than k, return all matching."""
        db = VectorDB(":memory:")
        col = db.collection("sparse")

        texts = [f"doc_{i}" for i in range(20)]
        metadatas = [{"color": "red"} if i < 3 else {"color": "blue"} for i in range(20)]
        embeddings = [_rand_embedding(i) for i in range(20)]
        col.add_texts(texts, metadatas=metadatas, embeddings=embeddings)

        results = col.similarity_search(_rand_embedding(42), k=10, filter={"color": "red"})
        assert len(results) == 3

    def test_no_filter_match_returns_empty(self):
        db = VectorDB(":memory:")
        col = db.collection("empty_filter")
        col.add_texts(
            ["a", "b", "c"],
            metadatas=[{"x": 1}] * 3,
            embeddings=[_rand_embedding(i) for i in range(3)],
        )
        results = col.similarity_search(_rand_embedding(0), k=5, filter={"x": 999})
        assert results == []


# ------------------------------------------------------------------ #
#  2. store_embeddings=False (default) and True
# ------------------------------------------------------------------ #


class TestStoreEmbeddings:
    def test_default_no_store_similarity_search_works(self):
        """With store_embeddings=False (default), similarity_search still works
        because it uses the usearch index, not SQLite BLOBs."""
        db = VectorDB(":memory:")
        col = db.collection("nostore")

        texts = ["alpha", "beta", "gamma"]
        embeddings = [_rand_embedding(i) for i in range(3)]
        col.add_texts(texts, embeddings=embeddings)

        results = col.similarity_search(embeddings[0], k=2)
        assert len(results) == 2
        assert results[0][0].page_content == "alpha"

    def test_default_no_store_mmr_search_works(self):
        """MMR search falls back to usearch get() when embeddings are not
        stored in SQLite."""
        db = VectorDB(":memory:")
        col = db.collection("nostore_mmr")

        texts = ["one", "two", "three", "four"]
        embeddings = [_rand_embedding(i) for i in range(4)]
        col.add_texts(texts, embeddings=embeddings)

        results = col.max_marginal_relevance_search(embeddings[0], k=2, fetch_k=4)
        assert len(results) == 2

    def test_no_store_rebuild_index_raises(self):
        """rebuild_index requires stored embeddings; without them it raises."""
        db = VectorDB(":memory:")
        col = db.collection("nostore_rebuild")

        col.add_texts(["x", "y"], embeddings=[_rand_embedding(0), _rand_embedding(1)])
        assert col.count() == 2

        with pytest.raises(RuntimeError, match="No embeddings found"):
            col.rebuild_index()

    def test_store_embeddings_true_rebuild_works(self):
        """With store_embeddings=True, rebuild_index succeeds."""
        db = VectorDB(":memory:")
        col = db.collection("stored", store_embeddings=True)

        embeddings = [_rand_embedding(i) for i in range(5)]
        col.add_texts(
            [f"doc_{i}" for i in range(5)],
            embeddings=embeddings,
        )
        assert col.count() == 5

        rebuilt = col.rebuild_index()
        assert rebuilt == 5

        # Search still works after rebuild
        results = col.similarity_search(embeddings[0], k=2)
        assert len(results) == 2


# ------------------------------------------------------------------ #
#  3. FLOAT16 quantization roundtrip
# ------------------------------------------------------------------ #


class TestFloat16Quantization:
    def test_serialize_deserialize_roundtrip(self):
        """FLOAT16 serialize -> deserialize preserves values within half-precision tolerance."""
        qs = QuantizationStrategy(Quantization.FLOAT16)
        original = np.array([0.1, -0.25, 0.5, 1.0, -1.0, 0.0, 0.333, -0.777], dtype=np.float32)

        blob = qs.serialize(original)
        recovered = qs.deserialize(blob, dim=len(original))

        np.testing.assert_allclose(recovered, original, atol=1e-3, rtol=1e-2)
        assert recovered.dtype == np.float32

    def test_float16_halves_storage(self):
        """FLOAT16 BLOBs should be half the size of FLOAT32."""
        qs32 = QuantizationStrategy(Quantization.FLOAT)
        qs16 = QuantizationStrategy(Quantization.FLOAT16)
        vec = np.random.randn(128).astype(np.float32)

        blob32 = qs32.serialize(vec)
        blob16 = qs16.serialize(vec)

        assert len(blob16) == len(blob32) // 2

    def test_float16_collection_search(self):
        """End-to-end: create a FLOAT16 collection, add texts, search."""
        db = VectorDB(":memory:", quantization=Quantization.FLOAT16)
        col = db.collection("f16")

        n = 20
        texts = [f"item_{i}" for i in range(n)]
        embeddings = [_rand_embedding(i) for i in range(n)]
        col.add_texts(texts, embeddings=embeddings)

        results = col.similarity_search(embeddings[0], k=3)
        assert len(results) == 3
        # The nearest neighbor to embeddings[0] should be item_0 itself
        assert results[0][0].page_content == "item_0"


# ------------------------------------------------------------------ #
#  4. Pagination on get_documents
# ------------------------------------------------------------------ #


class TestGetDocumentsPagination:
    @pytest.fixture()
    def col_20(self):
        """Collection with 20 documents, deterministic IDs."""
        self._db = VectorDB(":memory:")
        col = self._db.collection("paged")
        texts = [f"text_{i:02d}" for i in range(20)]
        embeddings = [_rand_embedding(i) for i in range(20)]
        col.add_texts(texts, embeddings=embeddings)
        assert col.count() == 20
        return col

    def test_limit_returns_exact_count(self, col_20):
        docs = col_20.get_documents(limit=5)
        assert len(docs) == 5

    def test_offset_returns_different_page(self, col_20):
        page1 = col_20.get_documents(limit=5)
        page2 = col_20.get_documents(limit=5, offset=5)
        assert len(page2) == 5
        ids_1 = {d[0] for d in page1}
        ids_2 = {d[0] for d in page2}
        assert ids_1.isdisjoint(ids_2), "Pages must not overlap"

    def test_last_page(self, col_20):
        page = col_20.get_documents(limit=5, offset=15)
        assert len(page) == 5

    def test_beyond_end_returns_empty(self, col_20):
        page = col_20.get_documents(limit=5, offset=20)
        assert page == []

    def test_no_limit_returns_all(self, col_20):
        docs = col_20.get_documents()
        assert len(docs) == 20

    def test_pagination_with_filter(self):
        db = VectorDB(":memory:")
        col = db.collection("paged_filter")

        texts = [f"text_{i}" for i in range(20)]
        metadatas = [{"type": "a"} if i % 2 == 0 else {"type": "b"} for i in range(20)]
        embeddings = [_rand_embedding(i) for i in range(20)]
        col.add_texts(texts, metadatas=metadatas, embeddings=embeddings)

        # 10 docs have type=a
        all_a = col.get_documents(filter_dict={"type": "a"})
        assert len(all_a) == 10

        page1 = col.get_documents(filter_dict={"type": "a"}, limit=5)
        assert len(page1) == 5

        page2 = col.get_documents(filter_dict={"type": "a"}, limit=5, offset=5)
        assert len(page2) == 5

        ids_1 = {d[0] for d in page1}
        ids_2 = {d[0] for d in page2}
        assert ids_1.isdisjoint(ids_2)

    def test_full_page_coverage(self, col_20):
        """Iterating through all pages should yield all 20 documents."""
        all_ids = set()
        for offset in range(0, 20, 5):
            page = col_20.get_documents(limit=5, offset=offset)
            for doc_id, _text, _meta in page:
                all_ids.add(doc_id)

        assert len(all_ids) == 20


# ------------------------------------------------------------------ #
#  5. Pagination on find_ids_by_texts and find_ids_by_filter
# ------------------------------------------------------------------ #


class TestCatalogPagination:
    @pytest.fixture()
    def col_with_metadata(self):
        """Collection with 15 docs, varied metadata."""
        self._db = VectorDB(":memory:")
        col = self._db.collection("catalog_paged")

        texts = [f"sentence_{i}" for i in range(15)]
        metadatas = [{"group": i % 3, "idx": i} for i in range(15)]
        embeddings = [_rand_embedding(i) for i in range(15)]
        col.add_texts(texts, metadatas=metadatas, embeddings=embeddings)
        return col

    def test_find_ids_by_texts_all(self, col_with_metadata):
        texts = [f"sentence_{i}" for i in range(15)]
        ids = col_with_metadata._catalog.find_ids_by_texts(texts)
        assert len(ids) == 15

    def test_find_ids_by_texts_with_limit(self, col_with_metadata):
        texts = [f"sentence_{i}" for i in range(15)]
        ids = col_with_metadata._catalog.find_ids_by_texts(texts, limit=5)
        assert len(ids) == 5

    def test_find_ids_by_texts_with_limit_offset(self, col_with_metadata):
        texts = [f"sentence_{i}" for i in range(15)]
        page1 = col_with_metadata._catalog.find_ids_by_texts(texts, limit=5)
        page2 = col_with_metadata._catalog.find_ids_by_texts(texts, limit=5, offset=5)
        assert len(page2) == 5
        assert set(page1).isdisjoint(set(page2))

    def test_find_ids_by_texts_offset_beyond_end(self, col_with_metadata):
        texts = [f"sentence_{i}" for i in range(15)]
        ids = col_with_metadata._catalog.find_ids_by_texts(texts, limit=5, offset=15)
        assert ids == []

    def test_find_ids_by_filter_all(self, col_with_metadata):
        catalog = col_with_metadata._catalog
        builder = catalog.build_filter_clause
        ids = catalog.find_ids_by_filter({"group": 0}, builder)
        assert len(ids) == 5  # 0, 3, 6, 9, 12

    def test_find_ids_by_filter_with_limit(self, col_with_metadata):
        catalog = col_with_metadata._catalog
        builder = catalog.build_filter_clause
        ids = catalog.find_ids_by_filter({"group": 0}, builder, limit=3)
        assert len(ids) == 3

    def test_find_ids_by_filter_with_limit_offset(self, col_with_metadata):
        catalog = col_with_metadata._catalog
        builder = catalog.build_filter_clause
        page1 = catalog.find_ids_by_filter({"group": 0}, builder, limit=3)
        page2 = catalog.find_ids_by_filter({"group": 0}, builder, limit=3, offset=3)
        assert len(page2) == 2  # only 2 remaining (5 total, took 3)
        assert set(page1).isdisjoint(set(page2))

    def test_find_ids_by_filter_offset_beyond_end(self, col_with_metadata):
        catalog = col_with_metadata._catalog
        builder = catalog.build_filter_clause
        ids = catalog.find_ids_by_filter({"group": 0}, builder, limit=5, offset=10)
        assert ids == []

    def test_find_ids_by_filter_empty_dict(self, col_with_metadata):
        catalog = col_with_metadata._catalog
        builder = catalog.build_filter_clause
        ids = catalog.find_ids_by_filter({}, builder)
        assert ids == []
