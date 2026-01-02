"""Tests for streaming insert API."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from typing import Iterator

from simplevecdb import VectorDB, StreamingProgress


class TestAddTextsStreaming:
    """Tests for VectorCollection.add_texts_streaming()."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_streaming.db"

    @pytest.fixture
    def dim(self) -> int:
        return 64

    def make_items(
        self, n: int, dim: int
    ) -> Iterator[tuple[str, dict | None, list[float] | None]]:
        """Generate test items as iterator."""
        for i in range(n):
            text = f"document {i}"
            meta = {"index": i, "category": "test"}
            embedding = np.random.randn(dim).astype(np.float32).tolist()
            yield (text, meta, embedding)

    def test_streaming_basic(self, db_path: Path, dim: int):
        """Basic streaming insert works."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        items = list(self.make_items(10, dim))
        progress_list = list(collection.add_texts_streaming(iter(items)))

        # Should have progress yields
        assert len(progress_list) >= 1

        # Final progress should show all docs
        final = progress_list[-1]
        assert final["docs_processed"] == 10

        # Verify data was inserted
        assert collection.count() == 10

        db.close()

    def test_streaming_batches(self, db_path: Path, dim: int):
        """Streaming respects batch_size."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        n_docs = 25
        batch_size = 10
        items = list(self.make_items(n_docs, dim))

        progress_list = list(
            collection.add_texts_streaming(iter(items), batch_size=batch_size)
        )

        # Should have 3 batches: 10, 10, 5
        assert len(progress_list) == 3
        assert progress_list[0]["docs_in_batch"] == 10
        assert progress_list[1]["docs_in_batch"] == 10
        assert progress_list[2]["docs_in_batch"] == 5

        assert collection.count() == n_docs
        db.close()

    def test_streaming_progress_callback(self, db_path: Path, dim: int):
        """Progress callback is invoked."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        callback_invocations: list[StreamingProgress] = []

        def on_progress(p: StreamingProgress):
            callback_invocations.append(p)

        items = list(self.make_items(15, dim))
        list(
            collection.add_texts_streaming(
                iter(items), batch_size=5, on_progress=on_progress
            )
        )

        # Callback should be invoked for each batch
        assert len(callback_invocations) == 3
        assert callback_invocations[0]["batch_num"] == 1
        assert callback_invocations[1]["batch_num"] == 2
        assert callback_invocations[2]["batch_num"] == 3

        db.close()

    def test_streaming_generator_input(self, db_path: Path, dim: int):
        """Works with true generator (not list)."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Pass generator directly, not list
        gen = self.make_items(20, dim)
        progress_list = list(collection.add_texts_streaming(gen, batch_size=7))

        # 20 docs with batch_size=7: batches of 7, 7, 6
        assert len(progress_list) == 3
        assert progress_list[-1]["docs_processed"] == 20

        db.close()

    def test_streaming_empty_input(self, db_path: Path):
        """Empty input produces no progress."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        progress_list = list(collection.add_texts_streaming(iter([])))

        assert len(progress_list) == 0
        assert collection.count() == 0

        db.close()

    def test_streaming_single_item(self, db_path: Path, dim: int):
        """Single item works."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        items = [("single doc", {"id": 1}, np.random.randn(dim).tolist())]
        progress_list = list(collection.add_texts_streaming(iter(items)))

        assert len(progress_list) == 1
        assert progress_list[0]["docs_processed"] == 1
        assert progress_list[0]["docs_in_batch"] == 1

        db.close()

    def test_streaming_none_metadata(self, db_path: Path, dim: int):
        """None metadata is handled."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        items = [
            ("doc1", None, np.random.randn(dim).tolist()),
            ("doc2", {"key": "value"}, np.random.randn(dim).tolist()),
            ("doc3", None, np.random.randn(dim).tolist()),
        ]
        progress_list = list(collection.add_texts_streaming(iter(items)))

        assert progress_list[-1]["docs_processed"] == 3

        # Search should work
        results = collection.similarity_search(np.random.randn(dim).tolist(), k=3)
        assert len(results) == 3

        db.close()

    def test_streaming_searchable(self, db_path: Path, dim: int):
        """Streamed documents are searchable."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        # Insert with known embedding
        target_vec = np.array([1.0] * dim, dtype=np.float32)
        target_vec = target_vec / np.linalg.norm(target_vec)  # Normalize

        items = [
            ("target document", {"target": True}, target_vec.tolist()),
            ("other doc 1", None, np.random.randn(dim).tolist()),
            ("other doc 2", None, np.random.randn(dim).tolist()),
        ]
        list(collection.add_texts_streaming(iter(items)))

        # Search with same vector should find target
        results = collection.similarity_search(target_vec.tolist(), k=1)
        assert len(results) == 1
        assert results[0][0].page_content == "target document"
        assert results[0][0].metadata.get("target") is True

        db.close()

    def test_streaming_batch_ids_returned(self, db_path: Path, dim: int):
        """Progress includes batch IDs."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        items = list(self.make_items(5, dim))
        progress_list = list(collection.add_texts_streaming(iter(items), batch_size=2))

        # Collect all IDs from progress
        all_ids = []
        for p in progress_list:
            assert "batch_ids" in p
            assert len(p["batch_ids"]) == p["docs_in_batch"]
            all_ids.extend(p["batch_ids"])

        # Should have 5 unique IDs
        assert len(all_ids) == 5
        assert len(set(all_ids)) == 5

        db.close()

    def test_streaming_large_batch(self, db_path: Path, dim: int):
        """Large ingestion works without memory issues."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        n_docs = 1000
        batch_size = 100

        def generate_large():
            for i in range(n_docs):
                yield (f"doc {i}", {"i": i}, np.random.randn(dim).tolist())

        progress_count = 0
        for progress in collection.add_texts_streaming(
            generate_large(), batch_size=batch_size
        ):
            progress_count += 1
            # Memory should stay bounded - generator processes lazily

        assert progress_count == 10  # 1000 / 100
        assert collection.count() == n_docs

        db.close()


class TestStreamingWithFilters:
    """Test streaming with metadata filters."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_streaming_filter.db"

    def test_filter_after_streaming(self, db_path: Path):
        """Metadata filters work on streamed documents."""
        db = VectorDB(db_path)
        collection = db.collection("test")
        dim = 32

        items = [
            ("cat doc", {"animal": "cat"}, np.random.randn(dim).tolist()),
            ("dog doc", {"animal": "dog"}, np.random.randn(dim).tolist()),
            ("cat doc 2", {"animal": "cat"}, np.random.randn(dim).tolist()),
        ]
        list(collection.add_texts_streaming(iter(items)))

        # Filter by animal=cat
        results = collection.similarity_search(
            np.random.randn(dim).tolist(), k=10, filter={"animal": "cat"}
        )
        assert len(results) == 2
        for doc, _ in results:
            assert doc.metadata["animal"] == "cat"

        db.close()
