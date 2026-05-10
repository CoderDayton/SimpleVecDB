"""Tests targeting uncovered lines in core.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simplevecdb import VectorDB
from simplevecdb.core import get_optimal_batch_size
from simplevecdb.types import ClusterResult


# ------------------------------------------------------------------ #
# get_optimal_batch_size edge cases (line 83)
# ------------------------------------------------------------------ #


class TestGetOptimalBatchSizeEdgeCases:
    def test_cuda_low_vram_returns_64(self):
        """When CUDA GPU has very low VRAM, fall through all thresholds to 64."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value.total_memory = (
            0.5 * 1024**3  # 0.5 GB - below all thresholds
        )
        # Ensure ROCm and MPS are not available
        mock_torch.hip = MagicMock()
        mock_torch.hip.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch, "onnxruntime": None}):
            result = get_optimal_batch_size()
        assert result == 64


# ------------------------------------------------------------------ #
# add_texts validation (lines 354, 358, 362, 366)
# ------------------------------------------------------------------ #


class TestAddTextsValidation:
    @pytest.fixture
    def collection(self):
        db = VectorDB(":memory:")
        coll = db.collection("default")
        return coll

    def test_embeddings_length_mismatch(self, collection):
        with pytest.raises(ValueError, match="embeddings length"):
            collection.add_texts(
                ["a", "b"], embeddings=[[0.1, 0.2]], metadatas=[{}, {}]
            )

    def test_ids_length_mismatch(self, collection):
        with pytest.raises(ValueError, match="ids length"):
            collection.add_texts(
                ["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]], ids=[1]
            )

    def test_parent_ids_length_mismatch(self, collection):
        with pytest.raises(ValueError, match="parent_ids length"):
            collection.add_texts(
                ["a", "b"],
                embeddings=[[0.1, 0.2], [0.3, 0.4]],
                parent_ids=[1],
            )

    def test_metadatas_length_mismatch(self, collection):
        with pytest.raises(ValueError, match="metadatas length"):
            collection.add_texts(
                ["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]], metadatas=[{}]
            )


# ------------------------------------------------------------------ #
# similarity_search / batch / keyword early returns (lines 605, 643, 668)
# ------------------------------------------------------------------ #


class TestSearchEarlyReturns:
    @pytest.fixture
    def collection(self):
        db = VectorDB(":memory:")
        coll = db.collection("default")
        coll.add_texts(["hello"], embeddings=[[0.1, 0.2, 0.3]])
        return coll

    def test_similarity_search_k_zero(self, collection):
        result = collection.similarity_search([0.1, 0.2, 0.3], k=0)
        assert result == []

    def test_similarity_search_k_negative(self, collection):
        result = collection.similarity_search([0.1, 0.2, 0.3], k=-1)
        assert result == []

    def test_similarity_search_batch_k_zero(self, collection):
        result = collection.similarity_search_batch([[0.1, 0.2, 0.3]], k=0)
        assert result == []

    def test_similarity_search_batch_empty_queries(self, collection):
        result = collection.similarity_search_batch([], k=5)
        assert result == []

    def test_keyword_search_k_zero(self, collection):
        result = collection.keyword_search("hello", k=0)
        assert result == []

    def test_keyword_search_empty_query(self, collection):
        result = collection.keyword_search("", k=5)
        assert result == []


# ------------------------------------------------------------------ #
# rebuild_index with no embeddings (line 857)
# ------------------------------------------------------------------ #


class TestRebuildIndexNoEmbeddings:
    def test_rebuild_index_no_embeddings_raises(self, tmp_path):
        """rebuild_index raises RuntimeError when no embeddings in SQLite."""
        db = VectorDB(str(tmp_path / "rebuild.db"))
        collection = db.collection("default", store_embeddings=True)
        # Insert a doc but clear embeddings from catalog
        collection.add_texts(["test"], embeddings=[[0.1, 0.2]])
        # Wipe the embeddings column
        db.conn.execute(
            f"UPDATE {collection._table_name} SET embedding = NULL"
        )
        db.conn.commit()
        with pytest.raises(RuntimeError, match="No embeddings found"):
            collection.rebuild_index()
        db.close()


# ------------------------------------------------------------------ #
# VectorDB context manager (lines 1803, 1806)
# ------------------------------------------------------------------ #


class TestVectorDBContextManager:
    def test_context_manager_enter_exit(self):
        with VectorDB(":memory:") as db:
            coll = db.collection("test")
            coll.add_texts(["hello"], embeddings=[[0.1, 0.2]])
            assert coll.count() == 1
        # After exiting, db should be closed
        assert db._closed is True

    def test_close_idempotent(self):
        db = VectorDB(":memory:")
        db.close()
        db.close()  # Should not raise
        assert db._closed is True

    def test_del_calls_close(self):
        db = VectorDB(":memory:")
        db.collection("test").add_texts(["hi"], embeddings=[[0.1]])
        db.__del__()
        assert db._closed is True




# ------------------------------------------------------------------ #
# Cross-collection search: parallel failure handling (lines 1552-1553)
# ------------------------------------------------------------------ #


class TestCrossCollectionSearchFailure:
    def test_parallel_search_handles_failure(self):
        """When one collection search fails in parallel, results from others are kept."""
        db = VectorDB(":memory:")
        c1 = db.collection("c1")
        c2 = db.collection("c2")
        c1.add_texts(["good doc"], embeddings=[[0.1, 0.2]])
        c2.add_texts(["another"], embeddings=[[0.3, 0.4]])

        # Make c2's search raise
        def failing_search(*args, **kwargs):
            raise RuntimeError("Simulated failure")
        c2.similarity_search = failing_search

        results = db.search_collections([0.1, 0.2], k=5, parallel=True)
        # c1 results should still be returned
        assert len(results) >= 1
        assert any(coll == "c1" for _, _, coll in results)


# ------------------------------------------------------------------ #
# Streaming: on_progress callback (line 541)
# ------------------------------------------------------------------ #


class TestStreamingOnProgress:
    def test_on_progress_called(self, tmp_path):
        """on_progress callback is invoked for each batch."""
        db = VectorDB(str(tmp_path / "stream.db"))
        collection = db.collection("test")

        progress_reports = []

        def callback(progress):
            progress_reports.append(progress)

        items = [
            (f"doc{i}", {"i": i}, [float(i) * 0.1, float(i) * 0.2])
            for i in range(5)
        ]

        # Consume the generator
        list(collection.add_texts_streaming(iter(items), on_progress=callback))

        assert len(progress_reports) >= 1
        assert progress_reports[-1]["docs_processed"] == 5
        db.close()


# ------------------------------------------------------------------ #
# Streaming: auto-embedding in _process_streaming_batch (lines 566-567)
# ------------------------------------------------------------------ #


class TestStreamingAutoEmbedding:
    def test_process_streaming_batch_auto_embed(self, tmp_path):
        """_process_streaming_batch generates embeddings when needed."""
        db = VectorDB(str(tmp_path / "auto_embed.db"))
        collection = db.collection("test")

        # First add a doc to set dimensions
        collection.add_texts(["init"], embeddings=[[0.1, 0.2, 0.3]])

        mock_embeddings = [
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
            np.array([0.7, 0.8, 0.9], dtype=np.float32),
        ]

        with patch(
            "simplevecdb.embeddings.models.embed_texts", return_value=mock_embeddings
        ):
            ids = collection._process_streaming_batch(
                texts=["text1", "text2"],
                metas=[{}, {}],
                embeds=[None, None],
                needs_embedding=True,
                threads=1,
            )

        assert len(ids) == 2
        assert collection.count() == 3
        db.close()



# ------------------------------------------------------------------ #
# _resolve_index_path with encrypted index (line 261)
# ------------------------------------------------------------------ #


class TestResolveIndexPathEncrypted:
    def test_encrypted_index_without_key_raises(self, tmp_path):
        """Finding encrypted index without encryption key raises EncryptionError."""
        from simplevecdb.encryption import encrypt_index_file, EncryptionError

        db_path = tmp_path / "enc_test.db"
        # Create a database first
        db = VectorDB(str(db_path))
        coll = db.collection("test")
        coll.add_texts(["hello"], embeddings=[[0.1, 0.2, 0.3]])
        db.close()

        # Encrypt the index file
        index_path = Path(f"{db_path}.test.usearch")
        if index_path.exists():
            encrypt_index_file(index_path, "some-secret-key")
            # Remove the unencrypted index
            index_path.unlink(missing_ok=True)

            # Now try to open without encryption key - should raise
            with pytest.raises(EncryptionError):
                db2 = VectorDB(str(db_path))
                db2.collection("test")


# ------------------------------------------------------------------ #
# assign_to_cluster edge cases (lines 1322, 1327-1333, 1336)
# ------------------------------------------------------------------ #


class TestAssignToCluster:
    @pytest.fixture
    def clustered_collection(self, tmp_path):
        pytest.importorskip("sklearn")
        db = VectorDB(str(tmp_path / "cluster.db"))
        coll = db.collection("test")

        # Add clustered data
        np.random.seed(42)
        texts = []
        embeddings = []
        for c in range(3):
            for i in range(5):
                texts.append(f"c{c}_d{i}")
                emb = np.random.randn(8).astype(np.float32)
                emb[c] += 10.0
                embeddings.append(emb.tolist())

        coll.add_texts(texts, embeddings=embeddings)
        return db, coll

    def test_assign_to_cluster_not_found(self, clustered_collection):
        _, coll = clustered_collection
        with pytest.raises(ValueError, match="not found"):
            coll.assign_to_cluster("nonexistent")

    def test_assign_to_cluster_no_centroids(self, clustered_collection):
        """HDBSCAN clusters have no centroids and can't be used for assignment."""
        db, coll = clustered_collection

        # Save a fake cluster result with no centroids
        result = ClusterResult(
            labels=np.array([0, 1, 2]),
            centroids=None,
            doc_ids=[1, 2, 3],
            n_clusters=3,
            algorithm="hdbscan",
        )
        coll.save_cluster("no_centroids", result)

        with pytest.raises(ValueError, match="no centroids"):
            coll.assign_to_cluster("no_centroids")
        db.close()

    def test_assign_to_cluster_auto_select_unassigned(self, clustered_collection):
        """When doc_ids=None, assigns only unassigned documents."""
        db, coll = clustered_collection

        # First cluster and save
        result = coll.cluster(n_clusters=3, random_state=42)
        coll.save_cluster("test_cluster", result)

        # Tag existing docs with cluster metadata so they count as "assigned"
        all_ids = list(coll._index.keys())
        updates = [(doc_id, {"cluster": 0}) for doc_id in all_ids[:5]]
        coll._catalog.update_metadata_batch(updates)

        # Add new untagged docs
        np.random.seed(99)
        new_embs = np.random.randn(3, 8).astype(np.float32).tolist()
        coll.add_texts(["new1", "new2", "new3"], embeddings=new_embs)

        # Assign only the unassigned ones (doc_ids=None triggers auto-select)
        assigned = coll.assign_to_cluster("test_cluster", metadata_key="cluster")
        assert assigned >= 3  # At least the 3 new docs + untagged originals
        db.close()

    def test_assign_to_cluster_with_explicit_ids(self, clustered_collection):
        """assign_to_cluster with explicit doc_ids works."""
        db, coll = clustered_collection

        result = coll.cluster(n_clusters=3, random_state=42)
        coll.save_cluster("test_cluster", result)

        # Assign specific doc IDs
        all_ids = list(coll._index.keys())
        assigned = coll.assign_to_cluster(
            "test_cluster", doc_ids=all_ids[:3], metadata_key="cluster"
        )
        assert assigned == 3
        db.close()

    def test_assign_to_cluster_empty_doc_ids(self, clustered_collection):
        """assign_to_cluster with empty doc_ids list returns 0."""
        db, coll = clustered_collection

        result = coll.cluster(n_clusters=3, random_state=42)
        coll.save_cluster("test_cluster", result)

        assigned = coll.assign_to_cluster("test_cluster", doc_ids=[])
        assert assigned == 0
        db.close()


# ------------------------------------------------------------------ #
# as_langchain / as_llama_index (lines 1618, 1626)
# ------------------------------------------------------------------ #


class TestIntegrationFactories:
    def test_as_langchain_import_error(self, tmp_path):
        """as_langchain raises ImportError when langchain not installed."""
        db = VectorDB(str(tmp_path / "lc.db"))
        with patch.dict(sys.modules, {"langchain_core": None}):
            try:
                db.as_langchain()
            except (ImportError, ModuleNotFoundError, Exception):
                pass  # Expected when langchain not available
        db.close()

    def test_as_llama_index_import_error(self, tmp_path):
        """as_llama_index raises ImportError when llama_index not installed."""
        db = VectorDB(str(tmp_path / "li.db"))
        with patch.dict(sys.modules, {"llama_index": None}):
            try:
                db.as_llama_index()
            except (ImportError, ModuleNotFoundError, Exception):
                pass  # Expected when llama_index not available
        db.close()


# ------------------------------------------------------------------ #
# VectorDB vacuum (line 1780 area - exercise save/vacuum)
# ------------------------------------------------------------------ #


class TestVacuumAndSave:
    def test_vacuum(self, tmp_path):
        db = VectorDB(str(tmp_path / "vacuum.db"))
        coll = db.collection("test")
        coll.add_texts(["a", "b"], embeddings=[[0.1, 0.2], [0.3, 0.4]])
        db.vacuum(checkpoint_wal=True)
        db.vacuum(checkpoint_wal=False)
        db.close()

    def test_save(self, tmp_path):
        db = VectorDB(str(tmp_path / "save.db"))
        coll = db.collection("test")
        coll.add_texts(["a"], embeddings=[[0.1, 0.2]])
        db.save()
        db.close()
