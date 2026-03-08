"""Tests for clustering and auto-tagging functionality."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from typing import Literal

from simplevecdb import VectorDB
from simplevecdb.types import ClusterResult


sklearn = pytest.importorskip("sklearn", reason="sklearn required for clustering tests")


class TestClustering:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_clustering.db"

    @pytest.fixture
    def dim(self) -> int:
        return 32

    def make_clustered_embeddings(
        self, n_per_cluster: int, n_clusters: int, dim: int
    ) -> tuple[list[str], np.ndarray]:
        np.random.seed(42)
        texts = []
        embeddings = []
        for c in range(n_clusters):
            for i in range(n_per_cluster):
                texts.append(f"cluster_{c}_doc_{i}")
                emb = np.random.randn(dim).astype(np.float32)
                emb[c] += 10.0
                embeddings.append(emb)
        return texts, np.array(embeddings)

    def test_cluster_basic(self, db_path: Path, dim: int):
        """Basic clustering returns ClusterResult."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts, embeddings = self.make_clustered_embeddings(5, 3, dim)
        collection.add_texts(texts, embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=3, random_state=42)

        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 3
        assert len(result.labels) == 15
        assert len(result.doc_ids) == 15
        assert result.centroids is not None
        assert result.centroids.shape == (3, dim)
        assert result.algorithm == "minibatch_kmeans"

        db.close()

    def test_cluster_empty_collection(self, db_path: Path):
        """Clustering empty collection returns empty result."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        result = collection.cluster(n_clusters=3)

        assert result.n_clusters == 0
        assert len(result.labels) == 0
        assert len(result.doc_ids) == 0

        db.close()

    def test_cluster_algorithms(self, db_path: Path, dim: int):
        """Test different clustering algorithms."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts, embeddings = self.make_clustered_embeddings(5, 3, dim)
        collection.add_texts(texts, embeddings=embeddings.tolist())

        algorithms: tuple[Literal["kmeans", "minibatch_kmeans"], ...] = (
            "kmeans",
            "minibatch_kmeans",
        )
        for algo in algorithms:
            result = collection.cluster(n_clusters=3, algorithm=algo, random_state=42)
            assert result.n_clusters == 3
            assert result.algorithm == algo

        db.close()

    def test_cluster_with_filter(self, db_path: Path, dim: int):
        """Clustering respects metadata filter."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts, embeddings = self.make_clustered_embeddings(5, 2, dim)
        metadatas = [{"group": "A"}] * 5 + [{"group": "B"}] * 5
        collection.add_texts(texts, embeddings=embeddings.tolist(), metadatas=metadatas)

        result = collection.cluster(
            n_clusters=2, filter={"group": "A"}, random_state=42
        )

        assert len(result.doc_ids) == 5

        db.close()

    def test_cluster_with_sample_size(self, db_path: Path, dim: int):
        """Clustering with sample_size clusters sample and assigns rest."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts, embeddings = self.make_clustered_embeddings(10, 3, dim)
        collection.add_texts(texts, embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=3, sample_size=9, random_state=42)

        assert len(result.doc_ids) == 30
        assert result.n_clusters == 3

        db.close()

    def test_cluster_result_summary(self, db_path: Path, dim: int):
        """ClusterResult.summary() returns cluster counts."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts, embeddings = self.make_clustered_embeddings(5, 3, dim)
        collection.add_texts(texts, embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=3, random_state=42)
        summary = result.summary()

        assert isinstance(summary, dict)
        assert sum(summary.values()) == 15

        db.close()

    def test_cluster_result_get_cluster_doc_ids(self, db_path: Path, dim: int):
        """ClusterResult.get_cluster_doc_ids() returns docs in cluster."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts, embeddings = self.make_clustered_embeddings(5, 3, dim)
        collection.add_texts(texts, embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=3, random_state=42)

        all_from_clusters = []
        for cluster_id in range(3):
            cluster_docs = result.get_cluster_doc_ids(cluster_id)
            all_from_clusters.extend(cluster_docs)

        assert set(all_from_clusters) == set(result.doc_ids)

        db.close()


class TestAutoTag:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_autotag.db"

    def test_auto_tag_keywords(self, db_path: Path):
        """auto_tag generates TF-IDF keywords."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        texts = [
            "machine learning neural network",
            "deep learning artificial intelligence",
            "database sql query optimization",
            "sql index performance tuning",
        ]
        np.random.seed(42)
        embeddings = np.random.randn(4, 32).astype(np.float32)
        embeddings[:2, 0] += 10.0
        embeddings[2:, 1] += 10.0

        collection.add_texts(texts, embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=2, random_state=42)
        tags = collection.auto_tag(result, n_keywords=3)

        assert isinstance(tags, dict)
        assert len(tags) == 2

        db.close()

    def test_auto_tag_custom_callback(self, db_path: Path):
        """auto_tag with custom callback."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(4, 32).astype(np.float32)
        collection.add_texts(["a", "b", "c", "d"], embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=2, random_state=42)

        def custom_tag(texts: list[str]) -> str:
            return f"custom_{len(texts)}"

        tags = collection.auto_tag(result, method="custom", custom_callback=custom_tag)

        for tag in tags.values():
            assert tag.startswith("custom_")

        db.close()


class TestClusterMetadataPersistence:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_cluster_meta.db"

    @pytest.fixture
    def dim(self) -> int:
        return 32

    def test_assign_cluster_metadata(self, db_path: Path, dim: int):
        """assign_cluster_metadata persists cluster IDs."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(6, dim).astype(np.float32)
        embeddings[:3, 0] += 10.0
        embeddings[3:, 1] += 10.0

        collection.add_texts(
            [f"doc_{i}" for i in range(6)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=2, random_state=42)
        updated = collection.assign_cluster_metadata(result)

        assert updated == 6

        db.close()

    def test_assign_cluster_metadata_with_tags(self, db_path: Path, dim: int):
        """assign_cluster_metadata persists tags too."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(4, dim).astype(np.float32)
        collection.add_texts(
            ["ml doc", "ml text", "db doc", "db text"],
            embeddings=embeddings.tolist(),
        )

        result = collection.cluster(n_clusters=2, random_state=42)
        tags = {0: "machine_learning", 1: "database"}
        collection.assign_cluster_metadata(result, tags)

        members = collection.get_cluster_members(0)
        for doc in members:
            assert "cluster" in doc.metadata
            assert "cluster_tag" in doc.metadata

        db.close()

    def test_get_cluster_members(self, db_path: Path, dim: int):
        """get_cluster_members retrieves docs by cluster."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(6, dim).astype(np.float32)
        embeddings[:3, 0] += 10.0
        embeddings[3:, 1] += 10.0

        collection.add_texts(
            [f"doc_{i}" for i in range(6)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=2, random_state=42)
        collection.assign_cluster_metadata(result)

        summary = result.summary()
        for cluster_id, count in summary.items():
            members = collection.get_cluster_members(cluster_id)
            assert len(members) == count

        db.close()

    def test_custom_metadata_keys(self, db_path: Path, dim: int):
        """Custom metadata keys work."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(4, dim).astype(np.float32)
        collection.add_texts(["a", "b", "c", "d"], embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=2, random_state=42)
        tags = {0: "first", 1: "second"}
        collection.assign_cluster_metadata(
            result, tags, metadata_key="my_cluster", tag_key="my_tag"
        )

        members = collection.get_cluster_members(0, metadata_key="my_cluster")
        for doc in members:
            assert "my_cluster" in doc.metadata
            assert "my_tag" in doc.metadata

        db.close()


class TestClusteringEdgeCases:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_cluster_edge.db"

    def test_cluster_requires_n_clusters_for_kmeans(self, db_path: Path):
        """kmeans algorithms require n_clusters."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(5, 32).astype(np.float32)
        collection.add_texts(["a", "b", "c", "d", "e"], embeddings=embeddings.tolist())

        with pytest.raises(ValueError, match="n_clusters required"):
            collection.cluster(algorithm="kmeans")

        db.close()

    def test_cluster_single_document(self, db_path: Path):
        """Clustering single doc works."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        collection.add_texts(["only"], embeddings=[np.random.randn(32).tolist()])

        result = collection.cluster(n_clusters=1, random_state=42)

        assert result.n_clusters == 1
        assert len(result.doc_ids) == 1

        db.close()

    def test_more_clusters_than_docs(self, db_path: Path):
        """Requesting more clusters than docs gives fewer clusters."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(3, 32).astype(np.float32)
        collection.add_texts(["a", "b", "c"], embeddings=embeddings.tolist())

        result = collection.cluster(n_clusters=10, random_state=42)

        assert result.n_clusters <= 3

        db.close()


class TestClusterMetrics:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_cluster_metrics.db"

    @pytest.fixture
    def dim(self) -> int:
        return 32

    def test_cluster_result_has_inertia(self, db_path: Path, dim: int):
        """K-means clustering populates inertia metric."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(20, dim).astype(np.float32)
        collection.add_texts(
            [f"doc_{i}" for i in range(20)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=3, algorithm="kmeans", random_state=42)

        assert result.inertia is not None
        assert result.inertia > 0

        db.close()

    def test_cluster_result_has_silhouette(self, db_path: Path, dim: int):
        """Clustering with enough samples populates silhouette score."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(20, dim).astype(np.float32)
        embeddings[:10, 0] += 10.0
        embeddings[10:, 1] += 10.0

        collection.add_texts(
            [f"doc_{i}" for i in range(20)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=2, random_state=42)

        assert result.silhouette_score is not None
        assert -1 <= result.silhouette_score <= 1

        db.close()

    def test_cluster_result_metrics_method(self, db_path: Path, dim: int):
        """ClusterResult.metrics() returns dict with available metrics."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(20, dim).astype(np.float32)
        collection.add_texts(
            [f"doc_{i}" for i in range(20)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=3, random_state=42)
        metrics = result.metrics()

        assert isinstance(metrics, dict)
        assert "inertia" in metrics
        assert "silhouette_score" in metrics
        assert metrics["inertia"] is not None
        assert metrics["inertia"] > 0

        db.close()

    def test_silhouette_none_for_single_cluster(self, db_path: Path, dim: int):
        """Silhouette is None when only one cluster."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(5, dim).astype(np.float32)
        collection.add_texts(
            [f"doc_{i}" for i in range(5)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=1, random_state=42)

        assert result.silhouette_score is None

        db.close()


class TestClusterPersistence:
    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "test_cluster_persist.db"

    @pytest.fixture
    def dim(self) -> int:
        return 32

    def test_save_and_load_cluster(self, db_path: Path, dim: int):
        """save_cluster and load_cluster round-trip correctly."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(10, dim).astype(np.float32)
        collection.add_texts(
            [f"doc_{i}" for i in range(10)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=3, random_state=42)
        collection.save_cluster("test_cluster", result, metadata={"version": 1})

        loaded = collection.load_cluster("test_cluster")

        assert loaded is not None
        loaded_result, loaded_meta = loaded
        assert loaded_result.n_clusters == 3
        assert loaded_result.algorithm == result.algorithm
        assert loaded_result.centroids is not None
        assert result.centroids is not None
        assert loaded_result.centroids.shape == result.centroids.shape
        assert loaded_meta == {"version": 1}

        db.close()

    def test_list_clusters(self, db_path: Path, dim: int):
        """list_clusters returns saved cluster info."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(10, dim).astype(np.float32)
        collection.add_texts(
            [f"doc_{i}" for i in range(10)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=2, random_state=42)
        collection.save_cluster("cluster_a", result)
        collection.save_cluster("cluster_b", result, metadata={"tag": "test"})

        clusters = collection.list_clusters()

        assert len(clusters) == 2
        names = {c["name"] for c in clusters}
        assert names == {"cluster_a", "cluster_b"}

        db.close()

    def test_delete_cluster(self, db_path: Path, dim: int):
        """delete_cluster removes saved cluster."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(10, dim).astype(np.float32)
        collection.add_texts(
            [f"doc_{i}" for i in range(10)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=2, random_state=42)
        collection.save_cluster("to_delete", result)

        assert collection.load_cluster("to_delete") is not None

        deleted = collection.delete_cluster("to_delete")

        assert deleted is True
        assert collection.load_cluster("to_delete") is None

        db.close()

    def test_load_nonexistent_cluster_returns_none(self, db_path: Path):
        """load_cluster returns None for unknown cluster."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        assert collection.load_cluster("nonexistent") is None

        db.close()

    def test_assign_to_cluster(self, db_path: Path, dim: int):
        """assign_to_cluster assigns docs using saved centroids."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        np.random.seed(42)
        embeddings = np.random.randn(10, dim).astype(np.float32)
        embeddings[:5, 0] += 10.0
        embeddings[5:, 1] += 10.0

        collection.add_texts(
            [f"doc_{i}" for i in range(10)], embeddings=embeddings.tolist()
        )

        result = collection.cluster(n_clusters=2, random_state=42)
        collection.save_cluster("saved", result)

        new_embs = np.random.randn(4, dim).astype(np.float32)
        new_embs[:2, 0] += 10.0
        new_embs[2:, 1] += 10.0
        new_ids = collection.add_texts(
            ["new_a", "new_b", "new_c", "new_d"], embeddings=new_embs.tolist()
        )

        assigned = collection.assign_to_cluster("saved", new_ids)

        assert assigned == 4

        cluster_0 = collection.get_cluster_members(0)
        cluster_1 = collection.get_cluster_members(1)
        all_assigned = len(cluster_0) + len(cluster_1)
        assert all_assigned >= 4

        db.close()

    def test_assign_to_cluster_raises_for_unknown(self, db_path: Path):
        """assign_to_cluster raises ValueError for unknown cluster."""
        db = VectorDB(db_path)
        collection = db.collection("test")

        with pytest.raises(ValueError, match="not found"):
            collection.assign_to_cluster("nonexistent", [1, 2, 3])

        db.close()


class TestClusterEngineDirectly:
    """Tests targeting ClusterEngine methods directly for coverage."""

    def test_cluster_vectors_empty(self):
        """Empty vectors returns empty ClusterResult."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        result = engine.cluster_vectors(
            np.array([], dtype=np.float32).reshape(0, 4),
            doc_ids=[],
        )
        assert result.n_clusters == 0
        assert len(result.labels) == 0
        assert result.doc_ids == []

    def test_cluster_vectors_unknown_algorithm(self):
        """Unknown algorithm raises ValueError."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        vectors = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            engine.cluster_vectors(vectors, list(range(10)), algorithm="bogus")  # type: ignore[arg-type]

    def test_cluster_vectors_minibatch_requires_n_clusters(self):
        """minibatch_kmeans without n_clusters raises ValueError."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        vectors = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ValueError, match="n_clusters required"):
            engine.cluster_vectors(
                vectors, list(range(10)), algorithm="minibatch_kmeans", n_clusters=None
            )

    def test_kmeans_missing_sklearn(self, monkeypatch):
        """_kmeans raises ImportError when sklearn is unavailable."""
        from simplevecdb.engine import clustering

        monkeypatch.setattr(clustering, "_import_optional", lambda name: None)
        engine = clustering.ClusterEngine()
        vectors = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ImportError, match="scikit-learn required"):
            engine._kmeans(vectors, 2, None)

    def test_minibatch_kmeans_missing_sklearn(self, monkeypatch):
        """_minibatch_kmeans raises ImportError when sklearn is unavailable."""
        from simplevecdb.engine import clustering

        monkeypatch.setattr(clustering, "_import_optional", lambda name: None)
        engine = clustering.ClusterEngine()
        vectors = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ImportError, match="scikit-learn required"):
            engine._minibatch_kmeans(vectors, 2, None)

    def test_hdbscan_missing(self, monkeypatch):
        """_hdbscan raises ImportError when hdbscan is unavailable."""
        from simplevecdb.engine import clustering

        monkeypatch.setattr(clustering, "_import_optional", lambda name: None)
        engine = clustering.ClusterEngine()
        vectors = np.random.randn(10, 4).astype(np.float32)
        with pytest.raises(ImportError, match="hdbscan required"):
            engine._hdbscan(vectors, 5)

    def test_generate_keywords_outlier_cluster(self):
        """Cluster -1 is tagged as 'outliers'."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        tags = engine.generate_keywords({-1: ["some text"], 0: ["hello world"] * 3})
        assert tags[-1] == "outliers"
        assert 0 in tags

    def test_generate_keywords_empty_texts(self):
        """Empty text list falls back to 'cluster_N'."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        tags = engine.generate_keywords({0: [], 1: ["hello world"] * 3})
        assert tags[0] == "cluster_0"

    def test_generate_keywords_missing_sklearn(self, monkeypatch):
        """generate_keywords raises ImportError when sklearn is unavailable."""
        from simplevecdb.engine import clustering

        monkeypatch.setattr(clustering, "_import_optional", lambda name: None)
        engine = clustering.ClusterEngine()
        with pytest.raises(ImportError, match="scikit-learn required"):
            engine.generate_keywords({0: ["hello world"]})

    def test_generate_keywords_value_error_fallback(self):
        """TF-IDF ValueError falls back to 'cluster_N'."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        # Single empty-ish doc that TF-IDF can't process
        tags = engine.generate_keywords({0: [""]})
        assert tags[0] == "cluster_0"

    def test_silhouette_single_cluster(self):
        """Silhouette returns None for < 2 clusters."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        vectors = np.random.randn(10, 4).astype(np.float32)
        labels = np.zeros(10, dtype=np.int32)
        assert engine._compute_silhouette(vectors, labels, 1) is None

    def test_assign_to_nearest_centroid(self):
        """Vectors are assigned to the nearest centroid."""
        from simplevecdb.engine.clustering import ClusterEngine

        engine = ClusterEngine()
        centroids = np.array([[0, 0], [10, 10]], dtype=np.float32)
        vectors = np.array([[1, 1], [9, 9], [0.5, 0.5]], dtype=np.float32)
        labels = engine.assign_to_nearest_centroid(vectors, centroids)
        assert list(labels) == [0, 1, 0]
