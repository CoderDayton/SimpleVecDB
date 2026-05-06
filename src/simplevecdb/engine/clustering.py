"""Vector clustering engine for SimpleVecDB."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .. import constants
from ..utils import _import_optional

if TYPE_CHECKING:
    from ..types import ClusterResult

_logger = logging.getLogger("simplevecdb.engine.clustering")

ClusterAlgorithm = Literal["kmeans", "minibatch_kmeans", "hdbscan"]


class ClusterEngine:
    """Handles vector clustering and tag generation."""

    def cluster_vectors(
        self,
        vectors: np.ndarray,
        doc_ids: list[int],
        algorithm: ClusterAlgorithm = "minibatch_kmeans",
        n_clusters: int | None = None,
        *,
        min_cluster_size: int = 5,
        random_state: int | None = None,
    ) -> ClusterResult:
        """
        Cluster vectors using the specified algorithm.

        Args:
            vectors: 2D array of shape (n_samples, n_features)
            doc_ids: Document IDs corresponding to each vector
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters (required for kmeans variants)
            min_cluster_size: Minimum cluster size (HDBSCAN only)
            random_state: Random seed for reproducibility

        Returns:
            ClusterResult with labels, centroids, and metadata
        """
        from ..types import ClusterResult

        if len(vectors) == 0:
            return ClusterResult(
                labels=np.array([], dtype=np.int32),
                centroids=None,
                doc_ids=[],
                n_clusters=0,
                algorithm=algorithm,
            )

        if algorithm == "hdbscan":
            labels, centroids, inertia = self._hdbscan(vectors, min_cluster_size)
        elif algorithm == "minibatch_kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters required for minibatch_kmeans")
            labels, centroids, inertia = self._minibatch_kmeans(
                vectors, n_clusters, random_state
            )
        elif algorithm == "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters required for kmeans")
            labels, centroids, inertia = self._kmeans(vectors, n_clusters, random_state)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)

        silhouette = self._compute_silhouette(vectors, labels, n_clusters_found)

        return ClusterResult(
            labels=labels,
            centroids=centroids,
            doc_ids=list(doc_ids),
            n_clusters=n_clusters_found,
            algorithm=algorithm,
            inertia=inertia,
            silhouette_score=silhouette,
        )

    def _kmeans(
        self,
        vectors: np.ndarray,
        n_clusters: int,
        random_state: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Standard K-means clustering."""
        sklearn_cluster = _import_optional("sklearn.cluster")
        if sklearn_cluster is None:
            raise ImportError(
                "scikit-learn required for clustering. Install with: "
                "pip install --force-reinstall simplevecdb"
            )

        kmeans = sklearn_cluster.KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )
        labels = kmeans.fit_predict(vectors)
        return (
            labels.astype(np.int32),
            kmeans.cluster_centers_.astype(np.float32),
            float(kmeans.inertia_),
        )

    def _minibatch_kmeans(
        self,
        vectors: np.ndarray,
        n_clusters: int,
        random_state: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Mini-batch K-means for large datasets."""
        sklearn_cluster = _import_optional("sklearn.cluster")
        if sklearn_cluster is None:
            raise ImportError(
                "scikit-learn required for clustering. Install with: "
                "pip install --force-reinstall simplevecdb"
            )

        batch_size = min(1024, len(vectors))
        kmeans = sklearn_cluster.MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init="auto",
        )
        labels = kmeans.fit_predict(vectors)
        return (
            labels.astype(np.int32),
            kmeans.cluster_centers_.astype(np.float32),
            float(kmeans.inertia_),
        )

    def _hdbscan(
        self,
        vectors: np.ndarray,
        min_cluster_size: int,
    ) -> tuple[np.ndarray, None, None]:
        """HDBSCAN density-based clustering (discovers natural clusters)."""
        hdbscan_mod = _import_optional("hdbscan")
        if hdbscan_mod is None:
            raise ImportError(
                "hdbscan required for density-based clustering. Install with: "
                "pip install --force-reinstall simplevecdb"
            )

        clusterer = hdbscan_mod.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(vectors)
        # HDBSCAN has no centroids or inertia
        return labels.astype(np.int32), None, None

    def _compute_silhouette(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> float | None:
        """Compute silhouette score; returns None if invalid clustering or sklearn unavailable."""
        if n_clusters < 2:
            return None

        sklearn_metrics = _import_optional("sklearn.metrics")
        if sklearn_metrics is None:
            return None

        mask = labels >= 0
        valid_vectors = vectors[mask]
        valid_labels = labels[mask]
        n_valid = len(valid_vectors)
        n_unique = len(set(valid_labels))

        if n_valid < 2 or n_unique < 2 or n_unique >= n_valid:
            return None

        # silhouette_score is O(n²) in time and memory because it computes a
        # full pairwise distance matrix. On collections >100k it OOMs. Cap
        # the sample for evaluation; sklearn does its own random sampling
        # internally when ``sample_size`` is set.
        kwargs: dict[str, Any] = {}
        if n_valid > constants.SILHOUETTE_MAX_SAMPLE:
            kwargs["sample_size"] = constants.SILHOUETTE_MAX_SAMPLE
            kwargs["random_state"] = 0
        return float(
            sklearn_metrics.silhouette_score(valid_vectors, valid_labels, **kwargs)
        )

    def generate_keywords(
        self,
        cluster_texts: dict[int, list[str]],
        n_keywords: int = 5,
    ) -> dict[int, str]:
        """
        Generate keyword tags for clusters using TF-IDF.

        Args:
            cluster_texts: Mapping of cluster_id -> list of document texts
            n_keywords: Number of keywords per cluster

        Returns:
            Mapping of cluster_id -> comma-separated keywords
        """
        sklearn_text = _import_optional("sklearn.feature_extraction.text")
        if sklearn_text is None:
            raise ImportError(
                "scikit-learn required for keyword extraction. Install with: "
                "pip install --force-reinstall simplevecdb"
            )

        tags: dict[int, str] = {}

        for cluster_id, texts in cluster_texts.items():
            if cluster_id == -1:
                tags[cluster_id] = "outliers"
                continue

            if not texts:
                tags[cluster_id] = f"cluster_{cluster_id}"
                continue

            try:
                vectorizer = sklearn_text.TfidfVectorizer(
                    max_features=100,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                )
                tfidf = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()

                scores = np.asarray(tfidf.sum(axis=0)).flatten()
                top_indices = scores.argsort()[-n_keywords:][::-1]
                keywords = [feature_names[i] for i in top_indices]

                tags[cluster_id] = ", ".join(keywords)
            except ValueError:
                tags[cluster_id] = f"cluster_{cluster_id}"

        return tags

    def assign_to_nearest_centroid(
        self,
        vectors: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Assign vectors to nearest centroid (for out-of-sample assignment)."""
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1).astype(np.int32)
