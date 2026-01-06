"""
Clustering analysis for economic regime detection.

Uses HDBSCAN, K-Means, and DBSCAN for unsupervised
regime identification with Observable-compatible output.
"""

from datetime import datetime
from typing import Any, Optional
import json

import numpy as np

from emergentomics.core.models import (
    ClusterAssignment,
    ClusteringResult,
    DimensionalityReduction,
)


class ClusterAnalyzer:
    """
    Performs clustering analysis on economic data.

    Detects economic regimes through unsupervised learning.
    Outputs Observable-compatible JSON for visualization.
    """

    def __init__(
        self,
        method: str = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: int = 3,
        n_clusters: Optional[int] = None,
    ):
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self._model = None
        self._reduction = None

    def fit_predict(
        self,
        data: np.ndarray,
        point_ids: Optional[list[str]] = None,
    ) -> ClusteringResult:
        """
        Fit clustering model and return results.
        """
        if point_ids is None:
            point_ids = [f"point_{i}" for i in range(len(data))]

        # Standardize data
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-10
        scaled = (data - mean) / std

        labels, centroids = self._cluster(scaled)

        # Build assignments
        assignments = []
        for i, (pid, label) in enumerate(zip(point_ids, labels)):
            # Calculate distance to centroid
            dist = None
            if label >= 0 and label < len(centroids):
                dist = float(np.linalg.norm(scaled[i] - centroids[label]))

            assignments.append(
                ClusterAssignment(
                    point_id=pid,
                    cluster_id=int(label),
                    membership_score=1.0 if label >= 0 else 0.0,
                    distance_to_centroid=dist,
                )
            )

        # Calculate cluster sizes
        unique_labels = sorted(set(labels))
        cluster_sizes = [
            sum(1 for l in labels if l == ul)
            for ul in unique_labels
            if ul >= 0
        ]

        # Calculate silhouette score
        silhouette = self._calc_silhouette(scaled, labels)

        return ClusteringResult(
            id=f"cluster_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.utcnow(),
            method=self.method,
            n_clusters=len([l for l in unique_labels if l >= 0]),
            assignments=assignments,
            silhouette_score=silhouette,
            centroids=[c.tolist() for c in centroids] if len(centroids) > 0 else [],
            cluster_sizes=cluster_sizes,
        )

    def _cluster(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform clustering with appropriate method."""
        try:
            if self.method == "hdbscan":
                try:
                    import hdbscan

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                    )
                    labels = clusterer.fit_predict(data)
                    # Calculate centroids
                    unique_labels = sorted(set(labels))
                    centroids = np.array([
                        data[labels == l].mean(axis=0)
                        for l in unique_labels
                        if l >= 0
                    ])
                    return labels, centroids

                except ImportError:
                    # Fall back to DBSCAN
                    self.method = "dbscan"

            if self.method == "dbscan":
                from sklearn.cluster import DBSCAN

                clusterer = DBSCAN(eps=0.5, min_samples=self.min_samples)
                labels = clusterer.fit_predict(data)

            elif self.method == "kmeans":
                from sklearn.cluster import KMeans

                n = self.n_clusters or min(5, len(data) // 10 + 1)
                clusterer = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = clusterer.fit_predict(data)
                return labels, clusterer.cluster_centers_

            else:
                # Default to KMeans
                from sklearn.cluster import KMeans

                n = self.n_clusters or 3
                clusterer = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = clusterer.fit_predict(data)
                return labels, clusterer.cluster_centers_

            # Calculate centroids for non-kmeans methods
            unique_labels = sorted(set(labels))
            centroids = np.array([
                data[labels == l].mean(axis=0)
                for l in unique_labels
                if l >= 0
            ]) if any(l >= 0 for l in labels) else np.array([])

            return labels, centroids

        except ImportError:
            # Fallback: simple threshold-based clustering
            return self._simple_cluster(data)

    def _simple_cluster(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple fallback clustering without sklearn."""
        # Use median split for 2 clusters
        if data.ndim == 1:
            median = np.median(data)
            labels = np.where(data > median, 1, 0)
            centroids = np.array([
                [np.mean(data[labels == 0])],
                [np.mean(data[labels == 1])],
            ])
        else:
            median = np.median(data[:, 0])
            labels = np.where(data[:, 0] > median, 1, 0)
            centroids = np.array([
                data[labels == 0].mean(axis=0),
                data[labels == 1].mean(axis=0),
            ])
        return labels, centroids

    def _calc_silhouette(self, data: np.ndarray, labels: np.ndarray) -> Optional[float]:
        """Calculate silhouette score."""
        try:
            from sklearn.metrics import silhouette_score

            unique_labels = set(labels)
            if len(unique_labels) < 2 or len(unique_labels) >= len(data):
                return None
            # Filter out noise points for silhouette
            mask = labels >= 0
            if sum(mask) < 2:
                return None
            return float(silhouette_score(data[mask], labels[mask]))
        except Exception:
            return None

    def reduce_dimensions(
        self,
        data: np.ndarray,
        point_ids: Optional[list[str]] = None,
        method: str = "pca",
        n_components: int = 2,
    ) -> DimensionalityReduction:
        """
        Reduce dimensionality for visualization.
        """
        if point_ids is None:
            point_ids = [f"point_{i}" for i in range(len(data))]

        original_dims = data.shape[1] if data.ndim > 1 else 1

        try:
            if method == "pca":
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=n_components)
                coords = reducer.fit_transform(data)
                explained = float(sum(reducer.explained_variance_ratio_))

            elif method == "tsne":
                from sklearn.manifold import TSNE

                reducer = TSNE(
                    n_components=n_components,
                    random_state=42,
                    perplexity=min(30, len(data) - 1),
                )
                coords = reducer.fit_transform(data)
                explained = None

            elif method == "umap":
                try:
                    import umap

                    reducer = umap.UMAP(
                        n_components=n_components,
                        random_state=42,
                        n_neighbors=min(15, len(data) - 1),
                    )
                    coords = reducer.fit_transform(data)
                    explained = None
                except ImportError:
                    # Fall back to PCA
                    return self.reduce_dimensions(data, point_ids, "pca", n_components)

            else:
                # Default PCA
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=n_components)
                coords = reducer.fit_transform(data)
                explained = float(sum(reducer.explained_variance_ratio_))

        except ImportError:
            # Simple projection fallback
            coords = data[:, :n_components] if data.ndim > 1 else data.reshape(-1, 1)
            explained = None
            method = "projection"

        return DimensionalityReduction(
            id=f"reduction_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            method=method,
            original_dims=original_dims,
            reduced_dims=n_components,
            coordinates=[c.tolist() for c in coords],
            point_ids=point_ids,
            explained_variance=explained,
        )

    def to_observable(
        self,
        clustering: ClusteringResult,
        reduction: Optional[DimensionalityReduction] = None,
    ) -> dict[str, Any]:
        """
        Export to Observable-compatible format.
        """
        # Build scatter plot data if we have reduction
        scatter_data = []
        if reduction:
            for i, (coord, pid) in enumerate(
                zip(reduction.coordinates, reduction.point_ids)
            ):
                # Find cluster for this point
                cluster_id = -1
                for assignment in clustering.assignments:
                    if assignment.point_id == pid:
                        cluster_id = assignment.cluster_id
                        break

                scatter_data.append({
                    "id": pid,
                    "x": coord[0] if len(coord) > 0 else 0,
                    "y": coord[1] if len(coord) > 1 else 0,
                    "cluster": cluster_id,
                })

        return {
            "metadata": {
                "method": clustering.method,
                "n_clusters": clustering.n_clusters,
                "silhouette_score": clustering.silhouette_score,
                "generated_at": clustering.timestamp.isoformat(),
                "cluster_sizes": clustering.cluster_sizes,
            },
            "assignments": [
                {
                    "point_id": a.point_id,
                    "cluster": a.cluster_id,
                    "membership": a.membership_score,
                    "distance": a.distance_to_centroid,
                }
                for a in clustering.assignments
            ],
            "centroids": clustering.centroids,
            "scatter": scatter_data,
            "reduction": {
                "method": reduction.method if reduction else None,
                "explained_variance": reduction.explained_variance if reduction else None,
            } if reduction else None,
        }

    def to_json(
        self,
        clustering: ClusteringResult,
        reduction: Optional[DimensionalityReduction] = None,
        path: Optional[str] = None,
    ) -> str:
        """Export to JSON for Observable."""
        data = self.to_observable(clustering, reduction)
        json_str = json.dumps(data, indent=2)

        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str


def cluster_gdelt_events(
    events: list[dict],
    features: list[str] = ["tone", "num_mentions", "goldstein_scale"],
) -> dict[str, Any]:
    """
    Convenience function to cluster GDELT events.

    Returns Observable-compatible output.
    """
    if not events:
        return {"metadata": {"error": "no_data"}, "assignments": [], "scatter": []}

    # Extract feature matrix
    data = []
    point_ids = []
    for i, event in enumerate(events):
        row = []
        valid = True
        for feat in features:
            val = event.get(feat)
            if val is None:
                valid = False
                break
            row.append(float(val))
        if valid:
            data.append(row)
            point_ids.append(event.get("id", f"event_{i}"))

    if len(data) < 5:
        return {"metadata": {"error": "insufficient_data"}, "assignments": [], "scatter": []}

    data = np.array(data)

    analyzer = ClusterAnalyzer(method="kmeans", n_clusters=3)
    clustering = analyzer.fit_predict(data, point_ids)
    reduction = analyzer.reduce_dimensions(data, point_ids, method="pca")

    return analyzer.to_observable(clustering, reduction)
