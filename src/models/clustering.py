"""K-Means clustering model."""

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import ClusteringConfig
from src.logger import get_logger

logger = get_logger(__name__)


class DocumentClusterer:
    """K-Means clustering for documents."""

    def __init__(self, config: ClusteringConfig = None):
        """
        Initialize clusterer.

        Args:
            config: ClusteringConfig instance
        """
        self.config = config or ClusteringConfig()
        self.model = None
        self.labels = None
        self.is_fitted = False

    def fit(self, tfidf_matrix) -> "DocumentClusterer":
        """
        Fit clustering model.

        Args:
            tfidf_matrix: TF-IDF matrix from vectorizer

        Returns:
            Self
        """
        # Cap n_clusters to number of samples
        n_samples = tfidf_matrix.shape[0]
        n_clusters = min(self.config.n_clusters, max(1, n_samples))
        
        logger.info(f"Fitting KMeans with {n_clusters} clusters (had {self.config.n_clusters}, {n_samples} samples)")
        
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            n_init=self.config.n_init,
        )
        self.labels = self.model.fit_predict(tfidf_matrix)
        self.is_fitted = True
        
        logger.info(f"KMeans fitted. Cluster distribution: {np.bincount(self.labels)}")
        return self

    def predict(self, tfidf_matrix):
        """Predict cluster labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.predict(tfidf_matrix)

    def get_cluster_centers(self):
        """Get cluster centers."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.cluster_centers_

    def get_silhouette_score(self, tfidf_matrix) -> float:
        """Calculate silhouette score."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return silhouette_score(tfidf_matrix, self.labels)

    def get_inertia(self) -> float:
        """Get inertia."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.inertia_

    def find_optimal_clusters(self, tfidf_matrix, k_range: tuple = (2, 10)) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            tfidf_matrix: TF-IDF matrix
            k_range: Range of k values to test

        Returns:
            Optimal number of clusters
        """
        logger.info(f"Finding optimal clusters in range {k_range}")
        
        n_samples = tfidf_matrix.shape[0]
        # Cap k_range to n_samples
        max_k = min(k_range[1], n_samples)
        min_k = min(k_range[0], max_k)
        
        best_k = min_k
        best_score = -1
        scores = {}

        for k in range(min_k, max_k + 1):
            if k < 1:
                continue
            kmeans = KMeans(
                n_clusters=k,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
            )
            labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, labels)
            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k

            logger.debug(f"k={k}: silhouette_score={score:.4f}")

        logger.info(f"Optimal clusters: {best_k} with score {best_score:.4f}")
        return best_k
