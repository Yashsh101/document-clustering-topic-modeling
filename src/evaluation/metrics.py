"""Evaluation metrics module."""

import logging
import numpy as np
from sklearn.metrics import davies_bouldin_score as db_score
from sklearn.metrics import silhouette_score as ss_score

from src.logger import get_logger

logger = get_logger(__name__)


class EvaluationMetrics:
    """Calculate evaluation metrics for clustering and topic modeling."""

    @staticmethod
    def silhouette_score(tfidf_matrix, labels: np.ndarray) -> float:
        """
        Calculate silhouette score.

        Args:
            tfidf_matrix: TF-IDF matrix
            labels: Cluster labels

        Returns:
            Silhouette score
        """
        score = ss_score(tfidf_matrix, labels)
        logger.info(f"Silhouette score: {score:.4f}")
        return score

    @staticmethod
    def davies_bouldin_score(tfidf_matrix, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin score (lower is better).

        Args:
            tfidf_matrix: TF-IDF matrix
            labels: Cluster labels

        Returns:
            Davies-Bouldin score
        """
        # Convert sparse matrix to dense if needed
        if hasattr(tfidf_matrix, 'toarray'):
            tfidf_matrix = tfidf_matrix.toarray()
        
        score = db_score(tfidf_matrix, labels)
        logger.info(f"Davies-Bouldin score: {score:.4f}")
        return score

    @staticmethod
    def inertia(kmeans_model) -> float:
        """
        Get inertia from KMeans model.

        Args:
            kmeans_model: Fitted KMeans model

        Returns:
            Inertia
        """
        inertia = kmeans_model.inertia_
        logger.info(f"Inertia: {inertia:.4f}")
        return inertia

    @staticmethod
    def calculate_all_metrics(tfidf_matrix, labels: np.ndarray, kmeans_model) -> dict:
        """
        Calculate all clustering metrics.

        Args:
            tfidf_matrix: TF-IDF matrix
            labels: Cluster labels
            kmeans_model: Fitted KMeans model

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "silhouette_score": EvaluationMetrics.silhouette_score(tfidf_matrix, labels),
            "davies_bouldin_score": EvaluationMetrics.davies_bouldin_score(tfidf_matrix, labels),
            "inertia": EvaluationMetrics.inertia(kmeans_model),
            "n_clusters": len(np.unique(labels)),
            "cluster_distribution": dict(zip(*np.unique(labels, return_counts=True))),
        }
        return metrics
