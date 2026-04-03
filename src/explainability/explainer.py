"""Explainability module."""

import logging
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class ClusterExplainer:
    """Explain clustering results."""

    def __init__(self, documents: list, labels: np.ndarray, feature_names: np.ndarray, tfidf_matrix):
        """
        Initialize explainer.

        Args:
            documents: List of documents
            labels: Cluster labels
            feature_names: Feature names from vectorizer
            tfidf_matrix: TF-IDF matrix
        """
        self.documents = documents
        self.labels = labels
        self.feature_names = feature_names
        self.tfidf_matrix = tfidf_matrix
        self.n_clusters = len(np.unique(labels))

    def get_cluster_top_terms(self, cluster_id: int, n_terms: int = 10) -> dict:
        """
        Get top terms for a cluster.

        Args:
            cluster_id: Cluster index
            n_terms: Number of terms to return

        Returns:
            Dictionary with top terms and their scores
        """
        # Get documents in cluster
        cluster_mask = self.labels == cluster_id
        cluster_vectors = self.tfidf_matrix[cluster_mask]
        
        # Calculate mean TF-IDF values
        mean_tfidf = np.asarray(cluster_vectors.mean(axis=0)).flatten()
        
        # Get top term indices
        top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
        
        terms = {}
        for idx in top_indices:
            if idx < len(self.feature_names):
                terms[self.feature_names[idx]] = float(mean_tfidf[idx])
        
        return terms

    def get_cluster_statistics(self, cluster_id: int) -> dict:
        """
        Get statistics for a cluster.

        Args:
            cluster_id: Cluster index

        Returns:
            Dictionary with cluster statistics
        """
        cluster_mask = self.labels == cluster_id
        cluster_docs = [doc for doc, is_in_cluster in zip(self.documents, cluster_mask) if is_in_cluster]
        
        return {
            "cluster_id": cluster_id,
            "n_documents": len(cluster_docs),
            "avg_length": np.mean([len(doc.split()) for doc in cluster_docs]),
            "top_terms": self.get_cluster_top_terms(cluster_id, 10),
        }

    def get_cluster_representative_docs(self, cluster_id: int, n_docs: int = 3) -> list:
        """
        Get representative documents from a cluster.

        Args:
            cluster_id: Cluster index
            n_docs: Number of documents to return

        Returns:
            List of representative documents
        """
        cluster_mask = self.labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return []
        
        cluster_vectors = self.tfidf_matrix[cluster_indices]
        
        # Calculate distance to cluster center (mean of vectors)
        cluster_center = np.asarray(cluster_vectors.mean(axis=0)).flatten()
        distances = np.zeros(len(cluster_indices))
        
        for i, idx in enumerate(cluster_indices):
            vec = np.asarray(cluster_vectors[i].todense()).flatten()
            distances[i] = np.linalg.norm(vec - cluster_center)
        
        # Get indices of closest documents
        closest_indices = np.argsort(distances)[:n_docs]
        
        return [self.documents[cluster_indices[i]] for i in closest_indices]

    def get_all_clusters_summary(self) -> dict:
        """Get summary of all clusters."""
        summary = {}
        for cluster_id in range(self.n_clusters):
            summary[f"cluster_{cluster_id}"] = self.get_cluster_statistics(cluster_id)
        return summary
