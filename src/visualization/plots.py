"""Visualization module."""

import logging
import matplotlib
matplotlib.use('Agg')  # Use file-based backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

from src.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Create visualizations for clustering and topic results."""

    def __init__(self, style: str = "seaborn-v0_8-darkgrid", dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
            dpi: DPI for saved figures
        """
        self.style = style
        self.dpi = dpi
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"Could not use style {style}: {e}")

    def plot_cluster_distribution(self, labels: np.ndarray, figsize: tuple = (10, 6)):
        """
        Plot cluster distribution.

        Args:
            labels: Cluster labels
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(unique, counts, color='steelblue', alpha=0.8)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Documents', fontsize=12)
        ax.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        return fig

    def plot_tsne_clusters(self, tfidf_matrix, labels: np.ndarray, figsize: tuple = (12, 8)):
        """
        Plot t-SNE visualization of clusters.

        Args:
            tfidf_matrix: TF-IDF matrix
            labels: Cluster labels
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        logger.info("Computing t-SNE embedding...")
        
        # Dynamically set perplexity for small datasets
        # perplexity must be < n_samples - 1
        n_samples = tfidf_matrix.shape[0]
        perplexity = min(30, max(2, (n_samples - 1) // 3))
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
        embeddings = tsne.fit_transform(tfidf_matrix.toarray())
        
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', s=50, alpha=0.6)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('t-SNE Cluster Visualization', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        return fig

    def plot_top_terms(self, terms_dict: dict, title: str = "Top Terms", figsize: tuple = (10, 6)):
        """
        Plot top terms for a cluster or topic.

        Args:
            terms_dict: Dictionary of term: score
            title: Plot title
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if not terms_dict:
            logger.warning("No terms to plot")
            return None
        
        terms = list(terms_dict.keys())[:10]
        scores = [terms_dict[t] for t in terms]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(terms, scores, color='steelblue', alpha=0.8)
        ax.set_xlabel('TF-IDF Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        return fig

    def plot_topic_distribution(self, doc_topic_matrix, figsize: tuple = (10, 6)):
        """
        Plot distribution of topics across documents.

        Args:
            doc_topic_matrix: Document-topic matrix from LDA
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        topic_sums = doc_topic_matrix.sum(axis=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(range(len(topic_sums)), topic_sums, color='steelblue', alpha=0.8)
        ax.set_xlabel('Topic ID', fontsize=12)
        ax.set_ylabel('Total Probability', fontsize=12)
        ax.set_title('Topic Distribution Across Documents', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        return fig

    def save_figure(self, fig, filepath: str):
        """
        Save figure to file.

        Args:
            fig: Matplotlib figure
            filepath: Path to save to
        """
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {filepath}")
        plt.close(fig)
