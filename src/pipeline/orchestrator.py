"""Pipeline orchestrator module."""

import logging
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from src.config import AppConfig, get_config
from src.data.loader import DocumentLoader
from src.preprocessing.text_processor import TextProcessor
from src.features.vectorizer import TFIDFVectorizer
from src.models.clustering import DocumentClusterer
from src.models.topic_modeling import TopicModeler
from src.evaluation.metrics import EvaluationMetrics
from src.explainability.explainer import ClusterExplainer
from src.visualization.plots import Visualizer
from src.logger import get_logger

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end pipeline for document clustering and topic modeling."""

    def __init__(self, config: Optional[AppConfig] = None, artifact_dir: str = "artifacts"):
        """
        Initialize pipeline.

        Args:
            config: AppConfig instance
            artifact_dir: Directory for artifacts
        """
        self.config = config or get_config()
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.loader = None
        self.preprocessor = None
        self.vectorizer = None
        self.clusterer = None
        self.topic_modeler = None
        self.visualizer = None

        # Data
        self.raw_documents = []
        self.processed_documents = []
        self.tfidf_matrix = None
        self.cluster_labels = None
        self.doc_topic_matrix = None
        self.metrics = {}

        logger.info("Pipeline initialized")

    def run(self, data_dir: str, save_artifacts: bool = True):
        """
        Run the complete pipeline.

        Args:
            data_dir: Directory containing documents
            save_artifacts: Whether to save artifacts

        Returns:
            Self
        """
        logger.info("=" * 50)
        logger.info("STARTING PIPELINE EXECUTION")
        logger.info("=" * 50)

        # Step 1: Load documents
        self.load_documents(data_dir)

        if len(self.raw_documents) == 0:
            logger.error("No documents loaded. Exiting.")
            return self

        # Step 2: Preprocess
        self.preprocess()

        # Step 3: Vectorize
        self.vectorize()

        # Step 4: Cluster
        self.cluster()

        # Step 5: Topic modeling
        self.extract_topics()

        # Step 6: Evaluate
        self.evaluate()

        # Step 7: Visualize
        self.visualize()

        if save_artifacts:
            self.save_artifacts()

        logger.info("=" * 50)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 50)

        return self

    def load_documents(self, data_dir: str):
        """Load documents from directory."""
        logger.info(f"Step 1: Loading documents from {data_dir}")
        self.loader = DocumentLoader()
        self.raw_documents, _ = self.loader.load_from_directory(data_dir)
        logger.info(f"Loaded {len(self.raw_documents)} documents")
        return self

    def preprocess(self):
        """Preprocess documents."""
        logger.info("Step 2: Preprocessing documents")
        self.preprocessor = TextProcessor(config=self.config.preprocessing)
        self.processed_documents = self.preprocessor.batch_process(self.raw_documents)
        logger.info(f"Preprocessed {len(self.processed_documents)} documents")
        return self

    def vectorize(self):
        """Vectorize documents."""
        logger.info("Step 3: Vectorizing documents")
        self.vectorizer = TFIDFVectorizer(self.config.vectorizer)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_documents)
        logger.info(f"Created TF-IDF matrix with shape {self.tfidf_matrix.shape}")
        return self

    def cluster(self):
        """Cluster documents."""
        logger.info("Step 4: Clustering documents")
        self.clusterer = DocumentClusterer(self.config.clustering)
        self.clusterer.fit(self.tfidf_matrix)
        self.cluster_labels = self.clusterer.labels
        logger.info(f"Clustering complete")
        return self

    def extract_topics(self):
        """Extract topics using LDA."""
        logger.info("Step 5: Extracting topics with LDA")
        self.topic_modeler = TopicModeler(self.config.topic)
        self.topic_modeler.fit(self.processed_documents)
        self.doc_topic_matrix = self.topic_modeler.doc_topic_matrix
        logger.info(f"Extracted {self.config.topic.n_topics} topics")
        return self

    def evaluate(self):
        """Evaluate clustering results."""
        logger.info("Step 6: Evaluating results")
        self.metrics = EvaluationMetrics.calculate_all_metrics(
            self.tfidf_matrix,
            self.cluster_labels,
            self.clusterer.model
        )
        logger.info(f"Metrics: {self.metrics}")
        return self

    def visualize(self):
        """Create visualizations."""
        logger.info("Step 7: Creating visualizations")
        self.visualizer = Visualizer()
        
        # Create plots
        plots_dir = self.artifact_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig1 = self.visualizer.plot_cluster_distribution(self.cluster_labels)
        self.visualizer.save_figure(fig1, str(plots_dir / "cluster_distribution.png"))

        fig2 = self.visualizer.plot_tsne_clusters(self.tfidf_matrix, self.cluster_labels)
        self.visualizer.save_figure(fig2, str(plots_dir / "tsne_clusters.png"))

        for cluster_id in range(len(set(self.cluster_labels))):
            explainer = ClusterExplainer(
                self.raw_documents,
                self.cluster_labels,
                self.vectorizer.vectorizer.get_feature_names_out(),
                self.tfidf_matrix
            )
            terms = explainer.get_cluster_top_terms(cluster_id)
            fig = self.visualizer.plot_top_terms(terms, f"Cluster {cluster_id} Top Terms")
            if fig:
                self.visualizer.save_figure(fig, str(plots_dir / f"cluster_{cluster_id}_terms.png"))

        logger.info(f"Saved visualizations to {plots_dir}")
        return self

    def save_artifacts(self):
        """Save trained models and results."""
        logger.info("Saving artifacts")
        
        models_dir = self.artifact_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        with open(models_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(models_dir / "clusterer.pkl", "wb") as f:
            pickle.dump(self.clusterer, f)

        with open(models_dir / "topic_modeler.pkl", "wb") as f:
            pickle.dump(self.topic_modeler, f)

        # Save config
        with open(models_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)

        # Save metrics (convert numpy types to native Python types)
        reports_dir = self.artifact_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to JSON-serializable format
        serializable_metrics = self._convert_to_serializable(self.metrics)
        
        with open(reports_dir / "metrics.json", "w") as f:
            json.dump(serializable_metrics, f, indent=4)

        logger.info(f"Artifacts saved to {self.artifact_dir}")
        return self

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy and other non-serializable types to native Python types."""
        import numpy as np
        
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): Pipeline._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [Pipeline._convert_to_serializable(item) for item in obj]
        return obj

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            "n_documents": len(self.raw_documents),
            "n_clusters": self.config.clustering.n_clusters,
            "n_topics": self.config.topic.n_topics,
            "metrics": self.metrics,
            "topics": self.topic_modeler.get_all_topics_summary() if self.topic_modeler else {},
        }
