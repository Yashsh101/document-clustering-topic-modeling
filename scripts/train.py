"""Training script - Load data, train models, save artifacts."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.pipeline.orchestrator import Pipeline
from src.logger import get_logger

logger = get_logger(__name__, log_file="artifacts/logs/train.log")


def main():
    """Run training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train document clustering and topic modeling pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sample",
        help="Directory containing documents",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Directory to save artifacts",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=5,
        help="Number of topics",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=500,
        help="Maximum TF-IDF features",
    )

    args = parser.parse_args()

    # Create config
    config = get_config()
    config.clustering.n_clusters = args.n_clusters
    config.topic.n_topics = args.n_topics
    config.vectorizer.max_features = args.max_features

    # Run pipeline
    logger.info("Starting training pipeline")
    pipeline = Pipeline(config=config, artifact_dir=args.artifact_dir)
    pipeline.run(args.data_dir, save_artifacts=True)

    # Print results
    results = pipeline.get_results_summary()
    logger.info("Training Results:")
    logger.info(f"  Documents: {results['n_documents']}")
    logger.info(f"  Clusters: {results['n_clusters']}")
    logger.info(f"  Topics: {results['n_topics']}")
    logger.info(f"  Silhouette Score: {results['metrics'].get('silhouette_score', 'N/A'):.4f}")
    logger.info(f"  Davies-Bouldin Score: {results['metrics'].get('davies_bouldin_score', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
