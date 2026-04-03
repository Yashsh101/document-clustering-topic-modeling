"""Evaluation script - Generate evaluation report."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.utils.nltk_utils import ensure_nltk_resources

logger = get_logger(__name__)

# Ensure NLTK resources are available
try:
    ensure_nltk_resources()
except Exception as e:
    logger.warning(f"NLTK resource initialization warning: {e}")


def load_metrics(artifact_dir: str) -> dict:
    """Load metrics from artifacts."""
    metrics_file = Path(artifact_dir) / "reports" / "metrics.json"

    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        return {}

    with open(metrics_file) as f:
        metrics = json.load(f)

    return metrics


def generate_report(artifact_dir: str, output_file: str = None):
    """Generate evaluation report."""
    metrics = load_metrics(artifact_dir)

    if not metrics:
        logger.error("No metrics found")
        return

    report = f"""
===========================================
CLUSTERING AND TOPIC MODELING EVALUATION REPORT
===========================================

CLUSTERING METRICS:
- Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}
  (Range: -1 to 1, Higher is better)
  Interpretation: {_interpret_silhouette(metrics.get('silhouette_score', 0))}

- Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 'N/A'):.4f}
  (Lower is better, typically 0-1)
  Interpretation: {_interpret_davies_bouldin(metrics.get('davies_bouldin_score', 1))}

- Inertia: {metrics.get('inertia', 'N/A'):.4f}
  (Within-cluster sum of squares, lower is better)

CLUSTER SUMMARY:
- Number of Clusters: {metrics.get('n_clusters', 'N/A')}
- Cluster Distribution: {metrics.get('cluster_distribution', {})}

INTERPRETATION:
These metrics measure the quality and compactness of the clusters:
- Higher silhouette scores indicate well-separated clusters
- Lower Davies-Bouldin scores indicate distinct cluster separation
- Even cluster distribution suggests balanced partitioning

===========================================
"""

    logger.info(report)

    if output_file:
        with open(output_file, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_file}")


def _interpret_silhouette(score: float) -> str:
    """Interpret silhouette score."""
    if score > 0.7:
        return "Strong separation (Excellent)"
    elif score > 0.5:
        return "Reasonable separation (Good)"
    elif score > 0.25:
        return "Weak structure (Fair)"
    else:
        return "No clear structure (Poor)"


def _interpret_davies_bouldin(score: float) -> str:
    """Interpret Davies-Bouldin score."""
    if score < 0.5:
        return "Excellent separation"
    elif score < 1.0:
        return "Good separation"
    elif score < 2.0:
        return "Fair separation"
    else:
        return "Poor separation"


def main():
    """Generate evaluation report."""
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--artifact-dir", type=str, default="artifacts", help="Artifact directory")
    parser.add_argument("--output", type=str, help="Output file for report")

    args = parser.parse_args()

    generate_report(args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
