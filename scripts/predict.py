"""Prediction script - Infer clusters and topics for new documents."""

import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.schema import PredictionResult
from src.utils.nltk_utils import ensure_nltk_resources

logger = get_logger(__name__)

# Ensure NLTK resources are available
try:
    ensure_nltk_resources()
except Exception as e:
    logger.warning(f"NLTK resource initialization warning: {e}")


def load_models(artifact_dir: str):
    """Load trained models from artifacts."""
    models_dir = Path(artifact_dir) / "models"

    with open(models_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(models_dir / "clusterer.pkl", "rb") as f:
        clusterer = pickle.load(f)

    with open(models_dir / "topic_modeler.pkl", "rb") as f:
        topic_modeler = pickle.load(f)

    return vectorizer, clusterer, topic_modeler


def predict(
    texts: list,
    vectorizer,
    clusterer,
    topic_modeler,
) -> list:
    """Predict clusters and topics for new texts."""
    # Vectorize
    tfidf_matrix = vectorizer.transform(texts)

    # Predict clusters
    cluster_ids = clusterer.predict(tfidf_matrix)

    # Transform to topic space
    from sklearn.feature_extraction.text import CountVectorizer
    count_matrix = topic_modeler.count_vectorizer.transform(texts)
    doc_topics = topic_modeler.transform(count_matrix)

    # Get feature names for keywords
    try:
        feature_names = vectorizer.get_feature_names_out()
    except (AttributeError, TypeError):
        feature_names = np.array([f"feature_{i}" for i in range(tfidf_matrix.shape[1])])

    results = []
    for i, text in enumerate(texts):
        cluster_id = int(cluster_ids[i])
        doc_topic_dist = doc_topics[i]
        top_topic = int(doc_topic_dist.argmax())
        top_topic_prob = float(doc_topic_dist[top_topic])

        # Extract top keywords from cluster center
        try:
            cluster_center = clusterer.model.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-5:][::-1]
            keywords = [str(feature_names[idx]) for idx in top_indices if idx < len(feature_names)]
        except (AttributeError, IndexError):
            keywords = []

        # Calculate similarity to cluster center
        try:
            doc_vector = tfidf_matrix[i].toarray().flatten()
            cluster_center = clusterer.model.cluster_centers_[cluster_id]
            similarity = float(1 / (1 + np.linalg.norm(doc_vector - cluster_center)))
        except (AttributeError, IndexError):
            similarity = None

        # Create prediction result using unified schema
        result = PredictionResult(
            cluster=cluster_id,
            topic=top_topic,
            topic_probability=top_topic_prob,
            keywords=keywords,
            topic_distribution={f"topic_{j}": float(p) for j, p in enumerate(doc_topic_dist)},
            cluster_similarity=similarity,
        )

        results.append(result)

    return results


def main():
    """Run prediction on new documents."""
    parser = argparse.ArgumentParser(description="Predict clusters and topics for new documents")
    parser.add_argument("--text", type=str, help="Single document text")
    parser.add_argument("--file", type=str, help="File containing documents (one per line)")
    parser.add_argument("--artifact-dir", type=str, default="artifacts", help="Artifact directory")
    parser.add_argument("--output", type=str, help="Output file for predictions")

    args = parser.parse_args()

    if not args.text and not args.file:
        logger.error("Please provide --text or --file")
        sys.exit(1)

    # Load models
    logger.info("Loading trained models...")
    vectorizer, clusterer, topic_modeler = load_models(args.artifact_dir)

    # Get texts
    if args.text:
        texts = [args.text]
    else:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]

    # Make predictions
    logger.info(f"Making predictions for {len(texts)} documents...")
    results = predict(texts, vectorizer, clusterer, topic_modeler)

    # Print results
    for i, result in enumerate(results):
        logger.info(f"Doc {i+1}: Cluster {result.cluster}, Topic {result.topic} "
                   f"(prob: {result.topic_probability:.4f}), Keywords: {', '.join(result.keywords)}")

    # Save results as JSON
    if args.output:
        with open(args.output, "w") as f:
            # Convert PredictionResult objects to dict for JSON serialization
            json_results = [r.to_dict() for r in results]
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
