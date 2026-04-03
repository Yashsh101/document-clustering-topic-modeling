#!/usr/bin/env python3
"""Professional Streamlit app for document clustering and topic modeling."""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger
from src.schema import PredictionResult

logger = get_logger(__name__)


def ensure_models_exist(artifact_dir: str = "artifacts") -> bool:
    """
    Ensure models are trained. If not, train them automatically.
    
    Args:
        artifact_dir: Path to artifacts directory
        
    Returns:
        True if models exist or were successfully trained
    """
    models_dir = Path(artifact_dir) / "models"
    required_files = ["vectorizer.pkl", "clusterer.pkl", "topic_modeler.pkl", "config.json"]
    
    # Check if all required files exist
    files_exist = all((models_dir / f).exists() for f in required_files)
    
    if files_exist:
        return True
    
    # Models don't exist - train them
    st.warning("⏳ **Initializing models for first run...**")
    st.info("Training the document clustering and topic modeling pipeline. This may take 30-60 seconds on first deployment...")
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Show status
        with status_placeholder.container():
            st.write("🔧 Training models on sample data...")
        
        # Run training script
        train_script = Path(__file__).parent.parent / "scripts" / "train.py"
        result = subprocess.run(
            [sys.executable, str(train_script), "--data-dir", "data/sample"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True
        )
        
        with status_placeholder.container():
            st.success("✅ Models trained successfully!")
        
        # Verify files exist
        if not all((models_dir / f).exists() for f in required_files):
            raise FileNotFoundError("Models were trained but files not found")
        
        return True
        
    except subprocess.TimeoutExpired:
        st.error("❌ Model training timed out. Please try refreshing the page.")
        logger.error("Training timed out")
        return False
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Model training failed: {e.stderr}")
        logger.error(f"Training failed: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"❌ Failed to ensure models: {e}")
        logger.error(f"Model ensure failed: {e}")
        return False


@st.cache_resource
def load_models(artifact_dir: str = "artifacts"):
    """Load trained models from artifacts (cached)."""
    models_dir = Path(artifact_dir) / "models"

    try:
        with open(models_dir / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open(models_dir / "clusterer.pkl", "rb") as f:
            clusterer = pickle.load(f)

        with open(models_dir / "topic_modeler.pkl", "rb") as f:
            topic_modeler = pickle.load(f)

        with open(models_dir / "config.json") as f:
            config = json.load(f)

        return vectorizer, clusterer, topic_modeler, config
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def make_prediction(text: str, vectorizer, clusterer, topic_modeler) -> PredictionResult:
    """Make predictions for a single document."""
    # Vectorize
    tfidf_matrix = vectorizer.transform([text])

    # Predict cluster
    cluster_id = int(clusterer.predict(tfidf_matrix)[0])

    # Transform to topic space
    count_matrix = topic_modeler.count_vectorizer.transform([text])
    doc_topics = topic_modeler.transform(count_matrix)[0]

    # Get top topic
    top_topic = int(doc_topics.argmax())
    top_topic_prob = float(doc_topics[top_topic])

    # Extract keywords from cluster center
    try:
        feature_names = vectorizer.get_feature_names_out()
        cluster_center = clusterer.model.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-5:][::-1]
        keywords = [str(feature_names[idx]) for idx in top_indices if idx < len(feature_names)]
    except (AttributeError, IndexError):
        keywords = []

    # Calculate similarity to cluster center
    try:
        doc_vector = tfidf_matrix.toarray()[0]
        cluster_center = clusterer.model.cluster_centers_[cluster_id]
        cluster_similarity = float(1 / (1 + np.linalg.norm(doc_vector - cluster_center)))
    except (AttributeError, IndexError):
        cluster_similarity = None

    return PredictionResult(
        cluster=cluster_id,
        topic=top_topic,
        topic_probability=top_topic_prob,
        keywords=keywords,
        topic_distribution={f"topic_{i}": float(p) for i, p in enumerate(doc_topics)},
        cluster_similarity=cluster_similarity,
    )


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Document Clustering & Topic Modeling",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Ensure models are trained (auto-train on first run)
    if not ensure_models_exist():
        st.error("❌ Failed to initialize models. Please check the logs.")
        return

    # Load models
    try:
        vectorizer, clusterer, topic_modeler, config = load_models()
        models_loaded = True
    except Exception as e:
        models_loaded = False
        logger.error(f"Failed to load models: {e}")

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview", "Analyze Document", "Cluster Details", "Topic Details", "Upload & Analyze"],
    )

    # Main header
    st.title("📚 Document Clustering & Topic Modeling")
    st.markdown(
        "A production-ready unsupervised NLP pipeline for clustering documents "
        "and extracting interpretable themes from unstructured text."
    )

    # Route to appropriate page

    if not models_loaded:
        st.error(
            "❌ **Failed to load models**\n\n"
            "The models were trained but could not be loaded. Please check the logs."
        )
        return

    if page == "Overview":
        show_overview_page(config)
    elif page == "Analyze Document":
        show_analyze_page(vectorizer, clusterer, topic_modeler, config)
    elif page == "Cluster Details":
        show_cluster_details_page(config)
    elif page == "Topic Details":
        show_topic_details_page(config)
    elif page == "Upload & Analyze":
        show_upload_page(vectorizer, clusterer, topic_modeler, config)


def show_overview_page(config):
    """Project overview page."""
    st.header("📋 Project Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clusters", config["clustering"]["n_clusters"])
    with col2:
        st.metric("Topics", config["topic"]["n_topics"])
    with col3:
        st.metric("Max Features", config["vectorizer"]["max_features"])
    with col4:
        st.metric("Min DF", config["vectorizer"]["min_df"])

    # Features
    st.markdown("### ✨ Key Features")
    features = [
        "🔹 **Document Clustering** using K-Means algorithm",
        "🔹 **Topic Extraction** using Latent Dirichlet Allocation (LDA)",
        "🔹 **TF-IDF Vectorization** with adaptive parameters",
        "🔹 **Interactive Analysis** with real-time predictions",
        "🔹 **Cluster Visualization** using t-SNE embedding",
        "🔹 **Model Evaluation** with silhouette and Davies-Bouldin scores",
        "🔹 **Batch Processing** for document collections",
    ]
    for feature in features:
        st.write(feature)

    # Usage instructions
    st.markdown("### 📖 How to Use")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        **Analyze Documents**
        1. Go to "Analyze Document" tab
        2. Enter or paste your text
        3. Click "Analyze" to get predictions
        """)
    with col2:
        st.write("""
        **Batch Analysis**
        1. Go to "Upload & Analyze" tab
        2. Upload a text file (one document per line)
        3. Click "Analyze All" for batch processing
        """)

    # Configuration
    st.markdown("### ⚙️ Configuration")
    with st.expander("View Configuration", expanded=False):
        st.json(config)


def show_analyze_page(vectorizer, clusterer, topic_modeler, config):
    """Single document analysis page."""
    st.header("🔍 Analyze Single Document")

    # Input section
    st.markdown("### Input")
    text = st.text_area(
        "Enter document text:",
        height=150,
        placeholder="Paste or type your document here...",
    )

    if text.strip():
        if st.button("🔎 Analyze", key="analyze_single"):
            with st.spinner("Analyzing document..."):
                try:
                    result = make_prediction(text, vectorizer, clusterer, topic_modeler)

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 🏷️ Cluster Assignment")
                        st.metric("Cluster ID", result.cluster)
                        if result.cluster_similarity is not None:
                            st.metric("Similarity to Cluster", f"{result.cluster_similarity:.2%}")

                    with col2:
                        st.markdown("### 📌 Topic Assignment")
                        st.metric("Topic ID", result.topic)
                        st.metric("Topic Probability", f"{result.topic_probability:.2%}")

                    # Keywords
                    st.markdown("### 🔑 Top Keywords")
                    if result.keywords:
                        keyword_cols = st.columns(len(result.keywords))
                        for i, keyword in enumerate(result.keywords):
                            with keyword_cols[i]:
                                st.info(f"**{keyword}**")
                    else:
                        st.info("No keywords extracted")

                    # Topic distribution
                    st.markdown("### 📊 Topic Distribution")
                    if result.topic_distribution:
                        topic_df = pd.DataFrame({
                            "Topic": list(result.topic_distribution.keys()),
                            "Probability": list(result.topic_distribution.values()),
                        })
                        st.bar_chart(topic_df.set_index("Topic"))
                    else:
                        st.info("No topic distribution available")

                    # Download results
                    st.markdown("### 📥 Export Results")
                    json_results = json.dumps(result.to_dict(), indent=2)
                    st.download_button(
                        "📄 Download as JSON",
                        json_results,
                        "prediction_result.json",
                        "application/json",
                    )

                except Exception as e:
                    st.error(f"❌ Error analyzing document: {e}")
                    logger.error(f"Prediction error: {e}", exc_info=True)
    else:
        st.info("👆 Enter text above to analyze")


def show_cluster_details_page(config):
    """Cluster analysis page."""
    st.header("📍 Cluster Details")

    st.markdown(f"**Number of Clusters:** {config['clustering']['n_clusters']}")

    st.info(
        "Cluster analysis shows representative documents and top terms for each cluster. "
        "This information is generated during training and saved in the artifacts directory."
    )

    # Display config
    with st.expander("View Cluster Configuration", expanded=False):
        st.json(config["clustering"])


def show_topic_details_page(config):
    """Topic analysis page."""
    st.header("🏷️ Topic Details")

    st.markdown(f"**Number of Topics:** {config['topic']['n_topics']}")
    st.markdown(f"**Top Words per Topic:** {config['topic']['n_top_words']}")

    st.info(
        "Topic analysis shows the top words and document distribution for each topic "
        "extracted by Latent Dirichlet Allocation (LDA). These topics are learned "
        "during training and can be explored via the Analyze Document page."
    )

    # Display config
    with st.expander("View Topic Configuration", expanded=False):
        st.json(config["topic"])


def show_upload_page(vectorizer, clusterer, topic_modeler, config):
    """Batch analysis page."""
    st.header("📤 Upload & Batch Analyze")

    st.markdown("Upload a text file with one document per line for batch analysis.")

    uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

    if uploaded_file:
        try:
            content = uploaded_file.read().decode("utf-8")
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            st.markdown(f"**File contains {len(lines)} documents**")

            if len(lines) > 0:
                if st.button("🔎 Analyze All", key="analyze_batch"):
                    with st.spinner(f"Analyzing {len(lines)} documents..."):
                        results = []
                        progress_bar = st.progress(0)

                        for i, line in enumerate(lines):
                            try:
                                result = make_prediction(line, vectorizer, clusterer, topic_modeler)
                                results.append({
                                    "document": line[:80] + "..." if len(line) > 80 else line,
                                    "cluster": result.cluster,
                                    "topic": result.topic,
                                    "topic_prob": result.topic_probability,
                                    "keywords": ", ".join(result.keywords[:3]),
                                })
                            except Exception as e:
                                logger.error(f"Error processing document {i}: {e}")
                                results.append({
                                    "document": line[:80],
                                    "cluster": "ERROR",
                                    "topic": "ERROR",
                                    "topic_prob": 0.0,
                                    "keywords": "N/A",
                                })

                            progress_bar.progress((i + 1) / len(lines))

                        # Display results table
                        st.markdown(f"### 📊 Results ({len(results)} documents)")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)

                        # Download results
                        json_output = json.dumps(results, indent=2)
                        st.download_button(
                            "📄 Download Results as JSON",
                            json_output,
                            "batch_analysis_results.json",
                            "application/json",
                        )
            else:
                st.warning("⚠️ No documents found in file")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            logger.error(f"File processing error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
