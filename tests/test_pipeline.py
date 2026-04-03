"""Tests for pipeline module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import Pipeline
from src.config import get_config


def test_pipeline_initialization():
    """Test Pipeline initialization."""
    config = get_config()
    pipeline = Pipeline(config=config)
    
    assert pipeline.config == config
    assert len(pipeline.raw_documents) == 0


def test_pipeline_load_documents(sample_dir):
    """Test loading documents in pipeline."""
    pipeline = Pipeline()
    pipeline.load_documents(str(sample_dir))
    
    assert len(pipeline.raw_documents) > 0


def test_pipeline_preprocess(sample_dir):
    """Test preprocessing in pipeline."""
    pipeline = Pipeline()
    pipeline.load_documents(str(sample_dir))
    pipeline.preprocess()
    
    assert len(pipeline.processed_documents) == len(pipeline.raw_documents)
    assert all(isinstance(doc, str) for doc in pipeline.processed_documents)


def test_pipeline_vectorize(sample_dir):
    """Test vectorization in pipeline."""
    pipeline = Pipeline()
    pipeline.load_documents(str(sample_dir))
    pipeline.preprocess()
    pipeline.vectorize()
    
    assert pipeline.tfidf_matrix is not None
    assert pipeline.tfidf_matrix.shape[0] == len(pipeline.raw_documents)


def test_pipeline_cluster(sample_dir):
    """Test clustering in pipeline."""
    pipeline = Pipeline()
    pipeline.load_documents(str(sample_dir))
    pipeline.preprocess()
    pipeline.vectorize()
    pipeline.cluster()
    
    assert pipeline.cluster_labels is not None
    assert len(pipeline.cluster_labels) == len(pipeline.raw_documents)


def test_full_pipeline_run(sample_dir):
    """Test full pipeline execution."""
    config = get_config()
    config.clustering.n_clusters = 2
    config.topic.n_topics = 2
    
    pipeline = Pipeline(config=config)
    pipeline.run(str(sample_dir), save_artifacts=False)
    
    assert len(pipeline.raw_documents) > 0
    assert pipeline.cluster_labels is not None
    assert pipeline.metrics is not None or len(pipeline.metrics) > 0
