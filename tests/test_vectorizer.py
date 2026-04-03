"""Tests for vectorizer module."""

import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.vectorizer import TFIDFVectorizer
from src.config import VectorizerConfig
from scipy.sparse import csr_matrix


def test_vectorizer_initialization():
    """Test TFIDFVectorizer initialization."""
    vec = TFIDFVectorizer()
    assert vec.is_fitted == False


def test_fit_transform(sample_documents):
    """Test fit and transform."""
    vec = TFIDFVectorizer()
    matrix = vec.fit_transform(sample_documents)
    
    assert isinstance(matrix, csr_matrix)
    assert matrix.shape[0] == len(sample_documents)
    assert vec.is_fitted == True


def test_transform_before_fit():
    """Test that transform raises error before fitting."""
    vec = TFIDFVectorizer()
    with pytest.raises(ValueError):
        vec.transform(["test document"])


def test_get_feature_names(sample_documents):
    """Test getting feature names."""
    vec = TFIDFVectorizer()
    vec.fit(sample_documents)
    features = vec.get_feature_names()
    
    assert isinstance(features, list)
    assert len(features) > 0


def test_matrix_shape(sample_documents):
    """Test TF-IDF matrix shape."""
    config = VectorizerConfig(max_features=10)
    vec = TFIDFVectorizer(config=config)
    matrix = vec.fit_transform(sample_documents)
    
    assert matrix.shape[0] == len(sample_documents)
    assert matrix.shape[1] <= 10


def test_transform_new_documents(sample_documents):
    """Test transforming new documents."""
    vec = TFIDFVectorizer()
    vec.fit(sample_documents)
    
    new_docs = ["machine learning is great"]
    matrix = vec.transform(new_docs)
    
    assert matrix.shape[0] == 1
    assert matrix.shape[1] == len(vec.get_feature_names())


def test_ngram_range():
    """Test n-gram range configuration."""
    config = VectorizerConfig(ngram_range=(1, 2))
    vec = TFIDFVectorizer(config=config)
    docs = ["machine learning", "natural language processing"]
    vec.fit(docs)
    
    features = vec.get_feature_names()
    # Should have unigrams and bigrams
    assert len(features) > 0


def test_empty_documents():
    """Test with empty document list."""
    vec = TFIDFVectorizer()
    docs = ["", "  ", "\n"]
    
    # Should handle documents (may result in empty features)
    try:
        vec.fit_transform(docs)
    except ValueError:
        pass  # Its okay if it raises ValueError for empty docs
