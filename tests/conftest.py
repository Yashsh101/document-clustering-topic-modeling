"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks",
        "natural language processing analyzes text",
        "computer vision processes images",
        "data preprocessing is important",
    ]


@pytest.fixture
def sample_dir(tmp_path):
    """Create a temporary directory with sample documents."""
    sample_docs = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks",
        "python is a great programming language",
    ]
    
    for i, doc in enumerate(sample_docs):
        file_path = tmp_path / f"sample_{i}.txt"
        file_path.write_text(doc)
    
    return tmp_path
