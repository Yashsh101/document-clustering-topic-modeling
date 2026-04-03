"""Tests for data loading module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DocumentLoader


def test_document_loader_initialization():
    """Test DocumentLoader initialization."""
    loader = DocumentLoader()
    assert loader.min_length == 10
    assert loader.documents == []
    assert loader.file_paths == []


def test_load_from_directory(sample_dir):
    """Test loading documents from directory."""
    loader = DocumentLoader(min_length=5)
    docs, paths = loader.load_from_directory(str(sample_dir))
    
    assert len(docs) == 3
    assert len(paths) == 3
    assert all(isinstance(doc, str) for doc in docs)


def test_load_from_list():
    """Test loading documents from list."""
    documents = ["document one", "document two", "document three"]
    loader = DocumentLoader()
    docs, paths = loader.load_from_list(documents)
    
    assert len(docs) == 3
    assert len(paths) == 3
    assert docs == documents


def test_filter_by_minimum_length(sample_dir):
    """Test filtering documents by minimum length."""
    loader = DocumentLoader(min_length=50)
    docs, paths = loader.load_from_directory(str(sample_dir))
    
    # Should filter out shorter documents
    assert len(docs) <= 3
    assert all(len(doc) >= 50 for doc in docs)


def test_empty_directory(tmp_path):
    """Test loading from empty directory."""
    loader = DocumentLoader()
    docs, paths = loader.load_from_directory(str(tmp_path))
    
    assert len(docs) == 0
    assert len(paths) == 0


def test_nonexistent_directory():
    """Test loading from nonexistent directory."""
    loader = DocumentLoader()
    docs, paths = loader.load_from_directory("/nonexistent/path")
    
    assert len(docs) == 0
    assert len(paths) == 0
