"""Tests for text preprocessing module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_processor import TextProcessor


def test_text_processor_initialization():
    """Test TextProcessor initialization."""
    processor = TextProcessor()
    assert processor.lowercase == True
    assert processor.remove_punctuation == True
    assert processor.remove_stopwords == True


def test_process_single_document():
    """Test processing a single document."""
    processor = TextProcessor()
    text = "Machine Learning is a great field of AI!"
    tokens = processor.process(text)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)


def test_lowercase():
    """Test lowercase conversion."""
    processor = TextProcessor(lowercase=True)
    tokens = processor.process("UPPERCASE TEXT")
    
    # All tokens should be lowercase
    assert all(token.islower() or not token.isalpha() for token in tokens)


def test_remove_punctuation():
    """Test punctuation removal."""
    processor = TextProcessor(remove_punctuation=True)
    tokens = processor.process("Hello, world! How are you?")
    
    # Should not contain punctuation
    import string
    assert not any(token in string.punctuation for token in tokens)


def test_remove_stopwords():
    """Test stopword removal."""
    processor = TextProcessor(remove_stopwords=True)
    tokens = processor.process("the quick brown fox")
    
    # "the" is a common stopword
    assert "the" not in tokens


def test_process_batch(sample_documents):
    """Test batch processing."""
    processor = TextProcessor()
    results = processor.process_batch(sample_documents)
    
    assert len(results) == len(sample_documents)
    assert all(isinstance(result, list) for result in results)


def test_process_to_string():
    """Test processing and returning as string."""
    processor = TextProcessor()
    text = "Python programming language"
    result = processor.process_to_string(text)
    
    assert isinstance(result, str)
    assert len(result) > 0


def test_empty_document():
    """Test processing empty document."""
    processor = TextProcessor()
    tokens = processor.process("")
    
    assert tokens == []


def test_invalid_input():
    """Test processing non-string input."""
    processor = TextProcessor()
    tokens = processor.process(None)
    
    assert tokens == []
