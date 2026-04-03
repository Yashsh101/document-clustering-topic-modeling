"""Text preprocessing pipeline."""

import re
import string
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.config import PreprocessingConfig
from src.utils.nltk_utils import ensure_nltk_resources

# Ensure all required NLTK resources are available
ensure_nltk_resources()


class TextProcessor:
    """Text preprocessing with configurable pipeline."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        tokenize: bool = True,
        lemmatize: bool = True,
        min_token_length: int = 2,
        language: str = "english",
        config: Optional[PreprocessingConfig] = None,
    ):
        """
        Initialize text processor.

        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric values
            remove_stopwords: Remove common stopwords
            tokenize: Tokenize text using word_tokenize
            lemmatize: Apply lemmatization
            min_token_length: Minimum token length to keep
            language: Language for stopwords
            config: Optional PreprocessingConfig (overrides individual params if provided)
        """
        # Support both config object and individual parameters
        if config is not None:
            self.lowercase = config.lowercase
            self.remove_punctuation = config.remove_punctuation
            self.remove_numbers = config.remove_numbers
            self.remove_stopwords = config.remove_stopwords
            self.tokenize = config.tokenize
            self.lemmatize = config.lemmatize
            self.min_token_length = config.min_token_length
            self.language = config.language
        else:
            self.lowercase = lowercase
            self.remove_punctuation = remove_punctuation
            self.remove_numbers = remove_numbers
            self.remove_stopwords = remove_stopwords
            self.tokenize = tokenize
            self.lemmatize = lemmatize
            self.min_token_length = min_token_length
            self.language = language

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = (
            set(stopwords.words(self.language)) if self.remove_stopwords else set()
        )

    def process(self, text: Optional[str]) -> list[str]:
        """
        Apply full preprocessing pipeline to text.

        Args:
            text: Raw text document (can be None)

        Returns:
            List of preprocessed tokens
        """
        # Handle None and empty strings
        if text is None or not isinstance(text, str):
            return []

        text = text.strip()
        if not text:
            return []

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Tokenize
        if self.tokenize:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Filter by minimum token length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def process_to_string(self, text: Optional[str]) -> str:
        """
        Process text and return as space-separated string.

        Args:
            text: Raw text document

        Returns:
            Preprocessed text as string (space-separated tokens)
        """
        tokens = self.process(text)
        return " ".join(tokens)

    def process_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Process multiple texts.

        Args:
            texts: List of text documents

        Returns:
            List of token lists
        """
        return [self.process(text) for text in texts]

    def batch_process(self, texts: list[str]) -> list[str]:
        """
        Process multiple texts and return as strings (legacy API).

        Args:
            texts: List of text documents

        Returns:
            List of processed documents (space-separated tokens)
        """
        return [self.process_to_string(text) for text in texts]
