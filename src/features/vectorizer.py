"""TF-IDF vectorization with save/load support."""

import logging
from typing import Optional

from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from joblib import dump, load

from src.config import VectorizerConfig

logger = logging.getLogger(__name__)


class TFIDFVectorizer:
    """Wrapper around sklearn TfidfVectorizer with save/load support."""

    def __init__(self, config: Optional[VectorizerConfig] = None):
        """
        Initialize vectorizer.

        Args:
            config: VectorizerConfig instance
        """
        self.config = config or VectorizerConfig()
        # Store original values for later adaptive adjustment
        self.original_min_df = self.config.min_df
        self.original_max_df = self.config.max_df
        self.vectorizer = None
        self.is_fitted = False

    def _get_adaptive_params(self, n_samples: int) -> dict:
        """
        Adapt min_df and max_df for small datasets.

        Args:
            n_samples: Number of documents

        Returns:
            Dictionary with adapted parameters
        """
        # For very small datasets, relax min_df/max_df constraints
        min_df = 1  # Allow terms appearing in just 1 document
        max_df = max(1.0, min(0.95, (n_samples - 1) / n_samples)) if n_samples > 1 else 1.0

        return {
            'max_features': self.config.max_features,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': self.config.ngram_range,
            'lowercase': self.config.lowercase,
            'stop_words': self.config.stop_words,
        }

    def fit(self, documents: list[str]) -> "TFIDFVectorizer":
        """
        Fit vectorizer on documents.

        Args:
            documents: List of text documents

        Returns:
            Self for chaining
        """
        n_samples = len(documents)
        params = self._get_adaptive_params(n_samples)
        
        self.vectorizer = SklearnTfidfVectorizer(**params)
        self.vectorizer.fit(documents)
        self.is_fitted = True
        logger.info(
            f"Vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}"
        )
        return self

    def transform(self, documents: list[str]) -> csr_matrix:
        """
        Transform documents to TF-IDF matrix.

        Args:
            documents: List of text documents

        Returns:
            Sparse TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: list[str]) -> csr_matrix:
        """
        Fit and transform documents.

        Args:
            documents: List of text documents

        Returns:
            Sparse TF-IDF matrix
        """
        self.fit(documents)
        return self.vectorizer.transform(documents)

    def get_feature_names(self) -> list[str]:
        """
        Get feature names (terms).

        Returns:
            List of terms in vocabulary
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()

    def save(self, path: str, vectorizer_name: str = "tfidf_vectorizer") -> None:
        """
        Save vectorizer to disk.

        Args:
            path: Directory path
            vectorizer_name: Name for saved file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")

        from pathlib import Path

        Path(path).mkdir(parents=True, exist_ok=True)
        vectorizer_path = f"{path}/{vectorizer_name}.pkl"
        dump(self.vectorizer, vectorizer_path)
        logger.info(f"Vectorizer saved to {vectorizer_path}")

    @classmethod
    def load(cls, path: str, vectorizer_name: str = "tfidf_vectorizer") -> "TFIDFVectorizer":
        """
        Load vectorizer from disk.

        Args:
            path: Directory path
            vectorizer_name: Name of saved file

        Returns:
            Loaded TFIDFVectorizer instance
        """
        vectorizer_path = f"{path}/{vectorizer_name}.pkl"
        sklearn_vectorizer = load(vectorizer_path)
        instance = cls()
        instance.vectorizer = sklearn_vectorizer
        instance.is_fitted = True
        logger.info(f"Vectorizer loaded from {vectorizer_path}")
        return instance
