"""LDA Topic modeling."""

import logging
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.config import TopicConfig
from src.logger import get_logger

logger = get_logger(__name__)


class TopicModeler:
    """LDA topic modeling for documents."""

    def __init__(self, config: TopicConfig = None):
        """
        Initialize topic modeler.

        Args:
            config: TopicConfig instance
        """
        self.config = config or TopicConfig()
        self.model = None
        self.count_vectorizer = None
        self.feature_names = None
        self.doc_topic_matrix = None
        self.is_fitted = False

    def fit(self, documents: list) -> "TopicModeler":
        """
        Fit LDA model on documents.

        Args:
            documents: List of preprocessed documents

        Returns:
            Self
        """
        logger.info(f"Fitting LDA with {self.config.n_topics} topics")
        
        # Create count vectorizer
        self.count_vectorizer = CountVectorizer(
            max_features=100,
            min_df=2,
            max_df=0.8,
            stop_words="english"
        )
        count_matrix = self.count_vectorizer.fit_transform(documents)
        self.feature_names = np.array(self.count_vectorizer.get_feature_names_out())
        
        # Fit LDA
        self.model = LatentDirichletAllocation(
            n_components=self.config.n_topics,
            max_iter=self.config.max_iter,
            learning_method=self.config.learning_method,
            random_state=self.config.random_state,
        )
        self.doc_topic_matrix = self.model.fit_transform(count_matrix)
        self.is_fitted = True
        
        logger.info(f"LDA fitted with {self.config.n_topics} topics")
        return self

    def transform(self, count_matrix):
        """Transform new documents."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.transform(count_matrix)

    def get_top_words(self, topic_id: int, n_words: int = 10) -> list:
        """
        Get top words for a topic.

        Args:
            topic_id: Topic index
            n_words: Number of words to return

        Returns:
            List of top words
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        top_indices = self.model.components_[topic_id].argsort()[-n_words:][::-1]
        return [self.feature_names[i] for i in top_indices]

    def get_all_topics_summary(self, n_words: int = 10) -> dict:
        """
        Get summary of all topics.

        Args:
            n_words: Number of top words per topic

        Returns:
            Dictionary with topic summaries
        """
        topics = {}
        for topic_id in range(self.config.n_topics):
            words = self.get_top_words(topic_id, n_words)
            topics[f"topic_{topic_id}"] = {
                "words": words,
                "summary": " ".join(words),
            }
        return topics

    def get_document_topics(self, doc_id: int, top_n: int = 3) -> list:
        """
        Get dominant topics for a document.

        Args:
            doc_id: Document index
            top_n: Number of top topics to return

        Returns:
            List of (topic_id, probability) tuples
        """
        if not self.is_fitted or self.doc_topic_matrix is None:
            raise ValueError("Model must be fitted first")
        
        doc_topics = self.doc_topic_matrix[doc_id]
        top_topic_indices = doc_topics.argsort()[-top_n:][::-1]
        return [(i, doc_topics[i]) for i in top_topic_indices]
