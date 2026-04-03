"""Configuration management for document clustering and topic modeling pipeline."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data loading."""

    input_dir: str = Field(default="data/sample", description="Directory for input documents")
    output_dir: str = Field(default="artifacts", description="Artifact output directory")
    supported_formats: list = Field(default=[".txt", ".csv", ".json"], description="Supported file formats")
    csv_text_column: str = Field(default="text", description="CSV column containing text")


class PreprocessingConfig(BaseModel):
    """Configuration for text preprocessing."""

    lowercase: bool = Field(default=True, description="Convert text to lowercase")
    remove_punctuation: bool = Field(default=True, description="Remove punctuation")
    remove_numbers: bool = Field(default=False, description="Remove numbers")
    remove_stopwords: bool = Field(default=True, description="Remove stopwords")
    tokenize: bool = Field(default=True, description="Tokenize text")
    lemmatize: bool = Field(default=True, description="Apply lemmatization")
    min_token_length: int = Field(default=2, description="Minimum token length")
    language: str = Field(default="english", description="Language for stopwords")


class VectorizerConfig(BaseModel):
    """Configuration for TF-IDF vectorization."""

    max_features: int = Field(default=500, description="Maximum number of features")
    min_df: int = Field(default=2, description="Minimum document frequency")
    max_df: float = Field(default=0.95, description="Maximum document frequency")
    ngram_range: tuple = Field(default=(1, 2), description="N-gram range")
    lowercase: bool = Field(default=True, description="Lowercase during vectorization")
    stop_words: str = Field(default="english", description="Stopwords language")


class ClusteringConfig(BaseModel):
    """Configuration for K-means clustering."""

    n_clusters: int = Field(default=5, description="Number of clusters")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    n_init: int = Field(default=10, description="Number of cluster initializations")
    max_iter: int = Field(default=300, description="Maximum iterations")
    search_k: bool = Field(default=False, description="Search optimal number of clusters")
    k_range: tuple = Field(default=(2, 10), description="Range for k search")


class TopicConfig(BaseModel):
    """Configuration for LDA topic modeling."""

    n_topics: int = Field(default=5, description="Number of topics")
    random_state: int = Field(default=42, description="Random state")
    max_iter: int = Field(default=20, description="Maximum iterations")
    learning_method: str = Field(default="batch", description="Learning method")
    n_top_words: int = Field(default=10, description="Number of top words per topic")


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    compute_silhouette: bool = Field(default=True, description="Compute silhouette score")
    compute_davies_bouldin: bool = Field(default=True, description="Compute Davies-Bouldin score")
    compute_inertia: bool = Field(default=True, description="Compute inertia")
    save_metrics: bool = Field(default=True, description="Save metrics to file")


class AppConfig(BaseModel):
    """Root configuration for the entire application."""

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    vectorizer: VectorizerConfig = Field(default_factory=VectorizerConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    topic: TopicConfig = Field(default_factory=TopicConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "data": self.data.model_dump(),
            "preprocessing": self.preprocessing.model_dump(),
            "vectorizer": self.vectorizer.model_dump(),
            "clustering": self.clustering.model_dump(),
            "topic": self.topic.model_dump(),
            "evaluation": self.evaluation.model_dump(),
        }


def get_config() -> AppConfig:
    """Get default application configuration."""
    return AppConfig()
