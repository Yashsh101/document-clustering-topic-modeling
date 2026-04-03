"""Unified prediction and output schemas."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Unified prediction result schema."""

    cluster: int = Field(..., description="Assigned cluster ID")
    topic: int = Field(..., description="Assigned topic ID")
    topic_probability: float = Field(..., description="Probability/score of top topic")
    keywords: List[str] = Field(default_factory=list, description="Top keywords for the cluster/topic")
    topic_distribution: Optional[Dict[str, float]] = Field(default=None, description="Full topic distribution")
    cluster_similarity: Optional[float] = Field(default=None, description="Similarity to cluster center")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "cluster": 2,
                "topic": 1,
                "topic_probability": 0.75,
                "keywords": ["customer", "billing", "refund", "issue"],
                "topic_distribution": {"topic_0": 0.2, "topic_1": 0.75, "topic_2": 0.05},
                "cluster_similarity": 0.82,
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(exclude_none=False)
