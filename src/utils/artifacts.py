"""
Artifact management for saving and loading trained models and components.
Provides serialization/deserialization for vectorizers, models, and metadata.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from src.logger import setup_logger

logger = setup_logger(__name__)


class ArtifactManager:
    """Manages saving and loading of ML artifacts."""

    def __init__(self, artifact_dir: str = "artifacts"):
        """
        Initialize artifact manager.

        Args:
            artifact_dir: Root directory for artifacts
        """
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.artifact_dir / "models"
        self.plots_dir = self.artifact_dir / "plots"
        self.reports_dir = self.artifact_dir / "reports"
        self.predictions_dir = self.artifact_dir / "predictions"

        for d in [self.models_dir, self.plots_dir, self.reports_dir, self.predictions_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: Any, name: str) -> str:
        """
        Save a model using pickle.

        Args:
            model: Model to save
            name: Model name (without .pkl extension)

        Returns:
            Path to saved model
        """
        filepath = self.models_dir / f"{name}.pkl"
        try:
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save model {name}: {e}")
            raise

    def load_model(self, name: str) -> Any:
        """
        Load a pickled model.

        Args:
            name: Model name (without .pkl extension)

        Returns:
            Loaded model
        """
        filepath = self.models_dir / f"{name}.pkl"
        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded model from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise

    def save_metrics(self, metrics: Dict[str, Any], name: str) -> str:
        """
        Save metrics as JSON.

        Args:
            metrics: Dictionary of metrics
            name: Metrics name (without .json extension)

        Returns:
            Path to saved metrics
        """
        filepath = self.reports_dir / f"{name}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Saved metrics to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save metrics {name}: {e}")
            raise

    def load_metrics(self, name: str) -> Dict[str, Any]:
        """
        Load metrics from JSON.

        Args:
            name: Metrics name (without .json extension)

        Returns:
            Loaded metrics dictionary
        """
        filepath = self.reports_dir / f"{name}.json"
        try:
            with open(filepath) as f:
                metrics = json.load(f)
            logger.info(f"Loaded metrics from {filepath}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to load metrics {name}: {e}")
            raise

    def save_predictions(self, predictions: Dict[str, Any], name: str) -> str:
        """
        Save predictions as JSON.

        Args:
            predictions: Dictionary of predictions
            name: Predictions name (without .json extension)

        Returns:
            Path to saved predictions
        """
        filepath = self.predictions_dir / f"{name}.json"
        try:
            with open(filepath, "w") as f:
                json.dump(predictions, f, indent=4)
            logger.info(f"Saved predictions to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save predictions {name}: {e}")
            raise

    def save_plot(self, fig: Any, name: str) -> str:
        """
        Save matplotlib figure.

        Args:
            fig: Matplotlib figure object
            name: Figure name (without .png extension)

        Returns:
            Path to saved figure
        """
        filepath = self.plots_dir / f"{name}.png"
        try:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"Saved plot to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save plot {name}: {e}")
            raise

    def list_models(self) -> list:
        """List all saved models."""
        return [f.stem for f in self.models_dir.glob("*.pkl")]

    def list_metrics(self) -> list:
        """List all saved metrics."""
        return [f.stem for f in self.reports_dir.glob("*.json")]

    def list_plots(self) -> list:
        """List all saved plots."""
        return [f.stem for f in self.plots_dir.glob("*.png")]
