"""Document loading from various formats (txt, csv, json)."""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from txt, csv, and json files."""

    def __init__(self, accepted_formats: Optional[list[str]] = None, min_length: int = 10):
        """
        Initialize loader.

        Args:
            accepted_formats: List of file extensions to load (default: txt, csv, json)
            min_length: Minimum document length in characters (default: 10)
        """
        self.accepted_formats = accepted_formats or ["txt", "csv", "json"]
        self.min_length = min_length
        self.documents: list[str] = []
        self.file_paths: list[str] = []

    def load_from_directory(self, directory: str, recursive: bool = True) -> tuple[list[str], list[str]]:
        """
        Load documents from a directory.

        Args:
            directory: Path to directory containing documents
            recursive: If True, search subdirectories

        Returns:
            Tuple of (documents, file_paths)
        """
        self.documents = []
        self.file_paths = []

        path = Path(directory)
        if not path.exists():
            logger.error(f"Directory not found: {directory}")
            return self.documents, self.file_paths

        logger.info(f"Loading documents from {directory}...")

        # Get files based on accepted formats
        pattern = "**/*" if recursive else "*"
        for fmt in self.accepted_formats:
            for file_path in path.glob(f"{pattern}.{fmt}"):
                if file_path.is_file():
                    self._load_file(file_path)

        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents, self.file_paths

    def load_from_list(self, documents: list[str]) -> tuple[list[str], list[str]]:
        """
        Load documents from a list of strings.

        Args:
            documents: List of document strings

        Returns:
            Tuple of (documents, file_paths) where file_paths are synthetic
        """
        self.documents = []
        self.file_paths = []

        for idx, doc in enumerate(documents):
            content = str(doc).strip() if doc else ""
            if content and len(content) >= self.min_length:
                self.documents.append(content)
                self.file_paths.append(f"list_item_{idx}")

        logger.info(f"Loaded {len(self.documents)} documents from list")
        return self.documents, self.file_paths

    def _load_file(self, file_path: Path) -> None:
        """Load a single file based on its format."""
        try:
            if file_path.suffix == ".txt":
                self._load_txt(file_path)
            elif file_path.suffix == ".csv":
                self._load_csv(file_path)
            elif file_path.suffix == ".json":
                self._load_json(file_path)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")

    def _load_txt(self, file_path: Path) -> None:
        """Load a single text file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content and len(content) >= self.min_length:
                    self.documents.append(content)
                    self.file_paths.append(str(file_path))
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

    def _load_csv(self, file_path: Path, text_column: str = "text") -> None:
        """Load documents from CSV file."""
        try:
            df = pd.read_csv(file_path)
            if text_column not in df.columns:
                logger.warning(
                    f"Column '{text_column}' not found in {file_path}. Columns: {df.columns.tolist()}"
                )
                return

            for idx, row in df.iterrows():
                content = str(row[text_column]).strip()
                if content and len(content) >= self.min_length:
                    self.documents.append(content)
                    self.file_paths.append(f"{file_path}:row_{idx}")
        except Exception as e:
            logger.warning(f"Error reading CSV {file_path}: {e}")

    def _load_json(self, file_path: Path) -> None:
        """Load documents from JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Handle both list and dict formats
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            content = item.strip()
                        elif isinstance(item, dict) and "text" in item:
                            content = str(item["text"]).strip()
                        else:
                            continue

                        if content and len(content) >= self.min_length:
                            self.documents.append(content)
                            self.file_paths.append(str(file_path))

                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            content = value.strip()
                        elif isinstance(value, dict) and "text" in value:
                            content = str(value["text"]).strip()
                        else:
                            continue

                        if content and len(content) >= self.min_length:
                            self.documents.append(content)
                            self.file_paths.append(f"{file_path}:key_{key}")

        except Exception as e:
            logger.warning(f"Error reading JSON {file_path}: {e}")

    def load(self, directory: str, recursive: bool = True) -> list[str]:
        """
        Load documents and return only document texts.

        Args:
            directory: Path to directory
            recursive: If True, search subdirectories

        Returns:
            List of documents
        """
        documents, _ = self.load_from_directory(directory, recursive)
        return documents
