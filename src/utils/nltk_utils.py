"""NLTK resource management utilities."""

import nltk
from src.logger import get_logger

logger = get_logger(__name__)

# List of required NLTK resources
REQUIRED_RESOURCES = [
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "corpora/wordnet",
]


def ensure_nltk_resources():
    """
    Ensure all required NLTK resources are available.
    
    Downloads missing resources using quiet mode.
    Idempotent - safe to call repeatedly.
    
    Raises:
        RuntimeError: If any resources fail to download
    """
    for resource in REQUIRED_RESOURCES:
        try:
            # Check if resource exists
            nltk.data.find(resource)
            logger.debug(f"NLTK resource available: {resource}")
        except LookupError:
            # Resource missing - download it
            logger.info(f"Downloading NLTK resource: {resource}")
            try:
                # Extract resource name (e.g., "punkt" from "tokenizers/punkt")
                resource_name = resource.split("/")[-1]
                nltk.download(resource_name, quiet=True)
                logger.info(f"✓ Downloaded NLTK resource: {resource}")
            except Exception as e:
                error_msg = f"Failed to download NLTK resource '{resource}': {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
