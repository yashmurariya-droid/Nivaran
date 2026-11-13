"""NIVARAN: Retrieval-Augmented Generation (RAG) System for Scholarship Assistance."""
from .utils.logger import setup_logger

# Initialize application-wide structured logger
logger = setup_logger()

__version__ = "0.1.0"
__all__ = ["logger", "__version__"]
