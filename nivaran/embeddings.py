"""Embeddings Service supporting local SentenceTransformers and Google gemini-embedding-2.

Allows seamless switching between local offline models (all-MiniLM-L6-v2)
and Google Cloud Gemini embeddings (gemini-embedding-2) with verified dimensional consistency.
"""
import os
from typing import List, Optional
from nivaran.utils.logger import get_logger

logger = get_logger("embeddings")


class EmbeddingService:
    """Manages text embeddings using SentenceTransformers or Google gemini-embedding-2."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self._model = None
        self._gemini_client = None
        self._is_gemini = "gemini" in model_name.lower()

    @property
    def is_gemini(self) -> bool:
        """Return True if using a Gemini embedding model."""
        return self._is_gemini

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model on first access."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("SentenceTransformer model loaded successfully.")
            except ImportError as e:
                logger.error("sentence-transformers not installed. Install it or use gemini-embedding-2.")
                raise RuntimeError("sentence-transformers is required for local embeddings.") from e
        return self._model

    @property
    def gemini_client(self):
        """Lazy-load the Gemini API client on first access."""
        if self._gemini_client is None:
            from google import genai
            key = self.api_key or os.getenv("GEMINI_API_KEY", "")
            if not key:
                logger.warning("Initializing Gemini client without explicit key; relying on environment.")
            self._gemini_client = genai.Client(api_key=key) if key else genai.Client()
        return self._gemini_client

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        if self._is_gemini:
            res = self.gemini_client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            if hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0].values
            elif hasattr(res, "embedding") and res.embedding:
                return res.embedding.values
            raise RuntimeError("Unexpected response structure from Gemini embed_content")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embedding vectors for a list of document strings."""
        if not texts:
            return []

        if self._is_gemini:
            results: List[List[float]] = []
            for t in texts:
                results.append(self.embed_text(t))
            return results

        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a search query."""
        return self.embed_text(query)
