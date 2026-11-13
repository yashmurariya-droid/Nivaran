"""Retrieval Module for NIVARAN RAG.

Connects query embedding with vector search to fetch relevant context chunks
with exact source attribution.
"""
from typing import List, Dict, Any, Optional
from nivaran.embeddings import EmbeddingService
from nivaran.vector_store import ChromaVectorStore
from nivaran.utils.logger import get_logger

logger = get_logger("retrieval")


class RAGRetriever:
    """Retrieves grounded context from ChromaDB using local SentenceTransformer embeddings."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_service: EmbeddingService,
        default_top_k: int = 4,
        similarity_threshold: float = 0.25
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant document chunks for a given query."""
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever.")
            return []

        k = top_k or self.default_top_k
        logger.info("Retrieving top %d context chunks for query: '%s'", k, query[:80])

        query_vec = self.embedding_service.embed_query(query)
        raw_results = self.vector_store.search_by_embedding(
            query_embedding=query_vec,
            top_k=k,
            where_filter=where_filter
        )

        # Filter by minimum similarity score
        filtered_results = [
            r for r in raw_results
            if r.get("similarity_score", 0.0) >= self.similarity_threshold
        ]

        logger.info(
            "Found %d candidate chunks, %d passed threshold (>= %.2f)",
            len(raw_results), len(filtered_results), self.similarity_threshold
        )
        return filtered_results

    def format_context_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a structured prompt context block with citations."""
        if not chunks:
            return "No relevant context found in the knowledge base."

        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown Source")
            page = meta.get("page", "?")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            score = chunk.get("similarity_score", 0.0)
            text = chunk.get("text", "").strip()

            block = (
                f"[Source {i}: {source} | Page {page} | ID: {chunk_id} | Relevance: {score:.2f}]\n"
                f"{text}\n"
            )
            context_blocks.append(block)

        return "\n---\n".join(context_blocks)
