"""Vector Store Module for ChromaDB persistent storage."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from nivaran.utils.logger import get_logger
from nivaran.ingestion import DocumentChunk

logger = get_logger("vector_store")


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage and similarity searches."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "nivaran_scholarship_rules"
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ChromaDB client at '%s'", self.persist_directory)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection '%s' ready. Current doc count: %d", self.collection_name, self.collection.count())

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Add document chunks and their pre-computed embeddings to the collection."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings ({len(embeddings)})")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info("Upserted %d chunks into '%s'. Total: %d", len(ids), self.collection_name, self.collection.count())
        return len(ids)

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB using pre-computed query embedding."""
        if self.collection.count() == 0:
            logger.warning("Search attempted on empty ChromaDB collection '%s'", self.collection_name)
            return []

        query_params: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            num_matches = len(results["ids"][0])
            for i in range(num_matches):
                chunk_id = results["ids"][0][i]
                doc_text = results["documents"][0][i] if results.get("documents") else ""
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                distance = results["distances"][0][i] if results.get("distances") else 1.0
                # Cosine distance to similarity score
                similarity = 1.0 - distance

                formatted_results.append({
                    "chunk_id": chunk_id,
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": distance,
                    "similarity_score": round(similarity, 4)
                })

        return formatted_results

    def delete_by_source(self, source_file: str) -> int:
        """Remove existing chunks for a given source filename before re-ingestion."""
        try:
            existing = self.collection.get(where={"source": source_file})
            if existing and existing.get("ids"):
                ids_to_del = existing["ids"]
                self.collection.delete(ids=ids_to_del)
                logger.info("Deleted %d existing chunks for source '%s'", len(ids_to_del), source_file)
                return len(ids_to_del)
        except Exception as e:
            logger.warning("Could not delete prior chunks for '%s': %s", source_file, str(e))
        return 0

    def count(self) -> int:
        """Return total number of chunks stored in collection."""
        return self.collection.count()

    def reset_collection(self):
        """Delete and recreate the current collection."""
        logger.warning("Resetting collection '%s'...", self.collection_name)
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
