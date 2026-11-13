"""Knowledge Base Ingestion Script for NIVARAN.

Ingests policy documents from data/knowledge_base/ into the persistent
ChromaDB vector store with local sentence-transformers embeddings.
Repeatable and idempotent.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from nivaran.utils.logger import get_logger
from nivaran.ingestion import DocumentIngestionService
from nivaran.embeddings import EmbeddingService
from nivaran.vector_store import ChromaVectorStore

logger = get_logger("kb_ingestion")


def ingest_knowledge_base(
    kb_dir: Path = Config.KNOWLEDGE_BASE_DIR,
    chunk_size: int = 250,
    chunk_overlap: int = 40,
    reset_collection: bool = False
) -> Dict[str, Any]:
    """Ingest all policy files in kb_dir into ChromaDB.
    
    Ensures repeatability by removing prior chunks for each file before re-indexing.
    """
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {kb_path}")

    # Discover candidate files (.pdf, .txt, .md)
    files = [
        f for f in kb_path.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md")
    ]
    if not files:
        logger.warning("No documents found in knowledge base: %s", kb_path)
        return {"status": "empty", "files_processed": 0, "total_chunks": 0}

    logger.info("Starting ingestion for %d file(s) in %s", len(files), kb_path)

    # Initialize services
    ingestion = DocumentIngestionService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = EmbeddingService(model_name=Config.EMBEDDING_MODEL_NAME)
    vector_store = ChromaVectorStore(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )

    if reset_collection:
        vector_store.reset_collection()

    total_chunks_indexed = 0
    file_details: List[Dict[str, Any]] = []

    for file_path in files:
        logger.info("Processing knowledge base document: %s", file_path.name)
        # Idempotence: remove any prior chunks from this file
        deleted = vector_store.delete_by_source(file_path.name)
        if deleted > 0:
            logger.info("Cleared %d outdated chunks for %s", deleted, file_path.name)

        # Ingest & chunk
        chunks = ingestion.process_file(file_path, document_type="policy")
        if not chunks:
            logger.warning("No chunks generated for %s", file_path.name)
            continue

        # Generate local embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_documents(texts)

        # Store in ChromaDB
        indexed = vector_store.add_chunks(chunks, embeddings)
        total_chunks_indexed += indexed

        file_details.append({
            "filename": file_path.name,
            "chunks_count": indexed,
            "pages": len(set(c.page_number for c in chunks)),
        })

    total_in_collection = vector_store.count()
    logger.info("Ingestion complete. Total collection count: %d", total_in_collection)

    return {
        "status": "success",
        "files_processed": len(files),
        "total_chunks_indexed": total_chunks_indexed,
        "total_collection_count": total_in_collection,
        "files": file_details
    }


if __name__ == "__main__":
    result = ingest_knowledge_base()
    print("Ingestion Result:", result)
