"""Document Ingestion Module for NIVARAN.

Handles PDF and text file extraction, page-level metadata tracking,
and chunking for the vector knowledge base.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import pymupdf  # PyMuPDF
from nivaran.utils.logger import get_logger

logger = get_logger("ingestion")


@dataclass
class DocumentChunk:
    """Represents a text chunk extracted from a document with full metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentIngestionService:
    """Extracts text and chunks documents for vector indexing."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text page-by-page from a PDF using PyMuPDF.
        
        Returns a list of dicts with 'page_number' and 'text'.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found at {path}")

        pages_data = []
        try:
            logger.info("Opening PDF: %s", path.name)
            doc = pymupdf.open(str(path))
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                page_text = page.get_text("text").strip()
                if page_text:
                    pages_data.append({
                        "page_number": page_index + 1,
                        "text": page_text
                    })
            doc.close()
            logger.info("Successfully extracted %d pages with text from %s", len(pages_data), path.name)
        except Exception as e:
            logger.error("Failed to extract text from PDF %s: %s", path.name, str(e))
            raise RuntimeError(f"Error processing PDF {path.name}: {str(e)}") from e

        return pages_data

    def extract_text_from_txt(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from a plain text file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found at {path}")

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            return [{"page_number": 1, "text": content}] if content else []
        except Exception as e:
            logger.error("Failed to read text file %s: %s", path.name, str(e))
            raise RuntimeError(f"Error reading file {path.name}: {str(e)}") from e

    def chunk_text(
        self,
        text: str,
        source_file: str,
        page_number: int,
        base_chunk_index: int = 0,
        document_type: str = "policy"
    ) -> List[DocumentChunk]:
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        current_idx = base_chunk_index

        for start_pos in range(0, len(words), step):
            end_pos = min(start_pos + self.chunk_size, len(words))
            chunk_words = words[start_pos:end_pos]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{Path(source_file).stem}_p{page_number}_c{current_idx}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=source_file,
                page_number=page_number,
                chunk_index=current_idx,
                metadata={
                    "source": source_file,
                    "page": page_number,
                    "chunk_id": chunk_id,
                    "document_type": document_type,
                    "word_count": len(chunk_words),
                }
            )
            chunks.append(chunk)
            current_idx += 1

            if end_pos >= len(words):
                break

        return chunks

    def process_file(self, file_path: Path, document_type: str = "policy") -> List[DocumentChunk]:
        """Process a file (PDF or TXT) and return all generated DocumentChunks."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = self.extract_text_from_pdf(path)
        elif ext in (".txt", ".md"):
            pages = self.extract_text_from_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Expected .pdf or .txt")

        all_chunks: List[DocumentChunk] = []
        chunk_counter = 0

        for page in pages:
            chunks = self.chunk_text(
                text=page["text"],
                source_file=path.name,
                page_number=page["page_number"],
                base_chunk_index=chunk_counter,
                document_type=document_type
            )
            all_chunks.extend(chunks)
            chunk_counter += len(chunks)

        logger.info("Generated %d chunks from %s", len(all_chunks), path.name)
        return all_chunks
