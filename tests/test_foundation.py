"""Foundation unit tests for NIVARAN modules."""
import tempfile
from pathlib import Path
import pytest
from config import Config
from nivaran.utils.logger import mask_sensitive_value
from nivaran.ingestion import DocumentIngestionService, DocumentChunk
from nivaran.embeddings import EmbeddingService
from nivaran.vector_store import ChromaVectorStore
from nivaran.retrieval import RAGRetriever
from nivaran.generator import GeminiRAGGenerator


def test_mask_sensitive_value():
    """Verify secrets are masked and never exposed."""
    assert mask_sensitive_value(None) == "<not set>"
    assert mask_sensitive_value("") == "<not set>"
    assert mask_sensitive_value("1234") == "***"
    assert mask_sensitive_value("AIzaSyB1234567890abcdef") == "AIza...cdef"


def test_config_paths_and_safety():
    """Verify configuration creates directories and produces safe summary."""
    Config.ensure_directories()
    assert Config.DATA_DIR.exists()
    assert Config.UPLOAD_FOLDER.exists()
    assert Config.CHROMA_PERSIST_DIRECTORY.exists()

    summary = Config.get_safe_summary()
    assert "gemini_key_preview" in summary
    # Ensure raw secret is not present in summary
    if Config.GEMINI_API_KEY:
        assert Config.GEMINI_API_KEY != summary["gemini_key_preview"]


def test_document_ingestion_pdf():
    """Verify document ingestion and extraction on real PDF files via PyMuPDF."""
    import pymupdf

    ingestion = DocumentIngestionService(chunk_size=50, chunk_overlap=10)

    with tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) as f:
        pdf_path = Path(f.name)

    try:
        # Create a test PDF with PyMuPDF
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text(
            (50, 72),
            "Merit-cum-Means Scholarship Guidelines: Clause 4.2.\n"
            "Applicants must submit an attested copy of the caste certificate "
            "issued by a competent authority. Applications missing this document "
            "will be summarily rejected."
        )
        doc.save(str(pdf_path))
        doc.close()

        # Ingest and process
        chunks = ingestion.process_file(pdf_path)
        assert len(chunks) >= 1
        assert "Merit-cum-Means Scholarship" in chunks[0].text
        assert chunks[0].metadata["page"] == 1
    finally:
        pdf_path.unlink(missing_ok=True)


def test_document_ingestion_text():
    """Verify document ingestion and chunking on text files."""
    ingestion = DocumentIngestionService(chunk_size=50, chunk_overlap=10)

    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("The National Means Scholarship requires a minimum of 60% aggregate marks in 10th standard. "
                "Annual family income must be below Rs. 2,50,000 from all sources. "
                "Income certificates must be issued by a Sub-Divisional Magistrate or equivalent officer.")
        temp_path = Path(f.name)

    try:
        chunks = ingestion.process_file(temp_path)
        assert len(chunks) >= 1
        first = chunks[0]
        assert isinstance(first, DocumentChunk)
        assert "National Means Scholarship" in first.text
        assert first.metadata["page"] == 1
        assert "chunk_id" in first.metadata
    finally:
        temp_path.unlink(missing_ok=True)


def test_embeddings_and_chroma_vector_store():
    """Verify sentence-transformers local embeddings with ChromaDB persistence."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        store = ChromaVectorStore(
            persist_directory=Path(temp_dir),
            collection_name="test_foundation_collection"
        )
        assert store.count() == 0

        # Test local embeddings
        embedder = EmbeddingService(model_name="all-MiniLM-L6-v2")
        text_samples = [
            "Income certificate must be signed by Revenue Officer or Tehsildar.",
            "Scholarship application submission deadline is October 31 annually."
        ]
        vectors = embedder.embed_documents(text_samples)
        assert len(vectors) == 2
        assert len(vectors[0]) == 384  # MiniLM dimension is 384

        # Test chunk upsert
        chunks = [
            DocumentChunk(
                chunk_id="test_doc_p1_c0",
                text=text_samples[0],
                source_file="guidelines.pdf",
                page_number=1,
                chunk_index=0,
                metadata={"source": "guidelines.pdf", "page": 1, "topic": "income"}
            ),
            DocumentChunk(
                chunk_id="test_doc_p1_c1",
                text=text_samples[1],
                source_file="guidelines.pdf",
                page_number=1,
                chunk_index=1,
                metadata={"source": "guidelines.pdf", "page": 1, "topic": "deadline"}
            ),
        ]

        inserted = store.add_chunks(chunks, vectors)
        assert inserted == 2
        assert store.count() == 2

        # Test RAG Retriever
        retriever = RAGRetriever(vector_store=store, embedding_service=embedder)
        results = retriever.retrieve("Who must sign the income certificate?", top_k=2)
        assert len(results) >= 1
        top_match = results[0]
        assert "Tehsildar" in top_match["text"]
        assert top_match["similarity_score"] > 0.3

        # Test prompt context formatting
        formatted = retriever.format_context_for_prompt(results)
        assert "Source 1: guidelines.pdf" in formatted
        assert "Page 1" in formatted


def test_gemini_generator_configuration_guard():
    """Verify Gemini generator safely guards against unconfigured keys."""
    # When unconfigured
    generator_unconfigured = GeminiRAGGenerator(api_key="")
    assert not generator_unconfigured.is_configured()

    res = generator_unconfigured.generate_rag_answer(
        query="Why was my form rejected?",
        context_text="Income limit exceeded."
    )
    assert res["success"] is False
    assert res["error"] == "MISSING_API_KEY"
    assert "Gemini API key is not configured" in res["answer"]
