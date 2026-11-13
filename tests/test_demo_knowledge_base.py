"""Retrieval and citation verification tests for the demo knowledge base.

Tests semantic retrieval against demo_scholarship_policy.pdf using local
embeddings and ChromaDB vector store without requiring any external API key.
"""
from pathlib import Path
import pytest
from config import Config
from nivaran.embeddings import EmbeddingService
from nivaran.vector_store import ChromaVectorStore
from nivaran.retrieval import RAGRetriever


@pytest.fixture(scope="module")
def retriever():
    """Provides a retriever pointing to the persistent demo knowledge base."""
    vector_store = ChromaVectorStore(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )
    embedder = EmbeddingService(model_name=Config.EMBEDDING_MODEL_NAME)
    return RAGRetriever(
        vector_store=vector_store,
        embedding_service=embedder,
        default_top_k=3,
        similarity_threshold=0.25
    )


def test_income_certificate_validity_retrieval(retriever):
    """Verify query about income certificate validity retrieves Section 4 (Page 2)."""
    query = "Who is the competent issuing authority and what is the validity period for income certificate?"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) >= 1, "Should retrieve at least one relevant chunk"
    top_chunk = results[0]

    # Verify primary citation points to Page 2 (Section 4: Income Certificate Validity)
    assert top_chunk["metadata"]["source"] == "demo_scholarship_policy.pdf"
    assert top_chunk["metadata"]["page"] == 2
    assert top_chunk["metadata"]["document_type"] == "policy"
    assert top_chunk["similarity_score"] > 0.40

    # Verify factual content across the retrieved context
    combined_text = " ".join([chunk["text"] for chunk in results])
    assert "Tehsildar" in combined_text or "Sub-Divisional Magistrate" in combined_text
    assert "2024-2025" in combined_text
    assert "six (6) months" in combined_text


def test_required_documents_retrieval(retriever):
    """Verify query about required documents checklist retrieves Section 3 (Page 2)."""
    query = "What mandatory application documents checklist must an applicant upload?"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) >= 1, "Should retrieve at least one relevant chunk"
    top_chunk = results[0]

    assert top_chunk["metadata"]["source"] == "demo_scholarship_policy.pdf"
    assert top_chunk["metadata"]["page"] == 2
    assert "MANDATORY APPLICATION DOCUMENTS CHECKLIST" in top_chunk["text"]
    assert "Aadhaar Card" in top_chunk["text"]
    assert top_chunk["similarity_score"] > 0.40


def test_rejection_grounds_and_remediation_retrieval(retriever):
    """Verify query about rejection reasons and correction procedure retrieves Section 6/7 (Page 3)."""
    query = "What are the common grounds for rejection and what is the resubmission procedure for Defect Code R-03 or R-02?"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) >= 1
    top_chunk = results[0]

    assert top_chunk["metadata"]["source"] == "demo_scholarship_policy.pdf"
    assert top_chunk["metadata"]["page"] == 3
    chunk_text = top_chunk["text"]
    assert "Defect Code" in chunk_text or "Resubmission" in chunk_text or "Annexure-IV" in chunk_text
    assert top_chunk["similarity_score"] > 0.35


def test_unrelated_query_produces_no_misleading_evidence(retriever):
    """Verify an unrelated query does not return false positive or misleading evidence."""
    unrelated_query = "How to bake a sourdough bread using wild yeast fermentation?"
    results = retriever.retrieve(unrelated_query, top_k=3)

    # All candidate chunks for sourdough baking should fail the similarity threshold
    assert len(results) == 0, f"Expected 0 chunks for unrelated query, got {len(results)}"


def test_citations_metadata_and_formatting(retriever):
    """Verify citation formatting produces accurate, transparent source references."""
    query = "What is the qualifying examination percentage and family income ceiling?"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) >= 1
    top_chunk = results[0]
    assert top_chunk["metadata"]["source"] == "demo_scholarship_policy.pdf"
    assert top_chunk["metadata"]["page"] == 1
    assert "2,50,000" in top_chunk["text"]
    assert "60.0%" in top_chunk["text"]

    # Verify prompt context format
    formatted_context = retriever.format_context_for_prompt(results)
    assert "[Source 1: demo_scholarship_policy.pdf | Page 1 |" in formatted_context
    assert "Relevance:" in formatted_context
