"""Unit and integration tests for grounded RAG generation using mocked Gemini.

Tests grounding, deterministic validation, remediation completeness,
and citation integrity without consuming live API quota.
"""
from unittest.mock import MagicMock, patch
import pytest
from app import create_app
from config import Config
from nivaran.generator import GeminiRAGGenerator, GroundingValidator
from nivaran.retrieval import RAGRetriever
from nivaran.vector_store import ChromaVectorStore
from nivaran.embeddings import EmbeddingService


@pytest.fixture
def mock_gemini_response():
    """Generates a mock Gemini GenerateContentResponse object."""
    mock_resp = MagicMock()
    mock_resp.text = (
        "Based on the provided policy, Aarav Sharma's application was rejected due to: "
        "1. Income certificate issued by a Notary Public instead of a Tehsildar (FY 2024-2025). "
        "2. Name mismatch between portal and Aadhaar card. "
        "He should obtain an updated Tehsildar certificate and submit an Annexure-IV Affidavit."
    )
    return mock_resp


def test_generator_prompt_grounding_and_structure(mock_gemini_response):
    """Verify Gemini generator structures prompt with context and applicant document without inventing data."""
    generator = GeminiRAGGenerator(api_key="mock-test-key-12345", model_name="gemini-3.5-flash-lite")

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_response

    with patch.object(GeminiRAGGenerator, "client", new=mock_client):
        context = "[Source 1: demo_scholarship_policy.pdf | Page 2] Must be issued by Tehsildar."
        doc = "Applicant: Aarav Sharma. Defect: Income certificate by Notary."
        query = "Why was Aarav Sharma's fictional scholarship application rejected, and what should he correct?"

        result = generator.generate_rag_answer(
            query=query,
            context_text=context,
            applicant_document_text=doc
        )

        assert result["success"] is True
        assert result["error"] is None
        assert result["model"] == "gemini-3.5-flash-lite"
        assert "Aarav Sharma" in result["answer"]

        # Verify call arguments
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.5-flash-lite"
        sent_content = call_kwargs["contents"]

        # Ensure retrieved context and applicant doc were injected into prompt
        assert "RETRIEVED POLICY EVIDENCE" in sent_content
        assert "demo_scholarship_policy.pdf" in sent_content
        assert "Aarav Sharma" in sent_content
        assert query in sent_content


def test_citations_derived_strictly_from_retrieval_metadata():
    """Verify citation metadata points strictly to ChromaDB metadata and cannot be hallucinated."""
    vector_store = ChromaVectorStore(
        persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )
    embedder = EmbeddingService(model_name=Config.EMBEDDING_MODEL_NAME)
    retriever = RAGRetriever(vector_store=vector_store, embedding_service=embedder, default_top_k=3)

    query = "Why was Aarav Sharma's fictional scholarship application rejected, and what should he correct?"
    chunks = retriever.retrieve(query, top_k=3)

    assert len(chunks) >= 1
    for chunk in chunks:
        meta = chunk["metadata"]
        assert "source" in meta
        assert meta["source"] == "demo_scholarship_policy.pdf"
        assert meta["page"] in (1, 2, 3)
        assert meta["document_type"] == "policy"
        assert chunk["similarity_score"] > 0.0


def test_api_query_route_with_mocked_gemini(mock_gemini_response):
    """Verify Flask /api/query endpoint end-to-end with verified citations."""
    app = create_app()
    app.config["TESTING"] = True

    with patch("nivaran.generator.GeminiRAGGenerator.generate_rag_answer") as mock_generate:
        mock_generate.return_value = {
            "answer": mock_gemini_response.text,
            "model": "gemini-3.5-flash-lite",
            "success": True,
            "error": None,
            "grounding_audit": {"is_grounded": True, "review_required": False, "warnings": []}
        }

        with app.test_client() as client:
            payload = {
                "query": "Why was Aarav Sharma's fictional scholarship application rejected, and what should he correct?",
                "applicant_document_text": "Notice for Aarav Sharma: Defect R-03 and Defect R-02."
            }
            res = client.post("/api/query", json=payload)

            assert res.status_code == 200
            data = res.get_json()
            assert data["success"] is True
            assert "Aarav Sharma" in data["answer"]
            assert len(data["citations"]) >= 1

            # Verify every citation has real metadata from ChromaDB
            first_cit = data["citations"][0]
            assert first_cit["source"] == "demo_scholarship_policy.pdf"
            assert first_cit["page"] in (1, 2, 3)
            assert first_cit["similarity_score"] > 0.25
            assert len(first_cit["snippet"]) > 0


def test_invented_deadlines_detected_and_flagged():
    """Verify that invented deadlines not in context are deterministically flagged."""
    context = "Policy Section 5: Defective Application Correction Window closes November 15, 2025."
    applicant = "Applicant: Aarav Sharma. Provisional rejection notice issued November 2, 2025."
    
    # Model generates an ungrounded deadline: December 31, 2025 and 'within 30 days'
    hallucinated_answer = (
        "You must submit your grievance on or before December 31, 2025 or within 30 days of this notice."
    )
    
    audit = GroundingValidator.audit_answer(
        answer=hallucinated_answer,
        context_text=context,
        applicant_text=applicant
    )
    
    assert audit["is_grounded"] is False
    assert audit["review_required"] is True
    assert any("December 31, 2025" in w for w in audit["warnings"])
    assert any("within 30 days" in w for w in audit["warnings"])
    assert "December 31, 2025" in audit["invented_deadlines"]


def test_unsupported_portal_detected_and_flagged():
    """Verify unsupported third-party portals are flagged as ungrounded."""
    context = "Resubmit documents via the State Unified Scholarship Portal."
    applicant = "Notice mentions State Grievance Portal."
    
    # Model hallucinates National Scholarship Portal or DigiLocker
    hallucinated_answer = (
        "Please log into the National Scholarship Portal (NSP) or upload documents via DigiLocker."
    )
    
    audit = GroundingValidator.audit_answer(
        answer=hallucinated_answer,
        context_text=context,
        applicant_text=applicant
    )
    
    assert audit["is_grounded"] is False
    assert audit["review_required"] is True
    assert any("National Scholarship Portal" in p for p in audit["unsupported_portals"])
    assert any("DigiLocker" in p for p in audit["unsupported_portals"])


def test_mandatory_accompanying_identity_evidence_preservation():
    """Verify that omitting Gazette notification or School Bonafide for Defect R-02 triggers omission warning."""
    context = (
        "Section 7: For Defect Code R-02 (Name Mismatch), the applicant must upload a signed "
        "Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper along with a copy of the "
        "official Gazette notification or School Bonafide confirming identity."
    )
    applicant = "Defect Code R-02 Name Mismatch: 'Aarav Sharma' vs 'Aarav Kumar Sharma'."
    
    # 1. Truncated answer omitting Gazette / School Bonafide and stamp paper
    incomplete_answer = (
        "To resolve Defect R-02, submit a formal Annexure-IV Name Discrepancy Affidavit."
    )
    audit_incomplete = GroundingValidator.audit_answer(
        answer=incomplete_answer,
        context_text=context,
        applicant_text=applicant
    )
    assert audit_incomplete["review_required"] is True
    assert any("Gazette notification or School Bonafide" in o for o in audit_incomplete["omissions"])
    assert any("Rs. 50 stamp paper" in o for o in audit_incomplete["omissions"])

    # 2. Complete answer preserving mandatory accompanying evidence
    complete_answer = (
        "To resolve Defect R-02, submit a signed Annexure-IV Name Discrepancy Affidavit on "
        "Rs. 50 stamp paper along with a certified copy of the official Gazette notification or School Bonafide."
    )
    audit_complete = GroundingValidator.audit_answer(
        answer=complete_answer,
        context_text=context,
        applicant_text=applicant
    )
    # The Annexure-IV specific omissions should be resolved
    assert "Gazette notification or School Bonafide" not in audit_complete["omissions"]
    assert "Rs. 50 stamp paper" not in audit_complete["omissions"]


def test_qr_barcode_requirement_verification():
    """Verify that omitting the machine-verifiable barcode/QR code requirement on income certificate triggers review."""
    context = (
        "Section 4: The income certificate must clearly state family income for Financial Year 2024-2025. "
        "The certificate must contain a machine-verifiable barcode or QR code linked to the revenue portal."
    )
    applicant = "Defect R-03: Invalid Income Certificate."

    # Answer discussing income certificate remediation without barcode/QR code
    ans_without_qr = "Apply for a digital income certificate for FY 2024-2025 from the Tehsildar."
    audit_no_qr = GroundingValidator.audit_answer(
        answer=ans_without_qr,
        context_text=context,
        applicant_text=applicant
    )
    assert any("barcode or QR code" in w for w in audit_no_qr["warnings"])
    assert "Machine-verifiable QR/barcode check" in audit_no_qr["omissions"]

    # Answer including QR/barcode check
    ans_with_qr = (
        "Apply for a digital income certificate for FY 2024-2025 from the Tehsildar, and verify "
        "that it contains a machine-verifiable barcode or QR code linked to the e-District portal."
    )
    audit_with_qr = GroundingValidator.audit_answer(
        answer=ans_with_qr,
        context_text=context,
        applicant_text=applicant
    )
    assert "Machine-verifiable QR/barcode check" not in audit_with_qr["omissions"]


def test_insufficient_evidence_produces_explicit_uncertainty():
    """Verify that when context is empty or lacks evidence, claims without uncertainty are flagged."""
    context = "No relevant policy evidence retrieved."
    applicant = ""

    # Answer pretending to know the answer without evidence
    unsupported_answer = "The scholarship reimburses hostel fees up to Rs. 50,000 per semester."
    audit_bad = GroundingValidator.audit_answer(
        answer=unsupported_answer,
        context_text=context,
        applicant_text=applicant
    )
    assert audit_bad["review_required"] is True
    assert any("Ungrounded Assertion" in w for w in audit_bad["warnings"])

    # Grounded answer stating insufficiency
    honest_answer = "The provided policy evidence does not contain information regarding hostel fee reimbursement."
    audit_honest = GroundingValidator.audit_answer(
        answer=honest_answer,
        context_text=context,
        applicant_text=applicant
    )
    assert not any("Ungrounded Assertion" in w for w in audit_honest["warnings"])


def test_full_aarav_sharma_compliant_remediation_passes_all_checks():
    """Verify that the fully compliant remediation instructions for Aarav Sharma pass all deterministic checks."""
    context = (
        "Section 4: The income certificate must be issued and digitally signed by an authorized Revenue Officer "
        "not below Tehsildar or SDM for Financial Year 2024-2025 with machine-verifiable barcode or QR code. "
        "Section 5: Defective application correction window closes November 15, 2025. "
        "Section 7: For Defect R-02, submit Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper along with "
        "a copy of official Gazette notification or School Bonafide. Navigate to 'Resubmit Defective Application' "
        "tab and enter grievance justification. Inform College Nodal Officer to re-verify before November 20, 2025."
    )
    applicant = (
        "Applicant: Aarav Sharma. Defect R-03: Notary certificate. Defect R-02: Name mismatch "
        "'Aarav Sharma' vs 'Aarav Kumar Sharma'. Resubmit on State Grievance Portal before November 15, 2025."
    )

    full_compliant_answer = (
        "Aarav Sharma's scholarship application was provisionally rejected due to two defects: "
        "Defect R-03 (income certificate issued by Notary instead of an authorized Revenue Officer for FY 2024-2025) "
        "and Defect R-02 (name discrepancy between 'Aarav Sharma' and 'Aarav Kumar Sharma').\n\n"
        "To rectify this, Aarav must execute the following remediation actions:\n"
        "1. Obtain a fresh digital income certificate for Financial Year 2024-2025 issued and digitally signed by "
        "an authorized Revenue Officer not below the rank of Tehsildar or SDM, ensuring it carries a machine-verifiable barcode or QR code.\n"
        "2. Prepare a signed Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper along with a certified copy "
        "of the official Gazette notification or School Bonafide confirming legal identity.\n"
        "3. Log into the State Grievance Portal, navigate to the 'Resubmit Defective Application' tab, upload the corrected documents, "
        "and enter the required grievance justification before November 15, 2025.\n"
        "4. Contact the College Scholarship Nodal Officer to complete institutional re-verification before November 20, 2025."
    )

    audit = GroundingValidator.audit_answer(
        answer=full_compliant_answer,
        context_text=context,
        applicant_text=applicant
    )

    assert audit["is_grounded"] is True, f"Expected grounded, got warnings: {audit['warnings']}"
    assert audit["review_required"] is False, f"Expected review_required=False, got: {audit['warnings']}"
    assert len(audit["warnings"]) == 0
    assert len(audit["omissions"]) == 0
    assert len(audit["invented_deadlines"]) == 0
    assert len(audit["unsupported_portals"]) == 0


def test_citation_metadata_derived_strictly_from_retrieval_never_invented():
    """Verify citation metadata is always derived from ChromaDB retrieval and never from generated text claims."""
    app = create_app()
    app.config["TESTING"] = True

    # Even if model output mentions a hallucinated source like "National Gazette 1999 Page 45"
    mock_hallucinated_citation_answer = (
        "According to National Gazette 1999 Page 45, your application was rejected. "
        "Please see Rule 88 from Department Order 1234."
    )

    with patch("nivaran.generator.GeminiRAGGenerator.generate_rag_answer") as mock_generate:
        mock_generate.return_value = {
            "answer": mock_hallucinated_citation_answer,
            "model": "gemini-3.5-flash-lite",
            "success": True,
            "error": None,
            "grounding_audit": {"is_grounded": False, "review_required": True, "warnings": ["Unverified sources"]}
        }

        with app.test_client() as client:
            res = client.post("/api/query", json={"query": "Why was my application rejected?"})
            assert res.status_code == 200
            data = res.get_json()

            # Ensure citations returned to client ONLY come from ChromaDB metadata, never model's hallucinated text
            assert len(data["citations"]) > 0
            for cit in data["citations"]:
                assert cit["source"] == "demo_scholarship_policy.pdf"
                assert cit["page"] in (1, 2, 3)
                assert "National Gazette" not in cit["source"]
