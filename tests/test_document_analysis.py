"""Unit and integration tests for Document-Driven Analysis in NIVARAN.

Tests server-side temporary upload storage, session isolation,
strict JSON schema validation, grounding checks, and dynamic defect extraction
without consuming any live Gemini API quota.
"""
import io
import json
import uuid
from unittest.mock import MagicMock, patch
import pytest

from app import create_app
from config import Config
from nivaran.generator import (
    GeminiRAGGenerator,
    GroundingValidator,
    DocumentAnalysisSchema,
    DefectFinding,
)


@pytest.fixture
def app_instance():
    """Create Flask application instance configured for testing."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key-12345"
    return app


@pytest.fixture
def client(app_instance):
    """Provide test client with session support."""
    with app_instance.test_client() as client:
        yield client


@pytest.fixture
def mock_gemini_analysis_response():
    """Provide a mock Gemini response adhering strictly to DocumentAnalysisSchema."""
    schema_payload = {
        "applicant_name": "Aarav Sharma",
        "document_reference": "SSP/2025-26/REJ-88412",
        "document_status": "Provisional Rejection",
        "summary": "Your application was provisionally rejected due to an invalid income certificate and a name mismatch.",
        "findings": [
            {
                "category": "confirmed_defect",
                "code": "R-03",
                "title": "Invalid Income Certificate Submitted",
                "what_was_submitted": "Income certificate issued by a Notary Public for Financial Year 2022-2023.",
                "policy_requirement": "Section 4: Income certificate must be issued by a Tehsildar or SDM for FY 2024-2025 with machine-verifiable barcode or QR code.",
                "remediation_steps": "Apply for digital income certificate for FY 2024-2025 from Tehsildar with digital signature and QR/barcode.",
                "supporting_evidence_needed": None
            },
            {
                "category": "confirmed_defect",
                "code": "R-02",
                "title": "Identity Discrepancy / Name Mismatch",
                "what_was_submitted": "Portal name 'Aarav Sharma' vs Aadhaar card 'Aarav Kumar Sharma'.",
                "policy_requirement": "Section 6 & 7: Names must match or be supported by Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper.",
                "remediation_steps": "Submit Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper along with Gazette notification or School Bonafide.",
                "supporting_evidence_needed": "Official Gazette notification or School Bonafide confirming identity on Rs. 50 stamp paper"
            }
        ],
        "evidence_gaps": []
    }

    mock_resp = MagicMock()
    mock_resp.text = json.dumps(schema_payload)
    return mock_resp


# =============================================================================
# 1. Upload Storage & Full Text Privacy
# =============================================================================

def test_upload_stores_document_server_side_and_returns_upload_id(client, app_instance):
    """Verify upload returns secure upload_id and preview, but never exposes full text to browser."""
    test_content = b"Rejection Notice: Name: Aarav Sharma. Defect R-03: Invalid Income Certificate."
    
    res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(test_content), "notice.txt")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 200
    data = res.get_json()

    assert data["success"] is True
    assert "upload_id" in data
    assert "full_text" not in data  # Never returned to client
    assert "preview" in data
    assert data["chunks_extracted"] >= 1

    # Verify document exists in server-side upload store
    upload_store = app_instance.extensions["nivaran"]["upload_store"]
    with client.session_transaction() as sess:
        user_sid = sess.get("user_sid")
        assert user_sid is not None

    stored = upload_store.get_document(data["upload_id"], user_sid)
    assert stored is not None
    assert "Aarav Sharma" in stored["full_text"]


# =============================================================================
# 2. Document Analysis on Fictional Rejection Notice
# =============================================================================

def test_analyze_fictional_rejection_notice_mocked_gemini(client, mock_gemini_analysis_response):
    """Verify /api/analyze parses structured findings, metadata and ChromaDB citations."""
    # 1. Upload notice
    test_content = (
        b"NOTICE OF DEFECTIVE APPLICATION. Applicant: Aarav Sharma. Ref: SSP/2025-26/REJ-88412. "
        b"DEFECT 1 (Code: R-03) Invalid Income Certificate from Notary. "
        b"DEFECT 2 (Code: R-02) Identity Discrepancy: Aarav Sharma vs Aarav Kumar Sharma."
    )
    upload_res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(test_content), "fictional_notice.txt")},
        content_type="multipart/form-data"
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.get_json()["upload_id"]

    # 2. Call /api/analyze with mocked Gemini
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gemini_analysis_response

    with patch.object(GeminiRAGGenerator, "client", new=mock_client):
        res = client.post("/api/analyze", json={"upload_id": upload_id})
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        assert data["applicant_name"] == "Aarav Sharma"
        assert data["document_reference"] == "SSP/2025-26/REJ-88412"
        assert data["document_status"] == "Provisional Rejection"
        
        # Verify confirmed defect findings
        findings = data["findings"]
        assert len(findings) == 2
        assert findings[0]["category"] == "confirmed_defect"
        assert findings[0]["code"] == "R-03"
        assert "Notary Public" in findings[0]["what_was_submitted"]
        assert findings[1]["category"] == "confirmed_defect"
        assert findings[1]["code"] == "R-02"
        assert "Aadhaar" in findings[1]["what_was_submitted"]

        # Verify citations derived strictly from ChromaDB metadata
        citations = data["citations"]
        assert len(citations) >= 1
        for cit in citations:
            assert cit["source"] == "demo_scholarship_policy.pdf"
            assert cit["page"] in (1, 2, 3)
            assert cit["similarity_score"] > 0.0


# =============================================================================
# 3. Unrelated Document & Insufficient Evidence
# =============================================================================

def test_analyze_unrelated_document_produces_insufficient_evidence(client):
    """Verify unrelated documents do not invent false defects and output insufficient_evidence."""
    upload_res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"Recipe for sourdough bread using flour, water, and salt."), "recipe.txt")},
        content_type="multipart/form-data"
    )
    upload_id = upload_res.get_json()["upload_id"]

    mock_unrelated_response = MagicMock()
    mock_unrelated_response.text = json.dumps({
        "applicant_name": None,
        "document_reference": None,
        "document_status": "Non-Scholarship Document",
        "summary": "This document does not contain scholarship application details or rejection notices.",
        "findings": [
            {
                "category": "insufficient_evidence",
                "code": None,
                "title": "Unrelated Document Content",
                "what_was_submitted": "Text contains a culinary recipe.",
                "policy_requirement": "Scholarship redressal requires a valid scholarship notice or marksheet.",
                "remediation_steps": "Please upload an official scholarship document.",
                "supporting_evidence_needed": None
            }
        ],
        "evidence_gaps": ["No scholarship application information detected."]
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_unrelated_response

    with patch.object(GeminiRAGGenerator, "client", new=mock_client):
        res = client.post("/api/analyze", json={"upload_id": upload_id})
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        assert data["applicant_name"] is None
        assert len(data["findings"]) == 1
        assert data["findings"][0]["category"] == "insufficient_evidence"


# =============================================================================
# 4. Blank or Empty Document Handling
# =============================================================================

def test_analyze_blank_or_empty_document(client):
    """Verify blank or empty text returns a safe insufficiency finding without calling Gemini."""
    upload_res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"   \n  \t   "), "blank.txt")},
        content_type="multipart/form-data"
    )
    upload_id = upload_res.get_json()["upload_id"]

    res = client.post("/api/analyze", json={"upload_id": upload_id})
    assert res.status_code == 200
    data = res.get_json()

    assert data["success"] is True
    assert data["document_status"] == "Blank Document"
    assert len(data["findings"]) == 1
    assert data["findings"][0]["category"] == "insufficient_evidence"
    assert "blank" in data["summary"].lower()


# =============================================================================
# 5. Malformed JSON Rejection Without Regex Repair
# =============================================================================

def test_analyze_malformed_json_returns_safe_failure_without_regex_repair(client):
    """Verify that non-JSON model output strictly fails and returns 502 with safe guidance."""
    upload_res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"Notice: Applicant rejected."), "notice.txt")},
        content_type="multipart/form-data"
    )
    upload_id = upload_res.get_json()["upload_id"]

    # Model returns conversational markdown instead of required JSON schema
    mock_malformed = MagicMock()
    mock_malformed.text = "Here is my analysis: 1. You were rejected because of your income certificate."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_malformed

    with patch.object(GeminiRAGGenerator, "client", new=mock_client):
        res = client.post("/api/analyze", json={"upload_id": upload_id})
        assert res.status_code == 502
        data = res.get_json()

        assert data["success"] is False
        assert data["error"] == "MALFORMED_ANALYSIS_OUTPUT"
        assert "temporarily unavailable" in data["message"]
        assert len(data["findings"]) == 0


# =============================================================================
# 6. Unsupported Claims Flagged by Grounding Audit & Citation Isolation
# =============================================================================

def test_unsupported_claims_flagged_by_grounding_audit_and_citations_isolated(client):
    """Verify ungrounded deadlines or outside portals trigger review_required while citations remain isolated."""
    upload_res = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"Applicant notice for Aarav Sharma."), "notice.txt")},
        content_type="multipart/form-data"
    )
    upload_id = upload_res.get_json()["upload_id"]

    # Model attempts to hallucinate December 31, 2025 and National Scholarship Portal
    mock_hallucinating_response = MagicMock()
    mock_hallucinating_response.text = json.dumps({
        "applicant_name": "Aarav Sharma",
        "document_reference": None,
        "document_status": "Provisional Rejection",
        "summary": "Application flagged.",
        "findings": [
            {
                "category": "confirmed_defect",
                "code": "R-03",
                "title": "Invalid Income Certificate",
                "what_was_submitted": "Notary certificate.",
                "policy_requirement": "Section 4 guidelines.",
                "remediation_steps": "Submit before December 31, 2025 via the National Scholarship Portal.",
                "supporting_evidence_needed": None
            }
        ],
        "evidence_gaps": []
    })

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_hallucinating_response

    with patch.object(GeminiRAGGenerator, "client", new=mock_client):
        res = client.post("/api/analyze", json={"upload_id": upload_id})
        assert res.status_code == 200
        data = res.get_json()

        assert data["success"] is True
        audit = data["grounding_audit"]
        assert audit["is_grounded"] is False
        assert audit["review_required"] is True
        assert any("December 31, 2025" in w for w in audit["warnings"])
        assert any("National Scholarship Portal" in w for w in audit["warnings"])

        # Citations still strictly point to ChromaDB metadata
        for cit in data["citations"]:
            assert cit["source"] == "demo_scholarship_policy.pdf"
            assert "National Scholarship Portal" not in cit["source"]


# =============================================================================
# 7. Upload ID Session Isolation & Cross-Session Substitution Prevention
# =============================================================================

def test_upload_id_session_isolation_and_cross_session_substitution(app_instance):
    """Verify session A's upload_id cannot be accessed by session B or with forged IDs."""
    client_a = app_instance.test_client()
    client_b = app_instance.test_client()

    # Session A uploads document
    res_a = client_a.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"Secret Document for Session A"), "session_a_doc.txt")},
        content_type="multipart/form-data"
    )
    assert res_a.status_code == 200
    upload_id_a = res_a.get_json()["upload_id"]

    # 1a. Session B without session attempts to analyze Session A's upload_id -> 403 Forbidden
    res_b = client_b.post("/api/analyze", json={"upload_id": upload_id_a})
    assert res_b.status_code == 403
    data_b = res_b.get_json()
    assert "access denied" in data_b["message"].lower()

    # 1b. Session B with its own session attempts cross-session substitution of Session A's upload_id -> 404 Not Found
    res_b_upload = client_b.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"Document for Session B"), "session_b_doc.txt")},
        content_type="multipart/form-data"
    )
    assert res_b_upload.status_code == 200
    res_b_substitute = client_b.post("/api/analyze", json={"upload_id": upload_id_a})
    assert res_b_substitute.status_code == 404
    data_substitute = res_b_substitute.get_json()
    assert "access denied" in data_substitute["message"].lower() or "not found" in data_substitute["message"].lower()

    # 2. Session A attempts to use an arbitrary/forged upload_id -> MUST fail 404
    forged_id = str(uuid.uuid4())
    res_forged = client_a.post("/api/analyze", json={"upload_id": forged_id})
    assert res_forged.status_code == 404

    # 3. Session A with empty upload_id -> MUST fail 400
    res_empty = client_a.post("/api/analyze", json={"upload_id": ""})
    assert res_empty.status_code == 400
