"""Tests for NIVARAN Multilingual UI Implementation.

Verifies:
1. Completeness and parity of English, Hindi, and Hinglish i18n dictionaries.
2. Generator prompt construction and strict evidence preservation guidance for en, hi, hinglish.
3. Language parameter handling and pass-through in /api/query and /api/analyze.
4. Strict preservation of original citations, applicant names, defect codes, and reference numbers.
5. GroundingValidator fidelity on multilingual responses (mocked Gemini, 0 live API calls).
6. HTML header language selector presence and data-i18n attribute coverage.
"""
import json
import re
import io
import pytest
from unittest.mock import MagicMock, patch
from app import create_app
from nivaran.generator import GeminiRAGGenerator, GroundingValidator, get_language_prompt_guidance, DocumentAnalysisSchema


@pytest.fixture
def app_instance():
    """Create Flask application instance configured for testing."""
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key-multilingual"
    return app


# =============================================================================
# 1. Dictionary Parity & Strict Evidence Preservation
# =============================================================================

def test_i18n_dictionaries_completeness_and_consistency():
    """Verify static/js/i18n.js defines en, hi, hinglish with complete key parity."""
    with open("static/js/i18n.js", "r", encoding="utf-8") as f:
        content = f.read()

    # Extract JSON-like window.I18N object
    match = re.search(r'window\.I18N\s*=\s*(\{[\s\S]+\});\s*$', content)
    assert match, "window.I18N declaration not found in static/js/i18n.js"

    js_text = match.group(1)
    assert "en:" in js_text or '"en":' in js_text
    assert "hi:" in js_text or '"hi":' in js_text
    assert "hinglish:" in js_text or '"hinglish":' in js_text

    # Extract exact language dictionary sections
    en_start = js_text.find("en:") if "en:" in js_text else js_text.find('"en":')
    hi_start = js_text.find("hi:") if "hi:" in js_text else js_text.find('"hi":')
    hinglish_start = js_text.find("hinglish:") if "hinglish:" in js_text else js_text.find('"hinglish":')

    assert en_start != -1 and hi_start != -1 and hinglish_start != -1

    en_section = js_text[en_start:hi_start]
    hi_section = js_text[hi_start:hinglish_start]
    hinglish_section = js_text[hinglish_start:]

    # Verify critical evidence preserved in both Hindi and Hinglish
    for lang, section in [("hi", hi_section), ("hinglish", hinglish_section)]:
        assert "SSP/2025-26/REJ-88412" in section, f"Ref number missing in {lang}"
        assert "R-03" in section and "R-02" in section, f"Defect codes missing in {lang}"
        assert "November 15, 2025" in section, f"Deadline missing in {lang}"
        assert "Annexure-IV" in section, f"Annexure-IV missing in {lang}"
        assert "Tehsildar" in section, f"Tehsildar missing in {lang}"
        assert "Aarav Sharma" in section, f"Applicant name missing in {lang}"


# =============================================================================
# 2. Generator Prompt Guidance Verification
# =============================================================================

def test_language_prompt_guidance_for_all_locales():
    """Verify prompt guidance enforces correct language and strict evidence preservation."""
    # English
    en_guidance = get_language_prompt_guidance("en")
    assert "ENGLISH" in en_guidance
    assert "student-friendly English" in en_guidance

    # Hindi
    hi_guidance = get_language_prompt_guidance("hi")
    assert "HINDI" in hi_guidance
    assert "Devanagari" in hi_guidance
    assert "CRITICAL EVIDENCE PRESERVATION" in hi_guidance
    assert "NEVER translate, transliterate, or alter" in hi_guidance
    assert "Aarav Sharma" in hi_guidance
    assert "Income Certificate" in hi_guidance
    assert "R-01" in hi_guidance and "R-05" in hi_guidance
    assert "State Grievance Portal" in hi_guidance
    assert "Tehsildar" in hi_guidance
    assert "November 15, 2025" in hi_guidance

    # Hinglish
    hinglish_guidance = get_language_prompt_guidance("hinglish")
    assert "HINGLISH" in hinglish_guidance
    assert "Roman script" in hinglish_guidance
    assert "conversational Hinglish" in hinglish_guidance
    assert "CRITICAL EVIDENCE PRESERVATION" in hinglish_guidance
    assert "Aarav Sharma" in hinglish_guidance
    assert "R-01" in hinglish_guidance and "R-05" in hinglish_guidance

    # Default / fallback on unknown language
    fallback_guidance = get_language_prompt_guidance("french")
    assert "ENGLISH" in fallback_guidance


# =============================================================================
# 3. Ask NIVARAN Query Endpoint with Language Selection
# =============================================================================

def test_api_query_with_hindi_selection_mocked(app_instance):
    """Test POST /api/query with language='hi' returning grounded Hindi response."""
    client = app_instance.test_client()

    hindi_answer = (
        "आपका छात्रवृत्ति आवेदन Defect Code R-03 और R-02 के कारण अस्वीकार किया गया है। "
        "सुधार के लिए Tehsildar से FY 2024-2025 का वैध Income Certificate प्राप्त करें और "
        "Rs. 50 stamp paper पर Annexure-IV Name Discrepancy Affidavit आधिकारिक Gazette notification "
        "के साथ State Grievance Portal पर November 15, 2025 से पहले जमा करें।"
    )

    mock_response = MagicMock()
    mock_response.text = hindi_answer

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        # Execute query with language='hi'
        res = client.post("/api/query", json={
            "query": "Why was Aarav Sharma's fictional scholarship application rejected, and what should he correct?",
            "applicant_document_text": "Applicant: Aarav Sharma. Defect Codes: R-03, R-02.",
            "language": "hi"
        })

        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["language"] == "hi"
        assert "R-03" in data["answer"]
        assert "R-02" in data["answer"]
        assert "November 15, 2025" in data["answer"]

        # Ensure citations are present and untouched
        assert len(data["citations"]) > 0
        assert data["citations"][0]["source"] == "demo_scholarship_policy.pdf"

        # Check that Gemini was called with Hindi prompt guidance
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args[1]["contents"] if "contents" in call_args[1] else call_args[0][1]
        assert "LANGUAGE & SCRIPT INSTRUCTION (HINDI)" in prompt_sent


def test_api_query_with_hinglish_selection_mocked(app_instance):
    """Test POST /api/query with language='hinglish' returning grounded Hinglish response."""
    client = app_instance.test_client()

    hinglish_answer = (
        "Aapka scholarship application Defect Code R-03 aur R-02 ke chalte reject hua hai. "
        "Aapko Tehsildar se FY 2024-2025 ka digital Income Certificate lena hoga aur "
        "Rs. 50 stamp paper par Annexure-IV Name Discrepancy Affidavit official Gazette notification "
        "ke saath State Grievance Portal par November 15, 2025 tak upload karna hoga."
    )

    mock_response = MagicMock()
    mock_response.text = hinglish_answer

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        res = client.post("/api/query", json={
            "query": "Why was my application rejected and what to fix?",
            "applicant_document_text": "Applicant: Aarav Sharma. Defect Codes: R-03, R-02.",
            "language": "hinglish"
        })

        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["language"] == "hinglish"
        assert "Aapka scholarship application" in data["answer"]
        assert "R-03" in data["answer"]

        # Check that Gemini was called with Hinglish prompt guidance
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args[1]["contents"] if "contents" in call_args[1] else call_args[0][1]
        assert "LANGUAGE & SCRIPT INSTRUCTION (HINGLISH)" in prompt_sent


# =============================================================================
# 4. Document Analysis Endpoint with Language Selection
# =============================================================================

def test_api_analyze_with_hindi_selection_mocked(app_instance):
    """Test POST /api/analyze with language='hi' returning structured schema in Hindi."""
    client = app_instance.test_client()

    # 1. Upload sample notice
    upload_res = client.post(
        "/api/upload",
        data={
            "file": (
                io.BytesIO(b"STATE SCHOLARSHIP PORTAL\nApplicant: Aarav Sharma\nRef: SSP/2025-26/REJ-88412\nDefect Code: R-03"),
                "fictional_notice.txt"
            )
        },
        content_type="multipart/form-data"
    )
    assert upload_res.status_code == 200
    upload_id = upload_res.get_json()["upload_id"]

    # 2. Mock Gemini structured JSON response in Hindi
    hindi_schema_data = {
        "applicant_name": "Aarav Sharma",
        "document_reference": "SSP/2025-26/REJ-88412",
        "document_status": "Provisional Rejection",
        "summary": "आपके नोटिस में 1 पुष्ट दोष पाया गया है। इसे November 15, 2025 से पहले ठीक किया जा सकता है।",
        "findings": [
            {
                "category": "confirmed_defect",
                "code": "R-03",
                "title": "अमान्य आय प्रमाण पत्र जमा किया गया",
                "what_was_submitted": "नोटरी द्वारा जारी पुराना आय प्रमाण पत्र",
                "policy_requirement": "Tehsildar द्वारा FY 2024-2025 के लिए QR कोड सहित जारी होना चाहिए",
                "remediation_steps": "Tehsildar से नया डिजिटल प्रमाण पत्र प्राप्त कर State Grievance Portal पर जमा करें",
                "supporting_evidence_needed": None
            }
        ],
        "evidence_gaps": []
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(hindi_schema_data)

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        res = client.post("/api/analyze", json={
            "upload_id": upload_id,
            "language": "hi"
        })

        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["language"] == "hi"
        # Original evidence preserved
        assert data["applicant_name"] == "Aarav Sharma"
        assert data["document_reference"] == "SSP/2025-26/REJ-88412"
        assert data["findings"][0]["code"] == "R-03"
        assert data["findings"][0]["category"] == "confirmed_defect"
        # Hindi text in title & summary
        assert "आय प्रमाण पत्र" in data["findings"][0]["title"]
        assert "पुष्ट दोष" in data["summary"]
        # Citations strictly preserved from ChromaDB
        assert len(data["citations"]) > 0
        assert data["citations"][0]["source"] == "demo_scholarship_policy.pdf"


# =============================================================================
# 16. Homepage Redesign & Trust Verification
# =============================================================================

def test_homepage_redesign_clean_student_civic_interface():
    """Verify that the homepage is clean, student-centric, free of promotional jargon, and has honest privacy notices."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Primary Headline & Clear Purpose
    assert "Your scholarship notice is confusing. Let&#39;s make it clear." in html or "Your scholarship notice is confusing. Let's make it clear." in html
    assert "Upload your rejection or defect notice." in html
    assert "NIVARAN will explain the document issues" in html

    # 2. Promotional and editorial AI slop must be absent
    removed_slogans = [
        "THE ANTI-BUREAUCRACY PROJECT",
        "ALL-INDIA SCHOLAR ASSIST",
        "काग़ज़ी भूलभुलैया (PAPER LABYRINTH)",
        "NON-COMMERCIAL PUBLIC CIVIC RESOURCE",
        "TRANSFORMATION PIPELINE",
        "MODULE 01",
        "RAG DECODER ENGINE",
        "The 5-Step NIVARAN RAG Workflow",
        "ACTION REQUIRED BEFORE NOV 15",
    ]
    for slogan in removed_slogans:
        assert slogan not in html, f"Found removed promotional slogan in HTML: {slogan}"

    # 3. Privacy Honesty: No false claims that data never leaves the computer
    assert "never leaves your computer" not in html
    assert "Privacy Note: Documents are parsed locally in memory and relevant excerpts are sent to Google's Gemini API for policy analysis." in html

    # 4. Single prominent upload section with required interactive elements
    assert 'id="upload-section"' in html
    assert 'id="dropzone"' in html
    assert 'id="file-input"' in html
    assert 'id="browse-btn"' in html
    assert 'id="file-feedback"' in html
    assert 'id="file-name-display"' in html
    assert 'id="process-file-btn"' in html
    assert 'id="upload-status"' in html

    # 5. Three simple direct guidance pathways
    assert "Check my documents" in html
    assert "Explain a rejection remark" in html
    assert "Show me my next steps" in html
    assert "switchScreen('screen-analysis')" in html
    assert "switchScreen('screen-ask')" in html
    assert "switchScreen('screen-steps')" in html

    # 6. Fictional demonstration disclaimer
    assert "Fictional Demonstration Disclaimer:" in html
    assert "fictional demonstration materials" in html



# =============================================================================
# 5. GroundingValidator on Multilingual Answers
# =============================================================================

def test_grounding_validator_catches_invented_deadlines_in_hindi_and_hinglish():
    """Verify GroundingValidator detects ungrounded dates even within Hindi or Hinglish text."""
    context = (
        "Correction deadline is November 15, 2025. "
        "Institutional verification deadline is November 20, 2025."
    )

    # Compliant Hindi answer with valid deadline -> Grounded
    valid_hi = (
        "सुधार की अंतिम तिथि November 15, 2025 है। "
        "Annexure-IV शपथ पत्र Rs. 50 stamp paper पर Gazette notification के साथ जमा करें।"
    )
    audit_valid = GroundingValidator.audit_answer(valid_hi, context_text=context)
    assert audit_valid["is_grounded"] is True
    assert len(audit_valid["invented_deadlines"]) == 0

    # Hallucinated date in Hindi answer -> Flagged
    hallucinated_hi = (
        "आप अपना आवेदन December 25, 2025 तक जमा कर सकते हैं।"
    )
    audit_invalid = GroundingValidator.audit_answer(hallucinated_hi, context_text=context)
    assert audit_invalid["is_grounded"] is False
    assert "December 25, 2025" in audit_invalid["invented_deadlines"]
    assert audit_invalid["review_required"] is True

    # Hallucinated portal in Hinglish answer -> Flagged
    hallucinated_portal_hinglish = (
        "Aapko National Scholarship Portal par jaakar upload karna hoga."
    )
    audit_portal = GroundingValidator.audit_answer(hallucinated_portal_hinglish, context_text=context)
    assert audit_portal["is_grounded"] is False
    assert "National Scholarship Portal" in audit_portal["unsupported_portals"]


# =============================================================================
# 6. HTML Template & Language Selector UI Elements
# =============================================================================

def test_html_contains_language_selector_and_i18n_script(app_instance):
    """Verify HTML template has language selector dropdown, options, and script tag."""
    client = app_instance.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. Header Language selector dropdown present
    assert 'id="language-selector"' in html
    assert '<option value="en">English</option>' in html
    assert '<option value="hi">हिंदी (Hindi)</option>' in html
    assert '<option value="hinglish">Hinglish</option>' in html

    # 2. i18n script loaded
    assert 'src="/static/js/i18n.js"' in html
    assert 'src="/static/js/app.js"' in html

    # 3. Key data-i18n tags present across all four screens
    assert 'data-i18n="nav.home"' in html
    assert 'data-i18n="nav.analysis"' in html
    assert 'data-i18n="nav.ask"' in html
    assert 'data-i18n="nav.steps"' in html
    assert 'data-i18n="home.title"' in html
    assert 'data-i18n="analysis.title"' in html
    assert 'data-i18n="ask.title"' in html
    assert 'data-i18n="steps.title"' in html
    assert 'data-i18n-placeholder="ask.input_placeholder"' in html


# =============================================================================
# 7. Language Sanitization & Fallback
# =============================================================================

def test_language_normalization_and_defaults(app_instance):
    """Verify case insensitivity and safe fallback to 'en' for unexpected inputs."""
    client = app_instance.test_client()

    mock_response = MagicMock()
    mock_response.text = "Answer in English."

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        # Case 1: Uppercase "HI"
        res_hi = client.post("/api/query", json={"query": "test query", "language": "HI"})
        assert res_hi.status_code == 200
        assert res_hi.get_json()["language"] == "hi"

        # Case 2: "hi-en" -> maps to hinglish
        res_hing = client.post("/api/query", json={"query": "test query", "language": "hi-en"})
        assert res_hing.status_code == 200
        assert res_hing.get_json()["language"] == "hinglish"

        # Case 3: Invalid / unrecognized language -> defaults safely to "en"
        res_unknown = client.post("/api/query", json={"query": "test query", "language": "es"})
        assert res_unknown.status_code == 200
        assert res_unknown.get_json()["language"] == "en"


# =============================================================================
# 8. All HTML Template i18n Keys Exist in All Dictionaries
# =============================================================================

def test_all_html_data_i18n_keys_present_in_all_locales():
    """Verify that every single data-i18n key in index.html is defined in en, hi, and hinglish."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    with open("static/js/i18n.js", "r", encoding="utf-8") as f:
        js_content = f.read()

    # Find all data-i18n keys
    data_keys = set(re.findall(r'data-i18n=["\']([^"\']+)["\']', html))
    placeholder_keys = set(re.findall(r'data-i18n-placeholder=["\']([^"\']+)["\']', html))
    all_keys = data_keys.union(placeholder_keys)

    assert len(all_keys) > 20, "Expected at least 20 unique i18n keys in template"

    # Separate dictionaries
    match = re.search(r'window\.I18N\s*=\s*(\{[\s\S]+\});\s*$', js_content)
    assert match, "window.I18N declaration not found in static/js/i18n.js"
    js_text = match.group(1)

    en_start = js_text.find("en:") if "en:" in js_text else js_text.find('"en":')
    hi_start = js_text.find("hi:") if "hi:" in js_text else js_text.find('"hi":')
    hinglish_start = js_text.find("hinglish:") if "hinglish:" in js_text else js_text.find('"hinglish":')

    en_dict_text = js_text[en_start:hi_start]
    hi_dict_text = js_text[hi_start:hinglish_start]
    hinglish_dict_text = js_text[hinglish_start:]

    for key in all_keys:
        quoted_key = f'"{key}":'
        assert quoted_key in en_dict_text, f"Key '{key}' missing from English (en) dictionary"
        assert quoted_key in hi_dict_text, f"Key '{key}' missing from Hindi (hi) dictionary"
        assert quoted_key in hinglish_dict_text, f"Key '{key}' missing from Hinglish (hinglish) dictionary"


# =============================================================================
# 9. Client-Side Safe Fallback in app.js
# =============================================================================

def test_app_js_safe_fallback_implementation():
    """Verify static/js/app.js implements safe dictionary lookup without exposing raw keys."""
    with open("static/js/app.js", "r", encoding="utf-8") as f:
        app_js = f.read()

    # 1. Verify warning is logged on missing key
    assert "console.warn" in app_js
    assert "Missing translation key" in app_js

    # 2. Verify fallback to English dictionary before empty string
    assert 'window.I18N["en"]' in app_js or "window.I18N['en']" in app_js

    # 3. Verify it does not return raw 'key' on missing key
    # Old faulty line was: 'return dict[key] !== undefined ? dict[key] : fallback || key;'
    assert "fallback || key" not in app_js, "app.js should not return raw key on missing translation"


# =============================================================================
# 10. Document Analysis Generator Language Fidelity & Mandatory Remediation
# =============================================================================

def test_analyze_document_defects_language_fidelity_and_remediation_completeness():
    """Verify analyze_document_defects mandates Hindi language fidelity and preserves all remediation requirements."""
    generator = GeminiRAGGenerator(api_key="mock-test-key")

    # Verify analyze_document_defects constructs prompt with instruction 7
    mock_response = MagicMock()
    hindi_analysis = {
        "applicant_name": "Aarav Sharma",
        "document_reference": "SSP/2025-26/REJ-88412",
        "document_status": "Provisional Rejection",
        "summary": "आपके आवेदन में 2 प्रमुख कमियाँ हैं।",
        "findings": [
            {
                "category": "confirmed_defect",
                "code": "R-03",
                "title": "अमान्य या पुराना आय प्रमाण पत्र (Invalid Income Certificate)",
                "what_was_submitted": "सक्षम प्राधिकारी द्वारा जारी नहीं किया गया पुराना प्रमाण पत्र",
                "policy_requirement": "Tehsildar या SDM द्वारा FY 2024-2025 के लिए QR कोड/बारकोड सहित डिजिटल प्रमाण पत्र होना चाहिए",
                "remediation_steps": "Tehsildar से FY 2024-2025 का डिजिटल आय प्रमाण पत्र प्राप्त करें और State Grievance Portal पर 'Resubmit Defective Application' टैब के तहत November 15, 2025 से पहले अपलोड करें।",
                "supporting_evidence_needed": "Tehsildar द्वारा जारी FY 2024-2025 का डिजिटल प्रमाण पत्र"
            },
            {
                "category": "confirmed_defect",
                "code": "R-02",
                "title": "नाम में विसंगति (Name Discrepancy)",
                "what_was_submitted": "आधार और अंकतालिका में नाम का बेमेल होना",
                "policy_requirement": "Rs. 50 stamp paper पर Annexure-IV शपथ पत्र और आधिकारिक Gazette notification या School Bonafide Certificate अनिवार्य है",
                "remediation_steps": "Rs. 50 stamp paper पर Annexure-IV शपथ पत्र बनवाएं, साथ में Gazette notification या School Bonafide लगाएं, State Grievance Portal पर ग्रीवेंस स्पष्टीकरण दर्ज करें, और College Nodal Officer को November 20, 2025 से पहले सूचित करें।",
                "supporting_evidence_needed": "Annexure-IV शपथ पत्र और Gazette Notification"
            }
        ],
        "evidence_gaps": []
    }
    mock_response.text = json.dumps(hindi_analysis)

    with patch.object(generator, "is_configured", return_value=True):
        with patch.object(generator.client.models, "generate_content", return_value=mock_response) as mock_gen:
            res = generator.analyze_document_defects(
                document_text="Applicant: Aarav Sharma. Defect Code: R-03, R-02.",
                context_text="Policy Clause 4.2: Income Certificate must be for FY 2024-2025 from Tehsildar with QR code. Clause 5.1: Name discrepancy requires Annexure-IV on Rs. 50 stamp paper with Gazette notification or School Bonafide. Deadline: November 15, 2025 on State Grievance Portal under Resubmit Defective Application. Nodal Officer deadline November 20, 2025.",
                language="hi"
            )

            # Check prompt contents
            call_kwargs = mock_gen.call_args[1]
            contents = call_kwargs["contents"]
            assert "7. LANGUAGE FIDELITY:" in contents
            assert "HINDI" in contents
            assert "LANGUAGE & SCRIPT INSTRUCTION (HINDI)" in contents

            # Verify response schema adherence
            assert res["success"] is True
            assert res["language"] == "hi"
            assert res["applicant_name"] == "Aarav Sharma"
            assert res["document_reference"] == "SSP/2025-26/REJ-88412"

            # Verify all 7 mandatory remediation prerequisites are present across the findings
            all_remediation_text = " ".join([
                f["remediation_steps"] + " " + f["policy_requirement"] for f in res["findings"]
            ])

            assert "Annexure-IV" in all_remediation_text
            assert "50" in all_remediation_text
            assert "Gazette" in all_remediation_text or "Bonafide" in all_remediation_text
            assert "FY 2024-2025" in all_remediation_text or "2024-2025" in all_remediation_text
            assert "Tehsildar" in all_remediation_text or "SDM" in all_remediation_text
            assert "QR" in all_remediation_text or "बारकोड" in all_remediation_text
            assert "State Grievance Portal" in all_remediation_text
            assert "Resubmit Defective Application" in all_remediation_text
            assert "November 15, 2025" in all_remediation_text
            assert "November 20, 2025" in all_remediation_text


# =============================================================================
# 11. Unsupported Reassurance Removal & Audit Check
# =============================================================================

def test_unsupported_reassurance_removed_from_all_locales():
    """Verify speculative reassurance is removed from all summaries and caught by validator."""
    with open("static/js/i18n.js", "r", encoding="utf-8") as f:
        js_content = f.read()

    # Verify speculative reassurance is absent from all dictionaries
    reassurance_phrases = [
        "Don't worry",
        "neither requires re-applying",
        "re-applying from scratch",
        "चिंता न करें",
        "नया आवेदन करने की आवश्यकता नहीं",
        "Fikr na karein",
        "shuru se dobara apply karne ki zaroorat nahi"
    ]
    for phrase in reassurance_phrases:
        assert phrase not in js_content, f"Unsupported reassurance phrase '{phrase}' still present in i18n.js"

    # Verify GroundingValidator detects invented reassurance
    context = "Section 7: Rectifications must be submitted via Resubmit Defective Application tab."
    unsupported_ans = (
        "Don't worry, neither requires re-applying from scratch. "
        "Just upload FY 2024-2025 income certificate."
    )
    audit = GroundingValidator.audit_answer(unsupported_ans, context_text=context)
    assert audit["review_required"] is True
    assert any("Unsupported Reassurance" in w for w in audit["warnings"])

    # Hindi unsupported reassurance detected
    unsupported_hi = (
        "चिंता न करें, आपको नया आवेदन करने की आवश्यकता नहीं है।"
    )
    audit_hi = GroundingValidator.audit_answer(unsupported_hi, context_text=context)
    assert audit_hi["review_required"] is True
    assert any("Unsupported Reassurance" in w for w in audit_hi["warnings"])


# =============================================================================
# 12. Historical Date Presentation & Date Safety Warnings
# =============================================================================

def test_historical_deadline_presentation_and_date_warnings(app_instance):
    """Verify past November 2025 dates are labeled as historical and warnings are present."""
    client = app_instance.test_client()
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)

    # 1. HTML contains date warning elements on Analysis and Steps screens
    assert 'id="analysis-date-warning"' in html
    assert 'data-i18n="analysis.date_warning"' in html
    assert 'id="steps-date-warning"' in html
    assert 'data-i18n="steps.date_warning"' in html

    with open("static/js/i18n.js", "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'window\.I18N\s*=\s*(\{[\s\S]+\});\s*$', content)
    assert match
    js_text = match.group(1)

    en_start = js_text.find("en:") if "en:" in js_text else js_text.find('"en":')
    hi_start = js_text.find("hi:") if "hi:" in js_text else js_text.find('"hi":')
    hinglish_start = js_text.find("hinglish:") if "hinglish:" in js_text else js_text.find('"hinglish":')

    en_section = js_text[en_start:hi_start]
    hi_section = js_text[hi_start:hinglish_start]
    hinglish_section = js_text[hinglish_start:]

    # 2. Date warnings exist in all three locales
    for lang, section in [("en", en_section), ("hi", hi_section), ("hinglish", hinglish_section)]:
        assert '"analysis.date_warning":' in section, f"analysis.date_warning missing in {lang}"
        assert '"steps.date_warning":' in section, f"steps.date_warning missing in {lang}"

    # 3. Warning explicitly mentions historical / fictional demonstration dates in Hindi and Hinglish
    assert "काल्पनिक 2025 प्रदर्शन नीति की पुरानी तिथियाँ" in hi_section
    assert "वर्तमान वास्तविक समय-सीमा नहीं" in hi_section
    assert "historical dates" in hinglish_section
    assert "not current real-world deadlines" in hinglish_section or "current real-world deadlines nahi" in hinglish_section

    # 4. Deadlines are labeled as historical demonstration dates rather than plain future deadlines
    assert "Historical Demonstration Date" in en_section
    assert "ऐतिहासिक प्रदर्शन तिथि" in hi_section
    assert "Historical Demonstration Date" in hinglish_section


# =============================================================================
# 13. Natural Hindi Translations Verification
# =============================================================================

def test_natural_hindi_translations():
    """Verify natural Hindi terms replace awkward bureaucratic or English terms."""
    with open("static/js/i18n.js", "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'window\.I18N\s*=\s*(\{[\s\S]+\});\s*$', content)
    js_text = match.group(1)
    hi_start = js_text.find("hi:") if "hi:" in js_text else js_text.find('"hi":')
    hinglish_start = js_text.find("hinglish:") if "hinglish:" in js_text else js_text.find('"hinglish":')
    hi_section = js_text[hi_start:hinglish_start]

    # Required translations
    assert "वित्तीय वर्ष" in hi_section, "Financial Year should be translated to वित्तीय वर्ष"
    assert "नोटरी पब्लिक" in hi_section, "Notary Public should be translated to नोटरी पब्लिक"
    assert "अस्थायी अस्वीकृति" in hi_section, "Provisional Rejection should be translated to अस्थायी अस्वीकृति"
    assert "नियमों के पालन" in hi_section or "नियमों का पालन" in hi_section, "Compliance should be translated to नियमों का पालन"
    assert "दस्तावेज़ों से जुड़ी छोटी प्रशासनिक कमियाँ" in hi_section, "Clerical compliance issues should be translated"

    # Official terms preserved
    assert "Tehsildar" in hi_section
    assert "SDM" in hi_section
    assert "Annexure-IV" in hi_section
    assert "Rs. 50 stamp paper" in hi_section
    assert "Gazette notification" in hi_section
    assert "School Bonafide" in hi_section
    assert "State Grievance Portal" in hi_section
    assert "Resubmit Defective Application" in hi_section


# =============================================================================
# 14. Remediation Completeness Retained in Hindi
# =============================================================================

def test_remediation_completeness_retained_in_hindi():
    """Verify all 8 mandatory remediation conditions are retained in Hindi checklist & guidance."""
    with open("static/js/i18n.js", "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'window\.I18N\s*=\s*(\{[\s\S]+\});\s*$', content)
    js_text = match.group(1)
    hi_start = js_text.find("hi:") if "hi:" in js_text else js_text.find('"hi":')
    hinglish_start = js_text.find("hinglish:") if "hinglish:" in js_text else js_text.find('"hinglish":')
    hi_section = js_text[hi_start:hinglish_start]

    # 1. Correct financial year
    assert "2024–2025" in hi_section or "2024-2025" in hi_section
    # 2. Authorized issuing officer
    assert "Tehsildar या SDM" in hi_section or "Tehsildar / SDM" in hi_section
    # 3. Digital signature and machine-verifiable QR/barcode
    assert "डिजिटल" in hi_section
    assert "बारकोड या QR कोड" in hi_section or "QR/बारकोड" in hi_section
    # 4. Annexure-IV affidavit on Rs. 50 stamp paper
    assert "Rs. 50 stamp paper पर Annexure-IV Name Discrepancy Affidavit" in hi_section
    # 5. Gazette notification OR School Bonafide
    assert "Gazette notification या School Bonafide" in hi_section
    # 6. Grievance justification
    assert "ग्रीवेंस औचित्य" in hi_section or "ग्रीवेंस स्पष्टीकरण" in hi_section
    # 7. Resubmission procedure
    assert "Resubmit Defective Application" in hi_section
    # 8. College Nodal Officer re-verification
    assert "नोडल अधिकारी" in hi_section


# =============================================================================
# 15. English Mode Unaffected
# =============================================================================

def test_english_mode_unaffected(app_instance):
    """Verify English mode retains full functionality, citations, and date safety without degradation."""
    client = app_instance.test_client()

    mock_response = MagicMock()
    mock_response.text = (
        "Your application was flagged for Defect Codes R-03 and R-02. "
        "Obtain a digital income certificate for FY 2024-2025 from the Tehsildar with a QR code, "
        "and execute an Annexure-IV Affidavit on Rs. 50 stamp paper with Gazette notification or School Bonafide. "
        "In this 2025 demonstration scenario, submit via the State Grievance Portal under Resubmit Defective Application "
        "before the historical deadline of November 15, 2025."
    )

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = mock_response

        res = client.post("/api/query", json={
            "query": "Why was Aarav Sharma's application rejected?",
            "language": "en"
        })

        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["language"] == "en"
        assert "R-03" in data["answer"]
        assert "R-02" in data["answer"]
        assert "Annexure-IV" in data["answer"]
        assert len(data["citations"]) > 0
        assert data["citations"][0]["source"] == "demo_scholarship_policy.pdf"


# =============================================================================
# 16. Homepage Redesign & Trust Verification
# =============================================================================

def test_homepage_redesign_clean_student_civic_interface():
    """Verify that the homepage is clean, student-centric, free of promotional jargon, and has honest privacy notices."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Primary Headline & Clear Purpose
    assert "Your scholarship notice is confusing. Let&#39;s make it clear." in html or "Your scholarship notice is confusing. Let's make it clear." in html
    assert "Upload your rejection or defect notice." in html
    assert "NIVARAN will explain the document issues" in html

    # 2. Promotional and editorial AI slop must be absent
    removed_slogans = [
        "THE ANTI-BUREAUCRACY PROJECT",
        "ALL-INDIA SCHOLAR ASSIST",
        "काग़ज़ी भूलभुलैया (PAPER LABYRINTH)",
        "NON-COMMERCIAL PUBLIC CIVIC RESOURCE",
        "TRANSFORMATION PIPELINE",
        "MODULE 01",
        "RAG DECODER ENGINE",
        "The 5-Step NIVARAN RAG Workflow",
        "ACTION REQUIRED BEFORE NOV 15",
    ]
    for slogan in removed_slogans:
        assert slogan not in html, f"Found removed promotional slogan in HTML: {slogan}"

    # 3. Privacy Honesty: No false claims that data never leaves the computer
    assert "never leaves your computer" not in html
    assert "Privacy Note: Documents are parsed locally in memory and relevant excerpts are sent to Google's Gemini API for policy analysis." in html

    # 4. Single prominent upload section with required interactive elements
    assert 'id="upload-section"' in html
    assert 'id="dropzone"' in html
    assert 'id="file-input"' in html
    assert 'id="browse-btn"' in html
    assert 'id="file-feedback"' in html
    assert 'id="file-name-display"' in html
    assert 'id="process-file-btn"' in html
    assert 'id="upload-status"' in html

    # 5. Three simple direct guidance pathways
    assert "Check my documents" in html
    assert "Explain a rejection remark" in html
    assert "Show me my next steps" in html
    assert "switchScreen('screen-analysis')" in html
    assert "switchScreen('screen-ask')" in html
    assert "switchScreen('screen-steps')" in html

    # 6. Fictional demonstration disclaimer
    assert "Fictional Demonstration Disclaimer:" in html
    assert "fictional demonstration materials" in html

