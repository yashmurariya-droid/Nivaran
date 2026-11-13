"""Gemini Generation Module for NIVARAN RAG.

Utilizes the official Google GenAI SDK to generate grounded responses
based strictly on retrieved knowledge base context, with deterministic
factual-grounding verification and compliance completeness checks.
"""
import re
import json
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from nivaran.utils.logger import get_logger, mask_sensitive_value

logger = get_logger("generator")


# =============================================================================
# Structured Document Analysis Schemas (Pydantic v2)
# =============================================================================

class DefectFinding(BaseModel):
    category: Literal["confirmed_defect", "possible_issue", "insufficient_evidence"] = Field(
        ...,
        description="Classification: 'confirmed_defect' (directly stated in notice), 'possible_issue' (discrepancy or warning), or 'insufficient_evidence' (lacks verifiable policy data)"
    )
    code: Optional[str] = Field(
        None,
        description="Defect code if explicitly cited in document or policy (e.g. 'R-03', 'R-02'), otherwise null"
    )
    title: str = Field(..., description="Short, plain-language description of the issue in the target requested language")
    what_was_submitted: str = Field(..., description="Factual description of what was submitted or flagged, in the target requested language")
    policy_requirement: str = Field(..., description="Factual policy requirement from retrieved evidence, in the target requested language")
    remediation_steps: str = Field(..., description="Complete step-by-step corrective actions grounded in policy, in the target requested language")
    supporting_evidence_needed: Optional[str] = Field(
        None,
        description="Mandatory accompanying documents (e.g. Gazette notification or School Bonafide, Rs. 50 stamp paper), in the target requested language, or null"
    )


class DocumentAnalysisSchema(BaseModel):
    applicant_name: Optional[str] = Field(None, description="Exact applicant name if stated in the document, otherwise null")
    document_reference: Optional[str] = Field(None, description="Exact document reference number if stated, otherwise null")
    document_status: Optional[str] = Field(None, description="Exact status stated in document (e.g. 'Provisional Rejection'), otherwise 'Unknown'")
    summary: str = Field(..., description="Empathetic, clear summary of document status and findings in plain words")
    findings: List[DefectFinding] = Field(default_factory=list, description="Itemized list of findings")
    evidence_gaps: List[str] = Field(default_factory=list, description="Specific areas where policy or document evidence is incomplete")


SYSTEM_PROMPT = """You are NIVARAN, an empathetic, clear AI scholarship assistant specializing in application compliance and grievance redressal.

GROUNDING & FACTUAL FIDELITY INSTRUCTIONS:
1. STRICT RETRIEVED CONTEXT ADHERENCE:
   - Base all statements, rules, requirements, and remediation actions EXCLUSIVELY on the provided retrieved policy evidence and applicant document.
   - All guidelines and rejection notices are from fictional demonstration materials. Treat them strictly as such.
   - Do NOT assume, extrapolate, or inject real-world external procedures (e.g., standard Indian administrative conventions, RTIs, court appeals, or notary rules) unless they explicitly appear in the provided text.

2. PROCEDURAL & REMEDIATION COMPLETENESS:
   - When explaining corrective steps or remediation actions, preserve ALL mandatory compound conditions without summarizing them away or omitting technical prerequisites.
   - NEVER drop accompanying/supporting documents (e.g., Gazette notification or School Bonafide).
   - NEVER omit stamp paper denominations (e.g., Rs. 50 stamp paper).
   - NEVER omit technical verification checks (e.g., machine-verifiable barcode or QR code on digital certificates).
   - NEVER omit specific portal navigation targets (e.g., the 'Resubmit Defective Application' tab) or mandatory form fields (e.g., entering grievance justification).

3. PROHIBITION OF INVENTED PROCEDURES & DEADLINES:
   - NEVER invent or synthesize deadlines, grace periods (e.g., "within 15 days", "within 30 days"), authorities, portals, stamp fees, or document names.
   - If a specific deadline, issuing office, or resubmission path is NOT in the retrieved evidence, explicitly state that the provided policy text does not specify it.

4. SEPARATION OF APPLICANT FACTS & POLICY RULES:
   - Clearly separate applicant-specific findings (such as applicant name, application ID, assigned defect codes, or submitted document defects) from general policy eligibility and validity rules.

5. EXPLICIT UNCERTAINTY & INSUFFICIENT EVIDENCE:
   - If the retrieved context is silent, insufficient, or ambiguous regarding any question, clearly state: "The provided policy evidence does not contain information regarding [topic]." Never guess or attempt to sound helpful by inventing plausible rules.

6. READABILITY WITHOUT TRUNCATION:
   - Write in simple, clear, student-friendly language with organized headings and bullet points. Never sacrifice factual completeness or drop prerequisite steps for brevity.

7. HISTORICAL DEMONSTRATION DATES & DEADLINE SAFETY:
   - All dates in the fictional policy (including November 15, 2025 and November 20, 2025) are historical dates from a past fictional 2025 demonstration scenario.
   - NEVER present them as current, active, or upcoming real-world deadlines. Clearly label them as historical demonstration dates from the fictional 2025 scenario.
   - NEVER invent replacement dates.

8. PROHIBITION OF UNSUPPORTED REASSURANCE:
   - NEVER assert or promise that an applicant does not need to submit a new application, or that approval is guaranteed, unless explicitly stated in the retrieved evidence. State only objective, factual findings and official resubmission procedures.
"""


def get_language_prompt_guidance(language: str = "en") -> str:
    """Return explicit prompt guidance for language selection, completeness, and evidence preservation."""
    clean_lang = (language or "en").lower().strip()

    if clean_lang in ("hi", "hindi"):
        return (
            "### LANGUAGE & SCRIPT INSTRUCTION (HINDI):\n"
            "- You MUST compose ALL explanatory and instructional content in simple, natural, student-friendly Hindi (हिन्दी) using Devanagari script.\n"
            "- Avoid archaic Sanskritized bureaucratic vocabulary; use everyday words that students and families easily understand.\n"
            "- NATURAL HINDI TERMINOLOGY:\n"
            "  Translate ordinary explanatory English terms into simple Hindi:\n"
            "  - Financial Year → वित्तीय वर्ष\n"
            "  - Notary Public → नोटरी पब्लिक (जहाँ आधिकारिक पदनाम आवश्यक हो)\n"
            "  - Provisional Rejection → अस्थायी अस्वीकृति\n"
            "  - Compliance → नियमों का पालन\n"
            "  - Clerical compliance issues → दस्तावेज़ों से जुड़ी छोटी प्रशासनिक कमियाँ\n"
            "- HISTORICAL DEMONSTRATION DATES (DATE SAFETY):\n"
            "  November 15, 2025 और November 20, 2025 काल्पनिक 2025 प्रदर्शन नीति की पुरानी (historical) तिथियाँ हैं। इन्हें कभी भी वर्तमान या आगामी समय-सीमा के रूप में प्रस्तुत न करें। हमेशा स्पष्ट रूप से दर्शाएँ कि ये काल्पनिक 2025 प्रदर्शन परिदृश्य की ऐतिहासिक तिथियाँ हैं (उदा. 'ऐतिहासिक प्रदर्शन तिथि: November 15, 2025')। कोई भी नई तारीख न जोड़ें।\n"
            "- PROHIBITION OF UNSUPPORTED REASSURANCE:\n"
            "  यह दावा कभी न करें कि आवेदक को नया आवेदन करने की आवश्यकता नहीं है। केवल नीति साक्ष्य के आधार पर तथ्यात्मक कमियों और 'Resubmit Defective Application' पोर्टल प्रक्रिया का उल्लेख करें।\n"
            "- MANDATORY REQUIREMENT FOR ALL EXPLANATORY FIELDS (DOCUMENT ANALYSIS & Q&A):\n"
            "  Every single explanatory field MUST be generated in Hindi (हिन्दी / Devanagari script):\n"
            "  - summary\n"
            "  - findings[].title (defect title translated into clear Hindi, retaining code e.g. '[Code: R-03]')\n"
            "  - findings[].what_was_submitted (factual description in Hindi)\n"
            "  - findings[].policy_requirement (policy requirement explained in Hindi)\n"
            "  - findings[].remediation_steps (complete step-by-step corrective actions in Hindi)\n"
            "  - findings[].supporting_evidence_needed (required accompanying documents in Hindi if applicable)\n"
            "  - evidence_gaps (in Hindi)\n"
            "  NEVER output title, what_was_submitted, policy_requirement, or remediation_steps in English when Hindi is selected!\n"
            "- REMEDIATION COMPLETENESS IN TRANSLATION (DO NOT SHORTEN OR OMIT PREREQUISITES):\n"
            "  When providing remediation steps in Hindi, preserve ALL mandatory conditions without simplification:\n"
            "  1. Valid digital Income Certificate issued by Tehsildar or SDM for FY 2024-2025 with machine-verifiable barcode or QR code\n"
            "  2. Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper\n"
            "  3. Mandatory accompanying identity proof: official Gazette notification OR School Bonafide\n"
            "  4. Resubmission via the State Grievance Portal under the 'Resubmit Defective Application' tab before November 15, 2025\n"
            "  5. Mandatory entry of grievance justification\n"
            "  6. Informing College Scholarship Nodal Officer before November 20, 2025 for institutional re-verification\n"
            "- CRITICAL EVIDENCE PRESERVATION (STRICT RULE):\n"
            "  NEVER translate, transliterate, or alter any of the following:\n"
            "  1. Applicant names (e.g., 'Aarav Sharma', 'Aarav Kumar Sharma')\n"
            "  2. Official document and certificate names (e.g., 'Income Certificate', 'Aadhaar Card', 'Annexure-IV Name Discrepancy Affidavit', 'School Bonafide', 'Gazette notification')\n"
            "  3. Exact reference numbers and dates (e.g., 'SSP/2025-26/REJ-88412', 'APP-2024-8891', 'November 15, 2025', 'November 20, 2025')\n"
            "  4. Defect and rejection codes (e.g., 'R-01', 'R-02', 'R-03', 'R-04', 'R-05')\n"
            "  5. Exact portal names, tab names, and legal values (e.g., 'State Grievance Portal', 'Resubmit Defective Application', 'Tehsildar', 'SDM', 'DRO', 'Rs. 50 stamp paper')\n"
            "  6. Document filenames and citations (e.g., 'demo_scholarship_policy.pdf', 'Section 4', 'Section 6', 'Section 7')\n"
            "  Keep all such names, codes, dates, and terms exactly as written in English; translate ONLY the surrounding explanations into Hindi."
        )
    elif clean_lang in ("hinglish", "hi-en"):
        return (
            "### LANGUAGE & SCRIPT INSTRUCTION (HINGLISH):\n"
            "- You MUST compose ALL explanatory and instructional content in natural, conversational Hinglish using Roman script (Latin alphabet).\n"
            "- Tone must be polite, direct, and conversational (e.g., 'Aapka application reject hua hai kyunki...', 'Aapko FY 2024-2025 ka valid digital income certificate Tehsildar se lena hoga...').\n"
            "- HISTORICAL DEMONSTRATION DATES (DATE SAFETY):\n"
            "  November 15, 2025 aur November 20, 2025 fictional 2025 demonstration scenario ki historical dates hain. Inhe kabhi bhi current ya upcoming deadlines ke roop mein present na karein. Clearly label karein ki yeh fictional demonstration dates hain. Koi new dates invent na karein.\n"
            "- PROHIBITION OF UNSUPPORTED REASSURANCE:\n"
            "  Bina policy evidence ke kabhi reassurance na dein ki naya application submit karne ki zaroorat nahi hai. Sirf policy-supported defects aur 'Resubmit Defective Application' tab ki procedure mention karein.\n"
            "- MANDATORY REQUIREMENT FOR ALL EXPLANATORY FIELDS (DOCUMENT ANALYSIS & Q&A):\n"
            "  Every single explanatory field MUST be generated in conversational Hinglish:\n"
            "  - summary\n"
            "  - findings[].title (defect title in Hinglish, retaining code e.g. '[Code: R-03]')\n"
            "  - findings[].what_was_submitted (in conversational Hinglish)\n"
            "  - findings[].policy_requirement (in conversational Hinglish)\n"
            "  - findings[].remediation_steps (complete step-by-step corrective actions in Hinglish)\n"
            "  - findings[].supporting_evidence_needed (in Hinglish if applicable)\n"
            "  - evidence_gaps (in Hinglish)\n"
            "  NEVER output title, what_was_submitted, policy_requirement, or remediation_steps in pure English when Hinglish is selected!\n"
            "- REMEDIATION COMPLETENESS IN TRANSLATION (DO NOT SHORTEN OR OMIT PREREQUISITES):\n"
            "  When providing remediation steps in Hinglish, preserve ALL mandatory conditions:\n"
            "  1. Valid digital Income Certificate issued by Tehsildar or SDM for FY 2024-2025 with machine-verifiable barcode or QR code\n"
            "  2. Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper\n"
            "  3. Mandatory accompanying identity proof: official Gazette notification OR School Bonafide\n"
            "  4. Resubmission via the State Grievance Portal under the 'Resubmit Defective Application' tab before November 15, 2025\n"
            "  5. Mandatory entry of grievance justification\n"
            "  6. Informing College Scholarship Nodal Officer before November 20, 2025 for institutional re-verification\n"
            "- CRITICAL EVIDENCE PRESERVATION (STRICT RULE):\n"
            "  NEVER alter, distort, or translate any of the following:\n"
            "  1. Applicant names (e.g., 'Aarav Sharma', 'Aarav Kumar Sharma')\n"
            "  2. Official document and certificate names (e.g., 'Income Certificate', 'Aadhaar Card', 'Annexure-IV Name Discrepancy Affidavit', 'School Bonafide', 'Gazette notification')\n"
            "  3. Exact reference numbers and dates (e.g., 'SSP/2025-26/REJ-88412', 'APP-2024-8891', 'November 15, 2025', 'November 20, 2025')\n"
            "  4. Defect and rejection codes (e.g., 'R-01', 'R-02', 'R-03', 'R-04', 'R-05')\n"
            "  5. Exact portal names, tab names, and legal values (e.g., 'State Grievance Portal', 'Resubmit Defective Application', 'Tehsildar', 'SDM', 'DRO', 'Rs. 50 stamp paper')\n"
            "  6. Document filenames and citations (e.g., 'demo_scholarship_policy.pdf', 'Section 4', 'Section 6', 'Section 7')\n"
            "  Keep all such terms intact; write only the explanations and instructions in conversational Hinglish."
        )
    else:
        return (
            "### LANGUAGE INSTRUCTION (ENGLISH):\n"
            "- Respond in simple, clear, student-friendly English.\n"
            "- Keep technical terms, applicant names, document references, and citations accurate and grounded."
        )


class GroundingValidator:
    """Performs deterministic validation and structured checks on generated answers."""

    DATE_PATTERNS = [
        re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:\d{4})?\b', re.IGNORECASE),
        re.compile(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s*(?:\d{4})?\b', re.IGNORECASE),
        re.compile(r'\bwithin\s+\d+\s+days\b', re.IGNORECASE),
    ]

    UNSUPPORTED_PORTAL_PATTERNS = [
        re.compile(r'\bNational Scholarship Portal\b', re.IGNORECASE),
        re.compile(r'\bNSP\b', re.IGNORECASE),
        re.compile(r'\bDigiLocker\b', re.IGNORECASE),
        re.compile(r'\bUGC Portal\b', re.IGNORECASE),
    ]

    UNSUPPORTED_REASSURANCE_PATTERNS = [
        re.compile(r'\b(?:do\s+not|don\'t|neither\s+requires)\s+(?:need\s+to\s+)?(?:submit|apply|re-apply|reapply)\s+(?:(?:a\s+)?(?:new|fresh|whole\s+new)\s+application|from\s+scratch)\b', re.IGNORECASE),
        re.compile(r'\b(?:re-applying|reapplying)\s+from\s+scratch\b', re.IGNORECASE),
        re.compile(r'नया\s+आवेदन\s+करने\s+की\s+(?:कोई\s+)?आवश्यकता\s+नहीं', re.IGNORECASE),
        re.compile(r'\bshuru\s+se\s+dobara\s+apply\s+karne\s+ki\s+zaroorat\s+nahi\b', re.IGNORECASE),
    ]

    UNCERTAINTY_PHRASES = [
        "does not provide",
        "does not contain",
        "not specified",
        "insufficient",
        "not found",
        "no information",
        "cannot be verified",
        "not mention",
        "कोई जानकारी नहीं",
        "उल्लेख नहीं",
        "अपर्याप्त साक्ष्य",
        "सत्यापित नहीं",
        "उपलब्ध नहीं",
        "जानकारी उपलब्ध नहीं",
        "koi information nahi",
        "information available nahi",
        "mention nahi",
        "verify nahi ho sakta",
    ]

    @classmethod
    def audit_answer(
        cls,
        answer: str,
        context_text: str,
        applicant_text: str = ""
    ) -> Dict[str, Any]:
        """Perform deterministic grounding checks against evidence."""
        warnings: List[str] = []
        invented_deadlines: List[str] = []
        unsupported_portals: List[str] = []
        omissions: List[str] = []

        combined_evidence = f"{context_text}\n{applicant_text}".lower()

        # 1. Date and Deadline Verification
        checked_dates = []
        for pattern in cls.DATE_PATTERNS:
            for match in pattern.finditer(answer):
                date_str = match.group(0).strip()
                checked_dates.append(date_str)
                clean_date = re.sub(r'(st|nd|rd|th|,)', '', date_str.lower()).strip()
                month_day_match = re.search(r'([a-z]+)\s+(\d{1,2})', clean_date)
                if month_day_match:
                    m, d = month_day_match.groups()
                    if f"{m} {d}" not in combined_evidence and f"{d} {m}" not in combined_evidence:
                        invented_deadlines.append(date_str)
                        warnings.append(f"Ungrounded deadline or timeframe detected: '{date_str}'")
                elif clean_date not in combined_evidence:
                    invented_deadlines.append(date_str)
                    warnings.append(f"Ungrounded deadline or timeframe detected: '{date_str}'")

        # 2. Unsupported Portals Check
        for portal_pat in cls.UNSUPPORTED_PORTAL_PATTERNS:
            match = portal_pat.search(answer)
            if match:
                portal_name = match.group(0)
                if portal_name.lower() not in combined_evidence:
                    unsupported_portals.append(portal_name)
                    warnings.append(f"Unsupported external portal or platform detected: '{portal_name}'")

        # 3. Remediation Completeness Check: Name Mismatch (R-02)
        if "annexure-iv" in combined_evidence:
            lower_ans = answer.lower()
            if "annexure-iv" in lower_ans or "name mismatch" in lower_ans or "name discrepancy" in lower_ans:
                if "gazette" not in lower_ans and "school bonafide" not in lower_ans:
                    omissions.append("Gazette notification or School Bonafide")
                    warnings.append(
                        "Remediation Incompleteness: Answer discusses Name Discrepancy remediation "
                        "but omits mandatory accompanying identity proof (official Gazette notification or School Bonafide)."
                    )
                if "rs. 50" not in lower_ans and "50 stamp" not in lower_ans and "50/-" not in lower_ans:
                    omissions.append("Rs. 50 stamp paper")
                    warnings.append(
                        "Remediation Incompleteness: Answer discusses Annexure-IV Affidavit without "
                        "specifying the mandatory Rs. 50 stamp paper requirement."
                    )

        # 4. Remediation Completeness Check: Income Certificate (R-03)
        if "barcode" in combined_evidence or "qr code" in combined_evidence:
            lower_ans = answer.lower()
            if "income certificate" in lower_ans and any(w in lower_ans for w in ["fix", "obtain", "apply", "correct", "digital"]):
                if "barcode" not in lower_ans and "qr" not in lower_ans:
                    omissions.append("Machine-verifiable QR/barcode check")
                    warnings.append(
                        "Remediation Notice: Answer discusses income certificate remediation but omits "
                        "the mandatory machine-verifiable barcode or QR code check."
                    )

        # 5. Remediation Completeness Check: Portal Tab & Justification
        if "resubmit defective application" in combined_evidence:
            lower_ans = answer.lower()
            if "upload" in lower_ans or "resubmit" in lower_ans:
                if "resubmit defective application" not in lower_ans:
                    omissions.append("Resubmit Defective Application portal tab")
                    warnings.append(
                        "Workflow Notice: Answer omits the specific portal tab ('Resubmit Defective Application') "
                        "required for resubmission."
                    )
                if "justification" not in lower_ans and "grievance justification" in combined_evidence:
                    omissions.append("Grievance justification entry")
                    warnings.append(
                        "Workflow Notice: Answer omits the requirement to enter grievance justification during resubmission."
                    )

        # 6. Unsupported Reassurance Check
        for reassur_pat in cls.UNSUPPORTED_REASSURANCE_PATTERNS:
            match = reassur_pat.search(answer)
            if match:
                phrase = match.group(0)
                if phrase.lower() not in combined_evidence:
                    warnings.append(
                        f"Unsupported Reassurance: Answer claims applicant does not need to submit a new application ('{phrase}'), which is not explicitly stated in policy evidence."
                    )

        # 7. Insufficient Context Verification
        if not context_text.strip() or "no relevant policy evidence" in context_text.lower():
            lower_ans = answer.lower()
            has_uncertainty = any(phrase in lower_ans for phrase in cls.UNCERTAINTY_PHRASES)
            if not has_uncertainty:
                warnings.append(
                    "Ungrounded Assertion: Context provided no relevant evidence, but response did not "
                    "express uncertainty or state that information was missing."
                )

        is_grounded = len(invented_deadlines) == 0 and len(unsupported_portals) == 0
        review_required = len(warnings) > 0

        return {
            "is_grounded": is_grounded,
            "review_required": review_required,
            "warnings": warnings,
            "invented_deadlines": invented_deadlines,
            "unsupported_portals": unsupported_portals,
            "omissions": omissions,
            "checked_dates": checked_dates,
        }

    @classmethod
    def audit_findings(
        cls,
        findings: List[DefectFinding],
        context_text: str,
        applicant_text: str = ""
    ) -> Dict[str, Any]:
        """Audit structured defect findings for factual grounding and mandatory policy conditions."""
        combined_text = "\n\n".join([
            f"{f.title} {f.what_was_submitted} {f.policy_requirement} {f.remediation_steps} {f.supporting_evidence_needed or ''}"
            for f in findings
        ])
        return cls.audit_answer(
            answer=combined_text,
            context_text=context_text,
            applicant_text=applicant_text
        )


class GeminiRAGGenerator:
    """Manages answer generation with Gemini using strict RAG context and grounding verification."""

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash"):
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model_name
        self._client: Optional[genai.Client] = None

    def is_configured(self) -> bool:
        """Check whether a valid API key is present."""
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    @property
    def client(self) -> genai.Client:
        """Lazily initialize Google GenAI client."""
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file."
            )
        if self._client is None:
            logger.info(
                "Initializing Google GenAI Client with model '%s' (Key: %s)",
                self.model_name,
                mask_sensitive_value(self.api_key),
            )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_rag_answer(
        self,
        query: str,
        context_text: str,
        applicant_document_text: Optional[str] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Generate a grounded answer using retrieved context and applicant's document in the requested language."""
        target_lang = (language or "en").lower().strip()
        if not self.is_configured():
            return {
                "answer": "Gemini API key is not configured. Please add GEMINI_API_KEY to your .env file to enable live AI generation.",
                "model": self.model_name,
                "language": target_lang,
                "success": False,
                "error": "MISSING_API_KEY",
                "grounding_audit": {
                    "is_grounded": False,
                    "review_required": True,
                    "warnings": ["API key missing."],
                    "invented_deadlines": [],
                    "unsupported_portals": [],
                    "omissions": [],
                    "checked_dates": []
                }
            }

        # Build user prompt
        prompt_sections = [
            "### RETRIEVED POLICY EVIDENCE (FICTIONAL DEMONSTRATION RULES):",
            context_text if context_text.strip() else "No relevant policy evidence retrieved.",
        ]

        if applicant_document_text and applicant_document_text.strip():
            prompt_sections.extend([
                "",
                "### APPLICANT REJECTION NOTICE / SUBMITTED DETAILS:",
                applicant_document_text.strip(),
            ])

        prompt_sections.extend([
            "",
            "### QUESTION:",
            query.strip(),
            "",
            "### RESPONSE GUIDANCE:",
            "- Answer in simple, accessible language without omitting any mandatory conditions.",
            "- Explain clearly why the application was flagged or rejected based strictly on the retrieved evidence.",
            "- In the correction checklist, detail EVERY required step: mandatory accompanying proofs (e.g. Gazette notification or School Bonafide), stamp paper values (e.g. Rs. 50), machine-verifiable barcode/QR code verification, specific portal tab names ('Resubmit Defective Application'), and grievance justification entry.",
            "- Quote exact policy deadlines only where supported by evidence. Never invent timeframes or external portals.",
            "- If any asked information is missing from the evidence, explicitly declare it.",
            "",
            get_language_prompt_guidance(target_lang),
        ])

        user_content = "\n".join(prompt_sections)

        try:
            logger.info("Sending generation request to Gemini model '%s' (language: %s)...", self.model_name, target_lang)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,  # Low temperature for factual fidelity
                ),
            )

            generated_text = response.text or "No response generated by model."
            logger.info("Successfully received response from Gemini (%d chars)", len(generated_text))

            # Perform deterministic grounding audit
            audit = GroundingValidator.audit_answer(
                answer=generated_text,
                context_text=context_text,
                applicant_text=applicant_document_text or ""
            )

            if audit["review_required"]:
                logger.warning("Grounding audit flagged items for review: %s", audit["warnings"])
                review_notice = (
                    "\n\n---\n"
                    "⚠️ **NIVARAN Policy Verification Notice:**\n"
                    "The following item(s) require verification against official policy rules:\n"
                    + "\n".join(f"- {w}" for w in audit["warnings"])
                )
                final_answer = generated_text + review_notice
            else:
                final_answer = generated_text

            return {
                "answer": final_answer,
                "raw_answer": generated_text,
                "model": self.model_name,
                "language": target_lang,
                "success": True,
                "error": None,
                "grounding_audit": audit,
            }

        except Exception as e:
            logger.error("Error calling Gemini API: %s", str(e))
            return {
                "answer": f"Error generating answer from Gemini API: {str(e)}",
                "model": self.model_name,
                "language": target_lang,
                "success": False,
                "error": str(e),
                "grounding_audit": {
                    "is_grounded": False,
                    "review_required": True,
                    "warnings": [str(e)],
                    "invented_deadlines": [],
                    "unsupported_portals": [],
                    "omissions": [],
                    "checked_dates": []
                }
            }

    def analyze_document_defects(
        self,
        document_text: str,
        context_text: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Perform structured, grounded document defect analysis returning validated JSON schema in the requested language."""
        target_lang = (language or "en").lower().strip()
        if not self.is_configured():
            return {
                "success": False,
                "error": "MISSING_API_KEY",
                "message": "Gemini API key is not configured. Please add GEMINI_API_KEY to your .env file.",
                "language": target_lang,
                "applicant_name": None,
                "document_reference": None,
                "document_status": None,
                "summary": "AI generation is not configured.",
                "findings": [],
                "evidence_gaps": ["API key missing."],
                "grounding_audit": {
                    "is_grounded": False,
                    "review_required": True,
                    "warnings": ["Missing API key"],
                    "invented_deadlines": [],
                    "unsupported_portals": [],
                    "omissions": [],
                    "checked_dates": []
                }
            }

        # Check for blank / empty document text
        if not document_text or not document_text.strip():
            return {
                "success": True,
                "error": None,
                "message": "Document is blank or unreadable.",
                "language": target_lang,
                "applicant_name": None,
                "document_reference": None,
                "document_status": "Blank Document",
                "summary": "The uploaded document contains no readable text or is blank.",
                "findings": [
                    {
                        "category": "insufficient_evidence",
                        "code": None,
                        "title": "Blank or Unreadable Document",
                        "what_was_submitted": "No readable text extracted from document.",
                        "policy_requirement": "All applications require legible original document uploads.",
                        "remediation_steps": "Please upload a clear, legible PDF or text notice.",
                        "supporting_evidence_needed": None
                    }
                ],
                "evidence_gaps": ["No text extracted from uploaded file."],
                "grounding_audit": {
                    "is_grounded": True,
                    "review_required": False,
                    "warnings": [],
                    "invented_deadlines": [],
                    "unsupported_portals": [],
                    "omissions": [],
                    "checked_dates": []
                }
            }

        prompt_sections = [
            "### RETRIEVED POLICY EVIDENCE (FICTIONAL DEMONSTRATION RULES):",
            context_text if context_text.strip() else "No relevant policy evidence retrieved.",
            "",
            "### UPLOADED APPLICANT DOCUMENT TEXT:",
            document_text.strip(),
            "",
            "### ANALYSIS INSTRUCTIONS:",
            "Perform an objective, factual scrutiny of the uploaded document against the retrieved policy rules.",
            "1. Output MUST be valid JSON adhering strictly to the provided schema.",
            "2. Distinguish three finding categories:",
            "   - 'confirmed_defect': Stated directly in the notice as an explicit defect or rule violation.",
            "   - 'possible_issue': A discrepancy, risk, or warning mentioned in the text.",
            "   - 'insufficient_evidence': Where the document mentions an item, but neither the document nor policy contains enough data to verify.",
            "3. DO NOT rely solely on codes R-01 to R-05; identify any procedural or document non-compliance.",
            "4. DO NOT invent applicant names, reference numbers, rejection statuses or deadlines. If absent from the document, set to null.",
            "5. In remediation steps, preserve ALL mandatory conditions: required Gazette notification or School Bonafide, Rs. 50 stamp paper, digital barcode/QR code verification, 'Resubmit Defective Application' tab, and grievance justification.",
            "6. Never interpret missing evidence as proof that no defects exist; record evidence gaps explicitly.",
            f"7. LANGUAGE FIDELITY: All explanatory text fields ('summary', 'title', 'what_was_submitted', 'policy_requirement', 'remediation_steps', 'supporting_evidence_needed') MUST be composed in {target_lang.upper()} (see detailed instructions below). Do not output these fields in English unless target language is 'en'.",
            "8. HISTORICAL DEMONSTRATION DATES: Treat all policy dates (November 15, 2025 and November 20, 2025) strictly as historical dates from a past fictional 2025 demonstration scenario. Never present them as current or upcoming deadlines, and do not invent replacement dates.",
            "9. NO UNSUPPORTED REASSURANCE: Never state or imply that the applicant does not need to submit a new application. State only objective, factual findings and official resubmission steps.",
            "",
            get_language_prompt_guidance(target_lang),
        ]

        user_content = "\n".join(prompt_sections)

        try:
            logger.info("Requesting structured document defect analysis from '%s' (language: %s)...", self.model_name, target_lang)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,  # Low temperature for deterministic schema adherence
                    response_mime_type="application/json",
                    response_schema=DocumentAnalysisSchema,
                ),
            )

            generated_text = (response.text or "").strip()
            logger.info("Received analysis response (%d chars)", len(generated_text))

            # Strict JSON validation without regex/markdown repair
            try:
                parsed_json = json.loads(generated_text)
                validated_analysis = DocumentAnalysisSchema.model_validate(parsed_json)
            except Exception as parse_err:
                logger.error("Strict JSON schema validation failed: %s", str(parse_err))
                return {
                    "success": False,
                    "error": "MALFORMED_ANALYSIS_OUTPUT",
                    "message": "Structured document analysis is temporarily unavailable due to a formatting error. Please consult the Ask Questions tab.",
                    "language": target_lang,
                    "applicant_name": None,
                    "document_reference": None,
                    "document_status": None,
                    "summary": "Document analysis could not be parsed into the required schema.",
                    "findings": [],
                    "evidence_gaps": ["Model returned malformed JSON output."],
                    "grounding_audit": {
                        "is_grounded": False,
                        "review_required": True,
                        "warnings": [f"Malformed JSON: {str(parse_err)}"],
                        "invented_deadlines": [],
                        "unsupported_portals": [],
                        "omissions": [],
                        "checked_dates": []
                    }
                }

            # Grounding verification on findings
            audit = GroundingValidator.audit_findings(
                findings=validated_analysis.findings,
                context_text=context_text,
                applicant_text=document_text
            )

            return {
                "success": True,
                "error": None,
                "message": None,
                "language": target_lang,
                "applicant_name": validated_analysis.applicant_name,
                "document_reference": validated_analysis.document_reference,
                "document_status": validated_analysis.document_status,
                "summary": validated_analysis.summary,
                "findings": [f.model_dump() for f in validated_analysis.findings],
                "evidence_gaps": validated_analysis.evidence_gaps,
                "grounding_audit": audit,
            }

        except Exception as e:
            logger.error("Error executing Gemini defect analysis: %s", str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Error generating document analysis: {str(e)}",
                "language": target_lang,
                "applicant_name": None,
                "document_reference": None,
                "document_status": None,
                "summary": "Analysis failed due to an API error.",
                "findings": [],
                "evidence_gaps": [str(e)],
                "grounding_audit": {
                    "is_grounded": False,
                    "review_required": True,
                    "warnings": [str(e)],
                    "invented_deadlines": [],
                    "unsupported_portals": [],
                    "omissions": [],
                    "checked_dates": []
                }
            }
