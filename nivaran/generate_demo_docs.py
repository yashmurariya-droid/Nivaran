"""Generates fictional demonstration documents for NIVARAN.

Produces:
1. data/knowledge_base/demo_scholarship_policy.pdf
2. data/raw/fictional_rejection_notice.pdf (and uploads copy)
"""
from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
RAW_DIR = BASE_DIR / "data" / "raw"
UPLOADS_DIR = BASE_DIR / "data" / "uploads"


def create_demo_scholarship_policy_pdf() -> Path:
    """Creates a 3-page fictional scholarship policy PDF."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = KB_DIR / "demo_scholarship_policy.pdf"

    doc = pymupdf.open()

    watermark = "FICTIONAL DEMONSTRATION POLICY — NOT AN OFFICIAL GOVERNMENT DOCUMENT."

    # PAGE 1: Overview & Eligibility
    page1 = doc.new_page(width=595, height=842)
    # Header watermark
    page1.insert_text((40, 35), watermark, fontsize=8, color=(0.6, 0.2, 0.2))
    page1.draw_line((40, 42), (555, 42), color=(0.7, 0.7, 0.7), width=0.5)

    p1_content = """PRAGATI MERIT-CUM-MEANS SCHOLARSHIP SCHEME (2025-2026)
OFFICIAL IMPLEMENTATION GUIDELINES & POLICY COMPENDIUM

Document Ref: PMS-DEMO-2025/REG-01
Effective Session: Academic Year 2025-2026

SECTION 1: SCHEME OBJECTIVE & SCOPE
The Pragati Merit-cum-Means Scholarship Scheme provides direct financial tuition assistance to meritorious students enrolled in accredited post-matric diploma and undergraduate degree programs who face severe economic constraints. The scheme is administered exclusively via the State Unified Scholarship Portal.

SECTION 2: COMPREHENSIVE ELIGIBILITY CRITERIA
Applicants must satisfy all of the following cumulative criteria on the date of online application submission:
1. Academic Merit: The candidate must have achieved a minimum of 60.0% aggregate marks (or equivalent CGPA 6.5/10.0) in the preceding qualifying examination (Class 10 Board for diploma courses, or Class 12 Higher Secondary for degree programs). No rounding-off of percentages is permitted.
2. Income Ceiling: The gross total annual family income from all combined sources (including salary, agriculture, business, and pension) must not exceed INR 2,50,000 (Two Lakh Fifty Thousand Rupees).
3. Enrollment Status: The student must be formally admitted as a regular, full-time candidate in a government or UGC/AICTE-recognized educational institution. Part-time, distance-mode, and correspondence courses are ineligible.
4. Citizenship and Domicile: The applicant must be a resident of the state and hold a valid unique identification number (Aadhaar).
5. Single Scheme Rule: The applicant must not be in receipt of any other central or state government scholarship or fellowship during the same academic year."""

    page1.insert_textbox(
        pymupdf.Rect(40, 55, 555, 780),
        p1_content,
        fontsize=10,
        color=(0.1, 0.1, 0.1)
    )
    page1.insert_text((275, 815), "Page 1 of 3", fontsize=9, color=(0.4, 0.4, 0.4))

    # PAGE 2: Required Documents, Income Validity, and Deadlines
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((40, 35), watermark, fontsize=8, color=(0.6, 0.2, 0.2))
    page2.draw_line((40, 42), (555, 42), color=(0.7, 0.7, 0.7), width=0.5)

    p2_content = """SECTION 3: MANDATORY APPLICATION DOCUMENTS CHECKLIST
All applicants must upload legible, color-scanned copies of the following original documents in PDF or JPEG format (max file size 2MB per document):
1. Online Application Form: System-generated printout duly signed by applicant and parent/guardian.
2. Competent Authority Income Certificate: Mandatory revenue authority certificate confirming family income.
3. Proof of Identity: Aadhaar Card or Aadhaar Enrollment Slip. The name on the Aadhaar card must match the student's legal academic name.
4. Academic Transcripts: Official Class 10 / Class 12 marksheet certified by the respective examination board.
5. Institutional Bonafide Certificate: Issued and stamped by the Head of the Institution / College Registrar.
6. Current Fee Receipt: Proof of paid tuition fee for the current 2025-2026 academic semester.
7. Active Bank Account Proof: Front page of nationalized bank passbook showing student's name, IFSC code, and active Aadhaar-seeding status.

SECTION 4: INCOME CERTIFICATE VALIDITY & COMPETENT ISSUING AUTHORITY RULES
Income certificates are subject to rigorous verification. Non-compliance results in automatic disqualification:
1. Designated Competent Authority: The income certificate must be issued and digitally signed by an authorized Revenue Officer not below the rank of Tehsildar, Sub-Divisional Magistrate (SDM), or District Revenue Officer.
2. Invalid Issuing Authorities: Certificates issued by Municipal Corporators, Gram Pradhans, Notaries Public, Gazetted Officers, or private employer salary slips are STRICTLY INVALID and will NOT be accepted.
3. Financial Year Validity: The certificate must clearly state the family income for the immediately preceding Financial Year 2024-2025.
4. Date of Issuance Limit: The certificate must be issued on or after April 1, 2024, and must not be older than six (6) months prior to the date of scholarship application submission.
5. Digital Authentication: The certificate must contain a machine-verifiable barcode or QR code linked to the state e-District or revenue portal.

SECTION 5: APPLICATION TIMELINES & IMPORTANT DEADLINES
- Online Application Window Opens: August 1, 2025
- Last Date for Online Submission: October 31, 2025 (5:00 PM IST)
- Institutional Verification Deadline: November 10, 2025
- Defective Application Correction Window: November 1, 2025 to November 15, 2025 (11:59 PM IST)"""

    page2.insert_textbox(
        pymupdf.Rect(40, 55, 555, 780),
        p2_content,
        fontsize=10,
        color=(0.1, 0.1, 0.1)
    )
    page2.insert_text((275, 815), "Page 2 of 3", fontsize=9, color=(0.4, 0.4, 0.4))

    # PAGE 3: Rejection Grounds & Resubmission Procedure
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text((40, 35), watermark, fontsize=8, color=(0.6, 0.2, 0.2))
    page3.draw_line((40, 42), (555, 42), color=(0.7, 0.7, 0.7), width=0.5)

    p3_content = """SECTION 6: COMMON GROUNDS FOR APPLICATION REJECTION
Applications are evaluated against automated verification filters and scrutinized by institutional nodal officers. The following defects trigger formal rejection:
- Defect Code R-01 (Income Ceiling Exceeded): Stated income on certificate exceeds INR 2,50,000 per annum.
- Defect Code R-02 (Identity Discrepancy / Name Mismatch): Any discrepancy between the student's name registered on the scholarship portal and the name recorded on the Aadhaar Card or Class 10 Board certificate.
- Defect Code R-03 (Non-Compliant Income Certificate): Submission of an income certificate issued by an unauthorized person (such as a notary or village sarpanch), an expired certificate older than 6 months, or a certificate for an outdated financial year.
- Defect Code R-04 (Academic Criteria Not Met): Qualifying examination aggregate score below 60.0%.
- Defect Code R-05 (Bank Account Non-Seeded): Bank account is inactive, inoperative, joint account without student as primary, or unlinked to Aadhaar for Direct Benefit Transfer (DBT).

SECTION 7: CORRECTION, GRIEVANCE REDRESSAL & RESUBMISSION PROCEDURE
Students whose applications receive a "Defective Application" or "Rejected with Remedy" notice may rectify defects during the designated correction window:
1. Step 1 - Defect Notice Review: Log into the student grievance portal and download the itemized Defect Notice specifying the assigned rejection code(s).
2. Step 2 - Corrective Remediation:
   - For Defect Code R-03 (Invalid Income Certificate): The applicant must apply for an expedited digital income certificate from the local Tehsildar / e-District portal for FY 2024-2025 and upload the fresh certificate.
   - For Defect Code R-02 (Name Mismatch): The applicant must upload a signed Annexure-IV Name Discrepancy Affidavit on Rs. 50 stamp paper along with a copy of the official Gazette notification or School Bonafide confirming identity.
3. Step 3 - Grievance Portal Resubmission: Navigate to the "Resubmit Defective Application" tab, upload the rectified documents, and enter the grievance justification.
4. Step 4 - Institutional Nodal Verification: Inform the College Scholarship Nodal Officer to re-verify the modified dossier before November 20, 2025. Late resubmissions after the correction cutoff will not be entertained."""

    page3.insert_textbox(
        pymupdf.Rect(40, 55, 555, 780),
        p3_content,
        fontsize=10,
        color=(0.1, 0.1, 0.1)
    )
    page3.insert_text((275, 815), "Page 3 of 3", fontsize=9, color=(0.4, 0.4, 0.4))

    doc.save(str(out_path))
    doc.close()
    return out_path


def create_fictional_rejection_notice_pdf() -> Path:
    """Creates a realistic fictional rejection notice PDF containing two identifiable defects."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "fictional_rejection_notice.pdf"

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    watermark = "FICTIONAL DEMONSTRATION DOCUMENT — NOT AN OFFICIAL REJECTION NOTICE."
    page.insert_text((40, 35), watermark, fontsize=8, color=(0.6, 0.2, 0.2))
    page.draw_line((40, 42), (555, 42), color=(0.7, 0.7, 0.7), width=0.5)

    notice_content = """STATE SCHOLARSHIP REDRESSAL & SANCTION AUTHORITY
DIRECTORATE OF HIGHER EDUCATION — SCHOLARSHIP CELL

NOTICE OF DEFECTIVE APPLICATION / PROVISIONAL REJECTION
Notice Reference: SSP/2025-26/REJ-88412
Date of Issuance: November 2, 2025

TO:
Applicant Name: Aarav Sharma
Application ID: SCH-2025-88412
Institution: City Institute of Science & Technology
Scheme Applied: Pragati Post-Matric Merit-cum-Means Scholarship Scheme (2025-2026)

Subject: Scrutiny Findings and Provisional Rejection of Scholarship Application

Dear Applicant,

Your online application for the Pragati Post-Matric Merit-cum-Means Scholarship (Application ID: SCH-2025-88412) has been scrutinized in accordance with the Scheme Implementation Guidelines. 

During institutional and state-level portal verification, your application was flagged with the following two (2) compliance defects:

DEFECT 1 (Code: R-03) — INVALID INCOME CERTIFICATE SUBMITTED:
The Income Certificate uploaded with your application (Document Ref: NOTARY-AFF-2023-410, dated March 15, 2023) is non-compliant with Section 4 of the Scheme Guidelines:
a) The certificate was issued and signed by a Notary Public, whereas Section 4, Clause 1 requires certification by a Revenue Officer not below the rank of Tehsildar or Sub-Divisional Magistrate (SDM).
b) The certificate states family income for Financial Year 2022-2023 instead of mandatory Financial Year 2024-2025.
c) The certificate was issued more than six months prior to application submission.

DEFECT 2 (Code: R-02) — IDENTITY DISCREPANCY / APPLICANT NAME MISMATCH:
A discrepancy was detected during automated Aadhaar verification:
a) Legal name entered on the Scholarship Application Portal: "Aarav Sharma"
b) Full name recorded in the uploaded Aadhaar Card: "Aarav Kumar Sharma"
Per Section 6 (Defect Code R-02), the applicant's name across the portal and identity proof must be identical, or supported by an Annexure-IV Name Discrepancy Undertaking.

REQUIRED REMEDIATION ACTION & RESUBMISSION WINDOW:
You are granted an opportunity to rectify these defects through the Scholarship Resubmission Window:
1. Obtain an official digital Income Certificate for FY 2024-2025 from your local Tehsildar office.
2. Submit a formal Annexure-IV Name Discrepancy Affidavit clarifying the name variation between "Aarav Sharma" and "Aarav Kumar Sharma".
3. Upload the corrected documents via the State Grievance Portal on or before November 15, 2025 (11:59 PM IST).
4. Request your College Nodal Officer to re-verify your dossier before November 20, 2025.

Failure to resubmit compliant documents within the 15-day window will result in permanent rejection.

Sd/-
Member Secretary / Nodal Verification Officer
State Scholarship Authority"""

    page.insert_textbox(
        pymupdf.Rect(40, 55, 555, 780),
        notice_content,
        fontsize=10,
        color=(0.1, 0.1, 0.1)
    )
    page.insert_text((275, 815), "Notice Page 1 of 1", fontsize=9, color=(0.4, 0.4, 0.4))

    doc.save(str(out_path))
    doc.close()

    # Also make a copy in uploads directory for convenient test uploads
    import shutil
    shutil.copyfile(out_path, UPLOADS_DIR / "fictional_rejection_notice.pdf")

    return out_path


if __name__ == "__main__":
    policy_pdf = create_demo_scholarship_policy_pdf()
    print(f"Created demo policy PDF at: {policy_pdf}")
    notice_pdf = create_fictional_rejection_notice_pdf()
    print(f"Created fictional rejection notice PDF at: {notice_pdf}")
