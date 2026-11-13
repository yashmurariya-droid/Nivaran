/**
 * NIVARAN Frontend Application Controller
 * Approved Stitch Editorial Design System • Vanilla JavaScript Client
 * Multilingual Architecture: English, Hindi (हिन्दी), Hinglish
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // State Management
  // =========================================================================
  const state = {
    currentScreen: "screen-home",
    currentLanguage: "en",
    lastAnalyzedLanguage: null,
    uploadId: null,
    uploadedFile: null,
    uploadedFileName: "fictional_rejection_notice.pdf",
    uploadedText: "",
    completedTasks: new Set(),
    totalTasks: 4,
    chatHistory: [],
  };

  // =========================================================================
  // DOM Elements
  // =========================================================================
  // Language Selector
  const languageSelector = document.getElementById("language-selector");

  // Nav & Screens
  const navButtons = document.querySelectorAll(".nav-btn");
  const screens = document.querySelectorAll(".screen-view");
  const kbPill = document.getElementById("kb-pill");
  const kbIndicator = document.getElementById("kb-indicator");
  const geminiPill = document.getElementById("gemini-pill");
  const geminiIndicator = document.getElementById("gemini-indicator");

  // Screen 1: Home / Upload & Direct Remark Query
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const dropzone = document.getElementById("dropzone");
  const fileFeedback = document.getElementById("file-feedback");
  const fileNameDisplay = document.getElementById("file-name-display");
  const fileStatusSub = document.getElementById("file-status-sub");
  const processFileBtn = document.getElementById("process-file-btn");
  const uploadStatus = document.getElementById("upload-status");
  const errorQueryInput = document.getElementById("error-query-input");
  const errorQueryBtn = document.getElementById("error-query-btn");
  const errorQueryBtnText = document.getElementById("error-query-btn-text");
  const queryResultBox = document.getElementById("query-result-box");
  const queryResultText = document.getElementById("query-result-text");
  const queryResultSource = document.getElementById("query-result-source");

  // Screen 2: Analysis
  const analysisFileName = document.getElementById("analysis-file-name");
  const dynamicFindingsContainer = document.getElementById("dynamic-findings-container");
  const analysisLoadingState = document.getElementById("analysis-loading-state");
  const analysisErrorState = document.getElementById("analysis-error-state");
  const analysisErrorMessage = document.getElementById("analysis-error-message");
  const analysisSummaryHeading = document.getElementById("analysis-summary-heading");
  const analysisSummaryText = document.getElementById("analysis-summary-text");
  const analysisStatus = document.getElementById("analysis-status");
  const analysisApplicant = document.getElementById("analysis-applicant");
  const analysisRef = document.getElementById("analysis-ref");
  const analysisGroundingNotice = document.getElementById("analysis-grounding-notice");
  const analysisGroundingWarnings = document.getElementById("analysis-grounding-warnings");
  const analysisCitationsContainer = document.getElementById("analysis-citations-container");
  const analysisCitationsList = document.getElementById("analysis-citations-list");

  // Screen 3: Ask NIVARAN
  const chatStream = document.getElementById("chat-stream");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const chatLoading = document.getElementById("chat-loading");
  const chatSubmitBtn = document.getElementById("chat-submit-btn");
  const pillButtons = document.querySelectorAll(".pill-btn");

  // Screen 4: Next Steps
  const taskToggles = document.querySelectorAll(".task-toggle");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const progressText = document.getElementById("progress-text");
  const progressMessage = document.getElementById("progress-message");

  // =========================================================================
  // 1. Localization / Translation Helpers
  // =========================================================================
  function t(key, fallback = "") {
    const lang = state.currentLanguage || "en";
    const currentDict = (window.I18N && window.I18N[lang]) || {};
    if (currentDict[key] !== undefined && currentDict[key] !== null && currentDict[key] !== "") {
      return currentDict[key];
    }
    const enDict = (window.I18N && window.I18N["en"]) || {};
    if (enDict[key] !== undefined && enDict[key] !== null && enDict[key] !== "") {
      return enDict[key];
    }
    console.warn(`[i18n] Missing translation key: '${key}' for language '${lang}'`);
    return fallback || "";
  }

  function applyLanguage(lang) {
    const validLangs = ["en", "hi", "hinglish"];
    const targetLang = validLangs.includes(lang) ? lang : "en";
    state.currentLanguage = targetLang;

    try {
      localStorage.setItem("nivaran_language", targetLang);
    } catch (e) {
      console.warn("localStorage not accessible for language preference");
    }

    if (languageSelector) {
      languageSelector.value = targetLang;
    }

    // Set document lang attribute for accessibility
    document.documentElement.lang = targetLang === "hi" ? "hi" : (targetLang === "hinglish" ? "hi-Latn" : "en");

    // 1. Translate all static data-i18n elements
    const i18nElements = document.querySelectorAll("[data-i18n]");
    i18nElements.forEach((el) => {
      const key = el.getAttribute("data-i18n");
      const defaultContent = el.getAttribute("data-fallback") || el.innerHTML.trim();
      const translation = t(key, defaultContent);
      if (translation) {
        el.innerHTML = translation;
      }
    });

    // 2. Translate all placeholder elements
    const placeholderElements = document.querySelectorAll("[data-i18n-placeholder]");
    placeholderElements.forEach((el) => {
      const key = el.getAttribute("data-i18n-placeholder");
      const translation = t(key, el.placeholder);
      if (translation) {
        el.placeholder = translation;
      }
    });

    // 3. Update suggested questions data-question attributes
    const p1 = document.getElementById("pill-btn-1");
    if (p1) p1.setAttribute("data-question", t("ask.pill_1_q"));
    const p2 = document.getElementById("pill-btn-2");
    if (p2) p2.setAttribute("data-question", t("ask.pill_2_q"));
    const p3 = document.getElementById("pill-btn-3");
    if (p3) p3.setAttribute("data-question", t("ask.pill_3_q"));
    const p4 = document.getElementById("pill-btn-4");
    if (p4) p4.setAttribute("data-question", t("ask.pill_4_q"));

    // 4. Update Checklist UI with new localized labels
    updateChecklistUI();

    // 5. If document was uploaded, re-run analysis in new language
    if (state.uploadId && (state.currentScreen === "screen-analysis" || (dynamicFindingsContainer && dynamicFindingsContainer.children.length > 0))) {
      runDocumentAnalysis(state.uploadId);
    }
  }

  // Language selector change listener
  if (languageSelector) {
    languageSelector.addEventListener("change", (e) => {
      applyLanguage(e.target.value);
    });
  }

  // =========================================================================
  // 2. Navigation Controller
  // =========================================================================
  window.switchScreen = function (targetScreenId) {
    if (!document.getElementById(targetScreenId)) return;

    // Hide all screens, show target
    screens.forEach((screen) => {
      if (screen.id === targetScreenId) {
        screen.classList.remove("hidden");
        screen.classList.add("active");
      } else {
        screen.classList.add("hidden");
        screen.classList.remove("active");
      }
    });

    // Update Nav buttons styling with Stitch styling
    navButtons.forEach((btn) => {
      const isTarget = btn.getAttribute("data-target") === targetScreenId;
      if (isTarget) {
        btn.className = "nav-btn px-3 py-1.5 border-2 border-ink bg-ink text-khadi shadow-[2px_2px_0px_#141414] transition-all flex items-center gap-1 cursor-pointer";
      } else {
        btn.className = "nav-btn px-3 py-1.5 border-2 border-ink bg-transparent hover:bg-khadi-subtle text-ink shadow-[2px_2px_0px_#141414] transition-all flex items-center gap-1 cursor-pointer";
      }
    });

    state.currentScreen = targetScreenId;
    window.location.hash = targetScreenId.replace("screen-", "");
    window.scrollTo({ top: 0, behavior: "smooth" });

    // If switching to analysis screen with an active upload and different language, re-run analysis
    if (targetScreenId === "screen-analysis" && state.uploadId && state.lastAnalyzedLanguage !== state.currentLanguage) {
      runDocumentAnalysis(state.uploadId);
    }
  };

  // Nav button click listeners
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-target");
      if (target) window.switchScreen(target);
    });
  });

  // Handle URL hash changes
  function handleHashChange() {
    const hash = window.location.hash.replace("#", "");
    const map = {
      home: "screen-home",
      welcome: "screen-home",
      analysis: "screen-analysis",
      "document-check": "screen-analysis",
      ask: "screen-ask",
      "ask-questions": "screen-ask",
      steps: "screen-steps",
      "next-steps": "screen-steps",
    };
    if (hash && map[hash]) {
      window.switchScreen(map[hash]);
    }
  }

  window.addEventListener("hashchange", handleHashChange);
  if (window.location.hash) {
    handleHashChange();
  }

  // =========================================================================
  // 3. Health & System Check
  // =========================================================================
  async function checkSystemHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("Health check unreachable");
      const data = await res.json();

      // Vector Store Status
      const count = data.vector_store_docs_count || 0;
      if (kbPill) {
        if (state.currentLanguage === "hi") {
          kbPill.textContent = `वेक्टर ज्ञानकोष: ${count} अंश`;
        } else if (state.currentLanguage === "hinglish") {
          kbPill.textContent = `Vector KB: ${count} chunks`;
        } else {
          kbPill.textContent = `Vector KB: ${count} chunks`;
        }
      }
      if (kbIndicator) {
        kbIndicator.className = count > 0 ? "w-2 h-2 rounded-full bg-stamp-green" : "w-2 h-2 rounded-full bg-marigold";
      }

      // Gemini Model Status
      if (geminiPill) {
        if (data.gemini_ready) {
          geminiPill.textContent = `Gemini: ${data.gemini_model}`;
          if (geminiIndicator) geminiIndicator.className = "w-2 h-2 rounded-full bg-stamp-green";
        } else {
          geminiPill.textContent = "Gemini: Key Required";
          if (geminiIndicator) geminiIndicator.className = "w-2 h-2 rounded-full bg-marigold";
        }
      }
    } catch (err) {
      console.warn("Health check error:", err);
      if (kbPill) kbPill.textContent = "Vector KB: Offline";
      if (kbIndicator) kbIndicator.className = "w-2 h-2 rounded-full bg-vermilion";
    }
  }

  // =========================================================================
  // 4. Document Upload Handling
  // =========================================================================
  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      handleFileSelected(file);
    });
  }

  // Drag and drop events
  if (dropzone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("border-marigold", "bg-cobalt/60");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("border-marigold", "bg-cobalt/60");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const file = dt.files[0];
      handleFileSelected(file);
    });
  }

  function handleFileSelected(file) {
    if (!file) return;

    state.uploadedFile = file;
    state.uploadedFileName = file.name;

    if (fileNameDisplay) fileNameDisplay.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    if (fileStatusSub) fileStatusSub.textContent = t("home.ready_to_analyze", "Ready to analyze with RAG");
    if (fileFeedback) fileFeedback.classList.remove("hidden");
    if (uploadStatus) uploadStatus.classList.add("hidden");
  }

  if (processFileBtn) {
    processFileBtn.addEventListener("click", async () => {
      if (!state.uploadedFile) return;

      processFileBtn.disabled = true;
      processFileBtn.textContent = state.currentLanguage === "hi" ? "प्रक्रिया जारी..." : (state.currentLanguage === "hinglish" ? "Processing ho rahi hai..." : "Processing...");
      showUploadStatus(state.currentLanguage === "hi" ? "दस्तावेज़ से सुरक्षित रूप से टेक्स्ट निकाला जा रहा है..." : "Extracting document text safely via PyMuPDF...", "info");

      const formData = new FormData();
      formData.append("file", state.uploadedFile);

      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.message || data.error || "Upload failed");
        }

        state.uploadedText = data.preview || "";
        state.uploadId = data.upload_id || null;
        showUploadStatus(`Extracted ${data.chunks_extracted} chunks from ${data.filename}`, "success");

        if (analysisFileName) {
          analysisFileName.textContent = data.filename;
        }

        // Smooth transition to Document Analysis screen and trigger analysis
        setTimeout(() => {
          window.switchScreen("screen-analysis");
          if (state.uploadId) {
            runDocumentAnalysis(state.uploadId);
          }
        }, 500);
      } catch (err) {
        console.error("Upload error:", err);
        showUploadStatus(`Error: ${err.message}`, "error");
      } finally {
        processFileBtn.disabled = false;
        processFileBtn.textContent = t("home.analyze_btn", "Analyze Document");
      }
    });
  }

  // =========================================================================
  // 5. Direct Remark Query Controller (Grounded RAG Integration)
  // =========================================================================
  window.handleDirectRemarkQuery = async function () {
    if (!errorQueryInput) return;
    const query = errorQueryInput.value.trim();
    if (!query) {
      alert("Please enter a remark snippet, rejection code, or defect reason to query the policy database.");
      return;
    }

    if (errorQueryBtn) errorQueryBtn.disabled = true;
    if (errorQueryBtnText) errorQueryBtnText.textContent = "Querying ChromaDB...";
    if (queryResultBox) {
      queryResultBox.classList.remove("hidden");
      if (queryResultText) queryResultText.textContent = "Searching policy circulars and generating grounded guidance note...";
    }

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          applicant_document_text: state.uploadedText || "Applicant: Aarav Sharma. Defect Codes: R-03 and R-02.",
          language: state.currentLanguage || "en",
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.message || data.error || "Query generation failed");
      }

      if (queryResultText) {
        queryResultText.textContent = data.answer || "No guidance note returned.";
      }
      if (queryResultSource && data.citations && data.citations.length > 0) {
        const topCitation = data.citations[0];
        queryResultSource.textContent = `Source: ${topCitation.source} (p. ${topCitation.page})`;
      }
    } catch (err) {
      console.error("Direct remark query error:", err);
      if (queryResultText) {
        queryResultText.textContent = `Error: ${err.message}`;
      }
    } finally {
      if (errorQueryBtn) errorQueryBtn.disabled = false;
      if (errorQueryBtnText) errorQueryBtnText.textContent = "Search Guidance Database";
    }
  };

  window.transferQueryToChat = function () {
    if (errorQueryInput && chatInput) {
      chatInput.value = errorQueryInput.value.trim();
    }
    window.switchScreen("screen-ask");
    if (chatForm) {
      chatForm.dispatchEvent(new Event("submit"));
    }
  };

  // Allow pressing Enter in textarea to submit direct query
  if (errorQueryInput) {
    errorQueryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        window.handleDirectRemarkQuery();
      }
    });
  }

  // =========================================================================
  // 6. Dynamic Document Analysis Controller
  // =========================================================================
  async function runDocumentAnalysis(uploadId) {
    if (!uploadId) return;

    // Show loading state
    if (analysisLoadingState) analysisLoadingState.classList.remove("hidden");
    if (dynamicFindingsContainer) dynamicFindingsContainer.classList.add("hidden");
    if (analysisErrorState) analysisErrorState.classList.add("hidden");
    if (analysisGroundingNotice) analysisGroundingNotice.classList.add("hidden");

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: uploadId,
          language: state.currentLanguage || "en"
        }),
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.message || data.error || "Analysis failed");
      }

      state.lastAnalyzedLanguage = state.currentLanguage;

      // 1. Update Metadata Card (Preserve original applicant name & reference)
      if (analysisApplicant) analysisApplicant.textContent = data.applicant_name || "Not specified";
      if (analysisRef) analysisRef.textContent = data.document_reference || "SSP/2025-26/REJ-88412";
      if (analysisStatus) {
        const rawStatus = data.document_status || "Reviewed";
        let displayStatus = rawStatus;
        if (rawStatus.toLowerCase().includes("rejection")) {
          displayStatus = state.currentLanguage === "hi" ? "अस्वीकृति सूचना" : (state.currentLanguage === "hinglish" ? "Rejection Notice" : "Rejection Notice");
          analysisStatus.className = "text-vermilion font-bold";
        } else if (rawStatus.toLowerCase().includes("blank")) {
          displayStatus = state.currentLanguage === "hi" ? "खाली दस्तावेज़" : (state.currentLanguage === "hinglish" ? "Blank Document" : "Blank Document");
          analysisStatus.className = "text-marigold font-bold";
        } else {
          analysisStatus.className = "text-marigold font-bold";
        }
        analysisStatus.textContent = displayStatus;
      }

      // 2. Update Advisory Summary Banner
      const findings = data.findings || [];
      const confirmedCount = findings.filter((f) => f.category === "confirmed_defect").length;
      const possibleCount = findings.filter((f) => f.category === "possible_issue").length;

      if (analysisSummaryHeading) {
        if (state.currentLanguage === "hi") {
          if (confirmedCount > 0) {
            analysisSummaryHeading.textContent = `हमने आपके दस्तावेज़ में ${confirmedCount} पुष्ट कमियों की पहचान की है।`;
          } else if (possibleCount > 0) {
            analysisSummaryHeading.textContent = `हमने जाँच के लिए ${possibleCount} संभावित मुद्दों की पहचान की है।`;
          } else {
            analysisSummaryHeading.textContent = "दस्तावेज़ समीक्षा परिणाम";
          }
        } else if (state.currentLanguage === "hinglish") {
          if (confirmedCount > 0) {
            analysisSummaryHeading.textContent = `Humne aapke document mein ${confirmedCount} confirmed defects identify kiye hain.`;
          } else if (possibleCount > 0) {
            analysisSummaryHeading.textContent = `Humne check karne ke liye ${possibleCount} potential issues identify kiye hain.`;
          } else {
            analysisSummaryHeading.textContent = "Document Review Findings";
          }
        } else {
          if (confirmedCount > 0) {
            analysisSummaryHeading.textContent = `We identified ${confirmedCount} confirmed defect${confirmedCount > 1 ? "s" : ""} in your document.`;
          } else if (possibleCount > 0) {
            analysisSummaryHeading.textContent = `We identified ${possibleCount} potential issue${possibleCount > 1 ? "s" : ""} to check.`;
          } else {
            analysisSummaryHeading.textContent = "Document Review Findings";
          }
        }
      }

      if (analysisSummaryText) {
        analysisSummaryText.textContent = data.summary || t("analysis.default_summary_text");
      }

      // 3. Render Dynamic Finding Cards in Stitch Aesthetic
      if (dynamicFindingsContainer) {
        dynamicFindingsContainer.innerHTML = "";

        if (findings.length === 0) {
          const noDefectTitle = state.currentLanguage === "hi" ? "कोई सक्रिय नीति दोष नहीं मिला" : (state.currentLanguage === "hinglish" ? "Koi policy defect detect nahi hua" : "No Active Policy Defects Detected");
          const noDefectDesc = state.currentLanguage === "hi" ? "वर्तमान दिशानिर्देशों के विरुद्ध इस दस्तावेज़ में कोई नीति उल्लंघन या अस्वीकृति आधार नहीं पाया गया।" : (state.currentLanguage === "hinglish" ? "Current guidelines ke against is document mein koi rejection grounds nahi mile." : "No policy violations or rejection grounds were identified in this document text against current guidelines.");
          dynamicFindingsContainer.innerHTML = `
            <div class="w-full bg-khadi border-2 border-ink p-8 shadow-[4px_4px_0px_#141414] text-center font-mono">
              <span class="material-symbols-outlined text-stamp-green text-[36px] mb-2">task_alt</span>
              <h3 class="font-display text-base font-bold text-ink mb-1">${escapeHtml(noDefectTitle)}</h3>
              <p class="text-xs text-ink-muted max-w-md mx-auto">
                ${escapeHtml(noDefectDesc)}
              </p>
            </div>
          `;
        } else {
          findings.forEach((finding, idx) => {
            const card = document.createElement("div");
            card.className = "finding-card w-full bg-khadi border-2 border-ink p-6 shadow-[4px_4px_0px_#141414] flex flex-col gap-4";

            // Category Styling & Localization
            let badgeClass = "rubber-stamp text-[9px] py-0.5 px-1.5";
            let badgeText = t("badge.confirmed", "Confirmed Defect");
            let numBg = "bg-vermilion text-khadi";

            if (finding.category === "possible_issue") {
              badgeClass = "rubber-stamp-cobalt text-[9px] py-0.5 px-1.5";
              badgeText = t("badge.possible", "Possible Issue");
              numBg = "bg-marigold text-ink";
            } else if (finding.category === "insufficient_evidence") {
              badgeClass = "rubber-stamp-green text-[9px] py-0.5 px-1.5";
              badgeText = t("badge.insufficient", "Insufficient Evidence");
              numBg = "bg-ink text-khadi";
            }

            const defectCodeLabel = finding.code ? `[Code: ${escapeHtml(finding.code)}] ` : "";
            const supportingEvidenceHtml = finding.supporting_evidence_needed
              ? `
              <div class="bg-marigold/10 border-2 border-marigold p-3 text-xs font-mono text-ink flex items-start gap-2 mt-1">
                <span class="material-symbols-outlined text-marigold text-[16px] shrink-0 mt-0.5">attach_file</span>
                <span><strong>${escapeHtml(t("analysis.finding_proof", "Required Proof:"))}</strong> ${escapeHtml(finding.supporting_evidence_needed)}</span>
              </div>
            `
              : "";

            card.innerHTML = `
              <div class="flex items-start justify-between gap-4">
                <div class="flex items-center gap-3">
                  <div class="w-8 h-8 ${numBg} flex items-center justify-center font-mono font-bold text-sm border border-ink">
                    ${idx + 1}
                  </div>
                  <h3 class="font-display font-bold text-base sm:text-lg text-ink">
                    ${defectCodeLabel}${escapeHtml(finding.title)}
                  </h3>
                </div>
                <span class="${badgeClass}">
                  ${escapeHtml(badgeText)}
                </span>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono mt-1">
                <div class="p-4 bg-khadi-subtle border border-ink/40">
                  <h4 class="font-bold text-ink mb-1 flex items-center gap-1.5 uppercase">
                    <span class="material-symbols-outlined text-vermilion text-[16px]">cancel</span>
                    ${escapeHtml(t("analysis.finding_submitted", "What was submitted / flagged"))}
                  </h4>
                  <p class="text-ink-muted text-xs leading-relaxed">
                    ${escapeHtml(finding.what_was_submitted)}
                  </p>
                </div>
                <div class="p-4 bg-khadi-subtle border border-ink/40">
                  <h4 class="font-bold text-ink mb-1 flex items-center gap-1.5 uppercase">
                    <span class="material-symbols-outlined text-cobalt text-[16px]">gavel</span>
                    ${escapeHtml(t("analysis.finding_policy", "Policy Requirement"))}
                  </h4>
                  <p class="text-ink-muted text-xs leading-relaxed">
                    ${escapeHtml(finding.policy_requirement)}
                  </p>
                </div>
              </div>

              <div class="bg-cobalt-light border border-cobalt p-4 flex items-start gap-3 mt-1 font-mono text-xs">
                <span class="material-symbols-outlined text-cobalt text-[18px] shrink-0 mt-0.5">build</span>
                <div>
                  <strong class="text-cobalt uppercase">${escapeHtml(t("analysis.finding_remediation", "Remediation steps:"))}</strong>
                  <span class="text-ink"> ${escapeHtml(finding.remediation_steps)}</span>
                </div>
              </div>
              ${supportingEvidenceHtml}
            `;
            dynamicFindingsContainer.appendChild(card);
          });
        }
        dynamicFindingsContainer.classList.remove("hidden");
      }

      // 4. Render Citations (Strictly preserved original filenames & excerpts)
      if (analysisCitationsContainer && analysisCitationsList) {
        if (data.citations && data.citations.length > 0) {
          analysisCitationsList.innerHTML = data.citations
            .map(
              (c) => `
              <div class="p-3 bg-khadi border border-ink text-xs font-mono">
                <div class="flex items-center justify-between mb-1 font-semibold text-ink">
                  <span class="truncate">📄 ${escapeHtml(c.source)} (p. ${c.page})</span>
                  <span class="text-cobalt font-bold">${(c.similarity_score * 100).toFixed(0)}% match</span>
                </div>
                <p class="text-ink-muted line-clamp-2">${escapeHtml(c.snippet)}...</p>
              </div>
            `
            )
            .join("");
          analysisCitationsContainer.classList.remove("hidden");
        }
      }

      // 5. Render Grounding Review Warnings
      if (data.grounding_audit && data.grounding_audit.review_required) {
        if (analysisGroundingNotice && analysisGroundingWarnings) {
          analysisGroundingWarnings.innerHTML = data.grounding_audit.warnings
            .map((w) => `<li>${escapeHtml(w)}</li>`)
            .join("");
          analysisGroundingNotice.classList.remove("hidden");
        }
      }
    } catch (err) {
      console.error("Analysis execution error:", err);
      if (analysisErrorState) {
        if (analysisErrorMessage) analysisErrorMessage.textContent = err.message || t("analysis.error_message");
        analysisErrorState.classList.remove("hidden");
      }
    } finally {
      if (analysisLoadingState) analysisLoadingState.classList.add("hidden");
    }
  }

  function showUploadStatus(msg, type) {
    if (!uploadStatus) return;
    uploadStatus.classList.remove("hidden", "bg-stamp-green/10", "border-stamp-green", "text-stamp-green", "bg-vermilion/10", "border-vermilion", "text-vermilion", "bg-marigold/10", "border-marigold", "text-ink");
    if (type === "success") {
      uploadStatus.classList.add("bg-stamp-green/10", "border-stamp-green", "text-stamp-green");
    } else if (type === "error") {
      uploadStatus.classList.add("bg-vermilion/10", "border-vermilion", "text-vermilion");
    } else {
      uploadStatus.classList.add("bg-marigold/10", "border-marigold", "text-ink");
    }
    uploadStatus.textContent = msg;
  }

  // =========================================================================
  // 7. Ask NIVARAN (Grounded Q&A Engine)
  // =========================================================================
  // Question Pill shortcuts
  pillButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.getAttribute("data-question");
      if (q && chatInput) {
        chatInput.value = q;
        chatForm.dispatchEvent(new Event("submit"));
      }
    });
  });

  if (chatForm && chatInput) {
    chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const query = chatInput.value.trim();
      if (!query) return;

      // 1. Append User Message
      appendUserMessage(query);
      chatInput.value = "";
      chatSubmitBtn.disabled = true;
      if (chatLoading) chatLoading.classList.remove("hidden");

      try {
        // 2. Dispatch to /api/query with grounded context and selected language
        const res = await fetch("/api/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query,
            applicant_document_text: state.uploadedText || "Applicant: Aarav Sharma. Defect Codes: R-03 and R-02.",
            language: state.currentLanguage || "en",
          }),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.message || data.error || "Query generation failed");
        }

        // 3. Append AI Response with Citations & Grounding Flags
        appendAiResponse(data);
      } catch (err) {
        console.error("Chat error:", err);
        appendErrorMessage(`Error retrieving grounded answer: ${err.message}`);
      } finally {
        if (chatLoading) chatLoading.classList.add("hidden");
        chatSubmitBtn.disabled = false;
      }
    });
  }

  function appendUserMessage(text) {
    const userRow = document.createElement("div");
    userRow.className = "flex items-start justify-end gap-3 pl-8";
    userRow.innerHTML = `
      <div class="flex flex-col items-end max-w-xl font-mono">
        <div class="bg-ink text-khadi border-2 border-ink px-4 py-3 shadow-[2px_2px_0px_#141414]">
          <p class="text-xs sm:text-sm font-medium leading-relaxed">${escapeHtml(text)}</p>
        </div>
        <span class="text-[10px] text-ink-muted mt-1 pr-1">${escapeHtml(t("ask.user_sent", "Sent just now • You"))}</span>
      </div>
      <div class="w-8 h-8 bg-marigold text-ink border border-ink flex items-center justify-center shrink-0 mt-0.5">
        <span class="material-symbols-outlined text-[18px]">person</span>
      </div>
    `;
    chatStream.appendChild(userRow);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  function appendAiResponse(data) {
    const aiRow = document.createElement("div");
    aiRow.className = "flex items-start gap-3 font-mono";

    // Format citations HTML (Original citations strictly preserved)
    let citationsHtml = "";
    if (data.citations && data.citations.length > 0) {
      citationsHtml = `
        <div class="mt-4 pt-4 border-t border-ink/20">
          <span class="text-xs font-bold uppercase tracking-wider text-cobalt block mb-2">${escapeHtml(t("ask.citations_header", "Verified Source Citations:"))}</span>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            ${data.citations
              .map(
                (c) => `
              <div class="p-3 bg-khadi border border-ink text-xs">
                <div class="flex items-center justify-between mb-1 font-semibold text-ink">
                  <span class="truncate">📄 ${escapeHtml(c.source)} (p. ${c.page})</span>
                  <span class="text-cobalt font-bold">${(c.similarity_score * 100).toFixed(0)}% match</span>
                </div>
                <p class="text-ink-muted line-clamp-2">${escapeHtml(c.snippet)}...</p>
              </div>
            `
              )
              .join("")}
          </div>
        </div>
      `;
    }

    // Format grounding audit notice if review required
    let auditNoticeHtml = "";
    if (data.grounding_audit && data.grounding_audit.review_required) {
      auditNoticeHtml = `
        <div class="mt-3 p-3 bg-amber-50 border-2 border-marigold text-xs text-ink">
          <strong class="block mb-1 text-vermilion">⚠️ Policy Review Notice:</strong>
          <ul class="list-disc pl-4 space-y-0.5 text-ink-muted">
            ${data.grounding_audit.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}
          </ul>
        </div>
      `;
    }

    aiRow.innerHTML = `
      <div class="w-8 h-8 bg-cobalt text-khadi border border-ink flex items-center justify-center shrink-0 mt-0.5">
        <span class="material-symbols-outlined text-[18px]">smart_toy</span>
      </div>
      <div class="flex flex-col max-w-2xl w-full">
        <div class="bg-khadi-subtle border-2 border-ink p-5 shadow-[4px_4px_0px_#141414] text-ink">
          <div class="text-xs sm:text-sm leading-relaxed whitespace-pre-wrap font-mono">${escapeHtml(data.answer || "")}</div>
          ${auditNoticeHtml}
          ${citationsHtml}
        </div>
        <span class="text-[10px] text-ink-muted mt-1.5 pl-1">
          NIVARAN RAG Advisor • Model: ${escapeHtml(data.model || "Gemini")}
        </span>
      </div>
    `;

    chatStream.appendChild(aiRow);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  function appendErrorMessage(msg) {
    const errRow = document.createElement("div");
    errRow.className = "flex items-start gap-3 font-mono";
    errRow.innerHTML = `
      <div class="w-8 h-8 bg-vermilion text-khadi border border-ink flex items-center justify-center shrink-0 mt-0.5">
        <span class="material-symbols-outlined text-[18px]">error</span>
      </div>
      <div class="flex flex-col max-w-2xl">
        <div class="bg-red-50 text-vermilion border-2 border-vermilion p-4 text-xs">
          ${escapeHtml(msg)}
        </div>
      </div>
    `;
    chatStream.appendChild(errRow);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  // =========================================================================
  // 8. Interactive Next Steps Checklist
  // =========================================================================
  function loadTaskState() {
    try {
      const saved = localStorage.getItem("nivaran_completed_tasks");
      if (saved) {
        state.completedTasks = new Set(JSON.parse(saved));
      }
    } catch (e) {
      console.warn("Storage unavailable");
    }
    updateChecklistUI();
  }

  function saveTaskState() {
    try {
      localStorage.setItem("nivaran_completed_tasks", JSON.stringify(Array.from(state.completedTasks)));
    } catch (e) {
      console.warn("Storage unavailable");
    }
  }

  taskToggles.forEach((btn) => {
    btn.addEventListener("click", () => {
      const taskId = btn.getAttribute("data-task");
      if (!taskId) return;

      if (state.completedTasks.has(taskId)) {
        state.completedTasks.delete(taskId);
      } else {
        state.completedTasks.add(taskId);
      }

      saveTaskState();
      updateChecklistUI();
    });
  });

  function updateChecklistUI() {
    const completedCount = state.completedTasks.size;
    const percentage = Math.round((completedCount / state.totalTasks) * 100);

    // Update Progress Bar
    if (progressBarFill) progressBarFill.style.width = `${percentage}%`;
    if (progressText) {
      if (state.currentLanguage === "hi") {
        progressText.textContent = `${state.totalTasks} में से ${completedCount} कार्य पूर्ण`;
      } else if (state.currentLanguage === "hinglish") {
        progressText.textContent = `${state.totalTasks} mein se ${completedCount} tasks complete`;
      } else {
        progressText.textContent = `${completedCount} of ${state.totalTasks} tasks completed`;
      }
    }

    // Update Message
    if (progressMessage) {
      if (completedCount === state.totalTasks) {
        progressMessage.innerHTML = t("steps.msg_complete");
      } else if (completedCount > 0) {
        const tpl = t("steps.msg_progress");
        progressMessage.innerHTML = tpl.replace("{count}", completedCount);
      } else {
        progressMessage.innerHTML = t("steps.msg_initial");
      }
    }

    // Update each task card
    taskToggles.forEach((btn) => {
      const taskId = btn.getAttribute("data-task");
      const isDone = state.completedTasks.has(taskId);
      const card = document.getElementById(`task-card-${taskId}`);
      const icon = btn.querySelector(".check-icon");
      const badge = card ? card.querySelector(".task-badge") : null;

      if (isDone) {
        btn.className = "task-toggle mt-1 w-7 h-7 bg-cobalt text-khadi border border-ink flex items-center justify-center shrink-0 transition-all cursor-pointer";
        if (icon) icon.classList.remove("opacity-0");
        if (card) card.classList.add("bg-cobalt-light/30");
        if (badge) {
          badge.className = "task-badge px-2 py-0.5 text-[10px] font-bold uppercase bg-cobalt text-khadi border border-ink self-start sm:self-auto";
          badge.textContent = t("steps.badge_completed", "Completed");
        }
      } else {
        btn.className = "task-toggle mt-1 w-7 h-7 bg-khadi-subtle text-ink border border-ink flex items-center justify-center shrink-0 hover:bg-khadi-dark transition-all cursor-pointer";
        if (icon) icon.classList.add("opacity-0");
        if (card) card.classList.remove("bg-cobalt-light/30");
        if (badge) {
          badge.className = "task-badge px-2 py-0.5 text-[10px] font-bold uppercase bg-khadi-dark border border-ink text-ink-muted self-start sm:self-auto";
          badge.textContent = t("steps.badge_pending", "Pending");
        }
      }
    });
  }

  // =========================================================================
  // 9. Helpers
  // =========================================================================
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial startup: Load language preference and task state
  let initialLang = "en";
  try {
    const saved = localStorage.getItem("nivaran_language");
    if (saved && ["en", "hi", "hinglish"].includes(saved)) {
      initialLang = saved;
    }
  } catch (e) {
    console.warn("Storage unavailable");
  }

  applyLanguage(initialLang);
  checkSystemHealth();
  loadTaskState();
});
