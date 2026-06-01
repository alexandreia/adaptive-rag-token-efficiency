(function () {
  // Bumped to v3: data format changed (15 items per dataset, dataset tabs).
  const STORAGE_KEY = "rag-annotation-form-state-v3";
  const samples = window.ANNOTATION_SAMPLES || [];
  const state = {
    currentIndex: 0,
    dataset: "",   // "" = All; "scifact" | "hotpotqa" | "bioasq" = filtered
    filter: "all",
    search: "",
    annotations: loadState(),
  };

  const els = {
    exportCsvBtn: document.getElementById("exportCsvBtn"),
    exportJsonBtn: document.getElementById("exportJsonBtn"),
    importCsvInput: document.getElementById("importCsvInput"),
    importJsonInput: document.getElementById("importJsonInput"),
    datasetTabs: document.getElementById("modelTabs"),  // reuses existing DOM element
    filterSelect: document.getElementById("filterSelect"),
    searchInput: document.getElementById("searchInput"),
    sampleList: document.getElementById("sampleList"),
    prevBtn: document.getElementById("prevBtn"),
    nextBtn: document.getElementById("nextBtn"),
    positionText: document.getElementById("positionText"),
    sampleId: document.getElementById("sampleId"),
    sampleGroup: document.getElementById("sampleGroup"),
    sampleMode: document.getElementById("sampleMode"),
    queryText: document.getElementById("queryText"),
    referenceAnswer: document.getElementById("referenceAnswer"),
    modelAnswer: document.getElementById("modelAnswer"),
    evidenceText: document.getElementById("evidenceText"),
    metricStrip: document.getElementById("metricStrip"),
    annotationForm: document.getElementById("annotationForm"),
  };

  function loadState() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch {
      return {};
    }
  }

  function saveState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.annotations));
  }

  function emptyAnnotation() {
    return { rating: "", notes: "" };
  }

  function getAnnotation(sample) {
    if (!state.annotations[sample.candidate_id]) {
      state.annotations[sample.candidate_id] = emptyAnnotation();
    }
    const existing = state.annotations[sample.candidate_id];
    state.annotations[sample.candidate_id] = {
      ...emptyAnnotation(),
      ...existing,
      rating: existing.rating || existing.annotator_1_rating || existing.final_rating || "",
      notes: existing.notes || existing.annotator_1_notes || existing.final_notes || "",
    };
    return state.annotations[sample.candidate_id];
  }

  function statusFor(sample) {
    return getAnnotation(sample).rating ? "complete" : "unstarted";
  }

  // Dataset tab options: "All" first, then one per unique dataset.
  function datasetOptions() {
    const options = [{ id: "", label: `All (${samples.length})` }];
    const seen = [];
    for (const sample of samples) {
      if (!sample.dataset || seen.includes(sample.dataset)) continue;
      seen.push(sample.dataset);
    }
    for (const dataset of seen) {
      const count = samples.filter((s) => s.dataset === dataset).length;
      const name = samples.find((s) => s.dataset === dataset)?.dataset_name || dataset;
      options.push({ id: dataset, label: `${name} (${count})` });
    }
    return options;
  }

  function filteredSamples() {
    const q = state.search.trim().toLowerCase();
    return samples.filter((sample) => {
      if (state.dataset && sample.dataset !== state.dataset) return false;
      const status = statusFor(sample);
      if (state.filter === "unstarted" && status !== "unstarted") return false;
      if (state.filter === "complete" && status !== "complete") return false;
      if (state.filter === "worst" && !sample.selection_reason.startsWith("worst")) return false;
      if (state.filter === "best" && !sample.selection_reason.startsWith("best")) return false;
      if (!q) return true;
      const annotation = getAnnotation(sample);
      const haystack = [
        sample.candidate_id,
        sample.query_id,
        sample.query_text,
        sample.reference_answer,
        sample.model_answer,
        annotation.notes,
      ].join(" ").toLowerCase();
      return haystack.includes(q);
    });
  }

  function currentSample() {
    const visible = filteredSamples();
    if (!visible.length) return null;
    const current = samples[state.currentIndex];
    if (current && visible.some((s) => s.candidate_id === current.candidate_id)) {
      return current;
    }
    state.currentIndex = samples.findIndex((s) => s.candidate_id === visible[0].candidate_id);
    return visible[0];
  }

  function setCurrentById(id) {
    const index = samples.findIndex((s) => s.candidate_id === id);
    if (index >= 0) {
      state.currentIndex = index;
      render();
    }
  }

  function renderList() {
    const visible = filteredSamples();
    els.sampleList.innerHTML = "";
    for (const sample of visible) {
      const button = document.createElement("button");
      const status = statusFor(sample);
      button.type = "button";
      button.className = `sample-row ${sample.candidate_id === currentSample()?.candidate_id ? "active" : ""}`;
      button.dataset.id = sample.candidate_id;
      button.innerHTML = `
        <span>
          <strong>${escapeHtml(sample.candidate_id)} · ${escapeHtml(sample.query_id)}</strong>
          <span>${escapeHtml(sample.dataset_name || sample.dataset || "")} · ${escapeHtml(sample.model_name || sample.model_family || "")} · ${escapeHtml(sample.selection_reason)} · ${escapeHtml(status)}</span>
        </span>
        <i class="status-dot ${escapeHtml(status)}" aria-hidden="true"></i>
      `;
      button.addEventListener("click", () => setCurrentById(sample.candidate_id));
      els.sampleList.appendChild(button);
    }
  }

  function renderDatasetTabs() {
    const options = datasetOptions();
    els.datasetTabs.innerHTML = "";
    for (const option of options) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `dataset-tab ${state.dataset === option.id ? "active" : ""}`;
      button.textContent = option.label;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", state.dataset === option.id ? "true" : "false");
      button.addEventListener("click", () => {
        state.dataset = option.id;
        const first = filteredSamples()[0];
        if (first) {
          state.currentIndex = samples.findIndex((s) => s.candidate_id === first.candidate_id);
        }
        render();
      });
      els.datasetTabs.appendChild(button);
    }
  }

  function renderSample() {
    const sample = currentSample();
    if (!sample) {
      els.sampleId.textContent = "";
      els.sampleGroup.textContent = "";
      els.sampleMode.textContent = "";
      els.queryText.textContent = "No samples match the current filters.";
      els.referenceAnswer.textContent = "";
      els.modelAnswer.textContent = "";
      els.evidenceText.textContent = "";
      els.metricStrip.innerHTML = "";
      els.positionText.textContent = "0 of 0";
      return;
    }
    const annotation = getAnnotation(sample);
    els.sampleId.textContent = sample.candidate_id;
    els.sampleGroup.textContent = sample.selection_reason;
    els.sampleMode.textContent = `${sample.dataset_name || sample.dataset || ""} · ${sample.model_name || sample.model_family || ""} · ${sample.method || ""}`;
    els.queryText.textContent = sample.query_text;
    els.referenceAnswer.textContent = sample.reference_answer;
    els.modelAnswer.textContent = sample.model_answer;
    els.evidenceText.textContent = sample.selected_document_text || "No retrieved evidence.";
    els.metricStrip.innerHTML = [
      ["F1", sample.answer_f1],
      ["Semantic", sample.semantic_similarity],
      ["nDCG@10", sample.ndcg_at_10],
      ["MRR@10", sample.mrr_at_10],
      ["Docs", sample.docs_used],
      ["Tokens", sample.total_tokens],
    ]
      .map(([label, value]) => `<span class="metric-pill">${label}: ${escapeHtml(value ?? "")}</span>`)
      .join("");

    els.annotationForm.elements.rating.value = annotation.rating || "";
    els.annotationForm.elements.notes.value = annotation.notes || "";
    const visible = filteredSamples();
    const visibleIndex = visible.findIndex((item) => item.candidate_id === sample.candidate_id);
    els.positionText.textContent = `${visibleIndex + 1} of ${visible.length}`;
  }

  function updateFromForm() {
    const sample = currentSample();
    if (!sample) return;
    const annotation = getAnnotation(sample);
    annotation.rating = els.annotationForm.elements.rating.value;
    annotation.notes = els.annotationForm.elements.notes.value;
    saveState();
    renderList();
  }

  function exportRows(rows = samples) {
    return rows.map((sample) => {
      const annotation = getAnnotation(sample);
      return { ...sample, rating: annotation.rating || "", notes: annotation.notes || "" };
    });
  }

  function download(filename, mimeType, content) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function exportJson() {
    const payload = { exported_at: new Date().toISOString(), rows: exportRows() };
    download("rag_annotations_export.json", "application/json", JSON.stringify(payload, null, 2));
  }

  function exportCsv() {
    downloadCsv("rag_annotations.csv", exportRows());
  }

  function downloadCsv(filename, rows) {
    if (!rows.length) {
      alert("No rows to export.");
      return;
    }
    const headers = Object.keys(rows[0]);
    const csv = [
      headers.join(","),
      ...rows.map((row) => headers.map((h) => csvEscape(row[h])).join(",")),
    ].join("\n");
    download(filename, "text/csv", csv);
  }

  function parseCsv(text) {
    // RFC-4180 compliant parser: handles quoted fields, embedded commas/newlines.
    const rows = [];
    let field = "";
    let inQuotes = false;
    let currentRow = [];
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const next = text[i + 1];
      if (inQuotes) {
        if (ch === '"' && next === '"') { field += '"'; i++; }
        else if (ch === '"') { inQuotes = false; }
        else { field += ch; }
      } else {
        if (ch === '"') { inQuotes = true; }
        else if (ch === ',') { currentRow.push(field); field = ""; }
        else if (ch === '\n' || (ch === '\r' && next === '\n')) {
          if (ch === '\r') i++;
          currentRow.push(field); field = "";
          rows.push(currentRow); currentRow = [];
        } else { field += ch; }
      }
    }
    if (field || currentRow.length) { currentRow.push(field); rows.push(currentRow); }
    if (!rows.length) return [];
    const headers = rows[0];
    return rows.slice(1).filter((r) => r.some((c) => c.trim())).map((r) => {
      const obj = {};
      headers.forEach((h, i) => { obj[h.trim()] = r[i] ?? ""; });
      return obj;
    });
  }

  function importCsv(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const rows = parseCsv(String(reader.result || ""));
        let imported = 0;
        for (const row of rows) {
          const id = row.candidate_id || row.annotation_id;
          if (!id) continue;
          const rating = row.rating || row.annotator_1_rating || row.final_rating || "";
          const notes  = row.notes  || row.annotator_1_notes  || row.final_notes  || "";
          if (!rating && !notes) continue;
          state.annotations[id] = { ...emptyAnnotation(), ...state.annotations[id], rating, notes };
          imported++;
        }
        saveState();
        render();
        alert(`Imported ${imported} annotation${imported !== 1 ? "s" : ""} from CSV.`);
      } catch (error) {
        alert(`CSV import failed: ${error.message}`);
      }
    };
    reader.readAsText(file);
  }

  function importJson(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result || "{}"));
        const rows = Array.isArray(parsed.rows) ? parsed.rows : [];
        for (const row of rows) {
          if (!row.candidate_id) continue;
          state.annotations[row.candidate_id] = {
            ...emptyAnnotation(),
            rating: row.rating || row.annotator_1_rating || row.final_rating || "",
            notes: row.notes || row.annotator_1_notes || row.final_notes || "",
          };
        }
        saveState();
        render();
      } catch (error) {
        alert(`Import failed: ${error.message}`);
      }
    };
    reader.readAsText(file);
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    if (/[",\n\r]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function render() {
    renderDatasetTabs();
    renderList();
    renderSample();
  }

  els.filterSelect.addEventListener("change", () => {
    state.filter = els.filterSelect.value;
    render();
  });
  els.searchInput.addEventListener("input", () => {
    state.search = els.searchInput.value;
    render();
  });
  els.nextBtn.addEventListener("click", () => {
    const visible = filteredSamples();
    const current = currentSample();
    const visibleIndex = visible.findIndex((s) => s.candidate_id === current?.candidate_id);
    const next = visible[Math.min(visible.length - 1, visibleIndex + 1)];
    if (next) {
      state.currentIndex = samples.findIndex((s) => s.candidate_id === next.candidate_id);
    }
    render();
  });
  els.prevBtn.addEventListener("click", () => {
    const visible = filteredSamples();
    const current = currentSample();
    const visibleIndex = visible.findIndex((s) => s.candidate_id === current?.candidate_id);
    const previous = visible[Math.max(0, visibleIndex - 1)];
    if (previous) {
      state.currentIndex = samples.findIndex((s) => s.candidate_id === previous.candidate_id);
    }
    render();
  });
  els.annotationForm.addEventListener("input", updateFromForm);
  els.exportCsvBtn.addEventListener("click", exportCsv);
  els.exportJsonBtn.addEventListener("click", exportJson);
  els.importCsvInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) importCsv(file);
    event.target.value = "";
  });
  els.importJsonInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) importJson(file);
    event.target.value = "";
  });

  render();
})();
