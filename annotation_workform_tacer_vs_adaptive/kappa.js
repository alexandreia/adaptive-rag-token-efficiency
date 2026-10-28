(function () {
  const VALID_RATINGS = new Set(["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "NOT_ENOUGH_INFO"]);
  const state = {
    fileA: null,
    fileB: null,
    rowsA: [],
    rowsB: [],
    mergedRows: [],
  };

  const els = {
    fileA: document.getElementById("fileA"),
    fileB: document.getElementById("fileB"),
    fileAName: document.getElementById("fileAName"),
    fileBName: document.getElementById("fileBName"),
    dropA: document.getElementById("dropA"),
    dropB: document.getElementById("dropB"),
    idColumn: document.getElementById("idColumn"),
    ratingColumn: document.getElementById("ratingColumn"),
    calculateBtn: document.getElementById("calculateBtn"),
    clearBtn: document.getElementById("clearBtn"),
    downloadBtn: document.getElementById("downloadBtn"),
    summaryPanel: document.getElementById("summaryPanel"),
    messagePanel: document.getElementById("messagePanel"),
    disagreementTable: document.querySelector("#disagreementTable tbody"),
  };

  function parseCsv(text) {
    const rows = [];
    let row = [];
    let cell = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      const next = text[i + 1];
      if (inQuotes) {
        if (char === "\"" && next === "\"") {
          cell += "\"";
          i += 1;
        } else if (char === "\"") {
          inQuotes = false;
        } else {
          cell += char;
        }
      } else if (char === "\"") {
        inQuotes = true;
      } else if (char === ",") {
        row.push(cell);
        cell = "";
      } else if (char === "\n") {
        row.push(cell);
        rows.push(row);
        row = [];
        cell = "";
      } else if (char !== "\r") {
        cell += char;
      }
    }

    if (cell || row.length) {
      row.push(cell);
      rows.push(row);
    }
    if (!rows.length) return [];

    const headers = rows[0].map((header) => header.trim());
    return rows.slice(1)
      .filter((values) => values.some((value) => value.trim()))
      .map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] || ""])));
  }

  async function loadFile(file, side) {
    const text = await file.text();
    state[`file${side}`] = file;
    state[`rows${side}`] = parseCsv(text);
    els[`file${side}Name`].textContent = `${file.name} (${state[`rows${side}`].length} rows)`;
    els.downloadBtn.disabled = true;
    clearResults();
  }

  function setupDrop(zone, input, side) {
    input.addEventListener("change", () => {
      const file = input.files && input.files[0];
      if (file) loadFile(file, side).catch(showError);
    });

    zone.addEventListener("dragover", (event) => {
      event.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (event) => {
      event.preventDefault();
      zone.classList.remove("drag-over");
      const file = event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) loadFile(file, side).catch(showError);
    });
  }

  function chooseColumn(rows, requested, candidates) {
    const headers = new Set(Object.keys(rows[0] || {}));
    if (requested !== "auto") {
      if (!headers.has(requested)) throw new Error(`Missing column: ${requested}`);
      return requested;
    }
    const found = candidates.find((candidate) => headers.has(candidate));
    if (!found) throw new Error(`Could not find any of these columns: ${candidates.join(", ")}`);
    return found;
  }

  function normalizeRating(value) {
    const rating = String(value || "").trim().toUpperCase();
    return VALID_RATINGS.has(rating) ? rating : rating;
  }

  function calculate() {
    if (!state.rowsA.length || !state.rowsB.length) {
      throw new Error("Please add both annotator CSV files first.");
    }

    const idCandidates = ["candidate_id", "sample_id", "query_id"];
    const ratingCandidates = ["rating", "annotator_rating", "annotator_1_rating", "annotator_2_rating", "final_rating", "annotator_1_label", "annotator_2_label", "final_label"];
    const idA = chooseColumn(state.rowsA, els.idColumn.value, idCandidates);
    const idB = chooseColumn(state.rowsB, els.idColumn.value, idCandidates);
    const ratingA = chooseColumn(state.rowsA, els.ratingColumn.value, ratingCandidates);
    const ratingB = chooseColumn(state.rowsB, els.ratingColumn.value, ratingCandidates);

    const mapB = new Map(state.rowsB.map((row) => [String(row[idB] || "").trim(), row]));
    const pairs = [];
    const missing = [];

    for (const rowA of state.rowsA) {
      const id = String(rowA[idA] || "").trim();
      if (!id) continue;
      const rowB = mapB.get(id);
      if (!rowB) {
        missing.push(id);
        continue;
      }
      const a = normalizeRating(rowA[ratingA]);
      const b = normalizeRating(rowB[ratingB]);
      if (!a || !b) continue;
      pairs.push({
        id,
        a,
        b,
        query: rowA.query_text || rowA.claim_or_question || rowB.query_text || rowB.claim_or_question || "",
      });
    }

    if (!pairs.length) {
      throw new Error("No matched rows had ratings in both files.");
    }

    const stats = agreementStats(pairs);
    state.mergedRows = pairs.map((pair) => ({
      item_id: pair.id,
      annotator_1_rating: pair.a,
      annotator_2_rating: pair.b,
      agreement: pair.a === pair.b ? "AGREE" : "DISAGREE",
      question_or_claim: pair.query,
    }));

    renderStats(stats, pairs, { idA, idB, ratingA, ratingB, missing });
    renderDisagreements(pairs.filter((pair) => pair.a !== pair.b));
    els.downloadBtn.disabled = false;
  }

  function agreementStats(pairs) {
    const total = pairs.length;
    const agreements = pairs.filter((pair) => pair.a === pair.b).length;
    const observed = agreements / total;
    const labels = Array.from(new Set(pairs.flatMap((pair) => [pair.a, pair.b]))).sort();
    let expected = 0;
    for (const label of labels) {
      const pA = pairs.filter((pair) => pair.a === label).length / total;
      const pB = pairs.filter((pair) => pair.b === label).length / total;
      expected += pA * pB;
    }
    const kappa = expected === 1 ? null : (observed - expected) / (1 - expected);
    return { total, agreements, disagreements: total - agreements, observed, expected, kappa, labels };
  }

  function renderStats(stats, pairs, columns) {
    els.summaryPanel.innerHTML = [
      ["Matched rated items", stats.total],
      ["Raw agreement", formatPercent(stats.observed)],
      ["Cohen's kappa", stats.kappa === null ? "n/a" : stats.kappa.toFixed(3)],
      ["Disagreements", stats.disagreements],
    ].map(([label, value]) => `<div class="stat-line"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");

    const invalid = pairs.filter((pair) => !VALID_RATINGS.has(pair.a) || !VALID_RATINGS.has(pair.b));
    const messages = [
      `Matched on ${columns.idA} / ${columns.idB}; compared ${columns.ratingA} / ${columns.ratingB}.`,
      columns.missing.length ? `${columns.missing.length} item(s) from annotator 1 were not found in annotator 2.` : "",
      invalid.length ? `${invalid.length} pair(s) used non-standard ratings. Check spelling if this is unexpected.` : "",
    ].filter(Boolean);
    els.messagePanel.innerHTML = messages.map((message) => `<p>${escapeHtml(message)}</p>`).join("");
  }

  function renderDisagreements(rows) {
    els.disagreementTable.innerHTML = rows.map((row) => `
      <tr>
        <td>${escapeHtml(row.id)}</td>
        <td>${escapeHtml(row.a)}</td>
        <td>${escapeHtml(row.b)}</td>
        <td>${escapeHtml(row.query)}</td>
      </tr>
    `).join("") || `<tr><td colspan="4">No disagreements among matched rated items.</td></tr>`;
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    return /[",\n]/.test(text) ? `"${text.replaceAll("\"", "\"\"")}"` : text;
  }

  function downloadMerged() {
    const headers = ["item_id", "annotator_1_rating", "annotator_2_rating", "agreement", "question_or_claim"];
    const csv = [
      headers.join(","),
      ...state.mergedRows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
    ].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "rag_kappa_merged_annotations.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function clearResults() {
    els.summaryPanel.innerHTML = "";
    els.messagePanel.innerHTML = "";
    els.disagreementTable.innerHTML = "";
  }

  function clearAll() {
    state.fileA = null;
    state.fileB = null;
    state.rowsA = [];
    state.rowsB = [];
    state.mergedRows = [];
    els.fileA.value = "";
    els.fileB.value = "";
    els.fileAName.textContent = "Drop or choose file";
    els.fileBName.textContent = "Drop or choose file";
    els.downloadBtn.disabled = true;
    clearResults();
  }

  function showError(error) {
    els.messagePanel.innerHTML = `<p class="error-message">${escapeHtml(error.message || error)}</p>`;
  }

  function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#39;",
    }[char]));
  }

  setupDrop(els.dropA, els.fileA, "A");
  setupDrop(els.dropB, els.fileB, "B");
  window.RAG_KAPPA_CALCULATOR = { parseCsv, agreementStats };
  els.calculateBtn.addEventListener("click", () => {
    try {
      calculate();
    } catch (error) {
      showError(error);
    }
  });
  els.clearBtn.addEventListener("click", clearAll);
  els.downloadBtn.addEventListener("click", downloadMerged);
  clearResults();
}());
