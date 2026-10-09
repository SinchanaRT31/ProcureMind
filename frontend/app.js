const state = {
  dashboard: null,
  transactions: [],
  investigationReport: null,
};

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatAmount(amount) {
  return currencyFormatter.format(amount);
}

function riskClass(level) {
  const normalized = String(level ?? "").toLowerCase().split(/\s+/)[0];
  return ["high", "medium", "low", "critical"].includes(normalized) ? normalized : "low";
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const payload = await response.json();
      if (payload.error) message = payload.error;
    } catch (_) {
      // Keep the HTTP status message when a response has no JSON body.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function renderStats(summary) {
  el("stats-grid").innerHTML = `
    <article class="stat-card">
      <div class="stat-label">Total transactions</div>
      <div class="stat-value">${summary.total_transactions.toLocaleString()}</div>
      <div class="stat-note">${summary.transactions_delta}</div>
    </article>
    <article class="stat-card">
      <div class="stat-label">Total anomalies</div>
      <div class="stat-value">${summary.total_anomalies.toLocaleString()}</div>
      <div class="stat-note">${summary.anomalies_delta}</div>
    </article>
    <article class="stat-card">
      <div class="stat-label">High-risk vendors</div>
      <div class="stat-value">${summary.high_risk_vendors.toLocaleString()}</div>
      <div class="stat-note">${summary.vendors_delta}</div>
    </article>
    <article class="stat-card">
      <div class="stat-label">Potential savings</div>
      <div class="stat-value">${formatAmount(summary.potential_savings)}</div>
      <div class="stat-note">${summary.savings_delta}</div>
    </article>
  `;
}

function renderRiskTrend(trend) {
  el("risk-trend").innerHTML = trend
    .map(
      (item) => `
        <div class="trend-item">
          <div class="trend-bar" style="height: ${item.score * 2}px"></div>
          <strong>${item.day}</strong>
          <span class="helper">${item.score}</span>
        </div>
      `,
    )
    .join("");
}

function renderWorkflow(items) {
  el("workflow-list").innerHTML = items.map((item) => `<li>${item}</li>`).join("");
}

function renderTransactions(transactions) {
  state.transactions = transactions;
  el("transactions-body").innerHTML = transactions
    .map(
      (item) => `
        <tr>
          <td>${item.invoice}</td>
          <td>${item.vendor}</td>
          <td>${item.department}</td>
          <td>${item.date}</td>
          <td>${formatAmount(item.amount)}</td>
          <td><span class="badge ${item.level}">${item.risk}/100</span></td>
          <td>${item.status}</td>
          <td><button class="button secondary open-investigation" data-transaction-id="${escapeHtml(item.id)}" type="button">Open case</button></td>
        </tr>
      `,
    )
    .join("");
}

function renderVendors(vendors) {
  el("vendors-list").innerHTML = vendors
    .map(
      (item) => `
        <article class="stack-item">
          <div class="stack-item-head">
            <strong>${item.name}</strong>
            <span class="badge ${riskClass(item.level)}">${item.score}/100</span>
          </div>
          <div class="stack-item-meta">
            <span>${item.level}</span>
            <span>${item.note}</span>
          </div>
          <div class="helper">
            Anomalies: ${item.anomalies} | Duplicate invoices: ${item.duplicate_invoices} | Price deviation: ${item.price_deviation}
          </div>
        </article>
      `,
    )
    .join("");
}

function renderHighlight(transaction) {
  const factors = transaction.explanation.factors
    .map(
      (item) => `
        <div class="factor-row">
          <span>${item.label}</span>
          <strong>${item.weight}</strong>
        </div>
      `,
    )
    .join("");

  el("highlight-panel").innerHTML = `
    <div class="highlight-score">
      <div class="score-box">${transaction.risk}</div>
      <div>
        <strong>${transaction.invoice} · ${transaction.vendor}</strong>
        <div class="helper">${transaction.reason}</div>
      </div>
    </div>
    <div>${transaction.explanation.summary}</div>
    <div class="factor-list">${factors}</div>
  `;
}

function renderDuplicates(duplicates) {
  el("duplicates-list").innerHTML = duplicates
    .map(
      (item) => `
        <article class="stack-item">
          <div class="stack-item-head">
            <strong>${item.primary_invoice} ↔ ${item.related_invoice}</strong>
            <span class="badge high">${item.similarity}%</span>
          </div>
          <div class="stack-item-meta">
            <span>${item.vendor}</span>
            <span>${item.days_apart} day gap</span>
          </div>
          <div class="helper">${item.note}</div>
        </article>
      `,
    )
    .join("");
}

function caseCard(caseItem) {
  const transaction = state.dashboard?.transactions.find((item) => item.id === caseItem.transaction_id);
  const events = (caseItem.review_history || []).map((event) => `
    <li class="review-event">
      <strong>${escapeHtml(event.event_type.replaceAll("_", " "))}</strong>
      <span class="helper">${escapeHtml(event.timestamp)}${event.reviewer_id ? ` · ${escapeHtml(event.reviewer_id)}` : ""}</span>
      <div>${escapeHtml(JSON.stringify(event.changes || {}))}</div>
    </li>
  `).join("");
  const decisions = ["", "CONFIRMED_ISSUE", "NO_ISSUE_FOUND", "NEEDS_MORE_INFORMATION", "ESCALATED"];
  const decisionOptions = decisions.map((decision) => `<option value="${decision}" ${decision === (caseItem.investigation_decision || "") ? "selected" : ""}>${decision || "No decision recorded"}</option>`).join("");
  return `
    <article class="stack-item case-card" data-case-id="${escapeHtml(caseItem.id)}">
      <div class="stack-item-head">
        <strong>Case #${escapeHtml(caseItem.id)}</strong>
        <span class="badge ${riskClass(caseItem.priority || "low")}">${escapeHtml(caseItem.priority || "No priority")}</span>
      </div>
      <div class="stack-item-meta">
        <span>${caseItem.owner ? `Owner: ${escapeHtml(caseItem.owner)} (sample)` : "Reviewer identity is optional"}</span>
        <span>Transaction: ${escapeHtml(caseItem.transaction_id)}</span>
      </div>
      ${transaction ? `<div class="review-transaction"><strong>${escapeHtml(transaction.invoice)} · ${escapeHtml(transaction.vendor)}</strong><div class="stack-item-meta"><span>${escapeHtml(transaction.department)}</span><span>${escapeHtml(transaction.date)}</span><span>${formatAmount(transaction.amount)}</span><span>Sample risk: ${escapeHtml(transaction.risk)}/100 (${escapeHtml(transaction.level)})</span></div><p>${escapeHtml(transaction.reason)}</p><p class="helper">${escapeHtml(transaction.explanation?.summary || "")}</p></div>` : ""}
      <div class="case-actions">
        <div class="review-controls">
          <label>Review status
            <select class="case-review-status">
              ${["OPEN", "REVIEWING", "RESOLVED"].map((status) => `<option ${caseItem.review_status === status ? "selected" : ""}>${status}</option>`).join("")}
            </select>
          </label>
          <label>Investigation decision
            <select class="case-decision">${decisionOptions}</select>
            <span class="helper">A recorded reviewer decision, not an automatic fraud finding.</span>
          </label>
          <label>Reviewer identifier (optional)
            <input class="case-reviewer" type="text" maxlength="200" value="${escapeHtml(caseItem.reviewer_id || "")}" placeholder="Not authenticated" />
          </label>
          <label>Existing investigation notes
            <div class="existing-notes">${escapeHtml(caseItem.investigation_notes || "No notes recorded.")}</div>
          </label>
          <label>Add a note
            <textarea class="case-note" rows="3" maxlength="10000" placeholder="Add an investigation note"></textarea>
          </label>
          <div class="case-status-row">
            <button class="button primary save-case" type="button">Save review</button>
            <span class="case-save-message" role="status"></span>
          </div>
          <details class="review-history">
            <summary>Review history (${(caseItem.review_history || []).length})</summary>
            <ol>${events || "<li>No review events yet.</li>"}</ol>
          </details>
      </div>
    </article>
  `;
}

function renderCases(cases) {
  const sampleIds = new Set((state.dashboard?.transactions || []).map((item) => item.id));
  const sampleCases = cases.filter((caseItem) => sampleIds.has(caseItem.transaction_id));
  el("cases-list").innerHTML = sampleCases.length
    ? sampleCases.map(caseCard).join("")
    : '<p class="helper">No sample review cases are available.</p>';
}

function createTextElement(tagName, text, className = "") {
  const node = document.createElement(tagName);
  node.textContent = text == null ? "" : String(text);
  if (className) node.className = className;
  return node;
}

function reportLabel(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function appendReportValue(container, value) {
  if (Array.isArray(value)) {
    if (!value.length) {
      container.append(createTextElement("p", "No items were returned.", "helper"));
      return;
    }
    const list = document.createElement("ul");
    for (const item of value) {
      const entry = document.createElement("li");
      appendReportValue(entry, item);
      list.append(entry);
    }
    container.append(list);
    return;
  }
  if (value && typeof value === "object") {
    const details = document.createElement("dl");
    details.className = "report-details";
    for (const [key, item] of Object.entries(value)) {
      const term = createTextElement("dt", reportLabel(key));
      const description = document.createElement("dd");
      appendReportValue(description, item);
      details.append(term, description);
    }
    container.append(details);
    return;
  }
  container.append(createTextElement("p", value == null ? "Not provided." : value));
}

function appendReportSection(container, title, value) {
  if (value === undefined || value === null) return;
  const section = document.createElement("section");
  section.className = "report-section";
  section.append(createTextElement("h4", title));
  appendReportValue(section, value);
  container.append(section);
}

function renderInvestigationReport(report) {
  const container = el("investigation-results");
  container.replaceChildren();
  const header = document.createElement("div");
  header.className = "report-heading";
  header.append(
    createTextElement("h3", `Trusted investigation · ${report.transaction_id || "Transaction"}`),
    createTextElement("p", `Report schema ${report.schema_version || "not specified"} · generated from the trusted dataset`, "helper"),
  );
  container.append(header);

  const whyFlagged = report.why_flagged;
  if (whyFlagged && typeof whyFlagged === "object") {
    appendReportSection(container, "Model anomaly signal", whyFlagged.model_prediction);
    appendReportSection(container, "SHAP explanation", whyFlagged.model_explanation);
  }
  appendReportSection(container, "Supporting findings and risk priority", report.supporting_findings);
  appendReportSection(container, "Verification and inconclusive or conflicting findings", report.inconclusive_or_conflicting);
  appendReportSection(container, "Missing information", report.missing_information);
  appendReportSection(container, "Follow-up checks", report.follow_up_checks);
  appendReportSection(container, "Limitations", report.limitations);
  container.append(createTextElement(
    "p",
    "Model anomaly signals and investigation evidence are not human decisions. An anomaly score is not a fraud probability and does not prove fraud.",
    "report-caution",
  ));
}

function appendReviewField(container, labelText, control) {
  const label = document.createElement("label");
  label.append(createTextElement("span", labelText), control);
  container.append(label);
}

function createReviewSelect(className, options, selectedValue) {
  const select = document.createElement("select");
  select.className = className;
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    element.selected = option.value === (selectedValue ?? "");
    select.append(element);
  }
  return select;
}

function renderReviewHistory(events) {
  const details = document.createElement("details");
  details.className = "review-history";
  details.append(createTextElement("summary", `Review history (${events.length})`));
  const list = document.createElement("ol");
  if (!events.length) list.append(createTextElement("li", "No review events yet."));
  for (const event of events) {
    const item = document.createElement("li");
    item.className = "review-event";
    const reviewer = event.reviewer_id ? ` · ${event.reviewer_id}` : "";
    item.append(
      createTextElement("strong", String(event.event_type || "Review event").replaceAll("_", " ")),
      createTextElement("span", `${event.timestamp || ""}${reviewer}`, "helper"),
      createTextElement("div", JSON.stringify(event.changes || {})),
    );
    list.append(item);
  }
  details.append(list);
  return details;
}

function renderTrustedReview(transactionId, caseItem = null, history = []) {
  const container = el("trusted-review");
  container.replaceChildren();
  delete container.dataset.caseId;
  const heading = createTextElement("h3", "Human review · separate from model findings");
  container.append(heading);
  if (!caseItem) {
    container.append(createTextElement("p", "No review case is recorded for this trusted transaction."));
    const openButton = createTextElement("button", "Open review case", "button primary open-trusted-review");
    openButton.type = "button";
    container.append(openButton);
    return;
  }

  container.dataset.caseId = caseItem.id;
  container.append(createTextElement("p", `Case ${caseItem.id} · ${transactionId}`, "helper"));
  const fields = document.createElement("dl");
  fields.className = "review-summary";
  for (const [label, value] of [
    ["Persisted review status", caseItem.review_status],
    ["Investigation decision", caseItem.investigation_decision || "No decision recorded"],
    ["Reviewer", caseItem.reviewer_id || "Not specified"],
  ]) {
    fields.append(createTextElement("dt", label), createTextElement("dd", value));
  }
  container.append(fields);
  container.append(createTextElement("h4", "Persisted investigation notes"));
  container.append(createTextElement("p", caseItem.investigation_notes || "No notes recorded.", "existing-notes"));

  const form = document.createElement("form");
  form.className = "trusted-review-form";
  const controls = document.createElement("div");
  controls.className = "trusted-review-controls";
  appendReviewField(controls, "Review status", createReviewSelect("trusted-review-status", [
    { value: "OPEN", label: "OPEN" },
    { value: "REVIEWING", label: "REVIEWING" },
    { value: "RESOLVED", label: "RESOLVED" },
  ], caseItem.review_status));
  appendReviewField(controls, "Investigation decision", createReviewSelect("trusted-review-decision", [
    { value: "", label: "No decision recorded" },
    { value: "CONFIRMED_ISSUE", label: "CONFIRMED_ISSUE" },
    { value: "NO_ISSUE_FOUND", label: "NO_ISSUE_FOUND" },
    { value: "NEEDS_MORE_INFORMATION", label: "NEEDS_MORE_INFORMATION" },
    { value: "ESCALATED", label: "ESCALATED" },
  ], caseItem.investigation_decision));
  const reviewer = document.createElement("input");
  reviewer.className = "trusted-reviewer-id";
  reviewer.type = "text";
  reviewer.maxLength = 200;
  reviewer.value = caseItem.reviewer_id || "";
  reviewer.placeholder = "Optional; this app does not authenticate reviewers";
  appendReviewField(controls, "Reviewer identifier (optional)", reviewer);
  const note = document.createElement("textarea");
  note.className = "trusted-review-note";
  note.rows = 3;
  note.maxLength = 10000;
  note.placeholder = "Add a review note";
  appendReviewField(controls, "Add a note", note);
  form.append(controls);
  const actions = document.createElement("div");
  actions.className = "case-status-row";
  const saveButton = createTextElement("button", "Save review", "button primary save-trusted-review");
  saveButton.type = "submit";
  actions.append(saveButton, createTextElement("span", "", "trusted-review-status-message"));
  form.append(actions);
  container.append(form, renderReviewHistory(history));

  if (caseItem.review_status === "RESOLVED") {
    const reopenButton = createTextElement("button", "Open a new review case", "button secondary open-trusted-review");
    reopenButton.type = "button";
    container.append(reopenButton);
  }
}

async function refreshTrustedReview(transactionId, preferredCaseId = null) {
  const container = el("trusted-review");
  container.setAttribute("aria-busy", "true");
  container.replaceChildren(createTextElement("p", "Loading persisted review information…", "helper"));
  try {
    const payload = await fetchJson("/api/cases");
    const matches = (Array.isArray(payload.cases) ? payload.cases : [])
      .filter((item) => item.transaction_id === transactionId);
    const active = matches.find((item) => item.review_status !== "RESOLVED");
    const selected = (preferredCaseId && matches.find((item) => item.id === preferredCaseId))
      || active
      || matches[matches.length - 1];
    if (!selected) {
      renderTrustedReview(transactionId);
      return true;
    }
    const caseId = encodeURIComponent(selected.id);
    const [detailPayload, historyPayload] = await Promise.all([
      fetchJson(`/api/cases/${caseId}`),
      fetchJson(`/api/cases/${caseId}/history`),
    ]);
    renderTrustedReview(transactionId, detailPayload.case, Array.isArray(historyPayload.review_history) ? historyPayload.review_history : []);
    return true;
  } catch (error) {
    container.replaceChildren(createTextElement("h3", "Human review"), createTextElement("p", `Could not load persisted review information: ${apiErrorMessage(error, "review")}`, "error-message"));
    return false;
  } finally {
    container.setAttribute("aria-busy", "false");
  }
}

function apiErrorMessage(error, operation) {
  if (error.status === 400) return error.message || "The transaction ID or review fields are invalid.";
  if (error.status === 404) return operation === "report" ? "No exact transaction ID match was found." : "The review case or transaction was not found.";
  if (error.status === 409) return "Multiple exact transaction IDs were found. The investigation cannot continue until the source data is unambiguous.";
  if (error.status === 503) return "The trusted transaction or report service is unavailable. Try again later.";
  if (error.status >= 500) return "The request could not be completed. Try again later.";
  if (!error.status) return "The ProcureMind service could not be reached.";
  return error.message || `Request failed (${error.status}).`;
}

async function submitInvestigation(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const input = form.elements.transaction_id;
  const transactionId = input.value.trim();
  const button = el("investigation-submit");
  const status = el("investigation-status");
  const results = el("investigation-results");
  const review = el("trusted-review");
  if (!transactionId) {
    status.textContent = "Enter a trusted transaction ID.";
    return;
  }
  button.disabled = true;
  status.className = "investigation-status";
  status.textContent = "Generating the investigation report…";
  results.replaceChildren();
  review.replaceChildren();
  results.setAttribute("aria-busy", "true");
  try {
    const params = new URLSearchParams({ transaction_id: transactionId });
    const payload = await fetchJson(`/api/investigations/report?${params.toString()}`);
    if (!payload.report || typeof payload.report !== "object" || payload.report.transaction_id !== transactionId) {
      const error = new Error("The report response did not match the requested transaction.");
      error.status = 500;
      throw error;
    }
    state.investigationReport = payload.report;
    renderInvestigationReport(payload.report);
    status.textContent = "Report loaded from the trusted transaction source.";
    await refreshTrustedReview(transactionId);
  } catch (error) {
    state.investigationReport = null;
    status.className = "investigation-status error-message";
    status.textContent = apiErrorMessage(error, "report");
  } finally {
    button.disabled = false;
    results.setAttribute("aria-busy", "false");
  }
}

async function openTrustedReview(event) {
  const button = event.target.closest(".open-trusted-review");
  if (!button || !state.investigationReport) return;
  const transactionId = state.investigationReport.transaction_id;
  button.disabled = true;
  button.textContent = "Opening review case…";
  try {
    const payload = await fetchJson("/api/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transaction_id: transactionId }),
    });
    const refreshed = await refreshTrustedReview(transactionId, payload.case?.id || null);
    el("investigation-status").textContent = refreshed
      ? "Review case opened or retrieved from persisted records."
      : "Review case saved, but its persisted details could not be refreshed.";
  } catch (error) {
    button.disabled = false;
    button.textContent = "Open review case";
    const message = createTextElement("p", apiErrorMessage(error, "review"), "error-message");
    el("trusted-review").append(message);
  }
}

async function saveTrustedReview(event) {
  if (!event.target.matches(".trusted-review-form")) return;
  event.preventDefault();
  const form = event.target;
  const container = form.closest("#trusted-review");
  const caseId = container.dataset.caseId;
  const transactionId = state.investigationReport?.transaction_id;
  const button = form.querySelector(".save-trusted-review");
  const message = form.querySelector(".trusted-review-status-message");
  if (!caseId || !transactionId) return;
  const body = {
    review_status: form.querySelector(".trusted-review-status").value,
    investigation_decision: form.querySelector(".trusted-review-decision").value || null,
    reviewer_id: form.querySelector(".trusted-reviewer-id").value,
  };
  const note = form.querySelector(".trusted-review-note").value;
  if (note.trim()) body.note = note;
  button.disabled = true;
  message.textContent = "Saving review…";
  try {
    await fetchJson(`/api/cases/${encodeURIComponent(caseId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const refreshed = await refreshTrustedReview(transactionId, caseId);
    el("investigation-status").textContent = refreshed
      ? "Review changes saved and reloaded from persisted records."
      : "Review changes saved, but persisted details could not be refreshed.";
  } catch (error) {
    message.textContent = apiErrorMessage(error, "review");
    button.disabled = false;
  }
}

async function loadDashboard() {
  const data = await fetchJson("/api/dashboard");
  state.dashboard = data;
  el("topbar-alert").textContent = `${data.alerts.last_24_hours} high-priority alerts in the last 24 hours. ${data.alerts.headline}`;
  renderStats(data.summary);
  renderRiskTrend(data.risk_trend);
  renderWorkflow(data.workflow);
  renderTransactions(data.transactions);
  renderVendors(data.vendors);
  renderHighlight(data.highlight_transaction);
  renderDuplicates(data.duplicates);
  renderCases(data.cases);
}

async function applyFilters(event) {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  const params = new URLSearchParams();
  for (const [key, value] of formData.entries()) {
    if (value) {
      params.append(key, value);
    }
  }
  const data = await fetchJson(`/api/transactions?${params.toString()}`);
  renderTransactions(data.transactions);
}

async function saveCase(event) {
  const button = event.target.closest(".save-case");
  if (!button) {
    return;
  }

  const card = button.closest("[data-case-id]");
  const caseId = card.dataset.caseId;
  const reviewStatus = card.querySelector(".case-review-status").value;
  const decision = card.querySelector(".case-decision").value;
  const reviewerId = card.querySelector(".case-reviewer").value;
  const note = card.querySelector(".case-note").value;
  const message = card.querySelector(".case-save-message");

  button.disabled = true;
  button.textContent = "Saving...";

  try {
    await fetchJson(`/api/cases/${encodeURIComponent(caseId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_status: reviewStatus, investigation_decision: decision || null, reviewer_id: reviewerId, ...(note.trim() ? { note } : {}) }),
    });
    message.textContent = "Saved.";
    button.textContent = "Saved";
    setTimeout(() => loadDashboard().catch(() => {
      message.textContent = "Saved, but the refreshed dashboard could not be loaded.";
      button.disabled = false;
      button.textContent = "Save review";
    }), 700);
  } catch (error) {
    message.textContent = error.message;
    button.disabled = false;
    button.textContent = "Save review";
  }
}

async function openInvestigation(event) {
  const button = event.target.closest(".open-investigation");
  if (!button) return;
  button.disabled = true;
  const originalText = button.textContent;
  button.textContent = "Opening...";
  try {
    await fetchJson("/api/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transaction_id: button.dataset.transactionId }),
    });
    await loadDashboard();
    el("cases-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    button.textContent = error.message;
    button.disabled = false;
    return;
  }
  button.textContent = originalText;
  button.disabled = false;
}

document.getElementById("filters-form").addEventListener("submit", applyFilters);
document.getElementById("investigation-form").addEventListener("submit", submitInvestigation);
document.getElementById("cases-list").addEventListener("click", saveCase);
document.getElementById("transactions-body").addEventListener("click", openInvestigation);
document.getElementById("trusted-review").addEventListener("click", openTrustedReview);
document.getElementById("trusted-review").addEventListener("submit", saveTrustedReview);

loadDashboard().catch(() => {
  el("topbar-alert").textContent = "Unable to load dashboard data.";
});
