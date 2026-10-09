const state = {
  dashboard: null,
  transactions: [],
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
  return level.toLowerCase().split(" ")[0];
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
    throw new Error(message);
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
  el("cases-list").innerHTML = cases.map(caseCard).join("");
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
document.getElementById("cases-list").addEventListener("click", saveCase);
document.getElementById("transactions-body").addEventListener("click", openInvestigation);

loadDashboard().catch(() => {
  el("topbar-alert").textContent = "Unable to load dashboard data.";
});
