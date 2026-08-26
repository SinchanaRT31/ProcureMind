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

function formatAmount(amount) {
  return currencyFormatter.format(amount);
}

function riskClass(level) {
  return level.toLowerCase().split(" ")[0];
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
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
  return `
    <article class="stack-item" data-case-id="${caseItem.id}">
      <div class="stack-item-head">
        <strong>Case #${caseItem.id}</strong>
        <span class="badge ${riskClass(caseItem.priority)}">${caseItem.priority}</span>
      </div>
      <div class="stack-item-meta">
        <span>Owner: ${caseItem.owner}</span>
        <span>Transaction: ${caseItem.transaction_id}</span>
      </div>
      <div class="case-actions">
        <div class="case-status-row">
          <select class="case-status">
            <option ${caseItem.status === "Under Investigation" ? "selected" : ""}>Under Investigation</option>
            <option ${caseItem.status === "Confirmed" ? "selected" : ""}>Confirmed</option>
            <option ${caseItem.status === "False Positive" ? "selected" : ""}>False Positive</option>
            <option ${caseItem.status === "Review Pending" ? "selected" : ""}>Review Pending</option>
          </select>
          <button class="button primary save-case" type="button">Save</button>
        </div>
        <textarea class="case-notes" rows="3">${caseItem.notes}</textarea>
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
  const status = card.querySelector(".case-status").value;
  const notes = card.querySelector(".case-notes").value;

  button.disabled = true;
  button.textContent = "Saving...";

  try {
    await fetchJson(`/api/cases/${caseId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, notes }),
    });
    button.textContent = "Saved";
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Save";
    }, 900);
  } catch (error) {
    button.disabled = false;
    button.textContent = "Retry";
  }
}

document.getElementById("filters-form").addEventListener("submit", applyFilters);
document.getElementById("cases-list").addEventListener("click", saveCase);

loadDashboard().catch(() => {
  el("topbar-alert").textContent = "Unable to load dashboard data.";
});
