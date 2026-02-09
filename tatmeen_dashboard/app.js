import {
  analyticsData,
  careData,
  employeeData,
  navItems,
  operationsData,
  overviewData,
  subscriptionData,
} from "./data.js";

const nav = document.getElementById("nav");
const content = document.getElementById("content");
const pageTitle = document.getElementById("page-title");
const pageSubtitle = document.getElementById("page-subtitle");

const state = {
  active: "overview",
};

const badgeForStatus = (status) => {
  if (status === "success" || status === "Healthy" || status === "Paid") {
    return "badge success";
  }
  if (status === "warning" || status === "Needs attention" || status === "Pending") {
    return "badge warning";
  }
  if (status === "Needs outreach") {
    return "badge danger";
  }
  return "badge";
};

const renderNav = () => {
  nav.innerHTML = "";
  navItems.forEach((item) => {
    const button = document.createElement("button");
    button.textContent = item.label;
    if (item.id === state.active) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => {
      state.active = item.id;
      pageTitle.textContent = item.label;
      pageSubtitle.textContent = item.subtitle;
      renderNav();
      renderContent();
    });
    nav.appendChild(button);
  });
};

const renderOverview = () => {
  content.innerHTML = `
    <div class="grid cols-3">
      ${overviewData.kpis
        .map(
          (kpi) => `
        <div class="card">
          <h3>${kpi.label}</h3>
          <div class="metric">
            <div class="value">${kpi.value}</div>
            <span class="badge">${kpi.change}</span>
          </div>
          <p class="footer-note">Updated daily from Tatmeen analytics warehouse.</p>
        </div>
      `
        )
        .join("")}
    </div>
    <div class="grid cols-2">
      <div class="card">
        <div class="section-header">
          <h2>Engagement trend</h2>
          <span class="filter-chip">Last 6 months</span>
        </div>
        <div class="chart">
          <div class="bars">
            ${overviewData.engagementTrend
              .map(
                (point) => `
              <div class="bar" style="height:${point.value}%">
                <span>${point.label}</span>
              </div>
            `
              )
              .join("")}
          </div>
        </div>
      </div>
      <div class="card">
        <div class="section-header">
          <h2>Health signals</h2>
          <span class="filter-chip">Live</span>
        </div>
        <div class="list">
          ${overviewData.healthSignals
            .map(
              (signal) => `
            <div class="list-item">
              <div>
                <strong>${signal.title}</strong>
                <p>${signal.desc}</p>
              </div>
              <span class="${badgeForStatus(signal.status)}">${signal.status}</span>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </div>
    <div class="card">
      <div class="section-header">
        <h2>Strategic initiatives</h2>
        <span class="filter-chip">Quarterly</span>
      </div>
      <table class="table">
        <thead>
          <tr>
            <th>Initiative</th>
            <th>Owner</th>
            <th>Status</th>
            <th>Impact</th>
          </tr>
        </thead>
        <tbody>
          ${overviewData.initiatives
            .map(
              (initiative) => `
            <tr>
              <td>${initiative.name}</td>
              <td>${initiative.owner}</td>
              <td><span class="${badgeForStatus(initiative.status)}">${initiative.status}</span></td>
              <td>${initiative.impact}</td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
};

const renderSubscriptions = () => {
  content.innerHTML = `
    <div class="notice">
      <strong>Business model hub:</strong> Mix PEPM, credit pools, and outcome-based
      incentives. Configure by cohort or legal entity.
    </div>
    <div class="grid cols-3">
      ${subscriptionData.models
        .map(
          (model) => `
        <div class="card">
          <h3>${model.name}</h3>
          <p class="footer-note">${model.summary}</p>
          <div class="metric">
            <div class="value">${model.price}</div>
            <span class="${badgeForStatus(model.status)}">${model.status}</span>
          </div>
          <div class="tag">${model.usage}</div>
        </div>
      `
        )
        .join("")}
    </div>
    <div class="split">
      <div class="card">
        <div class="section-header">
          <h2>Invoices & collections</h2>
          <span class="filter-chip">Auto-reconcile</span>
        </div>
        <table class="table">
          <thead>
            <tr>
              <th>Invoice</th>
              <th>Amount</th>
              <th>Due</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${subscriptionData.invoices
              .map(
                (invoice) => `
              <tr>
                <td>${invoice.id}</td>
                <td>${invoice.amount}</td>
                <td>${invoice.due}</td>
                <td><span class="${badgeForStatus(invoice.status)}">${invoice.status}</span></td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="section-header">
          <h2>Entitlements</h2>
          <span class="filter-chip">Policy ready</span>
        </div>
        <div class="list">
          ${subscriptionData.entitlements
            .map(
              (entitlement) => `
            <div class="list-item">
              <strong>${entitlement}</strong>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </div>
  `;
};

const renderEmployees = () => {
  content.innerHTML = `
    <div class="grid cols-2">
      <div class="card">
        <div class="section-header">
          <h2>Account linking strategy</h2>
          <span class="filter-chip">Saudi-first</span>
        </div>
        <div class="list">
          ${employeeData.linking
            .map(
              (linking) => `
            <div class="list-item">
              <div>
                <strong>${linking.method}</strong>
                <p>${linking.detail}</p>
              </div>
              <span class="${badgeForStatus(linking.status)}">${linking.status}</span>
            </div>
          `
            )
            .join("")}
        </div>
        <p class="footer-note">
          Each roster upload creates a hashed linking key and consent log to comply
          with PDPL and employer privacy boundaries.
        </p>
      </div>
      <div class="card">
        <div class="section-header">
          <h2>Onboarding actions</h2>
          <span class="filter-chip">Automated</span>
        </div>
        <div class="pill-group">
          ${employeeData.actions.map((action) => `<span class="tag">${action}</span>`).join("")}
        </div>
      </div>
    </div>
    <div class="card">
      <div class="section-header">
        <h2>Eligibility roster health</h2>
        <span class="filter-chip">Updated today</span>
      </div>
      <table class="table">
        <thead>
          <tr>
            <th>Location</th>
            <th>Eligible</th>
            <th>Linked</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${employeeData.roster
            .map(
              (row) => `
            <tr>
              <td>${row.name}</td>
              <td>${row.eligible.toLocaleString()}</td>
              <td>${row.linked.toLocaleString()}</td>
              <td><span class="${badgeForStatus(row.status)}">${row.status}</span></td>
            </tr>
          `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
};

const renderCare = () => {
  content.innerHTML = `
    <div class="grid cols-3">
      ${careData.programs
        .map(
          (program) => `
        <div class="card">
          <h3>${program.name}</h3>
          <div class="metric">
            <div class="value">${program.coverage}</div>
            <span class="${badgeForStatus(program.status)}">${program.status}</span>
          </div>
          <p class="footer-note">${program.cadence}</p>
        </div>
      `
        )
        .join("")}
    </div>
    <div class="split">
      <div class="card">
        <div class="section-header">
          <h2>Assessment completion</h2>
          <span class="filter-chip">Monthly</span>
        </div>
        <table class="table">
          <thead>
            <tr>
              <th>Assessment</th>
              <th>Completion</th>
              <th>Target</th>
            </tr>
          </thead>
          <tbody>
            ${careData.assessments
              .map(
                (assessment) => `
              <tr>
                <td>${assessment.name}</td>
                <td>${assessment.completion}%</td>
                <td>${assessment.target}%</td>
              </tr>
            `
              )
              .join("")}
          </tbody>
        </table>
      </div>
      <div class="card">
        <div class="section-header">
          <h2>Clinical governance</h2>
          <span class="filter-chip">Always on</span>
        </div>
        <div class="list">
          ${careData.governance
            .map(
              (item) => `
            <div class="list-item">
              <strong>${item}</strong>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </div>
  `;
};

const renderAnalytics = () => {
  content.innerHTML = `
    <div class="grid cols-3">
      ${analyticsData.insights
        .map(
          (insight) => `
        <div class="card">
          <h3>${insight.title}</h3>
          <div class="metric">
            <div class="value">${insight.value}</div>
            <span class="badge success">Outcome</span>
          </div>
          <p class="footer-note">${insight.detail}</p>
        </div>
      `
        )
        .join("")}
    </div>
    <div class="split">
      <div class="card">
        <div class="section-header">
          <h2>Risk segmentation</h2>
          <span class="filter-chip">AI assisted</span>
        </div>
        <div class="list">
          ${analyticsData.segments
            .map(
              (segment) => `
            <div class="list-item">
              <div>
                <strong>${segment.name}</strong>
                <p>${segment.count.toLocaleString()} members</p>
              </div>
              <span class="badge">${segment.trend}</span>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
      <div class="card">
        <div class="section-header">
          <h2>Service mix</h2>
          <span class="filter-chip">Last 30 days</span>
        </div>
        <div class="list">
          ${analyticsData.metrics
            .map(
              (metric) => `
            <div class="metric">
              <div>${metric.label}</div>
              <div class="value">${metric.value.toLocaleString()}</div>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </div>
  `;
};

const renderOperations = () => {
  content.innerHTML = `
    <div class="grid cols-3">
      ${operationsData.sla
        .map(
          (item) => `
        <div class="card">
          <h3>${item.label}</h3>
          <div class="metric">
            <div class="value">${item.value}</div>
            <span class="${badgeForStatus(item.status)}">${item.status}</span>
          </div>
        </div>
      `
        )
        .join("")}
    </div>
    <div class="split">
      <div class="card">
        <div class="section-header">
          <h2>Roadmap visibility</h2>
          <span class="filter-chip">Shared</span>
        </div>
        <div class="list">
          ${operationsData.roadmap
            .map(
              (item) => `
            <div class="list-item">
              <div>
                <strong>${item.title}</strong>
                <p>${item.desc}</p>
              </div>
              <span class="badge">${item.eta}</span>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
      <div class="card">
        <div class="section-header">
          <h2>Support playbooks</h2>
          <span class="filter-chip">Operational readiness</span>
        </div>
        <div class="list">
          ${operationsData.supportPlaybooks
            .map(
              (item) => `
            <div class="list-item">
              <strong>${item}</strong>
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    </div>
  `;
};

const renderContent = () => {
  switch (state.active) {
    case "subscriptions":
      renderSubscriptions();
      break;
    case "employees":
      renderEmployees();
      break;
    case "care":
      renderCare();
      break;
    case "analytics":
      renderAnalytics();
      break;
    case "operations":
      renderOperations();
      break;
    default:
      renderOverview();
  }
};

renderNav();
renderContent();
