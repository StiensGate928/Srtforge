export const navItems = [
  {
    id: "overview",
    label: "Overview",
    subtitle: "KPIs, outcomes, and engagement health",
  },
  {
    id: "subscriptions",
    label: "Subscriptions & Billing",
    subtitle: "Business models, entitlements, and invoices",
  },
  {
    id: "employees",
    label: "Employee Access",
    subtitle: "Roster linking, eligibility, and utilization",
  },
  {
    id: "care",
    label: "Care Programs",
    subtitle: "Journeys, assessments, and clinical governance",
  },
  {
    id: "analytics",
    label: "Analytics & Insights",
    subtitle: "ROI, claims avoidance, and productivity",
  },
  {
    id: "operations",
    label: "Operations",
    subtitle: "Support, SLAs, and incident readiness",
  },
];

export const overviewData = {
  kpis: [
    { label: "Active eligible employees", value: "18,420", change: "+6.4%" },
    { label: "Linked Tatmeen accounts", value: "14,772", change: "+9.1%" },
    { label: "Completed assessments", value: "6,830", change: "+12.8%" },
    { label: "Session credits used", value: "32,180", change: "78% burn" },
    { label: "Engagement NPS", value: "74", change: "+3 pts" },
    { label: "Clinical escalation rate", value: "1.8%", change: "-0.4%" },
  ],
  healthSignals: [
    {
      title: "High engagement cohorts",
      desc: "Young families & frontline staff show > 80% weekly active usage.",
      status: "success",
    },
    {
      title: "At-risk utilization",
      desc: "Night shift teams in Riyadh 2 HQ show delayed assessment completion.",
      status: "warning",
    },
    {
      title: "Care quality alerts",
      desc: "Escalation turnaround time improved to 2h 15m across all clinics.",
      status: "success",
    },
  ],
  initiatives: [
    {
      name: "Diabetes prevention sprint",
      owner: "Tatmeen Care Ops",
      status: "On track",
      impact: "Projected 18% risk reduction",
    },
    {
      name: "Maternal care concierge",
      owner: "HR Wellness",
      status: "Needs attention",
      impact: "Enrollment at 54% of target",
    },
    {
      name: "Behavioral health access",
      owner: "Compliance",
      status: "Live",
      impact: "96% satisfaction score",
    },
  ],
  engagementTrend: [
    { label: "Jan", value: 30 },
    { label: "Feb", value: 42 },
    { label: "Mar", value: 55 },
    { label: "Apr", value: 63 },
    { label: "May", value: 78 },
    { label: "Jun", value: 71 },
  ],
};

export const subscriptionData = {
  models: [
    {
      name: "Per Employee Per Month",
      summary: "Fixed PEPM with wellness bundle + behavioral care.",
      price: "SAR 38",
      status: "Active",
      usage: "18,420 eligible employees",
    },
    {
      name: "Credit Pool",
      summary: "Shared session credits for specialist consultations.",
      price: "12,000 credits / quarter",
      status: "Active",
      usage: "78% used",
    },
    {
      name: "Outcome-based",
      summary: "Shared savings tied to chronic condition improvements.",
      price: "10% of savings",
      status: "Pilot",
      usage: "3 cohorts running",
    },
  ],
  invoices: [
    {
      id: "INV-2025-06",
      amount: "SAR 1,920,000",
      status: "Paid",
      due: "Jul 12, 2025",
    },
    {
      id: "INV-2025-07",
      amount: "SAR 1,745,000",
      status: "Pending",
      due: "Aug 12, 2025",
    },
  ],
  entitlements: [
    "Unlimited primary care chat",
    "4 specialist sessions / employee / year",
    "Family add-on (up to 4 dependents)",
    "Corporate clinic scheduling integration",
    "24/7 urgent escalation line",
  ],
};

export const employeeData = {
  linking: [
    {
      method: "Phone + OTP (default)",
      detail:
        "Employees sign up with phone number. HR uploads roster; system matches by phone hash.",
      status: "Recommended",
    },
    {
      method: "Email roster sync",
      detail:
        "HR uploads verified emails. Tatmeen links accounts during first login and prompts phone verification.",
      status: "Available",
    },
    {
      method: "HRIS API sync",
      detail:
        "Automatic updates from SAP, Oracle, or Workday with eligibility logic and contract rules.",
      status: "Enterprise",
    },
  ],
  roster: [
    {
      name: "Riyadh HQ",
      eligible: 5200,
      linked: 4320,
      status: "Healthy",
    },
    {
      name: "Jeddah Plant",
      eligible: 6800,
      linked: 4832,
      status: "Needs outreach",
    },
    {
      name: "Remote Sales",
      eligible: 2100,
      linked: 1890,
      status: "Healthy",
    },
  ],
  actions: [
    "Launch SMS onboarding campaign",
    "Trigger WhatsApp reminder flow",
    "Send manager nudges for non-joined staff",
    "Schedule on-site activation day",
  ],
};

export const careData = {
  programs: [
    {
      name: "Chronic care management",
      coverage: "4,100 members",
      cadence: "Monthly care plan",
      status: "Stable",
    },
    {
      name: "Women’s health + maternity",
      coverage: "2,320 members",
      cadence: "Bi-weekly check-ins",
      status: "High demand",
    },
    {
      name: "Behavioral well-being",
      coverage: "1,140 members",
      cadence: "On-demand sessions",
      status: "On track",
    },
  ],
  assessments: [
    { name: "PHQ-9", completion: 82, target: 90 },
    { name: "GAD-7", completion: 78, target: 85 },
    { name: "Diabetes risk", completion: 64, target: 75 },
  ],
  governance: [
    "Clinical guidelines mapped to Saudi MOH protocols",
    "Escalation to partner clinics within 2 hours",
    "PDPL compliant data handling & consent logs",
    "NPHIES eligibility checks for reimbursements",
  ],
};

export const analyticsData = {
  insights: [
    {
      title: "Absenteeism reduction",
      value: "12.4%",
      detail: "vs baseline after 4 months of program roll-out.",
    },
    {
      title: "Claims avoidance",
      value: "SAR 2.8M",
      detail: "Projected savings from early interventions.",
    },
    {
      title: "Productivity uplift",
      value: "8.1%",
      detail: "Shift coverage improved in frontline units.",
    },
  ],
  segments: [
    {
      name: "High risk",
      count: 920,
      trend: "Down 11%",
    },
    {
      name: "Rising risk",
      count: 1860,
      trend: "Stable",
    },
    {
      name: "Preventive care",
      count: 5120,
      trend: "Up 17%",
    },
  ],
  metrics: [
    { label: "Teleconsultations", value: 3120 },
    { label: "In-clinic referrals", value: 840 },
    { label: "Nutrition sessions", value: 620 },
    { label: "Behavioral sessions", value: 1240 },
  ],
};

export const operationsData = {
  sla: [
    { label: "Average response time", value: "3 min", status: "success" },
    { label: "Critical incident uptime", value: "99.94%", status: "success" },
    { label: "Open escalations", value: "4", status: "warning" },
  ],
  roadmap: [
    {
      title: "GenAI care co-pilot",
      desc: "Arabic-first triage summaries and coaching nudges.",
      eta: "Q4 2025",
    },
    {
      title: "On-site clinic scheduling",
      desc: "Slot booking with clinic occupancy optimization.",
      eta: "Q1 2026",
    },
    {
      title: "Employer insights API",
      desc: "Secure data feed for HR analytics teams.",
      eta: "Q2 2026",
    },
  ],
  supportPlaybooks: [
    "Tiered support (L1-L3) with Arabic & English coverage",
    "Privacy breach runbook aligned to PDPL",
    "Employee grievance escalation to HR within 24h",
    "Quarterly business reviews + outcome scorecards",
  ],
};
