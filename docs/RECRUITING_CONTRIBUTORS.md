# Recruiting Contributors and Co-Maintainers

This document captures strategies for growing the contributor community and recruiting co-maintainers.

---

## Current Status

| Metric | Current | Target |
|--------|---------|--------|
| Maintainers | 1 | 3+ |
| Organizations | 1 | 3+ |
| Documented Adopters | 0 | 5+ |
| Active Contributors | 0 | 10+ |

---

## Why People Contribute to Open Source

Understanding motivations helps craft better outreach:

| Motivation | How We Address It |
|------------|-------------------|
| **Career advancement** | Portfolio project, co-authorship on publications |
| **Learning** | Real CDC data, production ML pipeline, modern stack |
| **Impact** | Public health preparedness, 10-17 day early warning |
| **Community** | Welcoming governance, clear contributor ladder |
| **Fun** | Interesting technical challenges, domain expertise |

---

## Where to Find Contributors

### Tier 1: Warm Leads (Highest Success Rate)

| Source | Why It Works | How to Approach |
|--------|--------------|-----------------|
| Existing network | Trust exists | Direct ask for co-maintainer |
| Conference connections | Shared interests | Follow up after epi/data science talks |
| Academic collaborators | Research incentive | Co-authorship opportunity |
| Work colleagues | Know your quality | Side project with portfolio value |

### Tier 2: Community Outreach

| Platform | Audience | Pitch Angle |
|----------|----------|-------------|
| r/epidemiology | Public health researchers | Domain expertise needed |
| r/datascience | ML practitioners | Real data, interesting models |
| Hacker News (Show HN) | Technical audience | Novel application, public good |
| Twitter/X #epitwitter | Epi community | Share findings, build following |
| Mastodon fosstodon.org | OSS enthusiasts | Open source values |

### Tier 3: Structured Programs

| Program | Description | Link |
|---------|-------------|------|
| Google Summer of Code | Paid student contributors | gsoc.withgoogle.com |
| Outreachy | Internships for underrepresented groups | outreachy.org |
| MLH Fellowship | Student OSS contributors | fellowship.mlh.io |
| NumFOCUS Small Dev Grants | Funding for contributors | numfocus.org |

### Tier 4: Domain-Specific Communities

| Community | Relevance |
|-----------|-----------|
| CDC Open Source | Government users/contributors |
| Reich Lab (COVID Forecast Hub) | Adjacent project, collaborators |
| WastewaterSCAN (Stanford) | Data source alignment |
| CSTE (State Epidemiologists) | End users |
| APHL (Public Health Labs) | Domain expertise |

---

## The Pitch

### One-Liner
> Open source tool that predicts respiratory hospitalizations from wastewater data, giving public health departments 10-17 days early warning.

### Value Propositions by Audience

**For Researchers:**
- Real CDC data pipeline (NWSS + NHSN)
- Publication opportunities
- Novel multi-pathogen forecasting approach
- Health equity analysis built-in

**For ML Engineers:**
- Production XGBoost/ARIMA pipeline
- Time series forecasting challenges
- Modern Python stack (pytest, black, ruff, mypy)
- OpenSSF certified codebase

**For Public Health Professionals:**
- Direct impact on preparedness
- County-level equity analysis
- Open methodology (transparent, auditable)
- Integration with CDC data sources

**For Open Source Enthusiasts:**
- Well-documented, Gold-certified project
- Clear governance and contributor ladder
- MIT licensed
- Welcoming community

---

## Outreach Template

```
Subject: Co-maintainer opportunity: Open source disease forecasting

Hi [Name],

I'm building an open source tool that predicts respiratory hospitalizations
from wastewater data—giving public health departments 10-17 days early warning.

**The project:**
- Uses CDC NWSS + NHSN data (public APIs)
- XGBoost/ARIMA forecasting with health equity analysis
- SvelteKit dashboard for visualization
- MIT licensed, OpenSSF certified

**Why I'm reaching out:**
[Personalized: their expertise, shared connection, relevant work]

**What I'm looking for:**
A co-maintainer to help with [specific area]. Time commitment ~2-5 hrs/week.

**What you get:**
- Co-authorship on any publications
- Real-world impact on public health preparedness
- Portfolio project with production-quality infrastructure

Repo: github.com/cschuman/wastewater-disease-prediction

Would you be interested in a 20-minute call to discuss?

[Your name]
```

---

## Content Calendar

### Launch Phase
- [ ] Show HN post
- [ ] r/epidemiology post
- [ ] r/datascience post
- [ ] Twitter thread with key findings

### Ongoing
- [ ] Monthly "interesting finding" posts
- [ ] Respond to all issues within 48 hours
- [ ] Thank contributors publicly
- [ ] Share adoption stories

---

## Tracking Outreach

| Date | Channel | Link/Details | Response |
|------|---------|--------------|----------|
| | | | |

---

## Success Metrics

| Metric | 30 Days | 90 Days | 1 Year |
|--------|---------|---------|--------|
| GitHub Stars | 50 | 200 | 1000 |
| Contributors | 2 | 5 | 15 |
| Maintainers | 1 | 2 | 3+ |
| Adopters | 1 | 3 | 10 |

---

## Key Insight

> **Finding maintainers is a sales job, not a technical job.**

The project is technically excellent. But maintainers appear because:

1. **They believe in the mission** → Public health impact story
2. **They get something out of it** → Publications, portfolio, community
3. **The barrier is low** → Clear good-first-issues, fast PR reviews
4. **They trust you** → Warm intros beat cold outreach 10:1

**Priority:** Start with your network. One warm intro is worth 100 cold GitHub stars.
