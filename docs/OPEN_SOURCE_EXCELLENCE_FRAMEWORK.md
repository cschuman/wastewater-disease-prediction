# Open Source Excellence Framework

> **The Platinum Standard for Open Source Projects**

This framework synthesizes best practices from CNCF, Apache Software Foundation, OpenSSF, and industry research to define what constitutes world-class open source.

---

## The Trust Pyramid

```
                    ┌─────────────┐
                    │  EXEMPLARY  │  ← "I'd bet my company on this"
                    │  Foundation │
                    │   Backing   │
                    ├─────────────┤
                    │ SUSTAINABLE │  ← "Safe for production"
                    │  Multi-org  │
                    │ Contributors│
                ┌───┴─────────────┴───┐
                │    PROFESSIONAL     │  ← "Safe to evaluate"
                │  Security + Tests   │
                │   + Versioning      │
            ┌───┴─────────────────────┴───┐
            │       FOUNDATIONAL          │  ← "Worth looking at"
            │   License + README + Code   │
            └─────────────────────────────┘
```

---

## Tier 1: Foundational (Required for Credibility)

| Dimension | Criteria | Evidence |
|-----------|----------|----------|
| **License Clarity** | OSI-approved license, clear in every file | LICENSE file, SPDX headers |
| **Basic Documentation** | README with purpose, install, usage | Working `quickstart` in README |
| **Code Access** | Public repo, full history | Git repository, no force-push rewrites |
| **Reproducible Build** | Anyone can build from source | Documented build process, CI badge |
| **Communication Channel** | Way to report issues and discuss | Issues enabled, discussions/Discord/Slack |

---

## Tier 2: Professional (Required for Production Use)

| Dimension | Criteria | Evidence |
|-----------|----------|----------|
| **Security Process** | Documented vulnerability reporting, response SLA | SECURITY.md, < 14 day response |
| **Versioning** | Semantic versioning, changelog | CHANGELOG.md, tagged releases |
| **Testing** | Automated test suite, CI/CD | Test coverage > 70%, green CI |
| **Code Quality** | Linting, formatting, static analysis | Pre-commit hooks, automated checks |
| **Release Process** | Documented, reproducible releases | Signed releases, checksums |
| **OpenSSF Badge** | At least Passing level | bestpractices.dev badge |

---

## Tier 3: Sustainable (Required for Long-term Trust)

| Dimension | Criteria | Evidence |
|-----------|----------|----------|
| **Multi-Maintainer** | Bus factor >= 3 | MAINTAINERS.md, commit diversity |
| **Cross-Org Contributors** | Not single-company controlled | Committers from 3+ organizations |
| **Governance** | Documented decision-making process | GOVERNANCE.md, public meetings |
| **Roadmap** | Public, community-influenced | ROADMAP.md or public project board |
| **Funding Transparency** | Clear sustainability model | OpenCollective, sponsorship, foundation |
| **Adopters List** | Proof of production usage | ADOPTERS.md with real companies |

---

## Tier 4: Exemplary (World-Class)

| Dimension | Criteria | Evidence |
|-----------|----------|----------|
| **Developer Experience** | TTFHW < 15 min, Time to First PR < 1 day | Measured, improved, dogfooded |
| **First-Class Docs** | Tutorial → How-to → Reference → Explanation | Diataxis framework, versioned docs |
| **Contributor Pipeline** | `good-first-issue` → mentorship → maintainer path | Documented contribution ladder |
| **Accessibility** | Inclusive design in docs and tooling | WCAG compliance, internationalization |
| **OpenSSF Gold** | Highest security practices | Gold badge achieved |
| **Foundation Graduated** | CNCF, Apache, or equivalent maturity | Graduated status, independent governance |
| **Backward Compat** | LTS branches, deprecation policy | 2+ year support windows |
| **Migration Tooling** | Can leave without lock-in | Export tools, data portability |

---

## The Three Pillars

```
┌─────────────────────────────────────────────────────────────┐
│                    OPEN SOURCE EXCELLENCE                   │
├───────────────────┬──────────────────┬──────────────────────┤
│    RELIABILITY    │   COMMUNITY      │    EXPERIENCE        │
│                   │                  │                      │
│ • Security        │ • Governance     │ • Onboarding         │
│ • Testing         │ • Diversity      │ • Documentation      │
│ • Versioning      │ • Sustainability │ • DX metrics         │
│ • Reproducibility │ • Transparency   │ • Accessibility      │
│                   │                  │                      │
│  "Can I trust     │  "Will it be     │  "Can I use it       │
│   this code?"     │   here tomorrow?"│   effectively?"      │
└───────────────────┴──────────────────┴──────────────────────┘
```

---

## The Contributor Flywheel

```
           ┌──────────────────┐
           │ Great DevEx      │
           │ (Low friction)   │
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │ More Contributors│
           │                  │◄────────────────┐
           └────────┬─────────┘                 │
                    │                           │
                    ▼                           │
           ┌──────────────────┐                 │
           │ Better Code      │                 │
           │ More Features    │                 │
           └────────┬─────────┘                 │
                    │                           │
                    ▼                           │
           ┌──────────────────┐                 │
           │ More Adopters    │                 │
           │ More Funding     │─────────────────┘
           └──────────────────┘
```

---

## Quantified Scorecard

| Category | Weight | Max Points |
|----------|--------|------------|
| **Security** | 25% | 25 |
| **Documentation** | 20% | 20 |
| **Community Health** | 20% | 20 |
| **Developer Experience** | 15% | 15 |
| **Governance** | 10% | 10 |
| **Sustainability** | 10% | 10 |
| **Total** | 100% | 100 |

**Scoring Thresholds:**
- **90+**: Platinum (World-class, CNCF Graduated equivalent)
- **75-89**: Gold (Production-ready, enterprise-grade)
- **60-74**: Silver (Professional, growing)
- **45-59**: Bronze (Foundational, emerging)
- **< 45**: Needs work

---

## Anti-Patterns

| Anti-Pattern | Why It Fails |
|--------------|--------------|
| "Perfect code, no community" | Code without users is a hobby project |
| "Viral growth, no governance" | LeftPad/colors.js incidents waiting to happen |
| "Corporate sponsorship, closed decisions" | Community will fork or leave |
| "Great docs, unmaintained code" | Promises without delivery destroy trust |
| "Security theater, no process" | Badges without substance are meaningless |
| "One brilliant maintainer" | Bus factor 1 = ticking time bomb |

---

## One-Liner Definition

> **The highest bar for open source: A project that a Fortune 500 company would stake critical infrastructure on, that a junior developer could contribute to in their first week, and that would survive if its largest contributor disappeared tomorrow.**

---

## Sources

- [CNCF Graduation Criteria](https://github.com/cncf/toc/blob/main/process/graduation_criteria.md)
- [Apache Project Maturity Model](https://community.apache.org/apache-way/apache-project-maturity-model.html)
- [OpenSSF Best Practices Badge](https://www.bestpractices.dev/en)
- [Linux Foundation - State of Open Source 2025](https://www.linuxfoundation.org/blog/the-state-of-open-source-software-in-2025)
- [CHAOSS Community Health Metrics](https://chaoss.community/)
