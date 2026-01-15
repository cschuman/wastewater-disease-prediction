# Contributor Ladder

This document describes the progression path from first-time contributor to project maintainer.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   User → Contributor → Trusted Contributor → Maintainer         │
│                                                                 │
│   "I use it"  "I help"    "I'm trusted"      "I lead"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

We believe in **earned trust**. As you contribute more, you gain more responsibility and access.

---

## Level 1: User

**You use the project.**

### What You Can Do
- Use the software
- Report bugs
- Request features
- Ask questions in Discussions
- Star the repo

### How to Level Up
- Submit your first PR (any size!)
- Help answer a question in Discussions
- Improve documentation

---

## Level 2: Contributor

**You've made at least one accepted contribution.**

### Recognition
- Listed in GitHub contributors
- Thanked in release notes (for significant contributions)

### What You Can Do
Everything a User can do, plus:
- Submit PRs
- Review others' PRs (non-binding)
- Participate in RFC discussions

### Expectations
- Follow the [Code of Conduct](../CODE_OF_CONDUCT.md)
- Follow contribution guidelines
- Be responsive to PR feedback

### How to Level Up
To become a Trusted Contributor:
- 3+ merged PRs over 2+ months
- Demonstrated understanding of project goals
- Constructive participation in discussions
- Quality code reviews on others' PRs

---

## Level 3: Trusted Contributor

**You've demonstrated consistent, quality contributions.**

### Recognition
- Listed in CONTRIBUTORS.md (if we create one)
- Invited to contributor-only discussions
- Given `triage` permission on GitHub

### What You Can Do
Everything a Contributor can do, plus:
- Triage issues (add labels, close duplicates)
- Request reviews from maintainers
- Be assigned to issues
- Represent the project in discussions

### Expectations
- Respond to assigned issues within 7 days
- Help onboard new contributors
- Participate in release testing

### How to Level Up
To become a Maintainer:
- 6+ months of consistent contribution
- Demonstrated technical judgment
- Positive community interactions
- Nominated by existing maintainer (or self-nominate)
- Approved by Lead Maintainer

---

## Level 4: Maintainer

**You help lead the project.**

### Recognition
- Listed in [GOVERNANCE.md](../GOVERNANCE.md)
- Write access to repository
- Decision-making authority

### What You Can Do
Everything a Trusted Contributor can do, plus:
- Merge PRs
- Cut releases
- Make architectural decisions
- Approve new Trusted Contributors
- Participate in governance decisions

### Expectations
- Respond to PRs/issues within 72 hours (or delegate)
- Participate in release planning
- Mentor contributors
- Uphold Code of Conduct
- Attend quarterly planning (async OK)

### Time Commitment
Estimated: 2-5 hours/week minimum

---

## Emeritus Status

Maintainers who can no longer actively contribute can become **Emeritus Maintainers**:

- Recognized for past contributions
- No active responsibilities
- Can return to active status by resuming contributions
- Listed separately in GOVERNANCE.md

To request Emeritus status, notify the Lead Maintainer.

---

## Getting Started: Your First Contribution

### Good First Issues

Look for issues labeled:
- `good first issue` - Simple, well-defined tasks
- `help wanted` - We'd love community help
- `documentation` - No code required

### First PR Checklist

1. Fork the repository
2. Create a branch (`feature/your-feature` or `fix/your-fix`)
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `black . && ruff check .`
6. Submit PR with clear description
7. Respond to review feedback

### What Makes a Great Contribution

| Do | Don't |
|----|-------|
| Small, focused PRs | Massive PRs touching everything |
| Clear commit messages | "Fixed stuff" |
| Include tests for new code | Skip tests |
| Update docs if needed | Leave docs outdated |
| Ask questions if stuck | Disappear for weeks |

---

## Recognition

We recognize contributors in several ways:

| Contribution Type | Recognition |
|-------------------|-------------|
| First PR merged | Welcome message + thank you |
| Bug fix | Mentioned in CHANGELOG |
| New feature | Mentioned in CHANGELOG + release notes |
| Significant contribution | Invited to become Trusted Contributor |
| Sustained contribution | Invited to become Maintainer |

---

## FAQ

### How long does it take to become a Maintainer?
Typically 6-12 months of consistent contribution. Quality matters more than quantity.

### Can I become a Maintainer without writing code?
Yes! Documentation, community support, and issue triage are all valuable. We need maintainers with diverse skills.

### What if I can only contribute occasionally?
That's fine! Any contribution is welcome. The ladder is about demonstrated trust over time, not constant activity.

### How do I nominate myself or someone else?
Open an issue with `[Nomination]` prefix, or email the Lead Maintainer directly.

### What if I disagree with a decision?
See the conflict resolution process in [GOVERNANCE.md](../GOVERNANCE.md).

---

## Current Opportunities

We're especially looking for contributors interested in:

- [ ] Machine learning model improvements
- [ ] Data visualization (D3.js, Svelte)
- [ ] Documentation and tutorials
- [ ] Testing and CI/CD
- [ ] Public health domain expertise

If any of these interest you, check out our [good first issues](https://github.com/cschuman/wastewater-disease-prediction/labels/good%20first%20issue) or introduce yourself in Discussions!

---

*This document is inspired by contributor ladders from [Kubernetes](https://github.com/kubernetes/community/blob/master/community-membership.md) and [Apache](https://community.apache.org/contributors/).*
