# Governance

This document describes the governance model for the Wastewater Disease Prediction project.

## Overview

This project uses a **Benevolent Dictator For Life (BDFL)** governance model with a clear path toward collaborative maintainership as the community grows. The goal is to transition to a **Consensus-Based** model once we have 3+ active maintainers.

## Roles

### Users
Anyone who uses the project. Users are encouraged to:
- Report bugs and request features via GitHub Issues
- Ask questions in GitHub Discussions
- Share how they use the project

### Contributors
Anyone who contributes to the project (code, documentation, issues, reviews). Contributors:
- Submit pull requests
- Review others' code
- Help answer community questions
- Improve documentation

### Maintainers
Contributors with commit access who actively maintain the project. Maintainers:
- Review and merge pull requests
- Triage issues
- Make release decisions
- Guide project direction

### Lead Maintainer (BDFL)
The project founder who has final decision-making authority. The Lead Maintainer:
- Sets overall project vision
- Resolves disputes when consensus cannot be reached
- Approves new maintainers
- Makes breaking change decisions

## Current Maintainers

| Name | GitHub | Role | Focus Areas |
|------|--------|------|-------------|
| Corey Schuman | [@cschuman](https://github.com/cschuman) | Lead Maintainer | All areas |

*We are actively seeking additional maintainers. See [CONTRIBUTOR_LADDER.md](docs/CONTRIBUTOR_LADDER.md).*

## Decision Making

### Day-to-Day Decisions
- Any maintainer can merge PRs that have been reviewed
- Bug fixes and minor improvements need one approval
- Documentation changes can be merged by any maintainer

### Significant Decisions
Decisions that affect the project's direction require broader input:

| Decision Type | Process |
|---------------|---------|
| New features | RFC discussion in GitHub Issues, maintainer approval |
| Breaking changes | RFC + 7-day comment period + Lead Maintainer approval |
| New maintainers | Nomination + Lead Maintainer approval |
| Dependency changes | PR review + security assessment |
| Release cuts | Any maintainer can release; major versions need Lead approval |

### RFC Process
For significant changes:
1. Open a GitHub Issue with `[RFC]` prefix
2. Describe the proposal, motivation, and alternatives considered
3. Allow 7 days for community feedback
4. Maintainers discuss and reach consensus
5. Lead Maintainer makes final call if no consensus

### Conflict Resolution
1. Discussion in the relevant Issue/PR
2. If unresolved: dedicated discussion thread
3. If still unresolved: maintainer vote (majority wins)
4. If tied: Lead Maintainer decides

## Becoming a Maintainer

We welcome new maintainers! The path is:

1. **Contribute regularly** - Multiple quality PRs over 3+ months
2. **Demonstrate judgment** - Good code review, helpful issue triage
3. **Show commitment** - Consistent engagement with the community
4. **Be nominated** - Self-nomination or nomination by existing maintainer
5. **Approval** - Lead Maintainer approves after reviewing contribution history

See [CONTRIBUTOR_LADDER.md](docs/CONTRIBUTOR_LADDER.md) for the detailed progression path.

## Code of Conduct

All participants must follow our [Code of Conduct](CODE_OF_CONDUCT.md). Violations should be reported to the Lead Maintainer.

## Changes to Governance

This governance document can be changed through the RFC process. Significant changes require:
- 14-day comment period
- Approval from all active maintainers
- Lead Maintainer final approval

## Transition Plan

As the project grows, we plan to transition governance:

| Stage | Maintainers | Model |
|-------|-------------|-------|
| Current | 1 | BDFL |
| Growing | 2-3 | BDFL + Maintainer Council |
| Mature | 4+ | Consensus-Based with Steering Committee |
| Foundation | N/A | Foundation governance (if applicable) |

## Contact

- **General questions**: GitHub Discussions
- **Security issues**: See [SECURITY.md](SECURITY.md)
- **Governance questions**: Open an issue with `[Governance]` tag

---

*This governance model is inspired by the [Node.js](https://github.com/nodejs/node/blob/main/GOVERNANCE.md) and [Rust](https://www.rust-lang.org/governance) projects.*
