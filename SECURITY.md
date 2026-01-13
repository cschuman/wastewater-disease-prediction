# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt within 48 hours
- **Assessment**: We will assess the vulnerability and determine its severity
- **Updates**: We will keep you informed of our progress
- **Resolution**: We aim to resolve critical vulnerabilities within 7 days
- **Credit**: We will credit you in our release notes (unless you prefer anonymity)

## Security Considerations for This Project

### Data Handling

This project fetches data from public CDC APIs. While the data is public, we implement:

- Input validation to prevent injection attacks
- Path validation to prevent directory traversal
- URL whitelisting for external data sources
- No storage of credentials in code

### Dependencies

We regularly update dependencies to address known vulnerabilities. Run:

```bash
pip install pip-audit
pip-audit
```

### Local Development

- Never commit `.env` files or API keys
- Use virtual environments to isolate dependencies
- Review dependencies before installation

## Known Limitations

- This software is for research purposes only
- Disease predictions should not be used for clinical decision-making without expert review
- See [DISCLAIMER.md](DISCLAIMER.md) for full limitations
