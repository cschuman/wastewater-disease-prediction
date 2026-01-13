# Contributing to Wastewater Disease Prediction

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a branch for your changes

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
- Check the existing issues to avoid duplicates
- Collect information about the bug (Python version, OS, steps to reproduce)

When submitting a bug report, include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant logs or error messages
- Your environment details

### Suggesting Features

Feature requests are welcome! Please:
- Check existing issues and discussions first
- Clearly describe the feature and its use case
- Explain why this would benefit the project

### Improving Documentation

Documentation improvements are always appreciated:
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve docstrings

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/wastewater-disease-prediction.git
cd wastewater-disease-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Fetching Data

```bash
# Fetch wastewater surveillance data
python scripts/fetch_nwss_data.py

# Fetch hospital admission data
python scripts/fetch_nhsn_data.py
```

## Coding Standards

### Style Guide

We use the following tools for code quality:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **Type hints** are encouraged for public APIs

```bash
# Format code
black .

# Run linter
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Docstrings

Use Google-style docstrings for functions and classes:

```python
def fetch_wastewater_data(state: str, start_date: str) -> pd.DataFrame:
    """Fetch wastewater surveillance data for a specific state.

    Args:
        state: Two-letter state code (e.g., "CA", "NY")
        start_date: Start date in YYYY-MM-DD format

    Returns:
        DataFrame with wastewater concentration data

    Raises:
        ValueError: If state code is invalid
        ConnectionError: If API is unreachable
    """
```

### Commit Messages

Write clear, concise commit messages:
- Use the imperative mood ("Add feature" not "Added feature")
- First line should be 50 characters or less
- Reference issues when relevant (e.g., "Fix data parsing bug (#42)")

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_data.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup

## Submitting Changes

### Pull Request Process

1. **Update your fork** with the latest upstream changes
2. **Create a feature branch** from `main`
3. **Make your changes** with clear, atomic commits
4. **Add or update tests** as needed
5. **Ensure all tests pass** locally
6. **Update documentation** if needed
7. **Submit a pull request** with a clear description

### Pull Request Guidelines

- Fill out the PR template completely
- Link related issues
- Keep PRs focused on a single change
- Be responsive to review feedback

### Review Process

- All PRs require at least one review
- Address review comments promptly
- Be open to suggestions and feedback

## Questions?

If you have questions, feel free to:
- Open a discussion on GitHub
- Ask in an issue

Thank you for contributing!
