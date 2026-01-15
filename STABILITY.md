# Stability Policy

This document describes our commitment to API stability and backward compatibility.

---

## Versioning

This project follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH

MAJOR - Breaking changes (incompatible API changes)
MINOR - New features (backward compatible)
PATCH - Bug fixes (backward compatible)
```

---

## Current Status

| Version | Status | Support |
|---------|--------|---------|
| 0.x.x | **Alpha** | Active development, APIs may change |
| 1.x.x | Stable (planned) | Full stability guarantees |

### Alpha Period (v0.x)

During the alpha period (versions 0.x.x):
- APIs may change between minor versions
- We will document breaking changes in CHANGELOG.md
- We will provide migration guides for significant changes
- We recommend pinning to exact versions in production

### Stable Period (v1.0+)

Once we reach v1.0.0:
- Public APIs will be stable within major versions
- Breaking changes only in major version bumps
- Deprecation warnings before removal
- 12-month support for each major version

---

## What We Consider "Public API"

### Stable (covered by semver)
- Python module interfaces in `src/`
- CLI commands and their arguments
- Data file formats (input/output schemas)
- Configuration file format (`config.yaml`)
- Web API endpoints (when released)

### Unstable (may change without major bump)
- Internal module structure
- Private functions (prefixed with `_`)
- Development tooling configuration
- CI/CD workflows
- Documentation structure

---

## Deprecation Policy

When we deprecate functionality:

### Timeline
1. **Deprecation Notice** - Feature marked deprecated with warning
2. **Migration Period** - Minimum 2 minor versions or 3 months
3. **Removal** - Feature removed in next major version

### Communication
- Deprecation warnings in code (Python `warnings` module)
- Notice in CHANGELOG.md
- Migration guide in documentation
- Announcement in GitHub Discussions

### Example Deprecation Warning
```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated and will be removed in v2.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

---

## Data Format Stability

### Input Data
- CDC API response formats may change (external dependency)
- We maintain adapters for known format versions
- Breaking CDC changes documented in release notes

### Output Data
- Parquet files: Schema changes are breaking changes
- JSON files: Additive changes (new fields) are non-breaking
- CSV files: Column additions are non-breaking; removals/renames are breaking

### Migration Support
For breaking data format changes, we provide:
- Migration scripts when feasible
- Documentation of schema changes
- Version tags in output files

---

## Dependency Policy

### Python Version
- We support Python versions with active security support
- Currently: Python 3.11, 3.12
- We provide 6 months notice before dropping a Python version

### Dependencies
- We pin minimum versions, not exact versions
- Major dependency updates may require minor version bump
- Security updates applied as patches when possible

---

## Support Policy

### During Alpha (current)
- Bug fixes for latest release only
- No backports to older versions
- Security fixes applied to latest version

### After v1.0 (planned)
| Version | Support Level | Duration |
|---------|---------------|----------|
| Latest major | Full support | Until next major |
| Previous major | Security fixes only | 12 months after next major |
| Older | No support | - |

---

## Breaking Change Checklist

Before making a breaking change, maintainers must:

- [ ] Document in RFC issue with `[Breaking]` label
- [ ] Allow 14-day comment period
- [ ] Write migration guide
- [ ] Update CHANGELOG.md with clear migration steps
- [ ] Increment major version (or minor in v0.x)
- [ ] Announce in GitHub Discussions

---

## Reporting Compatibility Issues

If you encounter unexpected breaking changes:

1. Check CHANGELOG.md for documented changes
2. Check if you're using a private/internal API
3. Open an issue with `[Compatibility]` label
4. Include: version numbers, code example, expected vs actual behavior

---

## Exceptions

We may make breaking changes without major version bump for:

- Security vulnerabilities (with disclosure)
- Legal/compliance requirements
- Bugs where current behavior is clearly wrong
- Changes required by external dependencies (CDC APIs)

All exceptions will be clearly documented in release notes.

---

*This policy is inspired by [NumPy NEP 23](https://numpy.org/neps/nep-0023-backwards-compatibility.html) and [Rust stability](https://rust-lang.github.io/rfcs/1122-language-semver.html).*
