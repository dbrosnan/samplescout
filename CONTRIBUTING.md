# Contributing to SampleScout

Thank you for your interest in contributing to SampleScout! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Python version and OS
   - Steps to reproduce
   - Expected vs actual behavior
   - Audio file details (format, duration, size)

### Suggesting Features

1. Check existing feature requests
2. Use the feature request template
3. Explain the use case and benefits

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `make check`
5. Commit with clear messages
6. Push and create a PR

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/samplescout.git
cd samplescout

# Install dev dependencies
make setup-dev

# Activate environment
source venv/bin/activate

# Run tests
make test
```

## Code Style

- We use **Black** for formatting (line length: 88)
- We use **isort** for import sorting
- We use **flake8** for linting
- We use **mypy** for type checking

Run `make format` before committing.

## Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `chore:` Maintenance

Example: `feat: add TensorRT support for Spleeter`

## Testing

- Write tests for new features
- Maintain or improve coverage
- Test with various audio formats

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
pytest tests/test_classifier.py -v
```

## Documentation

- Update README for user-facing changes
- Add docstrings to public functions
- Update CHANGELOG.md

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release PR
4. Tag the release after merge

## Questions?

Open a discussion or reach out to the maintainers.

Thank you for contributing!
