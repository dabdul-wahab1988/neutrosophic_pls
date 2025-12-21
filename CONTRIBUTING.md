# Contributing to Neutrosophic PLS

Thank you for your interest in contributing to Neutrosophic PLS! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/neutrosophic-pls.git
   cd neutrosophic-pls
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black neutrosophic_pls/ scripts/ tests/

# Lint code
ruff check neutrosophic_pls/ scripts/
```

### Type Checking

```bash
mypy neutrosophic_pls/
```

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass

## Code of Conduct

Please be respectful and constructive in all interactions.

## Questions?

Feel free to open an issue for questions or discussions.

## Authors

- Dickson Abdul-Wahab (dabdul-wahab@live.com)
- Ebenezer Aquisman Asare (aquisman1989@gmail.com)
