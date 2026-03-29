# Contributing to dataxid-syntheval

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/dataxid/dataxid-syntheval.git
cd dataxid-syntheval
uv sync
```

## Running Tests

```bash
pytest
```

## Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Code Style

- Line length: 99 characters
- Type hints everywhere, `Any` only for opaque state
- Minimal comments — code should be self-explanatory, comments explain *why*, not *what*
- Import order: stdlib → third-party → project (enforced by `ruff`)
- Polars-native: data processing uses Polars, algorithmic computations use battle-tested C libraries (scipy, sklearn)

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `ruff check` and `pytest` before submitting
4. Open a PR with a clear description of what and why

## Reporting Issues

Use [GitHub Issues](https://github.com/dataxid/dataxid-syntheval/issues) with the provided templates.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
