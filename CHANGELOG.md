# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project skeleton — `pyproject.toml`, `src/` layout, `uv`
- `SynthEvalConfig` — frozen dataclass for evaluation configuration
- `ingest()` — dual-input: accepts `pl.DataFrame` or `ProfileReport`
- `SynthEval` class skeleton — config + ingest integration
- Test suite — 22 tests covering config, ingest, and SynthEval
- GitHub templates — CODEOWNERS, PR template, issue templates
