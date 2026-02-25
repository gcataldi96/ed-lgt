# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project follows SemVer while
it is in the `0.x` development phase.

## [Unreleased]

### Added
- Modern `pip` packaging via `pyproject.toml`
- Runtime/dev/docs requirements files for maintainable installs
- Optional `simsio` extra (GitHub-backed): `pip install "edlgt[simsio]"`
- `requirements-simsio.txt` for direct GitHub installation of `simsio`
- `edlgt.__version__` sourced from installed package metadata

### Changed
- Public package/import name renamed from `ed_lgt` to `edlgt`
- Root `setup.py` reduced to a compatibility shim (metadata lives in `pyproject.toml`)

## [0.1.0] - 2026-02-25

### Added
- First structured public packaging/release setup for `edlgt`
- Core dependency declarations for `numpy`, `scipy`, `numba`, `sympy`, `matplotlib`

### Changed
- Prepared the library for `pip` installation and wheel builds
