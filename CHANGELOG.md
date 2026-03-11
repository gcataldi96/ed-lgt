# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project follows SemVer while
it is in the `0.x` development phase.

## [Unreleased]

### Added
- No user-facing changes yet.

## [0.2.1] - 2026-03-11

### Fixed
- Corrected `CITATION.cff` YAML formatting so release metadata is parsed reliably by Zenodo.

## [0.2.0] - 2026-03-11

### Added
- Generic N-body sparse assembly in symmetry-reduced bases for arbitrary operator list length (real-space and momentum-space paths).
- New validation coverage for generic N-body assembly, including finite-momentum checks in QED/SU2 benchmarks.
- Integrated 1+1D QED utilities to reconstruct link electric Casimir (`E2`) from measured matter density observables.
- Integrated-QED measurement support in workflows/examples with local-observable style reporting for reconstructed `E2`.

### Changed
- `NBodyTerm` compatibility with current `LocalTerm`/`TwoBodyTerm`/`PlaquetteTerm` conventions, including symmetry-aware generic routes.
- 1D integrated-QED reference initial states (`V`, `meson_center`) aligned with the Pauli-basis occupation convention used internally.
- QED static/dynamics workflows now handle integrated `E2` via reconstruction from matter observables instead of link-operator measurement.

### Fixed
- Explicit sparse-matrix dtype selection in Pauli operator construction to avoid SciPy `diags` dtype-cast warnings.

## [0.1.1] - 2026-02-26

### Added
- Modern `pip` packaging via `pyproject.toml`.
- Runtime/dev/docs requirements files for maintainable installs.
- Optional `simsio` extra (GitHub-backed): `pip install "edlgt[simsio]"`.
- `requirements-simsio.txt` for direct GitHub installation of `simsio`.
- `edlgt.__version__` sourced from installed package metadata.

### Changed
- Public package/import name renamed from `ed_lgt` to `edlgt`.
- Root `setup.py` reduced to a compatibility shim (metadata lives in `pyproject.toml`).

## [0.1.0] - 2026-02-25

### Added
- First structured public packaging/release setup for `edlgt`
- Core dependency declarations for `numpy`, `scipy`, `numba`, `sympy`, `matplotlib`

### Changed
- Prepared the library for `pip` installation and wheel builds
