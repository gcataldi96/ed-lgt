# edlgt

Exact diagonalization tools for lattice gauge theories and quantum many-body Hamiltonians.

Documentation: https://ed-su2.readthedocs.io/en/latest/

`pip install edlgt` and `import edlgt`

Supported Python: `>=3.10`

## Installation

### PyPI (recommended for users)

```bash
pip install edlgt
```

Optional Simsio workflows (installs `simsio` from GitHub):

```bash
pip install "edlgt[simsio]"
```

### From source (development)

```bash
git clone --recursive https://github.com/gcataldi96/ed-lgt.git
cd ed-lgt
pip install -e .
```

Development tools:

```bash
pip install -e ".[dev]"
```

Development tools + Simsio:

```bash
pip install -e ".[dev,simsio]"
```

## Quick Start

Check the installation:

```bash
python -c "import edlgt; print(edlgt.__version__)"
```

The best starting point is to run one of the example scripts and adapt it to your model:

```bash
python examples/example_QED_static.py
python examples/example_QED_dynamics.py
python examples/example_SU2_static.py
python examples/example_SU2_dynamics.py
```

Minimal import example:

```python
import edlgt
from edlgt.models import QED_Model, SU2_Model
from edlgt.modeling import diagonalize_density_matrix
```

## Optional Simsio Support

`simsio` is not currently available on PyPI, so the `simsio` extra installs it from GitHub.

If you prefer to install it manually, use one of:

```bash
pip install -r requirements-simsio.txt
```

```bash
pip install "git+https://github.com/rgbmrc/simsio.git"
```

If you work from this repository and use the submodule, also make sure it is present:

```bash
git submodule update --init --recursive
```

## Performance Notes

- For reproducible high-performance runs, prefer a dedicated Conda environment (for example with MKL).
- The `pip` package stays backend-agnostic for portability.
- After clearing caches, the first run can be slower because Numba recompiles kernels.
- Benchmark Numba-heavy code on a warm run (run twice).

## Project Layout

- `edlgt/`: library source code
- `examples/`: example scripts for QED, SU2, DFL, Zn, etc.
- `validation/`: validation scripts/tests
- `docs/`: documentation sources

## Citation

If you use `edlgt` in research, please cite it using the metadata in:

- `CITATION.cff`
- `CITATION.bib`

When Zenodo DOIs are available, use:

- the concept DOI for general software citation
- the version DOI for exact reproducibility

Current Zenodo concept DOI (all versions): `10.5281/zenodo.11145317`

## Maintenance (for contributors)

### Commit / Push Checklist

- [ ] Run at least one relevant example/script when changing package code
- [ ] Do a warm run before judging performance after Numba/kernel changes
- [ ] Check `import edlgt` after public API/import-path changes
- [ ] Keep `pyproject.toml` and `requirements*.txt` aligned when dependencies change
- [ ] Add user-facing changes to `CHANGELOG.md` under `Unreleased`
- [ ] Do not bump the version for normal commits/pushes (only for releases)
- [ ] Avoid committing generated artifacts (`dist/`, `__pycache__/`, large logs/outputs)

### Release Checklist

- [ ] Update `CHANGELOG.md` and choose the next version
- [ ] Bump version in `pyproject.toml`
- [ ] Clean build artifacts: `rm -rf dist build *.egg-info`
- [ ] Build package: `python -m build`
- [ ] Check metadata: `python -m twine check dist/*`
- [ ] Test install in a fresh environment
- [ ] Upload to TestPyPI, test install, then upload to PyPI
- [ ] Tag the release in git (for example `v0.1.0`)
