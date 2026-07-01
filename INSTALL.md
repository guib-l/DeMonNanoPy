# Installation of DeMonNanoPy

## Requirements

- Python 3
- deMonNano executable available on the machine
- basis/slater-koster files accessible from the chosen working directory

Python dependencies are declared in `pyproject.toml`:

- `numpy`
- `ase`
- `pytest`

## Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the Python dependencies declared in `pyproject.toml` and makes the
`deMonPy` package importable from the environment.

Build metadata is defined in `pyproject.toml` using `setuptools`.







