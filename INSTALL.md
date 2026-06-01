# Installation of DeMonNanoPy

## Requirements

- Python 3
- deMonNano executable available on the machine
- basis/slater-koster files accessible from the chosen working directory

Python dependencies listed in `requirements.txt`:

- `numpy`
- `ase`
- `pytest`

## Installation

Create and activate a virtual environment, then install the project in editable mode:

```bash
python -m venv .env
source .env/bin/activate
pip install -e .
```

This installs the Python dependencies declared in `pyproject.toml` and makes the
`deMonPy` package importable from the environment.

If you only want the raw dependencies without installing the package itself, you
can still use:

```bash
pip install -r requirements.txt
```

Build metadata is defined in [pyproject.toml](/home/pguibourg/Documents/DeMonNanoPy/pyproject.toml) using `setuptools`.







