# Installation of DeMonNanoPy

## Requirements

- Python >= 3.11
- a deMonNano executable (`deMon.x`) available on the machine
- basis / Slater-Koster files accessible from the chosen working directory

Python dependencies (NumPy and ASE) are declared in `pyproject.toml` and are
installed automatically. The `dev` optional group adds the `ruff` linter.

> **Note:** the distribution is named `DeMonNanoPy` but the import package is
> `deMonPy`. You install `DeMonNanoPy` and `import deMonPy`.

## Installation

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate

pip install .
# or, for development (adds the ruff linter):
pip install -e ".[dev]"
```

This installs the Python dependencies declared in `pyproject.toml` and makes the
`deMonPy` package importable from the environment. The installed version is
exposed as `deMonPy.__version__`.

Build metadata is defined in `pyproject.toml` using `setuptools` (the version is
derived from `deMonPy.__version__`).

## Tests

Run the testing commande `pytest` with options:

| keyword          | Description         | 
|------------------|---------------------|
| `--all`         | Beta keywords, use carfully | 
| `--forces`       | Gradient tests | 
| `--optim`        | Test about optimization of geometry | 
| `--dynamics`     | Dynamics calculation test | 
| `--references`   | Comparison with external references calculations | 
| `--optional`     | Optionnal test (some) | 
| `--freq`         | Test for frequencies analysis | 
| `--beta`         | Beta keywords, use carfully | 
| `--3OB=`         | 3OB parameters directory for some QMMM tests | 


To get all clean test output, try:

```bash
pytest -vv --tb=short --color=yes --all
```






