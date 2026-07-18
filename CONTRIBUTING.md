# Contributing

Thank you for your interest in contributing to this project!

Contributions of all kinds are welcome, including bug fixes, new features,
documentation improvements, and suggestions.

## Prerequisites

- Python 3.11 or later
- Git
- A working `deMon.x` binary and Slater-Koster parameter files, if you want to
  run the test suite (see [Configuration](#configuration))

## Setting up the development environment

Clone the repository:

```bash
git clone https://github.com/guib-l/DeMonNanoPy.git
cd DeMonNanoPy
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Then install the project in editable mode, including the development
dependencies:

```bash
pip install -e ".[dev]"
```

The `dev` extra adds `ruff`, which is required to lint your changes.

> **Note:** the distribution is named `DeMonNanoPy` but the import package is
> `deMonPy`. You install `DeMonNanoPy` and `import deMonPy`.

## Configuration

The test suite needs to know where your deMonNano executable and basis files
live. This is provided through a `global.json` file at the root of the
repository.

**This file is intentionally not versioned** (it is listed in `.gitignore`,
since the paths are specific to each machine), so a fresh clone does not have
one. You must create it yourself before running the tests:

```json
{
    "DEMON_EXECUTABLE": "../../DeMonNano/deMon.x",
    "DEMON_BASIS": "../../DeMonNano/basis"
}
```

Both paths are resolved with `os.path.abspath`, so they may be absolute or
relative to the directory you run from. Without this file, every test module
fails at collection time with `FileNotFoundError: global.json`.

## Running the tests

This project uses pytest.

**Tests must be run from the `test/` directory.** They load the configuration
via the relative path `../global.json`, so running `pytest` from the repository
root fails during collection:

```bash
cd test
pytest
```

A full run on a correctly configured machine takes a couple of minutes.
Some tests are expected to fail and are marked `xfail`; they do not indicate a
broken setup.

### Optional test groups

Several groups of tests are skipped by default and are enabled with dedicated
flags:

| Flag | What it runs |
| --- | --- |
| `--run-optional` | Tests marked `optional` (longer physical systems) |
| `--beta` | Beta features |
| `--dftbplus` | Tests requiring a DFTB+ installation |
| `--forces` | Numerical forces tests (slow) |

They can be combined:

```bash
pytest --run-optional --forces
```

Please run at least the default suite before opening a pull request, and the
relevant optional groups if your change touches them.

## Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for both linting and
formatting. Configuration lives in `pyproject.toml`: line length 100, with the
`E` (pycodestyle), `F` (pyflakes) and `I` (isort) rule sets enabled.

Lint the source and the tests:

```bash
ruff check deMonPy test
ruff format deMonPy test
```

> Restrict the checks to `deMonPy` and `test` rather than running `ruff check .`
> on the whole repository: the notebooks under `tutorial/` contain deliberately
> incomplete cells (placeholders the reader is meant to fill in), which ruff
> reports as `F821 undefined-name`. Those reports are expected and should not be
> "fixed".

## Contribution workflow

1. Open an issue first for anything substantial, so the approach can be
   discussed before you spend time on it:
   <https://github.com/guib-l/DeMonNanoPy/issues>
2. Create a branch from `main`:

   ```bash
   git checkout main
   git pull
   git checkout -b my-feature
   ```

3. Make your changes, and add or update tests when you change behaviour.
4. Before pushing, check that both of these are clean:

   ```bash
   ruff check deMonPy test
   cd test && pytest
   ```

   There is no continuous integration on this repository, so these local checks
   are the only safety net.
5. Open a pull request against `main`, describing what changed and why. Mention
   any test group your change requires (for example `--dftbplus`), so the
   reviewer can reproduce your run.

### Commit messages

Use a short `Type: summary` prefix, in the imperative mood:

```
Fix: incorrect unit conversion in dipole parser
Feat: add periodic boundary conditions support
Docs: document the optional pytest flags
Test: add numerical forces regression test
Core: bump version to 0.1.2
Lint: apply ruff formatting
```

Common types are `Fix`, `Feat`, `Docs`, `Test`, `Core`, `Lint` and `Refactor`.
Keep the summary line under about 72 characters, and use the commit body for
any further detail.

## Reporting bugs

When reporting a bug, please include:

- your Python version and `deMonPy.__version__`,
- the deMonNano version or build you are using,
- a minimal script that reproduces the problem,
- the full traceback or the relevant deMonNano output.
