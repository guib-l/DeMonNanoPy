# DeMonNanoPy

Python wrapper around deMonNano for preparing inputs, launching calculations,
and parsing results.

## Overview

DeMonNanoPy provides a small Python API to:

- write deMonNano input files from Python dictionaries,
- execute deMonNano calculations from a working directory,
- parse output data such as energies and geometries,
- drive predefined workflows such as optimization, PTMC, and molecular dynamics,
- integrate deMonNano usage with ASE structures.

The codebase is organized around three main layers:

- `deMonPy.input.write_input`: builds `deMon.inp` from Python parameters,
- `deMonPy.deMonNano.deMonNano`: runs a standard calculation,
- `deMonPy.deMonNano.Module_DeMonNano`: runs higher-level workflows through modules.

## Requirements

- Python 3
- deMonNano executable available on the machine
- basis/slater-koster files accessible from the chosen working directory

Python dependencies listed in `requirements.txt`:

- `numpy`
- `ase`

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

## Repository Layout

```text
deMonPy/
  deMonNano.py      Main execution classes
  input.py          Input writer
  output.py         Output parser
  profile.py        Process execution helpers and decorators
  modules/          Workflow modules (optimization, PTMC, dynamics, ...)
exemple/            Usage examples
test/               Pytest-based regression tests and test data
pyproject.toml      Packaging and dependency metadata
```

## Core Concepts

### Standard Calculation

Use `deMonPy.deMonNano.deMonNano` when you want to:

- prepare a single calculation,
- run deMonNano,
- collect parsed results in a Python dictionary.

The normal workflow is:

1. define the executable path and input parameters,
2. create a `deMonNano` instance,
3. call `calculate(symbols=..., positions=...)`,
4. read `calculator.results`.

### Module-Based Workflow

Use `deMonPy.deMonNano.Module_DeMonNano` when you want a higher-level workflow
implemented in `deMonPy.modules`, for example:

- `opt` for geometry optimization,
- `ptmc` for parallel tempering Monte Carlo,
- `md` for molecular dynamics.

Available module names are defined in `deMonPy.__init__.py`.

## Parameter Structure

DeMonNanoPy expects nested dictionaries. A minimal configuration looks like this:

```python
parameters = {
	"DEMON_EXECUTABLE": "~/Documents/dev_deMon/deMon.x",
	"BASIS": {
		"PTYPE": "BIO",
		"SKFILE": "../basis",
	},
	"DEMON_PARAMETERS": {
		"ACTIVE": {
			"DFTB": {
				"SCC": True,
			},
		},
	},
}
```

Main keys used in the project:

- `DEMON_EXECUTABLE`: path to the deMonNano executable,
- `BASIS`: basis and parameter-file configuration,
- `DEMON_PARAMETERS.ACTIVE`: active deMonNano sections such as `DFTB`, `CHARGE`, `MULTI`, `CI`, `CUTSYS`, `TD-DFTB`, `FREQ`, `QMMM`, and `PRINT`,
- `DEMON_MODULE.ACTIVE`: active module settings such as `OPT`, `PTMC`, `MD`, or `NEB`.

The repository contains realistic examples in [exemple/exemple_opt.py](/home/pguibourg/Documents/DeMonNanoPy/exemple/exemple_opt.py), [exemple/exemple_md.py](/home/pguibourg/Documents/DeMonNanoPy/exemple/exemple_md.py), [exemple/exemple_ptmc.py](/home/pguibourg/Documents/DeMonNanoPy/exemple/exemple_ptmc.py), and [test/config.json](/home/pguibourg/Documents/DeMonNanoPy/test/config.json).

## Basic Usage

### Single-Point Calculation

```python
import numpy as np
from ase.atoms import Atoms

from deMonPy.deMonNano import deMonNano


parameters = {
	"DEMON_EXECUTABLE": "~/Documents/dev_deMon/deMon.x",
	"BASIS": {
		"PTYPE": "BIO",
		"SKFILE": "../basis",
	},
	"DEMON_PARAMETERS": {
		"ACTIVE": {
			"DFTB": {
				"SCC": True,
			},
		},
	},
}

image = Atoms(
	["O", "H", "H", "O", "H", "H"],
	positions=np.array([
		[1.2478, -0.5185, 3.4049],
		[1.5946, -1.4204, 3.3886],
		[0.9008, -0.3341, 2.5062],
		[3.2478, -0.4185, 3.4049],
		[3.5946, -1.5204, 3.3886],
		[2.9008, -0.3341, 2.6062],
	]),
)

calculator = deMonNano(
	title="CALCULATION DEMONANO",
	workdir=".run/",
	**parameters,
)

calculator.calculate(
	symbols=image.symbols,
	positions=image.positions,
)

print(calculator.results)
calculator.print_results()
```

Typical parsed entries include:

- `results["energy"]`
- `results["input_geometry"]`
- `results["output_geometry"]`
- `results["trajectory"]` when trajectory output is enabled

### Module Workflow Example

```python
import numpy as np
from ase.atoms import Atoms

from deMonPy.deMonNano import Module_DeMonNano


parameters = {
	"DEMON_EXECUTABLE": "~/Documents/dev_deMon/deMon.x",
	"BASIS": {
		"PTYPE": "BIO",
		"SKFILE": "../basis",
	},
	"DEMON_PARAMETERS": {
		"ACTIVE": {
			"DFTB": {
				"SCC": True,
			},
		},
	},
}

image = Atoms(
	["O", "H", "H", "O", "H", "H"],
	positions=np.array([
		[1.2478, -0.5185, 3.4049],
		[1.5946, -1.4204, 3.3886],
		[0.9008, -0.3341, 2.5062],
		[3.2478, -0.4185, 3.4049],
		[3.5946, -1.5204, 3.3886],
		[2.9008, -0.3341, 2.6062],
	]),
)

mod = Module_DeMonNano(
	module="opt",
	title="CALCULATION DEMONANO",
	workdir=".run/",
	**parameters,
)

mod(image=image, max=10)
mod.print_results()
```

The module wrapper initializes a workflow object from the module registry and
forwards keyword arguments to that workflow.

## Running the Examples

Example scripts are available in the `exemple/` directory:

- [exemple/exemple_opt.py](/home/pguibourg/Documents/DeMonNanoPy/exemple/exemple_opt.py)
- [exemple/exemple_md.py](/home/pguibourg/Documents/DeMonNanoPy/exemple/exemple_md.py)
- [exemple/exemple_ptmc.py](/home/pguibourg/Documents/DeMonNanoPy/exemple/exemple_ptmc.py)

Run them from the repository root after adapting the executable path and basis path:

```bash
python exemple/exemple_opt.py
python exemple/exemple_md.py
python exemple/exemple_ptmc.py
```

## Tests

The project contains pytest-based tests in the `test/` directory. They assume:

- deMonNano is installed and runnable,
- the executable path is configured in the test configuration,
- basis files are present under `test/basis/`.

Run the test suite with:

```bash
pytest
```

If you are working from a fresh environment, install the package first:

```bash
pip install -e .
pytest
```

Useful entry points include [test/test_demon.py](/home/pguibourg/Documents/DeMonNanoPy/test/test_demon.py) and [test/test_module.py](/home/pguibourg/Documents/DeMonNanoPy/test/test_module.py).

## Current Capabilities

Based on the current source tree, the project already includes support for:

- DFTB input generation,
- charge and multiplicity settings,
- bond parameter writing,
- CI and TD-DFTB input and partial output parsing,
- geometry parsing,
- optimization, PTMC, and MD workflow entry points.

Some parser and workflow sections are still placeholders in the current codebase,
notably parts of frequency, PTMC, NEB, and debug parsing.

## Notes

- Working directories are created automatically when needed.
- Results are stored in Python dictionaries and can be serialized as JSON.
- Paths to the executable and basis files are user-provided and must be valid in
  the execution environment.
- The package metadata currently targets a lightweight editable-install workflow.

## License

This project is distributed under the terms of the license provided in [LICENSE](/home/pguibourg/Documents/DeMonNanoPy/LICENSE).

