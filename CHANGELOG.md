# Changelog

## [0.1.1] - 2026-07-18

### Fixed
- Restore Python 3.10/3.11 compatibility (f-string quoting in the PBC writer)
- ASE calculator now raises instead of returning silent zero forces
- `Module_DeMonNano` honours a caller-supplied module registry
- `initialize()` is idempotent (safe `reset()` -> `__call__()` cycles)
- `to_dict()` returns a cycle-free, serialisable shallow copy
- Preserve the `ase_obj` setting across `update()`
- Remove debug prints and stdout warnings from the library code

### Changed
- Single source of truth for the version (`deMonPy.__version__`)
- Clean `ruff` run across the package

## [0.1.0] - 2026-06-01

### Added
- Public version
- Creator of deMonNano input
- Execute deMonNano code from input files
- Parser of deMonNano output
- Unit test of deMonNano code
