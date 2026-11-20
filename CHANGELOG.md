# Changelog

## [0.2.0] - 2026-11-20

### Added
- Top-level public API: `deMonNano`, `Module_DeMonNano`, the ASE `DeMonNano`
  calculator and the `DeMonError` family are now importable from `deMonPy`
- Type hints on the public entry points (`__init__`, `calculate`, `forward`)
- Opt-in `timeout` to bound a calculation
- Binary-free unit test suite (no local deMon environment required)

### Changed
- Unified error handling under the `DeMonError` hierarchy
  (`ConfigError`, `InputError`, `ExecuteFailed`, `OutputParseError`); failed
  runs now surface their `ERROR :` lines instead of returning silently
- ASE calculator maps deMonNano failures to ASE `SCFError` / `CalculationFailed`
- Geometry is validated (length/shape); unknown `ACTIVE` sections and
  non-writable value types now emit a `UserWarning`
- `clean_workdir()` refuses the filesystem root and home; all I/O is UTF-8
- Python floor raised to 3.12

### Fixed
- A missing executable now reports a clear error instead of a cryptic failure
- ASE calculator: documented keys now match the real keys
  (`DEMON_EXECUTABLE` / `BASIS` / `DEMON_PARAMETERS`) — no silent fallback to globals
- The parser raises a diagnosable `OutputParseError` instead of broadcasting
  an undiagnosable failure
- `write_input` no longer mutates caller parameters; `write()` is reentrant
- Docs: fixed a non-existent module path, a dangling `TODO.md` reference, and
  inconsistent Python versions

### Removed
- The `ignore_fails` parameter

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
