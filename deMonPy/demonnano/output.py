#!/usr/bin/env python3

# Import standard de python3
import os
import re

import numpy as np

from deMonPy.exceptions import OutputParseError
from deMonPy.ioread import IOread, safe_parse
from deMonPy.molden import read_XYZ
from deMonPy.profile import assert_flags, exclude_flags


# =============================================================================
# deMon output format positions
#
# The parsers below read deMon output (deMon.out / deMon.mol / deMon.coef)
# by raw position.  Each constant names one magic index, split marker, slice
# bound or line offset and quotes the deMon output line it parses.  Guards
# built on these constants record a "parse" error (via ``_add_error``) when a
# line or block does not have the expected shape; they never change the
# valid-output result.
# =============================================================================

# --- read_basics --------------------------------------------------------------
# "Requested SCC tolerance :  0.10000E-07" -- value = first sci-notation token
BASICS_SCC_TOLERANCE_TOKEN = 0
# "SCC mixing paramter :   0.30" -- value = first decimal token
BASICS_SCC_MIXING_TOKEN = 0
# "Maximum number of SCC cycles :  200" -- value = last integer token
BASICS_MAX_SCC_CYCLES_TOKEN = -1
# "Number of DFTB atoms           :      6" -- split marker ":"
BASICS_COLON_SEPARATOR = ":"
# "Number of DFTB atoms           :      6" -- value = last ':'-separated field
BASICS_COLON_VALUE_TOKEN = -1

# --- read_energy / get_energies -----------------------------------------------
# "DFTB total energy         [Hartree]       =        -8.06209343" -- last token
ENERGY_VALUE_TOKEN = -1

# --- read_ci --------------------------------------------------------------------
# "for state   1 =     -23.34492577" -- state number
CI_STATE_NUMBER_TOKEN = 2
# "for state   1 =     -23.34492577" -- state energy
CI_STATE_ENERGY_TOKEN = 4
# "weight of conf   1 = 0.997473 (  0.997864 )" -- Mulliken weight
CI_CONF_WEIGHT_TOKEN = 5
# "molecule   1 = 0.997473" -- atomic charge
CI_ATOM_CHARGE_TOKEN = 3
# deMon numbers states from 1 ("for state 1"); np.argmin is 0-based
CI_STATE_NUMBER_OFFSET = 1
# "for state   1 =     -23.34492577" -- minimum tokens (guard)
CI_STATE_MIN_TOKENS = 5
# "weight of conf   1 = 0.997473 (  0.997864 )" -- minimum tokens (guard)
CI_CONF_WEIGHT_MIN_TOKENS = 6
# "molecule   1 = 0.997473" -- minimum tokens (guard)
CI_ATOM_CHARGE_MIN_TOKENS = 4

# --- read_tddftb -------------------------------------------------------------------
# "requested transitions to calculate     :       10" -- value = last token
TDDFTB_NTRANSITIONS_TOKEN = -1
# SUMMARY block = signature line + blank + column header + blank + N data lines
TDDFTB_SUMMARY_HEADER_LINES = 4
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- the "->" separator
TDDFTB_TRANSITION_ARROW = "->"
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- w [eV]
TDDFTB_STATE_W_TOKEN = 0
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- Osz. strength
TDDFTB_STATE_OSCILLATOR_TOKEN = 1
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- Transition (from-state)
TDDFTB_STATE_FROM_TOKEN = 2
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- the "->" separator
TDDFTB_TRANSITION_ARROW_TOKEN = 3
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- Transition (to-state)
TDDFTB_STATE_TO_TOKEN = 4
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- Weight
TDDFTB_STATE_WEIGHT_TOKEN = 5
# "8.813  0.0013149  8   ->     9  0.998  8.805  S" -- E_sp [eV]
TDDFTB_STATE_ENERGY_TOKEN = 6
# a transition line has 8 tokens; the parser uses up to token 6
TDDFTB_STATE_MIN_TOKENS = 7
# a line with >= 4 tokens and "->" at token 3 is a transition line (guard)
TDDFTB_TRANSITION_MIN_TOKENS = 4

# --- read_freq ------------------------------------------------------------------------
# " NUMBER OF ATOMS: 3" -- atom count = last token
FREQ_ATOMS_TOKEN = -1
# "ZERO-POINT ENERGY =   0.02151778 HARTREE   =   13.5026 KCAL/MOL" -- 4th token
FREQ_ZPE_TOKEN = 3
# "ZERO-POINT ENERGY =   0.02151778 HARTREE" -- minimum tokens (guard)
FREQ_ZPE_MIN_TOKENS = 4
# "MODE:    1" / "FREQUENCY:  1562.7  [1/CM]" / "INTENSITY:    44.0  [KM/MOL]"
# -- value = 2nd token
FREQ_LABEL_VALUE_TOKEN = 1
# "MODE:    1" / "FREQUENCY:  1562.7" / "INTENSITY:    44.0" -- minimum tokens (guard)
FREQ_LABEL_MIN_TOKENS = 2
# each MODE block = 6 header lines + Nbatm atom rows:
# "MODE:", "FREQUENCY:", "INTENSITY:", blank, "ATOM  INP.NR.  X  Y  Z", blank
FREQ_MODE_HEADER_LINES = 6
# MODE block -- atom-displacement rows start at line index 6
FREQ_VECT_TABLE_START = 6
# "O  1  0.0000  0.0716  0.0000" -- symbol + INP.NR. columns before x y z
FREQ_VECT_SKIP_TOKENS = 2
# "O  1  0.0000  0.0716  0.0000" -- minimum tokens (guard)
FREQ_VECT_MIN_TOKENS = 5

# --- read_AOM_matrix ----------------------------------------------------------------------
# "Orbit El  N  L   1  2 ..." -- chunk column numbers from token 4
AOM_COLUMN_TOKEN_START = 4
# "1  O  2  0  1.000000 ..." -- row index
AOM_ROW_TOKEN = 0
# "1  O  2  0  1.000000 ..." -- values from token 4
AOM_VALUE_TOKEN_START = 4
# "1  O  2  0  1.000000 ..." -- minimum tokens (guard)
AOM_ROW_MIN_TOKENS = 5
# "1  O  2  0  1.000000 ..." -- deMon rows and columns are numbered from 1
AOM_1BASED_OFFSET = 1

# --- read_debug (Gamma matrix) ----------------------------------------------------------------
# "Shell El         1  2 ..." -- column numbers from token 2
GAMMA_COLUMN_TOKEN_START = 2
# "1 O  0.52220 ..." -- row index
GAMMA_ROW_TOKEN = 0
# "1 O  0.52220 ..." -- element symbol
GAMMA_ELEMENT_TOKEN = 1
# "1 O  0.52220 ..." -- values from token 2
GAMMA_VALUE_TOKEN_START = 2
# "1 O  0.52220 ..." -- minimum tokens (guard)
GAMMA_MIN_TOKENS = 3

# --- read_pbc -----------------------------------------------------------------------------------
# " -0.790839405237485        2.00000000000000" -- eigenvalue
PBC_EIGENERGY_TOKEN = 0
# " -0.790839405237485        2.00000000000000" -- occupation
PBC_EIG_OCCUPATION_TOKEN = 1
# " -0.790839405237485        2.00000000000000" -- 2 floats per band line
PBC_EIG_TOKEN_COUNT = 2
# each real line occupies 2 entries in the splitlines() view ("\n".join of
# readlines() keeps the trailing newlines), so the scan window is nband * 2
PBC_EIG_SCAN_WINDOW_FACTOR = 2
# fractional k-point coordinates: last non-empty line within 3 lines before
# "kpoint number" (no in-repo sample prints them; shape unverified)
PBC_KCOORD_LOOKBACK = 3
# coordinates line -- tokens 1:4 hold the fractional coordinates
PBC_KCOORD_TOKEN_START = 1
PBC_KCOORD_TOKEN_STOP = 4

# --- read_print (deMon.coef) ---
# deMon.coef: the value follows on the line after the label
COEF_VALUE_LINE_OFFSET = 1
# "8  O  0.6550  -0.0544  5.9307  1  4" -- Z
COEF_ATOM_Z_TOKEN = 0
# "8  O  0.6550  -0.0544  5.9307  1  4" -- symbol
COEF_ATOM_SYMBOL_TOKEN = 1
# "8  O  0.6550  -0.0544  5.9307  1  4" -- x
COEF_ATOM_X_TOKEN = 2
# "8  O  0.6550  -0.0544  5.9307  1  4" -- y
COEF_ATOM_Y_TOKEN = 3
# "8  O  0.6550  -0.0544  5.9307  1  4" -- z
COEF_ATOM_ZCOORD_TOKEN = 4
# "8  O  0.6550  -0.0544  5.9307  1  4" -- basis start
COEF_ATOM_BASIS_START_TOKEN = 5
# "8  O  0.6550  -0.0544  5.9307  1  4" -- basis end
COEF_ATOM_BASIS_END_TOKEN = 6
# "8  O  0.6550  -0.0544  5.9307  1  4" -- minimum tokens (guard)
COEF_ATOM_MIN_TOKENS = 7
# "1 2.00  -0.8381" -- occupation
COEF_MOE_OCCUPATION_TOKEN = 1
# "1 2.00  -0.8381" -- MO energy
COEF_MOE_ENERGY_TOKEN = 2
# "1 2.00  -0.8381" -- minimum tokens (guard)
COEF_MOE_MIN_TOKENS = 3
# "llmos , ulmos" followed by "1  36" -- minimum tokens (guard)
COEF_LUMO_RANGE_MIN_TOKENS = 2

# --- read_errors ---
# "ERROR : loop over charges not converged" -- split marker ":"
ERROR_COLON_SEPARATOR = ":"
# "ERROR : loop over charges not converged" -- message = last ':'-separated field
ERROR_MESSAGE_TOKEN = -1

# --- read_geometry (deMon.mol MD frames) ---
# frame comment "BOMD INFORMATION  <e0> <e1> <e2> <e3>" (tokens 2..5, see
# molden._read_xyz_ext); deMon.mol of a BOMD run
XYZ_INFO_POTENTIAL_COL = 0
XYZ_INFO_KINETIC_COL = 1
XYZ_INFO_TOTAL_COL = 2
XYZ_INFO_TIME_COL = 3

# --- parse_tensors ---
# "(1, 2) = ..." -- matrix labels are 1-based
TENSOR_LABEL_OFFSET = 1

# --- _read_opt ---
# forces.out -- first line is a header, then one force vector per line
FORCES_HEADER_LINES = 1
# "Maximum gradient =  1.2e-05" -- value = last token
OPT_GRADIENT_TOKEN = -1

# --- _read_md ---
# "Average TOTAL ENERGY : <value>" -- value = 2nd-to-last token
# (no in-repo sample of the MD summary; shape unverified)
MD_ENERGY_VALUE_TOKEN = -2

# --- _read_ptmc ---
# deMon.NN.mol frame comment -- value = 3rd token (no in-repo sample; unverified)
PTMC_TRAJ_VALUE_TOKEN = 2


def convert_float(val, safe=False):
    try:
        return float(val)
    except ValueError:
        return val if not safe else None



class read_output(IOread):
    """Parser for deMonNano output files and derived results."""

    _criteria_energy_str = {
        "DFTB total energy": "energy",
        "DFTB electronic energy": "electronic_energy",
        "DFTB band energy": "band_energy",
        "DFTB repulsive energy": "repulsive_energy",
        "DFTB Coulomb energy": "coulomb_energy",
        "DFTB London energy": "london_energy",
        "DFTB MM+Mechanical Coupl.": "MM_coupl.",
        "DFTB electronic entropy": "electronic_entropy",
        "DFTB Polarisation energy": "polarisation_energy",
        "DFTB HOMO-LUMO gap": "HOMO-LUMO_gap",
        "DFTB (HOMO)-(HOMO-1) gap": "HOMO_gap",
        "DFTB Fermi energy level": "fermi_energy",
        "DFTB third order Coulomb energy": "3d_coulomb_energy",
    }
    _criteria_mm = {
        "TOTAL ENERGY                =": "qmmm-energy",
        "VAN DER WAALS ENERGY": "qmmm-VdW_energy",
        "ELECTROSTATIC CHARGE ENERGY": "qmmm-electrostatic_energy",
        "UREY BRADLEY ENERGY": "qmmm-UB_energy",
        "IMPROPER TORSION ENERGY": "qmmm-improper_torsion_energy",
        "TORSION ANGLE ENERGY": "qmmm-torsion_angle_energy",
        "BOND ANGLE ENERGY": "qmmm-bond_angle_energy",
        "BOND ENERGY": "qmmm-bond_energy",
    }

    def __init__(
        self,
        properties=["energy"],
        workdir="./",
        output="deMon.out",
        flags=set(),
        ase_obj=True,
        type_calculation={},
        ignore_parse_errors=True,
    ):
        """Initialize the output reader.

        Args:
            properties: Requested properties to parse.
            workdir: Directory containing output files.
            output: Main output file name.
            flags: Active parser flags.
            type_calculation: Reserved calculation metadata.
            ignore_parse_errors: If ``True`` (default), a reader that
                fails records a ``"parse"`` error and leaves its results
                set to ``None`` instead of raising.  Missing or
                unreadable output files always raise.
        """

        IOread.__init__(
            self,
        )

        self.workdir = workdir
        self.output = output

        self.flags = flags
        self.ignore_parse_errors = ignore_parse_errors

        self.properties = properties
        self.complet_results = {
            "errors": [],
        }
        self._ase_obj = ase_obj
        self.lines = []

    def read_file(
        self,
    ):
        """Load the main output file into memory.

        Raises:
            OutputParseError: If the output file is missing or
                unreadable (typical when the binary crashed before
                writing anything).
        """

        filename = os.path.join(self.workdir, self.output)
        try:
            with open(filename, "r", encoding="utf-8", errors="replace") as fd:
                self.lines.extend(fd.readlines())
        except FileNotFoundError as err:
            raise OutputParseError(
                f"deMon output not found at {filename}. "
                f"Did the calculation fail before writing output?"
            ) from err
        except OSError as err:
            raise OutputParseError(f"Cannot read deMon output at {filename}: {err}") from err

    def _add_error(self, kind, message, line=None):
        """Append a structured error entry to the results.

        Args:
            kind: Short error category (e.g. ``"mpi"``, ``"tddftb"``,
                ``"scf"``, ``"optimization"``).
            message: Human-readable description.
            line: Optional source line the error was extracted from.
        """
        self.complet_results.setdefault("errors", []).append(
            {
                "kind": kind,
                "message": message,
                "line": line,
            }
        )

    # ------------------------------------------------------------------ guards
    # Shape guards for the raw-position parsing below.  Each guard records a
    # "parse" error through the existing _add_error mechanism when a line or
    # block does not have the expected shape, and then leaves the original
    # parsing untouched so the reader degrades exactly as it did before
    # (natural exception -> safe_parse, or silent partial result).

    def _guard_basics_count(self, label, line):
        """Record a parse error if a ``read_basics`` "label : <int>" line is malformed."""
        field = line.split(BASICS_COLON_SEPARATOR)[BASICS_COLON_VALUE_TOKEN]
        try:
            int(field)
        except ValueError:
            self._add_error(
                "parse",
                f"read_basics: '{label}' expects an integer after "
                f"'{BASICS_COLON_SEPARATOR}', found {field.strip()!r}",
                line=line.strip(),
            )

    @staticmethod
    def _is_tddftb_transition(values):
        """True when a split SUMMARY line looks like a transition line."""
        return (
            len(values) >= TDDFTB_TRANSITION_MIN_TOKENS
            and values[TDDFTB_TRANSITION_ARROW_TOKEN] == TDDFTB_TRANSITION_ARROW
        )

    def _guard_tddftb_transition(self, signature, line, values):
        """Record a parse error if a ``read_tddftb`` transition line is malformed."""
        if len(values) < TDDFTB_STATE_MIN_TOKENS:
            self._add_error(
                "parse",
                f"read_tddftb: '{signature}' transition line expects at least "
                f"{TDDFTB_STATE_MIN_TOKENS} tokens, found {len(values)}: {line.strip()!r}",
                line=line.strip(),
            )
            return
        try:
            float(values[TDDFTB_STATE_W_TOKEN])
            float(values[TDDFTB_STATE_OSCILLATOR_TOKEN])
            int(values[TDDFTB_STATE_FROM_TOKEN])
            int(values[TDDFTB_STATE_TO_TOKEN])
            float(values[TDDFTB_STATE_WEIGHT_TOKEN])
            float(values[TDDFTB_STATE_ENERGY_TOKEN])
        except (ValueError, IndexError):
            self._add_error(
                "parse",
                f"read_tddftb: '{signature}' transition line has non-numeric fields: "
                f"{line.strip()!r}",
                line=line.strip(),
            )

    def _guard_freq_label(self, marker, line):
        """Record a parse error if a ``read_freq`` "LABEL: value" line is malformed."""
        values = line.split()
        if len(values) < FREQ_LABEL_MIN_TOKENS:
            self._add_error(
                "parse",
                f"read_freq: '{marker}' line expects at least "
                f"{FREQ_LABEL_MIN_TOKENS} tokens, found {len(values)}: {line.strip()!r}",
                line=line.strip(),
            )

    def _guard_gamma_header(self, line):
        """Record a parse error if a ``read_debug`` "Shell El" header is malformed."""
        cols = line.split()[GAMMA_COLUMN_TOKEN_START:]
        if not cols or not all(c.lstrip("-").isdigit() for c in cols):
            self._add_error(
                "parse",
                f"read_debug: 'Shell El' header expects integer column numbers from "
                f"token {GAMMA_COLUMN_TOKEN_START}, found: {line.strip()!r}",
                line=line.strip(),
            )

    def _guard_gamma_row(self, line, n_cols):
        """Record a parse error if a ``read_debug`` Gamma row is malformed."""
        parts = line.split()
        try:
            int(parts[GAMMA_ROW_TOKEN])
        except (ValueError, IndexError):
            self._add_error(
                "parse",
                f"read_debug: Gamma row expects an integer row index, found: "
                f"{line.strip()!r}",
                line=line.strip(),
            )
            return
        if len(parts[GAMMA_VALUE_TOKEN_START:]) != n_cols:
            self._add_error(
                "parse",
                f"read_debug: Gamma row expects {n_cols} values (one per column), "
                f"found {len(parts[GAMMA_VALUE_TOKEN_START:])}: {line.strip()!r}",
                line=line.strip(),
            )

    def _guard_coef_int_value(self, value_line, label):
        """Record a parse error if a deMon.coef integer value line is malformed."""
        try:
            int(value_line)
        except ValueError:
            self._add_error(
                "parse",
                f"read_print: deMon.coef '{label}' expects an integer value, "
                f"found {value_line.strip()!r}",
                line=value_line.strip(),
            )

    def _guard_coef_llmos(self, value_line):
        """Record a parse error if the deMon.coef "llmos" value line is malformed."""
        parts = value_line.split()
        if len(parts) < COEF_LUMO_RANGE_MIN_TOKENS or not all(
            p.lstrip("-").isdigit() for p in parts[:COEF_LUMO_RANGE_MIN_TOKENS]
        ):
            self._add_error(
                "parse",
                f"read_print: deMon.coef 'llmos' value line expects at least "
                f"{COEF_LUMO_RANGE_MIN_TOKENS} integers, found: {value_line.strip()!r}",
                line=value_line.strip(),
            )

    def _guard_coef_atom_row(self, line):
        """Record a parse error if a deMon.coef "atomic position" row is malformed."""
        parts = line.split()
        if len(parts) < COEF_ATOM_MIN_TOKENS:
            self._add_error(
                "parse",
                f"read_print: deMon.coef atom row expects at least "
                f"{COEF_ATOM_MIN_TOKENS} tokens (Z, symbol, x, y, z, basis start, "
                f"basis end), found {len(parts)}: {line.strip()!r}",
                line=line.strip(),
            )
            return
        try:
            int(parts[COEF_ATOM_Z_TOKEN])
            float(parts[COEF_ATOM_X_TOKEN])
            float(parts[COEF_ATOM_Y_TOKEN])
            float(parts[COEF_ATOM_ZCOORD_TOKEN])
            int(parts[COEF_ATOM_BASIS_START_TOKEN])
            int(parts[COEF_ATOM_BASIS_END_TOKEN])
        except (ValueError, IndexError):
            self._add_error(
                "parse",
                f"read_print: deMon.coef atom row has non-numeric fields: "
                f"{line.strip()!r}",
                line=line.strip(),
            )

    def _guard_coef_moe_row(self, line):
        """Record a parse error if a deMon.coef "occupation and MOE" row is malformed."""
        parts = line.split()
        if len(parts) < COEF_MOE_MIN_TOKENS:
            self._add_error(
                "parse",
                f"read_print: deMon.coef MOE row expects at least {COEF_MOE_MIN_TOKENS} "
                f"tokens (index, occupation, energy), found {len(parts)}: "
                f"{line.strip()!r}",
                line=line.strip(),
            )
            return
        try:
            float(parts[COEF_MOE_OCCUPATION_TOKEN])
            float(parts[COEF_MOE_ENERGY_TOKEN])
        except (ValueError, IndexError):
            self._add_error(
                "parse",
                f"read_print: deMon.coef MOE row has non-numeric fields: "
                f"{line.strip()!r}",
                line=line.strip(),
            )

    # =================================
    # READ ENERGY (basics)

    @safe_parse("properties")
    def read_basics(self):
        """Read basics things"""

        text = "\n".join(self.lines)

        data = {
            "scc_tolerance": None,
            "max_scc_cycles": None,
            "scc_mixing": None,
            "n_atoms": None,
            "n_atom_types": None,
            "matrix_dim": None,
            "n_electrons": None,
            "n_shells": None,
        }

        for line in text.splitlines():
            line = line.strip()

            # -------- SCC PARAMS --------
            if "Requested SCC tolerance" in line:
                val = re.findall(r"[-+]?\d+\.\d+E[-+]?\d+", line)
                if val:
                    data["scc_tolerance"] = float(val[BASICS_SCC_TOLERANCE_TOKEN])

            elif "Maximum number of SCC cycles" in line:
                val = re.findall(r"\d+", line)
                if val:
                    data["max_scc_cycles"] = int(val[BASICS_MAX_SCC_CYCLES_TOKEN])

            elif "SCC mixing" in line:
                val = re.findall(r"[-+]?\d*\.\d+", line)
                if val:
                    data["scc_mixing"] = float(val[BASICS_SCC_MIXING_TOKEN])

            # -------- ELECTRONIC INFO --------
            elif "Number of DFTB atoms" in line:
                self._guard_basics_count("Number of DFTB atoms", line)
                data["n_atoms"] = int(
                    line.split(BASICS_COLON_SEPARATOR)[BASICS_COLON_VALUE_TOKEN]
                )

            elif "Number of DFTB atom types" in line:
                self._guard_basics_count("Number of DFTB atom types", line)
                data["n_atom_types"] = int(
                    line.split(BASICS_COLON_SEPARATOR)[BASICS_COLON_VALUE_TOKEN]
                )

            elif "Dimension of DFTB matrices" in line:
                self._guard_basics_count("Dimension of DFTB matrices", line)
                data["matrix_dim"] = int(
                    line.split(BASICS_COLON_SEPARATOR)[BASICS_COLON_VALUE_TOKEN]
                )

            elif "Number of DFTB electrons" in line:
                self._guard_basics_count("Number of DFTB electrons", line)
                data["n_electrons"] = int(
                    line.split(BASICS_COLON_SEPARATOR)[BASICS_COLON_VALUE_TOKEN]
                )

            elif "Number of DFTB shells" in line:
                self._guard_basics_count("Number of DFTB shells", line)
                data["n_shells"] = int(
                    line.split(BASICS_COLON_SEPARATOR)[BASICS_COLON_VALUE_TOKEN]
                )

        self.complet_results["properties"] = data

    @safe_parse("energy")
    def read_energy(self):
        """Parse energy values from the loaded output lines."""

        start_search = False

        _state = {}
        self.complet_results["energy"] = {}

        for line in self.lines:
            _energy = self.get_energies(line, start_search=start_search)
            _state.update(_energy)

            if "energy" in self.complet_results.keys():
                start_search = True
        self.complet_results["energy"].update(_state)

    def get_energies(self, line, criteria="DFTB total energy", start_search=False):
        """Extract energy terms from a single output line.

        Args:
            line: Line to inspect.
            criteria: Primary energy marker used to seed parsing.
            start_search: Whether extended energy parsing is active.

        Returns:
            dict: Parsed energy terms found on the line.
        """

        _energy = {}

        _criteria_energy_str_bis = self._criteria_energy_str

        if "mm" in self.flags:
            _criteria_energy_str_bis = self._criteria_mm

        if "qmmm" in self.flags:
            _criteria_energy_str_bis = self._criteria_energy_str
            _criteria_energy_str_bis.update(self._criteria_mm)

        if self.is_inside(criteria, line):
            label = _criteria_energy_str_bis[criteria]
            _energy[label] = self.get_float(line, index=ENERGY_VALUE_TOKEN)

        if start_search:
            for cs, it in _criteria_energy_str_bis.items():
                if self.is_inside(cs, line):
                    _energy[it] = self.get_float(line, index=ENERGY_VALUE_TOKEN)
        return _energy

    # =================================
    # READ GEOMETRY (basics)

    @safe_parse("input_geometry", "output_geometry")
    @exclude_flags(["ptmc", "freq", "ptmd", "rttddftb"])
    def read_geometry(self, output="deMon.mol", is_charges=False, velocities=False, keep=1):
        """Read input, output, or trajectory geometries.

        Args:
            output: Geometry file name.
            is_charges: Whether the XYZ reader should parse charges.
            keep: Geometry sampling interval.

        Raises:
            OutputParseError: If the geometry file is missing (typically
                when the calculation crashed before emitting it).
        """

        filename = os.path.join(self.workdir, output)
        if not os.path.isfile(filename):
            raise OutputParseError(
                f"Geometry file not found at {filename}. The calculation likely did not complete."
            )
        try:
            data, info = read_XYZ(
                filename,
                is_charges=is_charges,
                velocities=velocities,
                keep=keep,
                ase_obj=self._ase_obj,
            )
        except (OSError, ValueError) as err:
            raise OutputParseError(f"Failed to parse geometry file {filename}: {err}") from err

        if len(data) == 1:
            if "opt" in self.flags:
                self.complet_results["input_geometry"] = data[0]
                self.complet_results["output_geometry"] = data[0]

        if len(data) == 2:
            self.complet_results["input_geometry"] = data[0]
            self.complet_results["output_geometry"] = data[-1]

        if len(data) > 2:
            self.complet_results["input_geometry"] = data[0]
            if "traj" in self.flags:
                self.complet_results["trajectory"] = data
            self.complet_results["output_geometry"] = data[-1]

        if "md" in self.flags:
            self.complet_results["time"] = info[:, XYZ_INFO_TIME_COL]
            self.complet_results["potential_energy"] = info[:, XYZ_INFO_POTENTIAL_COL]
            self.complet_results["kinetic_energy"] = info[:, XYZ_INFO_KINETIC_COL]
            self.complet_results["total_energy"] = info[:, XYZ_INFO_TOTAL_COL]

    # =================================
    # READ PARAMETERS (basics)

    @safe_parse("states", "charges")
    @assert_flags("ci")
    def read_ci(self):
        """Parse configuration interaction states and configurations."""
        start_search = True
        state_search = False

        _conf = {}
        count = 0
        state = {}

        self.complet_results["states"] = {}

        for line in self.lines:
            _energy = self.get_energies(line, start_search=start_search)
            _conf.update(_energy)

            if "*********   CONFIGURATIONS   *********" in line:
                if count > 0:
                    self.complet_results[f"configuration_{count}"] = _conf
                count += 1
                _conf = {}
                self.complet_results["energy"] = {"energy": None}

            if self.is_inside("************    STATES    ************", line):
                state_search = True

            if self.is_inside("for state", line) and state_search:
                sl = line.split()
                if len(sl) < CI_STATE_MIN_TOKENS:
                    self._add_error(
                        "parse",
                        f"read_ci: 'for state' line expects at least "
                        f"{CI_STATE_MIN_TOKENS} tokens (for state <n> = <e>), found "
                        f"{len(sl)}: {line.strip()!r}",
                        line=line.strip(),
                    )
                num = int(sl[CI_STATE_NUMBER_TOKEN])
                sub = float(sl[CI_STATE_ENERGY_TOKEN])
                wgh, chrg_mulliken, chrg_cm3 = [], [], []
                state = {f"state {num}": {"energy": sub}}

            if self.is_inside("weight of conf", line) and state_search:
                sl = line.split()
                if len(sl) < CI_CONF_WEIGHT_MIN_TOKENS:
                    self._add_error(
                        "parse",
                        f"read_ci: 'weight of conf' line expects at least "
                        f"{CI_CONF_WEIGHT_MIN_TOKENS} tokens, found {len(sl)}: "
                        f"{line.strip()!r}",
                        line=line.strip(),
                    )
                try:
                    wgh.append(float(sl[CI_CONF_WEIGHT_TOKEN]))
                except (ValueError, IndexError):
                    wgh.append(None)
                state[f"state {num}"].update({"weight": wgh})

            if self.is_inside("molecule", line) and state_search:
                sl = line.split()
                if len(sl) < CI_ATOM_CHARGE_MIN_TOKENS:
                    self._add_error(
                        "parse",
                        f"read_ci: 'molecule' line expects at least "
                        f"{CI_ATOM_CHARGE_MIN_TOKENS} tokens, found {len(sl)}: "
                        f"{line.strip()!r}",
                        line=line.strip(),
                    )
                chrg = chrg_mulliken
                label = "charges_mulliken"
                if len(chrg_mulliken) == len(wgh):
                    chrg = chrg_cm3
                    label = "charges_cm3"
                try:
                    chrg.append(float(sl[CI_ATOM_CHARGE_TOKEN]))
                except (ValueError, IndexError):
                    chrg.append(None)

                state[f"state {num}"].update({label: chrg})

            self.complet_results["states"].update(state)

        self.complet_results[f"configuration_{count}"] = _conf
        self.complet_results["charges"] = {}

        if "states" in self.complet_results.keys():
            if len(self.complet_results["states"].keys()) > 0:
                self.complet_results.pop("energy")

                idx = (
                    np.argmin(
                        [state["energy"] for k, state in self.complet_results["states"].items()]
                    )
                    + CI_STATE_NUMBER_OFFSET
                )

                energies = self.complet_results["states"][f"state {idx}"]["energy"]
                self.complet_results["energy"] = {"energy": energies}

                charges = self.complet_results["states"][f"state {idx}"]["charges_mulliken"]
                self.complet_results["charges"].update({"charges mulliken": charges})

                if "charges_cm3" in self.complet_results["states"][f"state {idx}"]:
                    charges = self.complet_results["states"][f"state {idx}"]["charges_cm3"]
                    self.complet_results["charges"].update({"charges cm3": charges})

    #@safe_parse("triplet", "singlet")
    @assert_flags("td-dftb")
    def read_tddftb(self):
        """Parse TD-DFTB singlet and triplet transitions."""
        
        msg = "LINEAR RESPONSE FOR CLOSED SHELL MOLECULES ONLY."
        for line in self.lines:
            if msg in line:
                self._add_error("tddftb", msg, line=line.strip())
                return

        for line in self.lines:
            if self.is_inside("requested transitions to calculate", line):
                try:
                    int(line.split()[TDDFTB_NTRANSITIONS_TOKEN])
                except (ValueError, IndexError):
                    self._add_error(
                        "parse",
                        "read_tddftb: 'requested transitions to calculate' expects a "
                        f"trailing integer, found {line.strip()!r}",
                        line=line.strip(),
                    )
                N = self.get_int(line, TDDFTB_NTRANSITIONS_TOKEN)
                break

        count = 0
        text = "\n".join(self.lines)
        for key, signature in zip(["triplet", "singlet"], ["SUMMARY TRIPLET:", "SUMMARY SINGLET"]):
            args = {}
            if signature not in text:
                continue
            for i, line in enumerate(self.lines):
                self.compute_block(line, signature, "", nb_line=N + TDDFTB_SUMMARY_HEADER_LINES)
                if self._tocken_block:
                    values = line.split()
                    if self._is_tddftb_transition(values):
                        self._guard_tddftb_transition(signature, line, values)
                    try:
                        args.update(
                            {
                                f"state_{count}": {
                                    "w": float(values[TDDFTB_STATE_W_TOKEN]),
                                    "oscillator": float(values[TDDFTB_STATE_OSCILLATOR_TOKEN]),
                                    "from": int(values[TDDFTB_STATE_FROM_TOKEN]),
                                    "to": int(values[TDDFTB_STATE_TO_TOKEN]),
                                    "weight": float(values[TDDFTB_STATE_WEIGHT_TOKEN]),
                                    "energy": float(values[TDDFTB_STATE_ENERGY_TOKEN]),
                                }
                            }
                        )
                        count += 1
                    except (ValueError, IndexError):
                        pass
            self.complet_results[key] = args

    @safe_parse("frequency", "zpe")
    @assert_flags("freq")
    def read_freq(self):
        """Parse vibrational frequency results."""
        freq, temp = [], []
        zpe, Nbatm = 0.00, 0

        for line in self.lines:
            if "NUMBER OF ATOMS:" in line:
                sl = line.split()
                if len(sl) < 2:
                    self._add_error(
                        "parse",
                        "read_freq: 'NUMBER OF ATOMS:' expects a trailing integer, "
                        f"found {line.strip()!r}",
                        line=line.strip(),
                    )
                else:
                    try:
                        int(sl[FREQ_ATOMS_TOKEN])
                    except ValueError:
                        self._add_error(
                            "parse",
                            "read_freq: 'NUMBER OF ATOMS:' expects a trailing integer, "
                            f"found {sl[FREQ_ATOMS_TOKEN].strip()!r}",
                            line=line.strip(),
                        )
                Nbatm = int(line.split()[FREQ_ATOMS_TOKEN])
            if "ZERO-POINT ENERGY =" in line:
                if len(line.split()) < FREQ_ZPE_MIN_TOKENS:
                    self._add_error(
                        "parse",
                        f"read_freq: 'ZERO-POINT ENERGY =' line expects at least "
                        f"{FREQ_ZPE_MIN_TOKENS} tokens, found {len(line.split())}: "
                        f"{line.strip()!r}",
                        line=line.strip(),
                    )
                zpe = float(line.split()[FREQ_ZPE_TOKEN])

            self.compute_block(line, "MODE:", "", FREQ_MODE_HEADER_LINES + Nbatm)

            if self.counter == 0 and self._block != "":
                temp.append(self._block)
                self._block = ""

        for frq in temp:
            mode, frequency, intensity = 0, 0.0, 0.0

            for line in frq.split("\n"):
                if self.is_inside(control="MODE:", line=line):
                    self._guard_freq_label("MODE:", line)
                    mode = int(line.split()[FREQ_LABEL_VALUE_TOKEN])
                if self.is_inside(control="FREQUENCY:", line=line):
                    self._guard_freq_label("FREQUENCY:", line)
                    frequency = float(line.split()[FREQ_LABEL_VALUE_TOKEN])
                if self.is_inside(control="INTENSITY:", line=line):
                    self._guard_freq_label("INTENSITY:", line)
                    intensity = float(line.split()[FREQ_LABEL_VALUE_TOKEN])

            table = []
            rows = frq.split("\n")[FREQ_VECT_TABLE_START:-1]
            if len(rows) < Nbatm:
                self._add_error(
                    "parse",
                    f"read_freq: MODE block expects {Nbatm} atom-displacement rows "
                    f"({FREQ_MODE_HEADER_LINES} header lines + {Nbatm} atom rows), "
                    f"found {len(rows)}",
                )
            for lin in rows:
                valeurs = lin.strip().split()
                if len(valeurs) < FREQ_VECT_MIN_TOKENS:
                    self._add_error(
                        "parse",
                        f"read_freq: atom-displacement row expects at least "
                        f"{FREQ_VECT_MIN_TOKENS} tokens (symbol, INP.NR., x, y, z), "
                        f"found {len(valeurs)}: {lin.strip()!r}",
                        line=lin.strip(),
                    )
                nombres = [convert_float(v, False) for v in valeurs]
                table.append(nombres)

            table = np.array(table)[:, FREQ_VECT_SKIP_TOKENS:]

            freq.append(
                {"mode": mode, "frequency": frequency, "intensity": intensity, "vect": table}
            )

        self.complet_results["frequency"] = freq

        self.complet_results["zpe"] = zpe

    def read_AOM_matrix(
        self,
        trigger_line="  S",
        keep_matrix="all",
    ):
        """Read the AOM matrix from a DFTB output file."""

        lines = self.lines.copy()
        if isinstance(keep_matrix, int):
            keep_matrix = [keep_matrix]

        start_idx = None
        for i, line in enumerate(lines):
            if trigger_line in line:
                start_idx = i + 1
                break

        if start_idx is None:
            raise ValueError(f"Section '{trigger_line}' not found in file.")

        data = []
        block_num = 0
        authorized = True
        symb_save = trigger_line.strip()
        i = start_idx

        size = self.complet_results["properties"]["matrix_dim"]
        tab = np.zeros((size, size))

        while i < len(lines):
            line = lines[i].strip()
            if symb_save == lines[i].strip():
                authorized = True
            if (not line or not line.startswith("Orbit")) or not authorized:
                i += 1
                continue

            parts = line.split()
            cols_tokens = parts[AOM_COLUMN_TOKEN_START:]
            if not cols_tokens or not all(
                x.lstrip("-").isdigit() for x in cols_tokens
            ):
                self._add_error(
                    "parse",
                    f"read_AOM_matrix: 'Orbit' header expects integer column numbers "
                    f"from token {AOM_COLUMN_TOKEN_START}, found: {line.strip()!r}",
                    line=line.strip(),
                )
            cols = [int(x) for x in cols_tokens]

            min_cols = min(cols)
            max_cols = max(cols)

            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("Orbit") or line.startswith("&&"):
                    break

                parts = line.split()
                try:
                    row = int(parts[AOM_ROW_TOKEN])
                except (ValueError, IndexError):
                    break

                if len(parts) < AOM_ROW_MIN_TOKENS or len(
                    parts[AOM_VALUE_TOKEN_START:]
                ) != len(cols):
                    self._add_error(
                        "parse",
                        f"read_AOM_matrix: data row expects {len(cols)} values (one per "
                        f"column {min_cols}..{max_cols}), found "
                        f"{len(parts[AOM_VALUE_TOKEN_START:])}: {line.strip()!r}",
                        line=line.strip(),
                    )
                    # Fail with a diagnosable error instead of letting the
                    # assignment below raise an opaque numpy broadcast error.
                    raise OutputParseError(
                        f"read_AOM_matrix: section '{symb_save}' row {row} has "
                        f"{len(parts[AOM_VALUE_TOKEN_START:])} values, header "
                        f"declares {len(cols)} columns: {line.strip()!r}"
                    )

                values = [convert_float(x, safe=True) for x in parts[AOM_VALUE_TOKEN_START:]]
                tab[
                    min_cols - AOM_1BASED_OFFSET : max_cols,
                    row - AOM_1BASED_OFFSET,
                ] = values

                i += 1

            if cols[-1] == size:
                tab = tab.T
                if keep_matrix == "all":
                    data.append(tab.copy())
                    tab = np.zeros((size, size))
                elif keep_matrix == [-1]:
                    data = [tab.copy()]
                    tab = np.zeros((size, size))
                else:
                    if block_num in keep_matrix:
                        data.append(tab.copy())
                        tab = np.zeros((size, size))

                block_num += 1
                authorized = False
        return data

    @safe_parse()
    @assert_flags("print")
    def read_debug(self, extract_data=False, keep_matrix=-1):

        if extract_data:
            matrix = {}
            labels = {}

            in_section = False
            text = "\n".join(self.lines)

            for line in text.splitlines():
                if "SCC ATOM-DEPENDENT GAMMA MATRIX" in line:
                    in_section = True
                    continue

                if in_section:
                    if not line:
                        continue
                    if "&&" in line:
                        break

                    if line.startswith("Shell El"):
                        self._guard_gamma_header(line)
                        parts = line.split()
                        current_cols = list(map(int, parts[GAMMA_COLUMN_TOKEN_START:]))
                        continue

                    parts = line.split()

                    if len(parts) < GAMMA_MIN_TOKENS:
                        if line.strip():
                            self._add_error(
                                "parse",
                                f"read_debug: Gamma row expects at least "
                                f"{GAMMA_MIN_TOKENS} tokens (row, element, values), "
                                f"found {len(parts)}: {line.strip()!r}",
                                line=line.strip(),
                            )
                        continue

                    self._guard_gamma_row(line, len(current_cols))

                    row_idx = int(parts[GAMMA_ROW_TOKEN])
                    element = parts[GAMMA_ELEMENT_TOKEN]
                    values = list(map(float, parts[GAMMA_VALUE_TOKEN_START:]))

                    labels[row_idx] = element

                    if row_idx not in matrix:
                        matrix[row_idx] = {}

                    for col_idx, val in zip(current_cols, values):
                        matrix[row_idx][col_idx] = val

            size = max(matrix.keys())
            full_matrix = np.zeros((size, size))

            for i in matrix:
                for j in matrix[i]:
                    full_matrix[i - 1][j - 1] = matrix[i][j]

            self.complet_results["Gamma"] = full_matrix.copy()

            tab = self.read_AOM_matrix(
                trigger_line="                    S ",
                keep_matrix=keep_matrix,
            )
            self.complet_results["Overlaps"] = tab.copy()

            tab = self.read_AOM_matrix(
                trigger_line="                    F0",
                keep_matrix=keep_matrix,
            )
            self.complet_results["Ham0"] = tab.copy()

            tab = self.read_AOM_matrix(
                trigger_line="                    F ",
                keep_matrix=keep_matrix,
            )
            self.complet_results["Hamiltonian"] = tab.copy()

            tab = self.read_AOM_matrix(
                trigger_line="                    P ",
                keep_matrix=keep_matrix,
            )
            self.complet_results["Density"] = tab.copy()

            tab = self.read_AOM_matrix(
                trigger_line="                    C ",
                keep_matrix=keep_matrix,
            )
            self.complet_results["Coefficients"] = tab.copy()

    @safe_parse("k-points", "eigenvalues", "kpts_coords")
    @assert_flags("pbc")
    def read_pbc(
        self,
    ):
        text = "\n".join(self.lines)

        energyK = []
        kcoords = []
        lines = text.splitlines()
        nband = self.complet_results["properties"]["matrix_dim"]

        count, nkpt, limit = False, 0, 0
        for i, line in enumerate(lines):

            if "kpoint number" in line:
                count = True
                limit = i + nband * PBC_EIG_SCAN_WINDOW_FACTOR
                nkpt += 1
                # Fractional coordinates are on the previous non-empty line
                prev = [
                    li
                    for li in lines[max(0, i - PBC_KCOORD_LOOKBACK):i]
                    if li.strip()
                ][-1]
                kcoords.append(
                    [float(x) for x in prev.split()[PBC_KCOORD_TOKEN_START:PBC_KCOORD_TOKEN_STOP]]
                )

            if count and i <= limit:
                li = line.split()
                try:
                    energyK.append(
                        [
                            float(li[PBC_EIGENERGY_TOKEN]),
                            float(li[PBC_EIG_OCCUPATION_TOKEN]),
                        ]
                    )
                except (ValueError, IndexError):
                    pass

        if nkpt > 0 and nband is not None and len(energyK) != nkpt * nband:
            self._add_error(
                "parse",
                f"read_pbc: expected {nkpt * nband} (energy, occupation) entries "
                f"({nkpt} k-points x {nband} bands x {PBC_EIG_TOKEN_COUNT} floats), "
                f"found {len(energyK)}",
            )

        energyK = np.array(energyK)
        energyK = np.reshape(energyK, (nkpt, nband, PBC_EIG_TOKEN_COUNT))
        eigenv = energyK[:, :, :1].reshape((nkpt, nband))
        
        self.complet_results["k-points"] = energyK
        self.complet_results["eigenvalues"] = eigenv
        self.complet_results["kpts_coords"] = np.array(kcoords)


    @safe_parse("moe", "mos")
    @assert_flags("print")
    def read_print(self):
        """Parse print output sections."""

        text = "\n".join(self.lines)

        occupied = []
        virtual = []

        in_section = False

        for line in text.splitlines():
            if "DFTB Eigen values" in line:
                in_section = True
                occupied = []
                virtual = []
                continue

            if in_section:
                if "Occupied Eigen values" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+", line)
                    occupied.extend(map(float, nums))

                elif "Virtual Eigen values" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+", line)
                    virtual.extend(map(float, nums))

            if "DFTB total energy" in line:
                in_section = False

        self.complet_results["moe"] = {"occupied": occupied, "virtual": virtual}

        filename = os.path.join(self.workdir, "deMon.coef")
        if os.path.exists(filename):
            data = {
                "n_atoms": None,
                "n_basis": None,
                "parameter_type": None,
                "homo": None,
                "lumo_range": None,
                "atoms": [],
                "mo_energies": [],
                "occupations": [],
                "mo_coefficients": [],
            }

            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # --------- HEADER ----------
                if line == "NAtoms":
                    self._guard_coef_int_value(lines[i + COEF_VALUE_LINE_OFFSET], "NAtoms")
                    data["n_atoms"] = int(lines[i + COEF_VALUE_LINE_OFFSET].strip())
                    i += 2
                    continue

                elif line == "N basis":
                    self._guard_coef_int_value(lines[i + COEF_VALUE_LINE_OFFSET], "N basis")
                    data["n_basis"] = int(lines[i + COEF_VALUE_LINE_OFFSET].strip())
                    i += 2
                    continue

                elif line == "PArameter type":
                    data["parameter_type"] = lines[i + COEF_VALUE_LINE_OFFSET].strip()
                    i += 2
                    continue

                elif line == "homo":
                    self._guard_coef_int_value(lines[i + COEF_VALUE_LINE_OFFSET], "homo")
                    data["homo"] = int(lines[i + COEF_VALUE_LINE_OFFSET].strip())
                    i += 2
                    continue

                elif "llmos" in line:
                    self._guard_coef_llmos(lines[i + COEF_VALUE_LINE_OFFSET])
                    parts = lines[i + COEF_VALUE_LINE_OFFSET].split()
                    data["lumo_range"] = tuple(map(int, parts))
                    i += 2
                    continue

                # --------- ATOMS ----------
                elif line.startswith("atomic position"):
                    i += 1
                    data["atoms"] = []
                    while i < len(lines) and lines[i].strip():
                        parts = lines[i].split()
                        if parts[0] == "occupation":
                            break
                        self._guard_coef_atom_row(lines[i])
                        atom = {
                            "Z": int(parts[COEF_ATOM_Z_TOKEN]),
                            "symbol": parts[COEF_ATOM_SYMBOL_TOKEN],
                            "x": float(parts[COEF_ATOM_X_TOKEN]),
                            "y": float(parts[COEF_ATOM_Y_TOKEN]),
                            "z": float(parts[COEF_ATOM_ZCOORD_TOKEN]),
                            "basis_start": int(parts[COEF_ATOM_BASIS_START_TOKEN]),
                            "basis_end": int(parts[COEF_ATOM_BASIS_END_TOKEN]),
                        }
                        data["atoms"].append(atom)
                        i += 1
                    continue

                # --------- MO ENERGIES ----------
                elif line.startswith("occupation and MOE"):
                    i += 1
                    data["occupations"] = []
                    data["mo_energies"] = []
                    while i < len(lines) and lines[i].strip():
                        parts = lines[i].split()
                        if parts[1] == "Coefficients":
                            break
                        self._guard_coef_moe_row(lines[i])
                        data["occupations"].append(float(parts[COEF_MOE_OCCUPATION_TOKEN]))
                        data["mo_energies"].append(float(parts[COEF_MOE_ENERGY_TOKEN]))
                        i += 1
                    data["occupations"] = np.array(data["occupations"])
                    data["mo_energies"] = np.array(data["mo_energies"])
                    continue

                # --------- MO COEFFICIENTS ----------
                elif line.startswith("MO Coefficients"):
                    i += 1
                    coeffs = []
                    data["mo_coefficients"] = []
                    while i < len(lines) and lines[i].strip():
                        parts = lines[i].split()
                        if parts[0] == "NAtoms":
                            break
                        nums = re.findall(r"[-+]?\d*\.\d+", lines[i])
                        coeffs.extend(map(float, nums))
                        i += 1

                    data["mo_coefficients"] = np.reshape(
                        np.array(coeffs), (data["n_basis"], data["n_basis"])
                    )
                    continue

                i += 1

            self.complet_results["mos"] = data

    @safe_parse()
    def read_errors(
        self,
    ):
        """Scan for textual errors emitted by deMonNano and record them.

        Appends structured entries produced by :meth:`_add_error`.
        """
        for line in self.lines:
            if "ERROR :" in line:
                _str = line.split(ERROR_COLON_SEPARATOR)[ERROR_MESSAGE_TOKEN].strip()
                self._add_error("demon", _str, line=line.strip())

        msg = ["optimization not converged"]
        for line in self.lines:
            for m in msg:
                if m in line:
                    self._add_error("optimization", m, line=line.strip())

    @safe_parse("tensors")
    @exclude_flags(["ptmc", "freq"])
    def parse_tensors(self):

        text = "\n".join(self.lines)
        data = {}

        # Helpers
        def extract_vector(pattern):
            match = re.search(pattern, text)
            if match:
                return [float(x) for x in match.groups()]
            return None

        # Scalars / vectors
        data["mass_center"] = extract_vector(
            r"mass center\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        )

        data["charge_center"] = extract_vector(
            r"charge center\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        )

        monopole = re.search(r"charge monopole\s*=\s*([-\d\.]+)", text)
        if monopole:
            data["charge_monopole"] = float(monopole.group(1))

        data["charge_dipole"] = extract_vector(
            r"charge dipole\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        )

        norme = re.search(r"norme dipole\s*=\s*([-\d\.]+)", text)
        if norme:
            data["dipole_norm"] = float(norme.group(1))

        # Matrices
        inertia = [[0] * 3 for _ in range(3)]
        quadrupole = [[0] * 3 for _ in range(3)]

        matrix_pattern = re.findall(
            r"\(\s*(\d)\s*,\s*(\d)\s*\)\s*=\s*([-\d\.]+)\s+([-\d\.]+)", text
        )

        for i, j, val1, val2 in matrix_pattern:
            i, j = int(i) - TENSOR_LABEL_OFFSET, int(j) - TENSOR_LABEL_OFFSET
            inertia[i][j] = float(val1)
            quadrupole[i][j] = float(val2)

        data["inertia_matrix"] = inertia
        data["charge_quadrupole"] = quadrupole

        # Eigenvalues
        eig_inertia = re.search(
            r"Inertia eigenvalues\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", text
        )
        if eig_inertia:
            data["inertia_eigenvalues"] = [float(x) for x in eig_inertia.groups()]

        eig_charge = re.search(
            r"Charges eigenvalues\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", text
        )
        if eig_charge:
            data["charge_eigenvalues"] = [float(x) for x in eig_charge.groups()]

        self.complet_results["tensors"] = data

    # =================================
    # READ MODULES (basics)

    @safe_parse()
    @assert_flags("ptmd")
    def _read_ptmd(self):
        """Parse PTMD response"""
        for line in self.lines:
            if "ERROR IN MPI_INIT" in line:
                self._add_error("mpi", "ERROR IN MPI_INIT", line=line.strip())
                return

    @safe_parse("optimization")
    @assert_flags("opt")
    def _read_opt(self):
        """Parse optimization-specific output sections."""

        if "doforces" in self.flags:
            forces = []
            filename = os.path.join(self.workdir, "forces.out")
            with open(filename, "r", encoding="utf-8", errors="replace") as fd:
                for i, line in enumerate(fd.readlines()):
                    if i < FORCES_HEADER_LINES:
                        continue
                    forces.append(list(map(float, line.split())))
            self.complet_results["forces"] = np.array(forces)

        self.complet_results["optimization"] = {}
        for line in self.lines:
            if "Maximum gradient =" in line:
                value = float(line.split()[OPT_GRADIENT_TOKEN])
                self.complet_results["optimization"]["grad_max"] = value
                break

    @safe_parse("ptmc")
    @assert_flags("ptmc")
    def _read_ptmc(self):
        """Parse PTMC-specific output sections."""
        import re

        text = "\n".join(self.lines)
        data = {}

        seed_match = re.findall(r"SEED\s*=\s*((?:\d+\s+)+)", text)
        if seed_match:
            seeds = list(map(int, seed_match[0].split()))
            data["seeds"] = seeds

        patterns = {
            "nb_step": r"NB STEP\s*=\s*(\d+)",
            "optout": r"OPTOUT\s*=\s*(\d+)",
            "nb_temp": r"NB TEMP\s*=\s*(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = int(match.group(1))

        temps = []
        temp_matches = re.findall(r"\n\s*(\d+)\s+([\d\.]+)\s+([\d\.E\-\+]+)", text)

        for idx, t, val in temp_matches:
            temps.append({"index": int(idx), "temperature": float(t), "value": float(val)})

        data["temperatures"] = temps

        exchange = {}

        # 👉 méthode
        method_match = re.search(r"Method\s*:\s*(.+)", text)
        if method_match:
            exchange["method"] = method_match.group(1).strip()

        start_match = re.search(r"Start after\s+(\d+)\s+steps", text)
        if start_match:
            exchange["start_after"] = int(start_match.group(1))

        each_match = re.search(r"Each\s+(\d+)\s+step", text)
        if each_match:
            exchange["each_step"] = int(each_match.group(1))

        prob_match = re.search(r"probability of swap\s*:\s*([\d\.]+)", text)
        if prob_match:
            exchange["swap_probability"] = float(prob_match.group(1))

        data["exchange"] = exchange

        self.complet_results["ptmc"] = data

        _traj = {}
        if "traj" in self.flags:
            for out in range(len(data["temperatures"])):
                output = f"deMon.{(out + 1):02}.mol"
                filename = os.path.join(self.workdir, output)
                data, info = read_XYZ(filename, is_charges=False, velocities=False, keep=1)

                energies = np.array(
                    list(map(lambda x: float(x.split()[PTMC_TRAJ_VALUE_TOKEN]), info))
                )

                _traj[output] = {
                    "trajectory": data,
                    "energies": energies,
                }

            self.complet_results["trajectory"] = _traj

    @safe_parse()
    @assert_flags("md")
    def _read_md(self):
        """Parse molecular dynamics summary values."""

        energies_flags = {
            "average_potential_energy": "Average POTENTIAL ENERGY",
            "std_potential_energy": "Std Dev POTENTIAL ENERGY",
            "average_kinetic_energy": "Average KINETIC ENERGY",
            "std_kinetic_energy": "Std Dev KINETIC ENERGY",
            "average_total_energy": "Average TOTAL ENERGY",
            "std_total_energy": "Std Dev TOTAL ENERGY",
            "average_temperature": "Average TEMPERATURE",
            "std_temperature": "Std Dev TEMPERATURE",
            "energy_from_thermostat": "Energy transfer from thermostat",
            "energy_loss": "Energy loss from constraints",
        }

        for line in self.lines:
            for it, _flag in energies_flags.items():
                if _flag in line:
                    self.complet_results["energy"].update(
                        {it: float(line.split()[MD_ENERGY_VALUE_TOKEN])}
                    )

    @safe_parse()
    @assert_flags("neb")
    def _read_neb(self):
        """Parse nudged elastic band output sections.

        NEB output parsing is not implemented yet.  Rather than returning
        silently (which would leave the caller thinking the run was fully
        parsed), an explicit error entry is recorded so that
        :meth:`has_errors` / :meth:`raise_on_errors` surface it.
        """
        self._add_error(
            "neb",
            "NEB output parsing is not implemented yet.",
        )
