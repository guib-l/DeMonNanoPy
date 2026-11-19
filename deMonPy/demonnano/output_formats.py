#!/usr/bin/env python3
"""deMonNano output line formats used by the output readers."""

from typing import NamedTuple

# =============================================================================
# deMon output line formats
#
# The parsers below read deMon output (deMon.out / deMon.mol / deMon.coef)
# by raw token position.  Each ``LineFormat`` describes one line shape once:
# a real sample line and, for every value, its token index and converter.
# ``read_output._parse_line`` converts a line against its format and records
# a "parse" error (via ``_add_error``) when the line does not have the
# expected shape; it never changes the valid-output result.
# =============================================================================


class LineFormat(NamedTuple):
    """Shape of one deMon output line parsed by raw token position.

    Attributes:
        name: Short description used in error messages.
        sample: A real deMon line of this shape.
        fields: ``{key: (token_index, converter)}``.
        sep: Token separator (``None`` = whitespace).
    """

    name: str
    sample: str
    fields: dict
    sep: str = None


# --- read_basics ---------------------------------------------------------------
BASICS_COUNT = LineFormat(
    "count", "Number of DFTB atoms           :      6", {"value": (-1, int)}, sep=":"
)

# --- read_ci -------------------------------------------------------------------
CI_STATE = LineFormat(
    "'for state' line",
    "for state   1 =     -23.34492577",
    {"num": (2, int), "energy": (4, float)},
)
CI_CONF_WEIGHT = LineFormat(
    "'weight of conf' line",
    "weight of conf   1 = 0.997473 (  0.997864 )",
    {"weight": (5, float)},
)
CI_ATOM_CHARGE = LineFormat(
    "'molecule' line", "molecule   1 = 0.997473", {"charge": (3, float)}
)

# --- read_tddftb ---------------------------------------------------------------
TDDFTB_NTRANSITIONS = LineFormat(
    "'requested transitions to calculate' line",
    "requested transitions to calculate     :       10",
    {"value": (-1, int)},
)
TDDFTB_TRANSITION = LineFormat(
    "transition line",
    "8.813  0.0013149  8   ->     9  0.998  8.805  S",
    {
        "w": (0, float),
        "oscillator": (1, float),
        "from": (2, int),
        "to": (4, int),
        "weight": (5, float),
        "energy": (6, float),
    },
)
# a SUMMARY line with "->" at token 3 is a transition line
TDDFTB_ARROW_TOKEN, TDDFTB_ARROW = 3, "->"
# SUMMARY block = signature line + blank + column header + blank + N data lines
TDDFTB_SUMMARY_HEADER_LINES = 4

# --- read_freq -----------------------------------------------------------------
FREQ_ATOMS = LineFormat("'NUMBER OF ATOMS:' line", " NUMBER OF ATOMS: 3", {"value": (-1, int)})
FREQ_ZPE = LineFormat(
    "'ZERO-POINT ENERGY =' line",
    "ZERO-POINT ENERGY =   0.02151778 HARTREE   =   13.5026 KCAL/MOL",
    {"value": (3, float)},
)
FREQ_MODE = LineFormat("'MODE:' line", "MODE:    1", {"value": (1, int)})
FREQ_FREQUENCY = LineFormat(
    "'FREQUENCY:' line", "FREQUENCY:  1562.7  [1/CM]", {"value": (1, float)}
)
FREQ_INTENSITY = LineFormat(
    "'INTENSITY:' line", "INTENSITY:    44.0  [KM/MOL]", {"value": (1, float)}
)
FREQ_VECT = LineFormat(
    "atom-displacement row",
    "O  1  0.0000  0.0716  0.0000",
    {"x": (2, float), "y": (3, float), "z": (4, float)},
)
# each MODE block = 6 header lines + Nbatm atom rows:
# "MODE:", "FREQUENCY:", "INTENSITY:", blank, "ATOM  INP.NR.  X  Y  Z", blank
FREQ_MODE_HEADER_LINES = 6

# --- read_AOM_matrix -----------------------------------------------------------
# "Orbit El  N  L   1  2 ..." header / "1  O  2  0  1.000000 ..." rows:
# column numbers and values start at token 4, rows/columns are 1-based
AOM_DATA_START = 4

# --- read_debug (Gamma matrix) -------------------------------------------------
# "Shell El  1  2 ..." header / "1 O  0.52220 ..." rows: data from token 2
GAMMA_DATA_START = 2

# --- read_pbc ------------------------------------------------------------------
# each real line occupies 2 entries in the splitlines() view ("\n".join of
# readlines() keeps the trailing newlines), so the scan window is nband * 2
PBC_EIG_SCAN_WINDOW_FACTOR = 2
# " -0.790839405237485        2.00000000000000" -- (eigenvalue, occupation)
PBC_EIG_TOKEN_COUNT = 2
# fractional k-point coordinates: last non-empty line within 3 lines before
# "kpoint number", tokens 1:4 (no in-repo sample prints them; shape unverified)
PBC_KCOORD_LOOKBACK = 3

# --- read_print (deMon.coef) ---------------------------------------------------
# header labels ("NAtoms", "homo", ...) are followed by their value line
COEF_INT_VALUE = LineFormat("integer value line", "6", {"value": (0, int)})
COEF_LUMO_RANGE = LineFormat("'llmos' value line", "1  36", {"lo": (0, int), "hi": (1, int)})
COEF_ATOM = LineFormat(
    "atom row",
    "8  O  0.6550  -0.0544  5.9307  1  4",
    {
        "Z": (0, int),
        "symbol": (1, str),
        "x": (2, float),
        "y": (3, float),
        "z": (4, float),
        "basis_start": (5, int),
        "basis_end": (6, int),
    },
)
COEF_MOE = LineFormat(
    "MOE row", "1 2.00  -0.8381", {"occupation": (1, float), "energy": (2, float)}
)

# --- read_geometry (deMon.mol MD frames) ---------------------------------------
# frame comment "BOMD INFORMATION  <e0> <e1> <e2> <e3>" (see molden._read_xyz_ext)
XYZ_INFO_POTENTIAL_COL = 0
XYZ_INFO_KINETIC_COL = 1
XYZ_INFO_TOTAL_COL = 2
XYZ_INFO_TIME_COL = 3
