"""Shape guards of the deMonNano output parsers (deMonPy/demonnano/output.py).

Each guarded parser is fed a minimal synthetic deMon output block directly
through ``read_output.lines`` (no binary, no file creation):

(a) a well-formed block parses to the exact expected value (read by eye from
    the synthetic text) and records no error;
(b) a malformed block (missing token / too few lines) records a ``"parse"``
    error through the existing mechanism (kind + message naming the section)
    and the reader degrades exactly as it did before the guard was added
    (``None`` via ``safe_parse`` or a partial result).

One test also pins a real in-repo sample (``.run/water_2``) to the reference
values produced by the unmodified parser.
"""

import os

import numpy as np
import pytest

from deMonPy.demonnano.output import read_output
from deMonPy.exceptions import OutputParseError


def _reader(text, flags=frozenset()):
    """Build a read_output whose content is *text* (no file involved)."""
    wo = read_output(workdir="./", flags=flags)
    wo.lines = text.splitlines(keepends=True)
    return wo


def _errors(wo):
    return wo.complet_results.get("errors", [])


# =============================================================================
# read_basics -- split(":") marker parser
# =============================================================================


class TestBasicsColonGuard:
    WELL_FORMED = (
        "Number of DFTB atoms           :      3\n"
        "Number of DFTB atom types      :      2\n"
        "Dimension of DFTB matrices     :      6\n"
        "Number of DFTB electrons       :     16\n"
        "Number of DFTB shells          :      4\n"
    )

    def test_well_formed_parses_counts(self):
        wo = _reader(self.WELL_FORMED)
        wo.read_basics()
        assert wo.complet_results["properties"] == {
            "scc_tolerance": None,
            "max_scc_cycles": None,
            "scc_mixing": None,
            "n_atoms": 3,
            "n_atom_types": 2,
            "matrix_dim": 6,
            "n_electrons": 16,
            "n_shells": 4,
        }
        assert _errors(wo) == []

    def test_malformed_count_records_and_degrades_to_none(self):
        wo = _reader(self.WELL_FORMED.replace(":      3\n", ":      three\n"))
        wo.read_basics()
        # degrades exactly as before: the natural int() failure is caught by
        # safe_parse, which resets the "properties" key to None
        assert wo.complet_results["properties"] is None
        guard_errors = [e for e in _errors(wo) if "Number of DFTB atoms" in e["message"]]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"
        assert "read_basics" in guard_errors[0]["message"]


# =============================================================================
# read_tddftb -- fixed-index values[...] parser
# =============================================================================


class TestTddftbFixedIndexGuard:
    WELL_FORMED = (
        " requested transitions to calculate     :       2\n"
        "\n"
        " SUMMARY SINGLET\n"
        "\n"
        "       w [eV]        Osz. strength      Transition      Weight     E_sp [eV]  Sym.\n"
        "\n"
        "      8.813          0.0013149          8   ->     9       0.998       8.805   S\n"
        "     11.399          0.0002187          7   ->     9       0.998      11.397   S\n"
    )

    def test_well_formed_parses_states(self):
        wo = _reader(self.WELL_FORMED, flags=frozenset({"td-dftb"}))
        wo.read_tddftb()
        assert wo.complet_results["singlet"] == {
            "state_0": {
                "w": 8.813,
                "oscillator": 0.0013149,
                "from": 8,
                "to": 9,
                "weight": 0.998,
                "energy": 8.805,
            },
            "state_1": {
                "w": 11.399,
                "oscillator": 0.0002187,
                "from": 7,
                "to": 9,
                "weight": 0.998,
                "energy": 11.397,
            },
        }
        assert "triplet" not in wo.complet_results
        assert _errors(wo) == []

    def test_malformed_transition_records_and_degrades_to_partial(self):
        line = (
            "     11.399          0.0002187          7   ->     9       0.998"
            "      11.397   S\n"
        )
        wo = _reader(
            self.WELL_FORMED.replace(
                line,
                "     11.399          0.0002187          7   ->\n",
            ),
            flags=frozenset({"td-dftb"}),
        )
        wo.read_tddftb()
        # the malformed line is skipped (as before) -> only one state
        assert list(wo.complet_results["singlet"]) == ["state_0"]
        guard_errors = [e for e in _errors(wo) if "SUMMARY SINGLET" in e["message"]]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"
        assert "read_tddftb" in guard_errors[0]["message"]
        assert "transition" in guard_errors[0]["message"]


# =============================================================================
# read_freq -- hard-coded slice parser
# =============================================================================


class TestFreqSliceGuard:
    WELL_FORMED = (
        " NUMBER OF ATOMS: 2\n"
        " MODE:    1\n"
        "  FREQUENCY:  1562.7  [1/CM]\n"
        "  INTENSITY:    44.0  [KM/MOL]\n"
        "\n"
        "      ATOM  INP.NR.      X            Y            Z\n"
        "\n"
        "      O         1       0.0000       0.0716       0.0000\n"
        "      H         2      -0.4181      -0.5681       0.0000\n"
        "ZERO-POINT ENERGY =   0.02151778 HARTREE   =   13.5026 KCAL/MOL\n"
    )

    def test_well_formed_parses_mode_and_zpe(self):
        wo = _reader(self.WELL_FORMED, flags=frozenset({"freq"}))
        wo.read_freq()
        freq = wo.complet_results["frequency"]
        assert len(freq) == 1
        assert freq[0]["mode"] == 1
        assert freq[0]["frequency"] == 1562.7
        assert freq[0]["intensity"] == 44.0
        assert freq[0]["vect"].shape == (2, 3)
        assert wo.complet_results["zpe"] == 0.02151778
        assert _errors(wo) == []

    def test_short_block_records_and_degrades_to_partial(self):
        # 3 atoms declared, but the file ends mid-block (the last atom row
        # has no trailing newline): the slice silently drops that row, as
        # before -- the guard now records it
        text = (
            " NUMBER OF ATOMS: 3\n"
            " MODE:    1\n"
            "  FREQUENCY:  1562.7  [1/CM]\n"
            "  INTENSITY:    44.0  [KM/MOL]\n"
            "\n"
            "      ATOM  INP.NR.      X            Y            Z\n"
            "\n"
            "      O         1       0.0000       0.0716       0.0000\n"
            "      H         2      -0.4181      -0.5681       0.0000\n"
            "      C         3       0.1000       0.2000       0.3000"  # no trailing newline
        )
        wo = _reader(text, flags=frozenset({"freq"}))
        wo.read_freq()
        # degrades as before: the table is simply built from the 2 rows
        freq = wo.complet_results["frequency"]
        assert len(freq) == 1
        assert freq[0]["mode"] == 1
        assert freq[0]["vect"].shape == (2, 3)
        assert wo.complet_results["zpe"] == 0.0
        guard_errors = [
            e for e in _errors(wo) if e["message"].startswith("read_freq: MODE block")
        ]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"

    def test_short_atom_row_records_and_degrades_to_none(self):
        # one atom row missing its coordinate tokens -> ragged table, the
        # natural np.array failure is caught by safe_parse (as before)
        wo = _reader(
            self.WELL_FORMED.replace(
                "      O         1       0.0000       0.0716       0.0000\n",
                "      O         1       0.0000\n",
            ),
            flags=frozenset({"freq"}),
        )
        wo.read_freq()
        assert wo.complet_results["frequency"] is None
        assert wo.complet_results["zpe"] is None
        guard_errors = [
            e for e in _errors(wo) if e["message"].startswith("read_freq: atom-displacement")
        ]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"


# =============================================================================
# read_ci -- fixed-index sl[...] parser
# =============================================================================


class TestCiFixedIndexGuard:
    WELL_FORMED = (
        " ************    STATES    ************\n"
        " for state   1 =     -23.34492577\n"
        "  weight of conf   1 = 0.997473 (  0.997864 )\n"
        "  weight of conf   2 = 0.002527 (  0.028022 )\n"
        "   molecule   1 = 0.997473\n"
        "   molecule   2 = 0.002527\n"
    )

    def test_well_formed_parses_state(self):
        wo = _reader(self.WELL_FORMED, flags=frozenset({"ci"}))
        wo.read_energy()  # mirrors parse_output: read_energy runs before read_ci
        wo.read_ci()
        assert wo.complet_results["states"] == {
            "state 1": {
                "energy": -23.34492577,
                "weight": [0.997473, 0.002527],
                "charges_mulliken": [0.997473, 0.002527],
            }
        }
        assert wo.complet_results["energy"] == {"energy": -23.34492577}
        assert wo.complet_results["charges"] == {"charges mulliken": [0.997473, 0.002527]}
        assert _errors(wo) == []

    def test_malformed_weight_records_and_degrades_to_partial(self):
        wo = _reader(
            self.WELL_FORMED.replace(
                "  weight of conf   1 = 0.997473 (  0.997864 )\n",
                "  weight of conf   1 =\n",
            ),
            flags=frozenset({"ci"}),
        )
        wo.read_energy()
        wo.read_ci()
        # degrades as before: the weight token is skipped and set to None
        assert wo.complet_results["states"]["state 1"]["weight"] == [None, 0.002527]
        guard_errors = [e for e in _errors(wo) if "weight of conf" in e["message"]]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"
        assert "read_ci" in guard_errors[0]["message"]


# =============================================================================
# read_pbc -- hard-coded i + nband*2 window parser
# =============================================================================


class TestPbcWindowGuard:
    WELL_FORMED = (
        "Dimension of DFTB matrices     :      2\n"
        "\n"
        "    1\n"
        " kpoint number           1\n"
        "  -0.500000000000000        2.00000000000000\n"
        "  -0.400000000000000        0.00000000000000\n"
        "\n"
        "    2\n"
        " kpoint number           2\n"
        "  -0.450000000000000        2.00000000000000\n"
        "  -0.300000000000000        0.00000000000000\n"
    )

    def _reader(self, text):
        wo = _reader(text, flags=frozenset({"pbc"}))
        wo.read_basics()
        return wo

    def test_well_formed_parses_eigenvalues(self):
        wo = self._reader(self.WELL_FORMED)
        wo.read_pbc()
        np.testing.assert_allclose(
            wo.complet_results["eigenvalues"],
            [[-0.5, -0.4], [-0.45, -0.3]],
        )
        np.testing.assert_allclose(
            wo.complet_results["k-points"],
            [[[-0.5, 2.0], [-0.4, 0.0]], [[-0.45, 2.0], [-0.3, 0.0]]],
        )
        # no in-repo sample prints the fractional coordinates: they parse to
        # an empty column set (nkpt, 0)
        assert wo.complet_results["kpts_coords"].shape == (2, 0)
        assert _errors(wo) == []

    def test_missing_eigenvalue_records_and_degrades_to_none(self):
        # drop the last (energy, occupation) line of the second k-point
        missing = "  -0.300000000000000        0.00000000000000\n"
        wo = self._reader(self.WELL_FORMED.replace(missing, ""))
        wo.read_pbc()
        # the natural reshape failure is caught by safe_parse (as before)
        assert wo.complet_results["k-points"] is None
        assert wo.complet_results["eigenvalues"] is None
        assert wo.complet_results["kpts_coords"] is None
        guard_errors = [
            e for e in _errors(wo) if e["message"].startswith("read_pbc: expected")
        ]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"


# =============================================================================
# read_AOM_matrix -- row value count vs header column count
# =============================================================================


class TestAomRowCountGuard:
    # single-header AOM block (matrix_dim <= 10 prints one "Orbit" header):
    # each file row fills one column of the tab, then the tab is transposed
    # when the last header column equals the matrix dimension
    WELL_FORMED = (
        "Dimension of DFTB matrices     :      4\n"
        "                    S \n"
        "Orbit El  N  L   1  2  3  4\n"
        "1  O  2  0  0.10  0.20  0.30  0.40\n"
        "2  O  2  0  0.50  0.60  0.70  0.80\n"
        "3  H  1  0  0.11  0.12  0.13  0.14\n"
        "4  H  1  0  0.21  0.22  0.23  0.24\n"
        "&&\n"
    )

    def _reader(self, text):
        wo = _reader(text)
        wo.read_basics()  # seeds complet_results["properties"]["matrix_dim"]
        return wo

    def test_well_formed_parses_matrix(self):
        wo = self._reader(self.WELL_FORMED)
        data = wo.read_AOM_matrix()
        assert len(data) == 1
        # values read by eye from the synthetic text above, after the
        # final tab.T applied when the last header column == matrix_dim
        np.testing.assert_allclose(
            data[0],
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.50, 0.60, 0.70, 0.80],
                [0.11, 0.12, 0.13, 0.14],
                [0.21, 0.22, 0.23, 0.24],
            ],
        )
        assert _errors(wo) == []

    def test_incoherent_row_raises_diagnosable_error(self):
        # header declares 3 data columns, row 1 carries 5 values: the count
        # mismatch is detected before the numpy assignment and raises a
        # diagnosable OutputParseError instead of an opaque broadcast error
        text = (
            "Dimension of DFTB matrices     :      3\n"
            "                    S \n"
            "Orbit El  N  L   1  2  3\n"
            "1  O  2  0  0.10  0.00  0.00  0.99  0.88\n"
            "2  O  2  0  0.00  0.20  0.00\n"
            "3  O  2  0  0.00  0.00  0.30\n"
            "&&\n"
        )
        wo = self._reader(text)
        with pytest.raises(OutputParseError) as excinfo:
            wo.read_AOM_matrix()
        msg = str(excinfo.value)
        assert "read_AOM_matrix" in msg
        assert "'S'" in msg
        assert "row 1 has 5 values" in msg
        assert "header declares 3 columns" in msg
        assert "0.99  0.88" in msg  # the offending line is quoted
        # the 4.1 record guard still records before the raise (keeps the
        # offending line in the "line" field of the entry)
        guard_errors = [
            e for e in _errors(wo) if e["message"].startswith("read_AOM_matrix: data row")
        ]
        assert len(guard_errors) == 1
        assert guard_errors[0]["kind"] == "parse"
        assert "expects 3 values" in guard_errors[0]["message"]
        assert "found 5" in guard_errors[0]["message"]
        assert guard_errors[0]["line"] == "1  O  2  0  0.10  0.00  0.00  0.99  0.88"


# =============================================================================
# real sample regression (.run artifacts shipped with the repo)
# =============================================================================


WATER_2 = os.path.join(".run", "water_2", "deMon.out")


@pytest.mark.skipif(not os.path.exists(WATER_2), reason="no .run/water_2 sample in this checkout")
def test_water_2_sample_parses_as_before():
    """Reference values captured with the unmodified parser on the same file."""
    wo = read_output(workdir=os.path.join(".run", "water_2"), flags=frozenset({"freq"}))
    wo.read_file()
    wo.read_basics()
    wo.read_freq()

    p = wo.complet_results["properties"]
    assert p["n_atoms"] == 3
    assert p["n_atom_types"] == 2
    assert p["matrix_dim"] == 6
    assert p["n_electrons"] == 8
    assert p["n_shells"] == 4

    assert wo.complet_results["zpe"] == 0.02151778
    freq = wo.complet_results["frequency"]
    assert [(m["mode"], m["frequency"], m["intensity"]) for m in freq] == [
        (1, 1562.7, 44.0),
        (2, 3804.2, 6.5),
        (3, 4078.2, 0.1),
        (4, 0.0, 0.0),
        (5, 0.0, 0.0),
        (6, 0.0, 0.0),
        (7, 0.0, 0.0),
        (8, 0.0, 0.0),
        (9, 0.0, 0.0),
    ]
    assert all(m["vect"].shape == (3, 3) for m in freq)
    assert wo.complet_results["errors"] == []
