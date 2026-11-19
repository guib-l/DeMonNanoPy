"""Explicit UTF-8 encoding of the demonnano file I/O.

Binary-free tests: every file involved is a small synthetic file written
with raw bytes into ``tmp_path`` (via ``Path.write_bytes``), so the
encoding behaviour of the ``open()`` calls in ``deMonPy/demonnano/`` and
``deMonPy/molden.py`` is
really exercised.  deMon output files are read with
``encoding="utf-8", errors="replace"``: a stray invalid byte must not
raise ``UnicodeDecodeError`` and must surface as U+FFFD, while the rest
of the file still parses to the exact expected values.  The written
input file (``deMon.inp``) must be decodable as UTF-8.
"""

import numpy as np

from deMonPy.deMonNano import deMonNano
from deMonPy.demonnano.output import read_output
from deMonPy.molden import read_XYZ, write_XYZ


class TestReadFileEncoding:
    """read_file: the main deMon output (deMon.out), a deMon-produced file."""

    def test_invalid_byte_becomes_replacement_char(self, tmp_path):
        (tmp_path / "deMon.out").write_bytes(
            b"# deMon output for caf\xc3\xa9 \xff (invalid byte)\n"
            b"Number of DFTB atoms           :      3\n"
            b"Number of DFTB atom types      :      2\n"
            b"Dimension of DFTB matrices     :      6\n"
            b"Number of DFTB electrons       :     16\n"
            b"Number of DFTB shells          :      4\n"
        )
        wo = read_output(workdir=str(tmp_path), flags=frozenset())
        wo.read_file()  # must not raise UnicodeDecodeError
        # the valid UTF-8 é survives, the raw 0xFF byte becomes U+FFFD
        assert wo.lines[0] == "# deMon output for caf\u00e9 \ufffd (invalid byte)\n"
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
        assert wo.complet_results["errors"] == []


class TestReadPrintCoefEncoding:
    """read_print: deMon.coef, a deMon-produced file."""

    def test_invalid_byte_in_parameter_type(self, tmp_path):
        (tmp_path / "deMon.coef").write_bytes(
            b"NAtoms\n"
            b"   1\n"
            b" N basis\n"
            b"   2\n"
            b" PArameter type\n"
            b"BIO caf\xc3\xa9 \xff\n"
            b" homo\n"
            b"   1\n"
            b" llmos , ulmos\n"
            b"    1   2\n"
            b" atomic position\n"
            b"    8  O      0.6550     -0.0544      5.9307    1    2\n"
            b" occupation and MOE\n"
            b"    1 2.00     -0.9132\n"
            b"    2 1.00      0.5231\n"
            b" MO Coefficients\n"
            b"  0.7071  0.7071\n"
            b"  0.7071 -0.7071\n"
        )
        wo = read_output(workdir=str(tmp_path), flags=frozenset({"print"}))
        wo.read_print()  # must not raise UnicodeDecodeError
        mos = wo.complet_results["mos"]
        assert mos["n_atoms"] == 1
        assert mos["n_basis"] == 2
        # the é (valid UTF-8) survives, the raw 0xFF byte becomes U+FFFD
        assert mos["parameter_type"] == "BIO caf\u00e9 \ufffd"
        assert mos["homo"] == 1
        assert mos["lumo_range"] == (1, 2)
        atom = mos["atoms"][0]
        assert (atom["Z"], atom["symbol"]) == (8, "O")
        assert (atom["x"], atom["y"], atom["z"]) == (0.6550, -0.0544, 5.9307)
        assert (atom["basis_start"], atom["basis_end"]) == (1, 2)
        np.testing.assert_allclose(mos["occupations"], [2.0, 1.0])
        np.testing.assert_allclose(mos["mo_energies"], [-0.9132, 0.5231])
        np.testing.assert_allclose(
            mos["mo_coefficients"], [[0.7071, 0.7071], [0.7071, -0.7071]]
        )
        assert wo.complet_results["errors"] == []


class TestReadOptForcesEncoding:
    """_read_opt: forces.out, a deMon-produced file (opt + doforces flags)."""

    def test_invalid_byte_in_header_line(self, tmp_path):
        (tmp_path / "forces.out").write_bytes(
            b"FORCES for caf\xc3\xa9 \xff (eV/Bohr)\n"
            b" 0.10  0.20  0.30\n"
            b" 0.40  0.50  0.60\n"
        )
        wo = read_output(workdir=str(tmp_path), flags=frozenset({"opt", "doforces"}))
        wo._read_opt()  # must not raise UnicodeDecodeError
        np.testing.assert_allclose(
            wo.complet_results["forces"], [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]
        )
        assert wo.complet_results["optimization"] == {}
        assert wo.complet_results["errors"] == []


class TestWriteInputEncoding:
    """write(): the deMon.inp input file must be written as UTF-8."""

    def test_deMon_inp_is_utf8_decodable(self, tmp_path):
        calc = deMonNano(
            execut=None,
            workdir=str(tmp_path),
            basis={"PTYPE": "BIO", "SKFILE": "slako"},
            DEMON_PARAMETERS={"ACTIVE": {"DFTB": {"SCC": True}}},
        )
        calc.write_input(["C", "C"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

        inp = tmp_path / "deMon.inp"
        assert inp.exists()
        text = inp.read_text(encoding="utf-8")  # must not raise UnicodeDecodeError
        lines = text.splitlines()
        assert lines[0] == "TITLE"
        assert any(line.startswith("DFTB SCC") for line in lines)
        assert "GEOMETRY" in text


class TestReadXyzMoldenEncoding:
    """molden.read_XYZ: deMon.mol, a deMon-produced file."""

    XYZ = (
        b"    2\n"
        b"energy (Ha) : -1.2345 | caf\xc3\xa9 \xff\n"
        b"   C      0.000000    0.000000    0.000000\n"
        b"   C      1.400000    0.000000    0.000000\n"
    )

    def test_invalid_byte_in_comment_does_not_raise(self, tmp_path):
        path = tmp_path / "deMon.mol"
        path.write_bytes(self.XYZ)
        images, info = read_XYZ(str(path), ase_obj=False)  # no UnicodeDecodeError
        np.testing.assert_array_equal(images[0]["symbols"], ["C", "C"])
        np.testing.assert_allclose(images[0]["positions"], [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
        # the valid UTF-8 é survives, the raw 0xFF byte becomes U+FFFD
        assert info[0] == "energy (Ha) : -1.2345 | caf\u00e9 \ufffd\n"


class TestWriteXyzMoldenEncoding:
    """molden.write_XYZ: the user-facing XYZ output must be UTF-8."""

    def test_written_file_is_utf8_decodable(self, tmp_path):
        src = tmp_path / "deMon.mol"
        src.write_bytes(TestReadXyzMoldenEncoding.XYZ)
        images, _ = read_XYZ(str(src))

        out = tmp_path / "out.xyz"
        write_XYZ(str(out), images, comment="caf\u00e9")

        text = out.read_bytes().decode("utf-8")  # must not raise UnicodeDecodeError
        assert text.splitlines()[0].strip() == "2"
        assert "caf\u00e9" in text
