"""Tests of the deMon.mol reader :func:`deMonPy.molden.read_XYZ`."""

import numpy as np
import pytest

from deMonPy.molden import read_XYZ

# Graphite AB written by deMonNano (.run/pbc/deMon.mol): input geometry, then
# the BOMD frame with charges and the 9 Xx (8 cell corners + translation).
GRAPHITE_MOL = """\
    4
 INPUT GEOMETRY
  C      1.234088    2.137500    0.000000
  C      1.234091    0.712500    0.000000
  C      0.000000    1.425000    3.131000
  C      0.000003    0.000000    3.131000
      13
 BOMD INFORMATION      -6.95502185      0.00000000      0.00000000
  C      0.617042    1.068750   -1.565500      0.001837 0.00000000E+00 0.00000000E+00 0.00000000E+00
  C      0.617046   -0.356250   -1.565500     -0.001837 0.00000000E+00 0.00000000E+00 0.00000000E+00
  C     -0.617046    0.356250    1.565500     -0.001837 0.00000000E+00 0.00000000E+00 0.00000000E+00
  C     -0.617042   -1.068750    1.565500      0.001837 0.00000000E+00 0.00000000E+00 0.00000000E+00
 Xx     -0.617044   -1.068750   -3.131000
 Xx     -0.617044   -1.068750    3.131000
 Xx     -1.851135    1.068750   -3.131000
 Xx     -1.851135    1.068750    3.131000
 Xx      1.851135   -1.068750   -3.131000
 Xx      1.851135   -1.068750    3.131000
 Xx      0.617044    1.068750   -3.131000
 Xx      0.617044    1.068750    3.131000
 Xx      0.617046    1.068750    1.565500
"""

CELL = [[2.468179, 0.0, 0.0], [-1.234091, 2.1375, 0.0], [0.0, 0.0, 6.262]]


@pytest.fixture
def graphite_mol(tmp_path):
    filename = tmp_path / "deMon.mol"
    filename.write_text(GRAPHITE_MOL)
    return str(filename)


def test_cell_from_xx(graphite_mol):
    """Cell rebuilt from the Xx corners, Xx removed, input frame restored."""
    images, _ = read_XYZ(graphite_mol, is_charges=True)
    inp, out = images

    assert not inp.pbc.any()
    assert out.pbc.all()
    assert len(out) == 4
    assert set(out.get_chemical_symbols()) == {"C"}

    np.testing.assert_allclose(np.array(out.cell), CELL, atol=2e-6)
    np.testing.assert_allclose(out.positions, inp.positions, atol=2e-6)
    np.testing.assert_allclose(
        out.get_initial_charges(), [0.001837, -0.001837, -0.001837, 0.001837]
    )


def test_cell_dict_output(graphite_mol):
    """Same cell with ase_obj=False."""
    images, _ = read_XYZ(graphite_mol, is_charges=True, ase_obj=False)
    np.testing.assert_allclose(images[1]["cell"], CELL, atol=2e-6)
    assert images[1]["pbc"] is True


def test_velocities_with_xx(graphite_mol):
    """Xx lines (4 fields) do not break the velocity reading."""
    content = open(graphite_mol).read().splitlines()
    # read_XYZ with velocities parses the comment line as numbers: keep the BOMD frame
    with open(graphite_mol, "w") as fd:
        fd.write("\n".join(content[6:]) + "\n")

    images, _ = read_XYZ(graphite_mol, is_charges=True, velocities=True)
    np.testing.assert_allclose(images[0].get_velocities(), np.zeros((4, 3)))
    np.testing.assert_allclose(np.array(images[0].cell), CELL, atol=2e-6)


def test_wrong_number_of_xx(graphite_mol):
    """Anything else than 9 Xx raises ValueError."""
    content = open(graphite_mol).read().splitlines()
    content[6] = "      12"
    with open(graphite_mol, "w") as fd:
        fd.write("\n".join(content[:-1]) + "\n")

    with pytest.raises(ValueError, match="9 expected"):
        read_XYZ(graphite_mol, is_charges=True)
