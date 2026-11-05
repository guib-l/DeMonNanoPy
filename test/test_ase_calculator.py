"""Tests of the ASE calculator :class:`deMonPy.ase_calculator.DeMonNano`.

* ``TestUnit`` (A1-A7): the deMonNano engine is replaced by a fake one
  returning hand-written raw results; no executable is needed.
* ``TestIntegration`` (B1, B3, B5, B10): real deMonNano calculations.
"""

import os
from copy import deepcopy

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import SCFError
from ase.calculators.fd import calculate_numerical_forces
from ase.units import Bohr, Hartree

import deMonPy
import deMonPy.ase_calculator as ase_calculator
from deMonPy.ase_calculator import DeMonNano
from deMonPy.deMonNano import deMonNano
from deMonPy.exceptions import OutputParseError

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True, "DISP": 2},
            "CHARGE": 0.0,
        },
    },
}

parameters_pbc = deepcopy(parameters)
parameters_pbc["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"]["FERMI"] = 50
parameters_pbc["DEMON_PARAMETERS"]["ACTIVE"]["PBC"] = {
    "TYPE": "GENERAL",
    "LATTICE": np.array([[2.46, 0.0, 0.0], [-1.23, 2.130422, 0.0], [0.0, 0.0, 100.0]]),
    "KPTS": [10, 10, 1],
}

water_dimer = Atoms(
    ["O", "H", "H", "O", "H", "H"],
    positions=[
        [1.2478, -0.5185, 3.4049],
        [1.5946, -1.4204, 3.3886],
        [0.9008, -0.3341, 2.5062],
        [3.2478, -0.4185, 3.4049],
        [3.5946, -1.5204, 3.3886],
        [2.9008, -0.3341, 2.6062],
    ],
)

graphene = Atoms(
    ["C"] * 2,
    positions=[[0.0, 0.0, 0.0], [1.23, 0.71, 0.0]],
    cell=[[2.46, 0.0, 0.0], [-1.23, 2.130422, 0.0], [0.0, 0.0, 100.0]],
    pbc=True,
)

WORKDIR = ".run/ase-calculator"


# =============================================================================
# A. Unit tests (fake deMonNano engine)
# =============================================================================


class FakeDeMonNano:
    """Stand-in for :class:`deMonPy.deMonNano.deMonNano`.

    ``raw`` is returned as ``results``; ``kwargs`` keeps the constructor
    arguments given by the ASE calculator.
    """

    raw = {}
    kwargs = None
    calc_kwargs = None

    def __init__(self, **kwargs):
        FakeDeMonNano.kwargs = kwargs

    def calculate(self, **kwargs):
        FakeDeMonNano.calc_kwargs = kwargs
        self.results = deepcopy(FakeDeMonNano.raw)
        # deMonPy always returns the geometry read from deMon.mol
        self.results.setdefault(
            "output_geometry", Atoms(kwargs["symbols"], positions=kwargs["positions"])
        )


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ase_calculator, "deMonNano", FakeDeMonNano)
    FakeDeMonNano.raw = {}
    FakeDeMonNano.kwargs = None
    FakeDeMonNano.calc_kwargs = None
    return FakeDeMonNano


def _water():
    return Atoms(["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])


def _graphene_raw(kcoords, fermi=None):
    """Raw periodic results: 4 k-points (2x2x1 grid), 2 bands."""
    eps = np.array([[-0.50, 0.10], [-0.40, 0.20], [-0.45, 0.15], [-0.30, 0.25]])
    occ = np.array([[2.0, 0.0]] * 4)
    energy = {"energy": -10.0}
    if fermi is not None:
        energy["fermi_energy"] = fermi
    return {
        "energy": energy,
        "k-points": np.stack([eps, occ], axis=-1),
        "kpts_coords": kcoords,
    }


class TestUnit:
    def test_units(self, fake, tmp_path):
        """A1: energy Ha -> eV, forces = -gradient * Hartree / Bohr."""
        grad = np.array([[0.1, -0.2, 0.3], [0.0, 0.05, -0.1], [-0.1, 0.15, -0.2]])
        fake.raw = {"energy": {"energy": -1.5}, "forces": grad}

        calc = DeMonNano(directory=str(tmp_path), **parameters)
        calc.calculate(_water(), properties=["energy", "forces"])

        assert calc.results["energy"] == pytest.approx(-1.5 * Hartree, rel=1e-12)
        assert calc.results["free_energy"] == pytest.approx(-1.5 * Hartree, rel=1e-12)
        np.testing.assert_allclose(calc.results["forces"], -grad * Hartree / Bohr, rtol=1e-12)

    def test_scf_error(self, fake, tmp_path):
        """A2: an "ERROR :" line of deMon.out raises SCFError."""
        fake.raw = {
            "energy": {"energy": -1.5},
            "errors": [
                {
                    "kind": "demon",
                    "message": "loop over charges not converged",
                    "line": "ERROR : loop over charges not converged",
                }
            ],
        }
        calc = DeMonNano(directory=str(tmp_path), **parameters)
        with pytest.raises(SCFError, match="loop over charges not converged"):
            calc.calculate(_water(), properties=["energy"])

    def test_optimization_error_ignored(self, fake, tmp_path):
        """A2: errors of kind "optimization" do not raise."""
        fake.raw = {
            "energy": {"energy": -1.5},
            "errors": [
                {"kind": "optimization", "message": "optimization not converged", "line": ""}
            ],
        }
        calc = DeMonNano(directory=str(tmp_path), **parameters)
        calc.calculate(_water(), properties=["energy"])
        assert calc.results["energy"] == pytest.approx(-1.5 * Hartree, rel=1e-12)

    def test_forces_unreadable(self, fake, tmp_path):
        """A3: unreadable gradient block raises OutputParseError."""
        fake.raw = {"energy": {"energy": -1.5}, "forces": None}
        calc = DeMonNano(directory=str(tmp_path), **parameters)
        with pytest.raises(OutputParseError):
            calc.calculate(_water(), properties=["energy", "forces"])

    @pytest.mark.parametrize(
        "properties, module",
        [
            (["energy"], {}),
            (["energy", "forces"], {"ACTIVE": {"OPT": {"SP": True}}}),
        ],
    )
    def test_opt_sp_module(self, fake, tmp_path, properties, module):
        """A4: the OPT SP module is active only when forces are requested."""
        fake.raw = {"energy": {"energy": -1.5}, "forces": np.zeros((3, 3))}
        calc = DeMonNano(directory=str(tmp_path), **parameters)
        calc.calculate(_water(), properties=properties)
        assert fake.kwargs["DEMON_MODULE"] == module

    def test_charges_read(self, fake, tmp_path):
        """A8: charges read from deMon.mol without forces, absent with OPT SP."""
        geometry = _water()
        geometry.set_initial_charges([-0.6, 0.3, 0.3])
        fake.raw = {
            "energy": {"energy": -1.5},
            "forces": np.zeros((3, 3)),
            "output_geometry": geometry,
        }

        calc = DeMonNano(directory=str(tmp_path), **parameters)
        calc.calculate(_water(), properties=["energy"])
        assert fake.calc_kwargs["read_charges"] is True
        np.testing.assert_allclose(calc.results["charges"], [-0.6, 0.3, 0.3])

        calc = DeMonNano(directory=str(tmp_path), **parameters)
        calc.calculate(_water(), properties=["energy", "forces"])
        assert fake.calc_kwargs["read_charges"] is False
        assert "charges" not in calc.results

    def test_fermi_level(self, fake, tmp_path):
        """A5: Fermi level read from the output, else highest occupied eigenvalue."""
        kcoords = np.array([[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
        atoms = graphene.copy()

        fake.raw = _graphene_raw(kcoords, fermi=-0.2)
        calc = DeMonNano(directory=str(tmp_path), kpts=(2, 2, 1), **parameters_pbc)
        calc.calculate(atoms, properties=["energy"])
        assert calc.results["fermi_level"] == pytest.approx(-0.2 * Hartree, rel=1e-12)

        fake.raw = _graphene_raw(kcoords)
        calc = DeMonNano(directory=str(tmp_path), kpts=(2, 2, 1), **parameters_pbc)
        calc.calculate(atoms, properties=["energy"])
        assert calc.results["fermi_level"] == pytest.approx(-0.30 * Hartree, rel=1e-12)

    def test_kpoints_rebuilt(self, fake, tmp_path):
        """A6: k-point grid rebuilt (k1 outer, k3 inner) when not printed."""
        fake.raw = _graphene_raw(np.array([]))
        calc = DeMonNano(directory=str(tmp_path), kpts=(2, 2, 1), **parameters_pbc)
        calc.calculate(graphene.copy(), properties=["energy"])

        expected = np.array([[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0], [0.5, 0.5, 0]])
        np.testing.assert_allclose(calc.results["ibz_kpoints"], expected)
        np.testing.assert_allclose(calc.results["kpoint_weights"], np.full(4, 0.25))
        assert calc.results["eigenvalues"].shape == (1, 4, 2)
        assert calc.results["occupations"].shape == (1, 4, 2)

    def test_kpoints_mismatch(self, fake, tmp_path):
        """A7: number of k-points != number of k-point coordinates."""
        fake.raw = _graphene_raw(np.array([[0, 0, 0], [0, 0.5, 0], [0.5, 0, 0]]))
        calc = DeMonNano(directory=str(tmp_path), kpts=(2, 2, 1), **parameters_pbc)
        with pytest.raises(OutputParseError):
            calc.calculate(graphene.copy(), properties=["energy"])


# =============================================================================
# B. Integration tests (deMonNano executable)
# =============================================================================


class TestIntegration:
    @pytest.mark.forces
    def test_numerical_forces(self):
        """B1: analytical forces = finite-difference forces (water dimer)."""
        atoms = water_dimer.copy()
        atoms.calc = DeMonNano(directory=os.path.join(WORKDIR, "forces"), **parameters)

        forces = atoms.get_forces()
        numerical = calculate_numerical_forces(atoms, eps=1e-3)

        np.testing.assert_allclose(forces, numerical, rtol=1e-3, atol=1e-3)

    @pytest.mark.dynamics
    def test_periodic_verlet(self):
        """B2: NVE VelocityVerlet on graphene conserves Etot (periodic forces)."""
        from ase import units
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet

        atoms = graphene.copy()
        atoms.calc = DeMonNano(
            directory=os.path.join(WORKDIR, "verlet"), kpts=(10, 10, 1), **parameters_pbc
        )
        MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))

        dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
        etot = []
        dyn.attach(lambda: etot.append(atoms.get_total_energy()), interval=1)
        dyn.run(40)

        etot = np.array(etot)
        assert len(etot) == 41
        assert np.abs(etot - etot[0]).max() < 1e-4  # eV

    @pytest.mark.parametrize(
        "atoms, params, kpts",
        [
            (water_dimer, parameters, None),
            (graphene, parameters_pbc, (10, 10, 1)),
        ],
        ids=["water-dimer", "graphene"],
    )
    def test_same_energy_as_demonnano(self, atoms, params, kpts):
        """B3: ASE energy = deMonNano energy (Hartree) * Hartree."""
        name = "pbc" if atoms.pbc.any() else "mol"

        calc = DeMonNano(directory=os.path.join(WORKDIR, f"ase-{name}"), kpts=kpts, **params)
        calc.calculate(atoms.copy(), properties=["energy"])

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=os.path.join(WORKDIR, f"direct-{name}"),
            **deepcopy(params),
        )
        if atoms.pbc.any():
            dem.calculate(
                symbols=list(atoms.symbols),
                positions=atoms.positions,
                cell=np.array(atoms.cell),
                kpts=kpts,
            )
        else:
            dem.calculate(symbols=list(atoms.symbols), positions=atoms.positions)

        energy_ha = dem.results["energy"]["energy"]
        assert calc.results["energy"] == pytest.approx(energy_ha * Hartree, rel=1e-10)

    def test_charges(self):
        """B5: Mulliken charges of the water dimer, also after get_forces()."""
        reference = [-0.556548, 0.286685, 0.269881, -0.528002, 0.218527, 0.309458]

        atoms = water_dimer.copy()
        atoms.calc = DeMonNano(directory=os.path.join(WORKDIR, "charges"), **parameters)
        charges = atoms.get_charges()

        np.testing.assert_allclose(charges, reference, rtol=1e-5)
        assert charges.sum() == pytest.approx(0.0, abs=1e-5)

        # Forces first (OPT SP, no charges in deMon.mol): ASE recomputes
        atoms = water_dimer.copy()
        atoms.calc = DeMonNano(directory=os.path.join(WORKDIR, "charges"), **parameters)
        atoms.get_forces()
        np.testing.assert_allclose(atoms.get_charges(), reference, rtol=1e-5)

    def test_scf_not_converged(self):
        """B10: a nearly dissociated water dimer (SCC not converged) raises SCFError."""
        atoms = water_dimer.copy()
        atoms.positions = [
            [0.554, -0.537, 3.578],
            [1.593, -0.864, 3.256],
            [-0.128, 0.224, 3.021],
            [3.642, -0.528, 3.194],
            [4.223, -1.023, 3.819],
            [5.303, 1.336, 2.867],
        ]
        atoms.calc = DeMonNano(directory=os.path.join(WORKDIR, "scf"), **parameters)

        with pytest.raises(SCFError):
            atoms.get_potential_energy()
