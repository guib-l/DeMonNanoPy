import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.ase_calculator import DeMonNano

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True, "DISP": 2, "FERMI": 50},
            "PBC": {
                "TYPE": "GENERAL",
                "LATTICE": np.array(
                    [[2.46, 0.0, 0.0], [-1.230000, 2.130422, 0], [0.0, 0.0, 100.0]]
                ),
                "KPTS": [20, 20, 1],
            },
        },
    },
}

# Graphene (model system)
graphene = Atoms(
    ["C"] * 2,
    positions=[
        [0.000000, 0.000000, 0.000000], 
        [1.230000, 0.710000, 0.000000]
    ],
    cell=[
        [2.460000, 0.000000, 0.000000], 
        [-1.230000, 2.130422, 0.000000], 
        [0.0, 0.0, 100.0]
    ],
    pbc=True,
)

# Phonons: supercell (5, 5, 1) and k-points reduced accordingly (20 / 5 = 4)
PH_SUPERCELL = (5, 5, 1)
PH_KPTS = (4, 4, 1)


def _save(fig, filename):
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Figure : {filename}")


def exemple_run_bs():
    """Electronic band structure along G-M-K-G.

    deMonNano only computes a uniform N x N x 1 grid. With N a multiple of 6,
    the path G-M-K-G goes exactly through grid points: their eigenvalues are
    used directly (no interpolation, no non-SCF calculation).
    """
    from ase.dft.kpoints import BandPath
    from ase.spectrum.band_structure import BandStructure

    print("=" * 40)
    print(" Electronic band structure : graphene")

    N = 60
    atoms = graphene.copy()
    atoms.calc = DeMonNano(
        directory=".run/example-bs", kpts=(N, N, 1), **copy.deepcopy(parameters)
    )
    atoms.get_potential_energy()
    calc = atoms.calc
    kpts = calc.get_ibz_k_points()
    eigenvalues = calc.results["eigenvalues"]  # (nspins, nkpts, nbands), eV

    # Path in fractional coordinates: G (0,0) -> M (1/2,0) -> K (1/3,1/3) -> G
    n = N // 6
    path_kpts = np.array(
        [[i / N, 0.0, 0.0] for i in range(3 * n + 1)]  # G -> M
        + [[0.5 - j / N, 2 * j / N, 0.0] for j in range(1, n + 1)]  # M -> K
        + [[t / N, t / N, 0.0] for t in range(2 * n - 1, -1, -1)]  # K -> G
    )
    index = [np.where(np.abs(kpts - k).max(axis=1) < 1e-4)[0][0] for k in path_kpts]

    path = BandPath(
        cell=atoms.cell,
        kpts=path_kpts,
        special_points=atoms.cell.bandpath().special_points,
        path="GMKG",
    )
    bs = BandStructure(path, eigenvalues[:, index, :], reference=calc.get_fermi_level())
    bs = bs.subtract_reference()  # energies relative to the Fermi level

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bs.plot(ax=ax, emin=-20.0, emax=10.0)
    ax.set_ylabel(r"$E - E_F$ (eV)")
    _save(fig, "./graphene-bs.png")


def example_relax():
    """Relax graphene: lattice parameter from a E(a) scan, then positions with BFGS.

    deMonNano gives no stress, so the cell is relaxed by scanning the in-plane
    lattice parameter (a = b, gamma = 120 deg, vacuum along z kept fixed).
    """
    from ase.optimize import BFGS

    print("=" * 40)
    print(" Relaxation cell/positions : graphene")

    atoms = graphene.copy()
    atoms.calc = DeMonNano(
        directory=".run/example-relax", kpts=(33, 33, 1), **copy.deepcopy(parameters)
    )

    # --- Cell: E(a) scan + polynomial fit (degree 3) ---
    cell0 = np.array(atoms.cell)
    a0 = atoms.cell.lengths()[0]
    lattice = np.arange(2.40, 2.52 + 1e-9, 0.01)
    energies = []
    for a in lattice:
        cell = cell0.copy()
        cell[:2] *= a / a0
        atoms.set_cell(cell, scale_atoms=True)
        energies.append(atoms.get_potential_energy())

    fit = np.polyfit(lattice, energies, 3)
    roots = np.roots(np.polyder(fit))
    roots = roots[np.isreal(roots)].real
    a_opt = roots[np.polyval(np.polyder(fit, 2), roots) > 0][0]
    print(f" > Optimal lattice parameter : {a_opt:.4f} Ang")

    cell = cell0.copy()
    cell[:2] *= a_opt / a0
    atoms.set_cell(cell, scale_atoms=True)

    # --- Positions: BFGS ---
    BFGS(atoms, trajectory=".run/relax.traj").run(fmax=0.001)
    print(f" > Energy : {atoms.get_potential_energy():.6f} eV")
    print(f" > Distance C-C : {atoms.get_distance(0, 1, mic=True):.4f} Ang")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    grid = np.linspace(lattice[0], lattice[-1], 200)
    ax.plot(lattice, energies, "s", color="black", label="deMonNano")
    ax.plot(grid, np.polyval(fit, grid), color="grey", label="fit (deg. 3)")
    ax.axvline(a_opt, color="grey", ls="--")
    ax.set_xlabel(r"Lattice parameter a ($\AA$)")
    ax.set_ylabel("Energy (eV)")
    ax.legend()
    _save(fig, "./graphene-relax.png")

    relaxed = atoms.copy()
    relaxed.calc = None
    return relaxed


def example_dos():
    """Electronic density of states with ase.dft.dos.DOS."""
    from ase.dft.dos import DOS

    print("=" * 40)
    print(" Electronic DOS : graphene")

    atoms = graphene.copy()
    atoms.calc = DeMonNano(
        directory=".run/example-dos", kpts=(21, 21, 1), **copy.deepcopy(parameters)
    )
    atoms.get_potential_energy()

    dos = DOS(atoms.calc, width=0.5)
    energies = dos.get_energies()  # relative to the Fermi level
    weights = dos.get_dos()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(energies, weights, color="black")
    ax.set_xlim(min(energies), 13.0)
    ax.axvline(0.0, color="grey", ls="--")
    ax.set_xlabel(r"$E - E_F$ (eV)")
    ax.set_ylabel("DOS (states/eV)")
    _save(fig, "./graphene-dos.png")


def _phonons():
    """Run (or reload from cache) the finite-displacement phonon calculation."""
    from ase.phonons import Phonons

    atoms = graphene.copy()
    calc = DeMonNano(directory=".run/example-phonon", kpts=PH_KPTS, **copy.deepcopy(parameters))

    # An interrupted run leaves empty cache files that ASE treats as "in
    # progress": they must be deleted, otherwise the displacement is skipped.
    for filename in glob.glob(".run/phonon-relaxed/cache.*.json"):
        if os.path.getsize(filename) == 0:
            os.remove(filename)

    ph = Phonons(atoms, calc, supercell=PH_SUPERCELL, delta=0.05, name=".run/phonon-relaxed")
    ph.run()
    ph.read(acoustic=True)
    return atoms, ph


def example_phonon():
    """Phonon calculation: frequencies at Gamma."""

    print("=" * 40)
    print(" Phonon calculation : graphene")

    _, ph = _phonons()
    omega = ph.band_structure([[0.0, 0.0, 0.0]])[0]  # eV
    for i, w in enumerate(omega):
        print(f" > mode {i} : {w * 1000:10.3f} meV")


def example_phonon_bs():
    """Phonon band structure along G-M-K-G."""

    print("=" * 40)
    print(" Phonon band structure : graphene")

    atoms, ph = _phonons()
    path = atoms.cell.bandpath("GMKG", npoints=100)
    bs = ph.get_band_structure(path)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bs.plot(ax=ax, emin=0.0, emax=0.25)
    ax.set_ylabel("Energy (eV)")
    _save(fig, "./graphene-phonon-bs.png")


def example_phonon_dos():
    """Phonon density of states."""

    print("=" * 40)
    print(" Phonon DOS : graphene")

    _, ph = _phonons()
    dos = ph.get_dos(kpts=(20, 20, 1)).sample_grid(npts=200, width=2e-3)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.fill_between(dos.get_energies(), dos.get_weights(), color="grey", edgecolor="k", lw=1)
    ax.set_xlim(0.0, 0.25)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Phonon DOS")
    _save(fig, "./graphene-phonon-dos.png")


if __name__ == "__main__":

    # Band structure, DOS and phonons below use the relaxed structure
    graphene = example_relax()

    exemple_run_bs()

    example_dos()

    example_phonon()

    example_phonon_bs()

    example_phonon_dos()
