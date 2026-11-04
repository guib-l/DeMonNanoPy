import os

import copy
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
            "DFTB": {"SCC": True, "DISP":2},
            "CHARGE": 0.0,
        },
    },
}
image = Atoms(
    ["O", "H", "H", "O", "H", "H"],
    positions=np.array(
        [
            [1.2478, -0.5185, 3.4049],
            [1.5946, -1.4204, 3.3886],
            [0.9008, -0.3341, 2.5062],
            [3.2478, -0.4185, 3.4049],
            [3.5946, -1.5204, 3.3886],
            [2.9008, -0.3341, 2.6062],
        ]
    ),
)



def exemple_run_ase():

    WORKDIR = ".run/example-ase"

    calc = DeMonNano(
        omp_threads=1,
        title="CALCULATION DEMONANO",
        directory=WORKDIR,
        **parameters,
    )

    calc.calculate(atoms=image, properties=["energy", "forces"])

    print(" RESULTS : ")
    print(calc.results)
    print()


def exemple_run_bfgs():

    WORKDIR = ".run/example-bfgs"

    from ase.optimize import BFGS

    calc = DeMonNano(
        omp_threads=1,
        title="CALCULATION DEMONANO",
        directory=WORKDIR,
        **parameters,
    )
    image.calc = calc
    opt = BFGS(image, trajectory=os.path.join(WORKDIR, "H2O.traj"))
    opt.run(fmax=0.01)


def exemple_run_dynamics():
    """Langevin dynamics (NVT, 300 K) of the water dimer with ASE."""

    WORKDIR = ".run/example-dynamics"

    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import BFGS

    atoms = image.copy()
    atoms.calc = DeMonNano(
        omp_threads=1,
        title="CALCULATION DEMONANO",
        directory=WORKDIR,
        **parameters,
    )
    BFGS(atoms).run(fmax=0.01)  # start from the relaxed dimer
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    dyn = Langevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300,
        friction=0.01 / units.fs,
        trajectory=os.path.join(WORKDIR, "md.traj"),
    )

    time, epot, ekin = [], [], []

    def record():
        time.append(dyn.get_time() / units.fs)
        epot.append(atoms.get_potential_energy())
        ekin.append(atoms.get_kinetic_energy())

    dyn.attach(record, interval=1)
    dyn.run(200)

    epot, ekin = np.array(epot), np.array(ekin)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(time, epot - epot[0], label=r"$E_{pot}$")
    ax.plot(time, ekin, label=r"$E_{kin}$")
    ax.plot(time, epot + ekin - epot[0], label=r"$E_{tot}$")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(r"$E - E_{pot}(t=0)$ (eV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("./water-dynamics.png", dpi=150)


def exemple_run_neb():
    """NEB: exchange of the H-bond donor/acceptor roles in the water dimer.

    The final state is the relaxed initial state mirrored through the plane
    perpendicular to O-O (same energy), with the two molecules relabelled
    so that each O atom stays in place.
    """

    WORKDIR = ".run/example-neb"

    from ase.mep import NEB
    from ase.mep.neb import NEBTools
    from ase.optimize import BFGS, FIRE

    def new_calc(name):
        return DeMonNano(
            omp_threads=1,
            title="CALCULATION DEMONANO",
            directory=os.path.join(WORKDIR, name),
            **parameters,
        )

    # Initial state: relaxed water dimer
    initial = image.copy()
    initial.calc = new_calc("initial")
    BFGS(initial).run(fmax=0.01)

    # Final state: mirror + relabelling (O0 <-> O3, H1,H2 <-> H4,H5)
    pos = initial.positions
    center = 0.5 * (pos[0] + pos[3])
    normal = (pos[3] - pos[0]) / np.linalg.norm(pos[3] - pos[0])
    mirror = pos - 2.0 * np.outer((pos - center) @ normal, normal)
    final = initial.copy()
    final.positions = mirror[[3, 4, 5, 0, 1, 2]]
    final.calc = new_calc("final")

    images = [initial]
    for i in range(5):
        img = initial.copy()
        img.calc = new_calc(f"image-{i}")
        images.append(img)
    images.append(final)

    neb = NEB(images, climb=True)
    neb.interpolate(method="idpp")
    FIRE(neb, trajectory=os.path.join(WORKDIR, "neb.traj")).run(fmax=0.05, steps=200)

    nebtools = NEBTools(images)
    barrier, delta_e = nebtools.get_barrier()
    print(f" > Barrier : {barrier:.4f} eV, Delta E : {delta_e:.4f} eV")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    nebtools.plot_band(ax=ax)
    fig.tight_layout()
    fig.savefig("./water-neb.png", dpi=150)


def exemple_run_bassinHopping():
    """Basin-Hopping search on the water dimer (BFGS local optimisations)."""

    WORKDIR = ".run/example-bh"

    from ase.optimize import BFGS
    from ase.optimize.basin import BasinHopping

    os.makedirs(WORKDIR, exist_ok=True)
    logfile = os.path.join(WORKDIR, "bh.log")

    # The calculator gets its own sub-directory: deMonNano deletes the
    # "*.log" files of its working directory at each calculation.
    atoms = image.copy()
    atoms.calc = DeMonNano(
        omp_threads=1,
        title="CALCULATION DEMONANO",
        directory=os.path.join(WORKDIR, "calc"),
        **parameters,
    )

    bh = BasinHopping(
        atoms,
        temperature=0.1,  # kT in eV
        dr=0.1,
        optimizer=BFGS,
        fmax=0.05,
        logfile=logfile,
        trajectory=os.path.join(WORKDIR, "lowest.traj"),
        optimizer_logfile=os.path.join(WORKDIR, "opt.log"),
        local_minima_trajectory=os.path.join(WORKDIR, "local_minima.traj"),
    )
    bh.run(20)

    # Lines: "BasinHopping: step N, energy E, emin Emin"
    energy, emin = [], []
    with open(logfile) as f:
        for line in f:
            words = line.replace(",", " ").split()
            energy.append(float(words[4]))
            emin.append(float(words[6]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(energy, "o-", label="Local minimum")
    ax.plot(emin, "k--", label="Lowest minimum")
    ax.set_xlabel("Basin-Hopping step")
    ax.set_ylabel("Energy (eV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("./water-bh.png", dpi=150)



if __name__ == "__main__":
    exemple_run_ase()

    exemple_run_bfgs()

    exemple_run_dynamics()

    exemple_run_neb()

    exemple_run_bassinHopping()
