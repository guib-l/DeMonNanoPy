import copy
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
        },
    },
}

WORKDIR = ".run/examples-rttddftb"


def _save(fig, filename):
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"Figure : {filename}")


def read_rttddftb_charges(filename, natoms=10, nstep=1000):
    icount = 0
    charges = np.zeros((nstep, natoms + 1))

    with open(filename, "r") as fd:
        for line in fd.readlines():
            _line = line.split()

            if "CHARGE" in _line and "CUT_SYS" in _line:
                charges[icount] = list(map(float, _line[2:]))
                icount += 1
    return charges


def example_rttddfttb_glycine_N():

    table = []
    with open("test/data_test/qmmm/mb4") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)
    _images = Atoms(
        [
            "O",
            "O",
            "C",
            "N",
            "H",
            "H",
            "C",
            "H",
            "H",
            "H",
        ],
        positions=table[:10, :3],
    )

    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {
                        "SCC": True,
                    },
                    "RTTDDFTB": {
                        "COLL": 10,
                        "PROJPT": True,
                        "QPROJ": 1.0,
                        "FCOLL": 4.0,
                        "MPROJ": 1.0,
                        "PDIST": 7.0,
                        "POS-PROJ": [-0.897913, -0.137568, -0.418123],
                        "B-PARAM": [0.0, 0.0, 0.0],
                    },
                    "CUTSYS": {
                        "FRAGMENT": [1] * 10,
                    },
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:10, 3],
                        "TYPEMM": [5, 63, 3, 1, 6, 6, 2, 4, 4, 64],
                        "QM": "1-10",
                        "MM": "",
                        "FORCEFIELD": {"FF": "AMBER-FF99SB"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "MD": {
                        "MDSTEP": {
                            "MAX": 2000,
                            "OUT": 1,
                        },
                        "TIMESTEP": 0.001,
                        "TRAJECTORY": True,
                    },
                },
            }
        }
    )

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
    shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=_images.symbols, positions=_images.positions)

    _charges = read_rttddftb_charges(os.path.join(WORKDIR, "deMon.out"), 10, 2000)
    return _charges


def example_rttddfttb_glycine_CC():

    table = []
    with open("test/data_test/qmmm/mb4") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)
    _images = Atoms(
        [
            "O",
            "O",
            "C",
            "N",
            "H",
            "H",
            "C",
            "H",
            "H",
            "H",
        ],
        positions=table[:10, :3],
    )

    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {
                        "SCC": True,
                    },
                    "RTTDDFTB": {
                        "COLL": 10,
                        "PROJPT": True,
                        "QPROJ": 1.0,
                        "FCOLL": 0.0,
                        "MPROJ": 1.0,
                        "PDIST": 7.0,
                        "POS-PROJ": [-0.897913, -0.137568, -0.418123],
                        "B-PARAM": [-0.24996835, 0.2240185, 0.1653195],
                    },
                    "CUTSYS": {
                        "FRAGMENT": [1] * 10,
                    },
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:10, 3],
                        "TYPEMM": [5, 63, 3, 1, 6, 6, 2, 4, 4, 64],
                        "QM": "1-10",
                        "MM": "",
                        "FORCEFIELD": {"FF": "AMBER-FF99SB"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "MD": {
                        "MDSTEP": {
                            "MAX": 2000,
                            "OUT": 1,
                        },
                        "TIMESTEP": 0.001,
                        "TRAJECTORY": True,
                    },
                },
            }
        }
    )

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
    shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=_images.symbols, positions=_images.positions)

    _charges = read_rttddftb_charges(os.path.join(WORKDIR, "deMon.out"), 10, 2000)
    return _charges


def example_rttddfttb_glycine_O():

    table = []
    with open("test/data_test/qmmm/mb4") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)
    _images = Atoms(
        [
            "O",
            "O",
            "C",
            "N",
            "H",
            "H",
            "C",
            "H",
            "H",
            "H",
        ],
        positions=table[:10, :3],
    )
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {
                        "SCC": True,
                    },
                    "RTTDDFTB": {
                        "COLL": 10,
                        "PROJPT": True,
                        "QPROJ": 1.0,
                        "FCOLL": 2,
                        "MPROJ": 1.0,
                        "PDIST": 7.0,
                        "POS-PROJ": [-0.753375, -0.0482745, -0.655817],
                        "B-PARAM": [0.0, 0.0, 0.0],
                    },
                    "CUTSYS": {
                        "FRAGMENT": [1] * 10,
                    },
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:10, 3],
                        "TYPEMM": [5, 63, 3, 1, 6, 6, 2, 4, 4, 64],
                        "QM": "1-10",
                        "MM": "",
                        "FORCEFIELD": {"FF": "AMBER-FF99SB"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "MD": {
                        "MDSTEP": {
                            "MAX": 2000,
                            "OUT": 1,
                        },
                        "TIMESTEP": 0.001,
                        "TRAJECTORY": True,
                    },
                },
            }
        }
    )

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
    shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=_images.symbols, positions=_images.positions)

    _charges = read_rttddftb_charges(os.path.join(WORKDIR, "deMon.out"), 10, 2000)
    return _charges


if __name__ == "__main__":
    charge_N = example_rttddfttb_glycine_N()

    charge_O = example_rttddfttb_glycine_O()

    charge_CC = example_rttddfttb_glycine_CC()

    fig = plt.figure(
        figsize=(6.5, 6.5),
    )
    gs = fig.add_gridspec(3, hspace=0)
    ax = gs.subplots(sharex="all")

    for i in range(1, 11):
        ax[1].plot(charge_N[:, 0], charge_N[:, i])
        ax[0].plot(charge_CC[:, 0], charge_CC[:, i])
        ax[2].plot(charge_O[:, 0], charge_O[:, i])

    plt.xlabel("Time (fs)")
    ax[1].set_ylabel("Charge variation")

    _save(fig, "./charge_variation.png")
