import copy
import shutil
from pathlib import Path

import ase
import numpy as np

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "3OB", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
        },
    },
}

WORKDIR = ".run/examples-qmmm"


# =============================================================================
# NH3 -------------------------------------------------------------------------
# =============================================================================


def example_qmmm_amoniac(parameters_3ob):

    if not Path(parameters_3ob).is_dir():
        raise FileExistsError(f"Folder {parameters_3ob} didn't exists.")

    DEMON_BASIS = parameters_3ob
    table = []
    with open("test/data_test/qmmm/mb7") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)
    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:, 3],
                        "TYPEMM": [78, 79, 79, 79] + [63, 64, 64] * 215,
                        "QM": "1-4",
                        "MM": "5-649",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = [
        "N",
        "H",
        "H",
        "H",
    ] + ([["O", "H", "H"] * 215])[0]

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:, :3])

    results = mod.results
    total_energy = results["energy"]["energy"]
    opt_pos = results["output_geometry"].positions
    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:4, 3],
                        "TYPEMM": [78, 79, 79, 79],
                        "QM": "1-4",
                        "MM": "",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = [
        "N",
        "H",
        "H",
        "H",
    ]
    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:4, :3])
    results = mod.results
    amoniac_energy = results["energy"]["energy"]

    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[4:, 3],
                        "TYPEMM": [63, 64, 64] * 215,
                        "QM": "",
                        "MM": "1-645",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    symbols = ([["O", "H", "H"] * 215])[0]
    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=opt_pos[4:])
    results = mod.results
    water_energy = results["energy"]["qmmm-energy"]
    value = total_energy - amoniac_energy - water_energy

    print(f" > Solvation energy (NH3) : {value * ase.units.Hartree}")


# =============================================================================
# C2H5NO2 ----------------------------------------------------------------------
# =============================================================================


def example_qmmm_glycine(parameters_3ob):

    if not Path(parameters_3ob).is_dir():
        raise FileExistsError(f"Folder {parameters_3ob} didn't exists.")

    DEMON_BASIS = parameters_3ob
    table = []
    with open("test/data_test/qmmm/mb6") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)
    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:, 3],
                        "TYPEMM": [784, 784, 782, 783, 780, 780, 781, 64, 785, 785]
                        + [63, 64, 64] * 210,
                        "QM": "1-10",
                        "MM": "11-640",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = [
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
    ] + ([["O", "H", "H"] * 210])[0]

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:, :3])

    results = mod.results
    total_energy = results["energy"]["energy"]
    opt_pos = results["output_geometry"].positions
    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:10, 3],
                        "TYPEMM": [784, 784, 782, 783, 780, 780, 781, 64, 785, 785],
                        "QM": "1-10",
                        "MM": "",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = [
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
    ]
    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:10, :3])
    results = mod.results
    molecule_energy = results["energy"]["energy"]

    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[10:, 3],
                        "TYPEMM": [63, 64, 64] * 210,
                        "QM": "",
                        "MM": "1-630",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    symbols = ([["O", "H", "H"] * 210])[0]
    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=opt_pos[10:])
    results = mod.results
    water_energy = results["energy"]["qmmm-energy"]
    value = total_energy - molecule_energy - water_energy

    print(f" > Solvation energy (C2H5NO2) : {value * ase.units.Hartree}")


# =============================================================================
# C5H12 -----------------------------------------------------------------------
# =============================================================================


def example_qmmm_pentane(parameters_3ob):

    if not Path(parameters_3ob).is_dir():
        raise FileExistsError(f"Folder {parameters_3ob} didn't exists.")

    DEMON_BASIS = parameters_3ob
    table = []
    with open("test/data_test/qmmm/mb5") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)
    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:, 3],
                        "TYPEMM": [
                            80,
                            81,
                            85,
                            85,
                            85,
                            85,
                            81,
                            85,
                            85,
                            85,
                            81,
                            85,
                            80,
                            85,
                            85,
                            85,
                            85,
                        ]
                        + [63, 64, 64] * 205,
                        "QM": "1-17",
                        "MM": "18-632",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = [
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "C",
        "H",
        "H",
        "H",
        "C",
        "H",
        "C",
        "H",
        "H",
        "H",
        "H",
    ] + ([["O", "H", "H"] * 205])[0]

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:, :3])

    results = mod.results
    total_energy = results["energy"]["energy"]
    opt_pos = results["output_geometry"].positions
    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:17, 3],
                        "TYPEMM": [
                            80,
                            81,
                            85,
                            85,
                            85,
                            85,
                            81,
                            85,
                            85,
                            85,
                            81,
                            85,
                            80,
                            85,
                            85,
                            85,
                            85,
                        ],
                        "QM": "1-17",
                        "MM": "",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = ["C", "C", "H", "H", "H", "H", "C", "H", "H", "H", "C", "H", "C", "H", "H", "H", "H"]
    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:17, :3])
    results = mod.results
    molecule_energy = results["energy"]["energy"]

    # ---------------------------------------------------------------
    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[17:, 3],
                        "TYPEMM": [63, 64, 64] * 205,
                        "QM": "",
                        "MM": "1-615",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    symbols = ([["O", "H", "H"] * 205])[0]
    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=opt_pos[17:])
    results = mod.results
    water_energy = results["energy"]["qmmm-energy"]
    value = total_energy - molecule_energy - water_energy

    print(f" > Solvation energy (C5H12) : {value * ase.units.Hartree}")


# =============================================================================
# H2O -------------------------------------------------------------------------
# =============================================================================


def example_qmmm_water(parameters_3ob):

    if not Path(parameters_3ob).is_dir():
        raise FileExistsError(f"Folder {parameters_3ob} didn't exists.")

    DEMON_BASIS = parameters_3ob
    table = []
    with open("test/data_test/qmmm/mb3") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)

    # ---------------------------------------------------------------

    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4.0},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "FF",
                        "CHARGES": table[:, -1],
                        "TYPEMM": {"O": 63, "H": 64},
                        "QM": "1-3",
                        "MM": "4-639",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )

    copy_parameters.update(
        {
            "DEMON_MODULE": {
                "ACTIVE": {
                    "OPT": {"MAX": 5000, "TRAJECTORY": True},
                },
            }
        }
    )
    symbols = [
        "O",
        "H",
        "H",
    ] + ([["O", "H", "H"] * 212])[0]

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:, :3])

    results = mod.results
    total_energy = results["energy"]["energy"]
    opt_pos = results["output_geometry"].positions

    # ---------------------------------------------------------------
    table = []
    with open("test/data_test/qmmm/mb2") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)

    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4.0},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "INPUT",
                        "CHARGES": table[:, -1],
                        "TYPEMM": {"O": 63, "H": 64},
                        "QM": "1-3",
                        "MM": "",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    symbols = ["O", "H", "H"]

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=table[:, :3])
    results = mod.results
    water_energy = results["energy"]["energy"]

    # ---------------------------------------------------------------
    table = []
    with open("test/data_test/qmmm/mb1") as fd:
        for line in fd.readlines():
            table.append(list(map(float, line.split())))
    table = np.array(table)

    copy_parameters = copy.deepcopy(parameters)
    copy_parameters.update(
        {
            "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
            "DEMON_PARAMETERS": {
                "ACTIVE": {
                    "DFTB": {"SCC": True, "THIRD": True, "GCOR": 4.0},
                    "QMMM": {
                        "COUPLING": "ELECTROSTATIC",
                        "CHR": "INPUT",
                        "CHARGES": table[:, -1],
                        "TYPEMM": {"O": 63, "H": 64},
                        "QM": "",
                        "MM": "1-636",
                        "FORCEFIELD": {"FF": "OPLS-AA"},
                    },
                },
            },
        }
    )
    symbols = ([["O", "H", "H"] * 212])[0]

    mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

    shutil.copy2("example/data_test/3ord_param", f"{WORKDIR}/3ord_param")
    shutil.copy2("example/data_test/FFDS-OPLS", f"{WORKDIR}/FFDS")

    mod.calculate(symbols=symbols, positions=opt_pos[3:])
    results = mod.results
    qmmm_water_energy = results["energy"]["qmmm-energy"]

    value = total_energy - water_energy - qmmm_water_energy

    print(f" > Solvation energy (H2O) : {value * ase.units.Hartree}")


if __name__ == "__main__":
    import sys

    try:
        parameters_3ob = str(sys.argv[1]) 
    except:
        txt = "Please, run `python3 example/example_qmmm.py <3OB-basis directory>`."
        raise FileNotFoundError(f"Parameters not found.\n {txt}")
        

    example_qmmm_amoniac(parameters_3ob)
    example_qmmm_water(parameters_3ob)
    example_qmmm_glycine(parameters_3ob)
    example_qmmm_pentane(parameters_3ob)
