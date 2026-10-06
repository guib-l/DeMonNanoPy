import copy
import os
import shutil

import numpy as np
import pytest
from ase import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "MAT", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
            }
        },
    },
}


WORKDIR = ".run/rtdftb/"


positions = np.array(
    [
        [-1.73285200000000, -3.36104900000000, 2.96902200000000],
        [-1.69544600000000, 0.288990000000000, 0.277373000000000],
        [-2.23577300000000, -1.30990800000000, -1.83152900000000],
        [-2.36313200000000, -0.184144000000000, -4.18288200000000],
        [0.455692000000000, -2.59211500000000, 1.77269800000000],
        [0.211189000000000, -1.57350600000000, -0.717333000000000],
        [2.38612300000000, -0.613224000000000, -1.99981700000000],
        [-1.39774800000000, -0.782792000000000, 2.73912400000000],
        [-0.583086000000000, 4.34215500000000, 1.98005900000000],
        [0.628119000000000, 1.21781700000000, -1.07590200000000],
        [2.204800000000000e-002, -0.407691000000000, -3.14680300000000],
        [-1.01173700000000, 1.80323000000000, 2.40707500000000],
        [-2.02400400000000, -2.37662200000000, 0.569303000000000],
        [1.24652000000000, 2.70594400000000, 1.09244500000000],
        [0.856043000000000, 6.687200000000000e-002, 1.51589200000000],
        [-1.23263500000000, 2.92205600000000, -0.111302000000000],
        [-1.83022900000000, 1.40252700000000, -2.18021800000000],
        [4.67892100000000, -0.796699000000000, -0.766229000000000],
        [3.01297600000000, 0.980847000000000, 0.170411000000000],
        [2.60817200000000, -1.73131100000000, 0.518719000000000],
    ]
)
symbols = ["Au"] * 20


base = Atoms(
    ["O", "O", "C", "N", "H", "H", "C", "H", "H", "H"],
    positions=np.array(
        [
            [2.082633, -2.033395, -0.701809],
            [1.904313, -2.187495, 1.519431],
            [1.720950, -1.513026, 0.440466],
            [1.101965, 0.493364, -0.862487],
            [0.062217, -0.235367, 0.848594],
            [1.629754, 0.475622, 1.230297],
            [1.094008, -0.130999, 0.482069],
            [1.940946, -1.432516, -1.475023],
            [1.882500, 1.148872, -0.953228],
            [0.236814, 1.022592, -1.005798],
        ]
    ),
    charges=np.array(
        [
            -0.382294,
            -0.868006,
            0.754458,
            -0.522439,
            0.149845,
            0.095409,
            -0.096701,
            0.399571,
            0.229906,
            0.240251,
        ]
    ),
)


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


def read_dipol_from_binary(filname, n_step=100):
    from scipy.io import FortranFile

    assert os.path.isfile(filname)

    f = FortranFile(filname, "r")
    step, dipole = np.zeros(n_step), np.zeros((n_step, 3))

    for i in range(n_step):
        record = f.read_reals(dtype="float64")
        step[i] = record[0]
        dipole[i] = record[1:]
    f.close()

    return dipole


class TestRTDFTB:
    def _test_tddftb(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["BASIS"] = {"PTYPE": "", "SKFILE": ""}
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"DFTB": {"SCC": True}, "TD-DFTB": {"LRESP": 25, "NO_TRIP": True}}
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        shutil.copy2(
            "test/basis-test/Au-modified/Au-Au_modified.skf",
            f"{WORKDIR}/Au-Au_modified.skf",
        )
        shutil.copy2("test/basis-test/Au-modified/SCC-SLAKO", f"{WORKDIR}/SCC-SLAKO")

        mod.calculate(symbols=symbols, positions=positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -57.09364137, atol=1e-7)

    @pytest.mark.dynamics
    def test_basic_rtdftb(self):

        n_step = 200

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["BASIS"] = {"PTYPE": "", "SKFILE": ""}
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"DFTB": {"SCC": True}, "RTTDDFTB": {"KICK": 0.003, "KICKAXIS": 1}}
        )
        copy_parameters["DEMON_MODULE"] = {"ACTIVE": {}}
        copy_parameters["DEMON_MODULE"]["ACTIVE"].update(
            {
                "MD": {
                    "TIMESTEP": 0.05,
                    "MDSTEP": {"MAX": n_step, "OUT": 100, "SOUT": 100},
                    "TRAJECTORY": True,
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2(
            "test/basis-test/Au-modified/Au-Au_modified.skf",
            f"{WORKDIR}/Au-Au_modified.skf",
        )
        shutil.copy2("test/basis-test/Au-modified/SCC-SLAKO", f"{WORKDIR}/SCC-SLAKO")

        mod.calculate(symbols=symbols, positions=positions)

        results = mod.results

        new_dipole = read_dipol_from_binary(f"{WORKDIR}/deMon.dip.bin.1", n_step)
        ref_dipole = read_dipol_from_binary("test/data_test/deMon.dip.bin.1", 2000)

        step = min(n_step, 2000)
        assert np.allclose(ref_dipole[:step], new_dipole, atol=1e-4)
        assert np.allclose(results["energy"]["energy"], -57.09364137, atol=1e-7)

    def test_qmmm_rtdtdftb(self):

        add_water = Atoms(
            ["O", "H", "H"],
            positions=np.array(
                [
                    [26.250000, 30.030001, 30.610000],
                    [26.370000, 29.080000, 30.710000],
                    [26.570000, 30.390000, 31.440001],
                ]
            ),
        )
        _images = base + add_water

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
                            "FCOLL": 1.0,
                            "MPROJ": 1.0,
                            "PDIST": 7.6,
                            "POS-PROJ": [-0.897913, -0.137568, -0.418123],
                            "B-PARAM": [0.0, 0.0, 0.0],
                        },
                        "CUTSYS": {
                            "FRAGMENT": [1] * 10,
                        },
                        "QMMM": {
                            "COUPLING": "ELECTROSTATIC",
                            "CHR": "FF",
                            "CHARGES": None,
                            "TYPEMM": [5, 63, 3, 1, 6, 6, 2, 4, 4, 64, 2001, 2002, 2002],
                            "QM": "1-10",
                            "MM": "11-13",
                            "FORCEFIELD": {"FF": "AMBER-FF99SB"},
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")
        mod.calculate(symbols=_images.symbols, positions=_images.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -14.31818879, atol=1e-7)

    @pytest.mark.dynamics
    def test_qmmm_rtdtdftb_md(self):

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
                "O",
                "H",
                "H",
            ],
            positions=table[:, :3],
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
                            "CHR": "INPUT",
                            "CHARGES": None,
                            "TYPEMM": [5, 63, 3, 1, 6, 6, 2, 4, 4, 64, 2001, 2002, 2002],
                            "QM": "1-10",
                            "MM": "11-13",
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

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")

        mod.calculate(symbols=_images.symbols, positions=_images.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -14.33112067, atol=1e-7)

        new_charges = read_rttddftb_charges(os.path.join(WORKDIR, "deMon.out"), 10, 2000)
        ref_charges = read_rttddftb_charges("test/data_test/charges.txt-RTTDDFTB", 10, 2000)

        assert np.allclose(new_charges, ref_charges, atol=1e-2)
