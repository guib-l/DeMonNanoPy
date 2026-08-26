import copy

import ase

# import configs
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
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "MAX": 1000,
                "TOL": 1e-7,
            },
            "WMULL": {"BONDPARAMS": {"C H": 0.0, "O H": 0.39, "O C": 0.0}},
        },
    },
}

pyrene = Atoms(
    [
        "O",
        "H",
        "H",
    ]
    + ["C"] * 16
    + ["H"] * 10,
    positions=np.array(
        [
            [-0.170316, -3.187170, -0.126262],
            [-0.897048, -2.620657, 0.162745],
            [0.454556, -2.630128, -0.607749],
            [0.175933, 0.335146, 1.793713],
            [1.360050, 0.325800, 1.110006],
            [-1.072593, 0.111218, 1.119476],
            [-2.302066, 0.111541, 1.806122],
            [1.397620, 0.091799, -0.306827],
            [2.606846, 0.072980, -1.028286],
            [0.172008, -0.133217, -1.000407],
            [0.181765, -0.372636, -2.406258],
            [-1.063252, -0.123546, -0.287152],
            [-2.288443, -0.353337, -0.979928],
            [-3.496740, -0.344828, -0.256738],
            [-3.495810, -0.114190, 1.119551],
            [1.412155, -0.383289, -3.091169],
            [2.606303, -0.162069, -2.403831],
            [-1.066353, -0.601267, -3.079682],
            [-2.250464, -0.592049, -2.395959],
            [-4.444118, -0.521654, -0.784767],
            [1.426319, -0.567696, -4.174401],
            [-1.053082, -0.786725, -4.163133],
            [3.557977, -0.173091, -2.952575],
            [-3.197581, -0.770057, -2.924850],
            [-4.446870, -0.110240, 1.669451],
            [0.163224, 0.514366, 2.878219],
            [2.307746, 0.497406, 1.639983],
            [3.554629, 0.245151, -0.499450],
            [-2.315834, 0.291268, 2.890140],
        ]
    ),
)

WORKDIR = ".run/water_pyren/"


class TestWaterPyrene:
    def test_pyrene_wmull(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99999,
                            "TOL": 3e-7,
                            "GRADTOL": 1e-5,
                            "OUT": 1,
                            "TRAJECTORY": False,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene.symbols[3:], positions=pyrene.positions[3:], read_charges=True)

        results = mod.results
        results["output_geometry"]

        assert np.allclose(results["energy"]["energy"], -31.34507905, atol=1e-7)

    def test_water_pyrene_wmull(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99999,
                            "TOL": 3e-7,
                            "GRADTOL": 1e-5,
                            "OUT": 1,
                            "TRAJECTORY": False,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene.symbols, positions=pyrene.positions, read_charges=True)

        results = mod.results
        results["output_geometry"]

        assert np.allclose(results["energy"]["energy"], -35.40543335, atol=1e-7)

    def test_water_pyrene_wmull_charged(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CHARGE": 1.0,
                "CI": {
                    "CONST": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 26],
                },
            }
        )
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99999,
                            "TOL": 3e-7,
                            "GRADTOL": 1e-5,
                            "OUT": 1,
                            "TRAJECTORY": False,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene.symbols, positions=pyrene.positions, read_charges=True)

        results = mod.results
        results["output_geometry"]

        assert np.allclose(results["energy"]["energy"], -35.12370722, atol=1e-7)

    def _test_water_pyrene_IP(self):

        copy_parameters = copy.deepcopy(parameters)
        positions_neutrals = np.array(
            [
                [-0.170454, -3.187228, -0.126485],
                [-0.896954, -2.620636, 0.162938],
                [0.454588, -2.630104, -0.607665],
                [0.175935, 0.335117, 1.793697],
                [1.360066, 0.325812, 1.109990],
                [-1.072604, 0.111210, 1.119472],
                [-2.302079, 0.111597, 1.806121],
                [1.397608, 0.091798, -0.306829],
                [2.606835, 0.073021, -1.028294],
                [0.172001, -0.133295, -1.000397],
                [0.181768, -0.372743, -2.406241],
                [-1.063259, -0.123595, -0.287154],
                [-2.288451, -0.353375, -0.979924],
                [-3.496750, -0.344867, -0.256718],
                [-3.495818, -0.114177, 1.119565],
                [1.412163, -0.383358, -3.091147],
                [2.606307, -0.162069, -2.403821],
                [-1.066350, -0.601295, -3.079696],
                [-2.250477, -0.592022, -2.395975],
                [-4.444108, -0.521741, -0.784775],
                [1.426343, -0.567703, -4.174394],
                [-1.053031, -0.786682, -4.163160],
                [3.558002, -0.173017, -2.952534],
                [-3.197623, -0.769897, -2.924881],
                [-4.446856, -0.110207, 1.669494],
                [0.163187, 0.514275, 2.878211],
                [2.307770, 0.497441, 1.639940],
                [3.554638, 0.245212, -0.499501],
                [-2.315834, 0.291355, 2.890147],
            ]
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(
            symbols=pyrene.symbols,
            positions=positions_neutrals,
        )

        results = mod.results
        neutral_energy = results["energy"]["energy"]

        positions_ionic = np.array(
            [
                [-0.202746, -2.943113, -0.185431],
                [-0.827824, -3.503765, 0.291176],
                [0.520803, -3.512324, -0.476420],
                [0.161124, 0.377830, 1.785722],
                [1.361068, 0.368478, 1.092865],
                [-1.072329, 0.162952, 1.110251],
                [-2.314990, 0.165242, 1.801605],
                [1.390142, 0.143757, -0.311588],
                [2.610050, 0.126823, -1.042152],
                [0.165541, -0.073498, -1.006896],
                [0.173704, -0.308757, -2.412206],
                [-1.064888, -0.063877, -0.296445],
                [-2.288774, -0.289501, -0.990355],
                [-3.508157, -0.278548, -0.258811],
                [-3.510530, -0.052381, 1.117327],
                [1.416892, -0.317062, -3.102551],
                [2.612504, -0.100209, -2.418147],
                [-1.059035, -0.531810, -3.086317],
                [-2.258989, -0.522408, -2.393457],
                [-4.454868, -0.451234, -0.789604],
                [1.428427, -0.497280, -4.186616],
                [-1.051472, -0.715523, -4.169556],
                [3.562854, -0.109018, -2.966584],
                [-3.203028, -0.698668, -2.927244],
                [-4.460639, -0.046349, 1.666214],
                [0.154551, 0.550312, 2.870805],
                [2.306089, 0.533560, 1.628482],
                [3.557178, 0.294672, -0.510570],
                [-2.326095, 0.340527, 2.886484],
            ]
        )
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CHARGE": 1.0,
                "CI": {
                    "CONST": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [3, 26],
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(
            symbols=pyrene.symbols,
            positions=positions_ionic,
        )

        results = mod.results
        ionic_energy = results["energy"]["energy"]

        ips = (ionic_energy - neutral_energy) * ase.units.Hartree
        assert np.allclose(ips, 7.63, 5e-2)
