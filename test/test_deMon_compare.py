import copy

import numpy as np
import pytest
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file("global.json")

deMonPy.DEMON_BASIS = "../../test/basis-test"


parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
        },
    },
}

water = Atoms(
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

pyrene = Atoms(
    ["C"] * 16 + ["H"] * 10,
    positions=np.array(
        [
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


pyrene_water = Atoms(
    [
        "O",
        "H",
        "H",
    ]
    + ["C"] * 16
    + ["H"] * 10,
    positions=np.array(
        [
            [-0.170316, -2.187170, -0.126262],
            [-0.897048, -1.620657, 0.162745],
            [0.454556, -1.630128, -0.607749],
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

DEMON_BASIS = "../../test/basis-test"
WORKDIR = ".run/compare/"


class TestComparison:
    @pytest.mark.dftbplus
    def test_scc_water_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-10,
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=water.symbols, positions=water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.067517, atol=1e-6)

    @pytest.mark.dftbplus
    def test_scc_pyrene_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-10,
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene.symbols, positions=pyrene.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -31.33937716, atol=1e-6)

    @pytest.mark.dftbplus
    def test_scc_pyrene_water_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-10,
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene_water.symbols, positions=pyrene_water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -35.3706300998, atol=1e-7)

    @pytest.mark.dftbplus
    @pytest.mark.xfail(reason="TODO: DISPERSION BENCHMARK")
    def test_disp_bio_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "TOL": 1e-10, "DISP": 2},
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=water.symbols, positions=water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.067517, atol=1e-6)
        assert np.allclose(results["energy"]["london_energy"], -9.096e-5, atol=1e-6)

    @pytest.mark.dftbplus
    def test_tddftb_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-10,
                        },
                        "TD-DFTB": 15,
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=water.symbols, positions=water.positions)

        results = mod.results

        dftbp_results = {
            "state_0": {
                "w": 8.912,
                "oscillator": 0.00145313,
                "from": 8,
                "to": 9,
                "weight": 1.000,
                "energy": 8.904,
            },
            "state_1": {
                "w": 11.495,
                "oscillator": 0.00020729,
                "from": 7,
                "to": 9,
                "weight": 1.000,
                "energy": 11.493,
            },
            "state_2": {
                "w": 11.995,
                "oscillator": 0.01382197,
                "from": 6,
                "to": 9,
                "weight": 0.995,
                "energy": 11.824,
            },
            "state_3": {
                "w": 12.804,
                "oscillator": 0.03615028,
                "from": 5,
                "to": 9,
                "weight": 0.994,
                "energy": 12.550,
            },
            "state_4": {
                "w": 14.923,
                "oscillator": 0.00402463,
                "from": 3,
                "to": 9,
                "weight": 0.966,
                "energy": 15.014,
            },
            "state_5": {
                "w": 15.704,
                "oscillator": 0.00101974,
                "from": 8,
                "to": 10,
                "weight": 1.000,
                "energy": 15.694,
            },
            "state_6": {
                "w": 16.495,
                "oscillator": 0.51500594,
                "from": 4,
                "to": 9,
                "weight": 0.952,
                "energy": 14.473,
            },
            "state_7": {
                "w": 18.316,
                "oscillator": 0.00360137,
                "from": 7,
                "to": 10,
                "weight": 1.000,
                "energy": 18.283,
            },
            "state_8": {
                "w": 18.743,
                "oscillator": 0.06703579,
                "from": 6,
                "to": 10,
                "weight": 0.999,
                "energy": 18.614,
            },
            "state_9": {
                "w": 19.505,
                "oscillator": 0.06665403,
                "from": 5,
                "to": 10,
                "weight": 0.999,
                "energy": 19.340,
            },
        }
        singlet = results["singlet"]

        for (ka, ia), (kb, ib) in zip(singlet.items(), dftbp_results.items()):
            if ka == kb:
                assert np.allclose(ia["w"], ib["w"], atol=2e-3)
                assert np.allclose(ia["oscillator"], ib["oscillator"], atol=1e-5), f"Error in {ka}"
                assert np.allclose(ia["from"], ib["from"], atol=1e-1)
                assert np.allclose(ia["to"], ib["to"], atol=1e-1)
                assert np.allclose(ia["weight"], ib["weight"], atol=1e-3)
                assert np.allclose(ia["energy"], ib["energy"], atol=1e-3)

    @pytest.mark.dftbplus
    def test_scc_water_ldep(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "TOL": 1e-10, "L-DEP": True},
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=water.symbols, positions=water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.0597927515, atol=1e-7)
