import copy

import numpy as np
import pytest
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano
from demon_env import configure_or_skip

configure_or_skip("global.json")


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

WORKDIR = ".run/compare/"


class TestComparison:
    @pytest.mark.references
    def test_scc_water_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "TOL": 1e-10, "FHUB": True},
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=water.symbols, positions=water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.056172815, atol=1e-6)

    @pytest.mark.references
    def test_scc_pyrene_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "TOL": 1e-10, "FHUB": True},
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene.symbols, positions=pyrene.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -31.337463608, atol=1e-6)

    @pytest.mark.references
    def test_scc_pyrene_water_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "TOL": 1e-10, "FHUB": True},
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=pyrene_water.symbols, positions=pyrene_water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -35.3609083, atol=1e-7)

    @pytest.mark.references
    #@pytest.mark.xfail(reason="TODO: DISPERSION BENCHMARK")
    def test_disp_bio_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "TOL": 1e-10, "DISP": 2, "FHUB": True},
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=water.symbols, positions=water.positions)

        results = mod.results
        assert np.allclose(results["energy"]["electronic_energy"], -8.212962, atol=1e-6)
        assert np.allclose(results["energy"]["band_energy"], -8.256901, atol=1e-6)
        assert np.allclose(results["energy"]["coulomb_energy"], 0.043939, atol=1e-6)
        assert np.allclose(results["energy"]["repulsive_energy"], 0.1567898, atol=1e-6)

        # Not equivalent dispersion value
        #assert np.allclose(results["energy"]["energy"], -8.067517, atol=1e-6)
        #assert np.allclose(results["energy"]["london_energy"], -9.096e-5, atol=1e-6)

    @pytest.mark.references
    def test_tddftb_dftbp(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
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
                "w": 8.813,
                "oscillator": 0.00131492,
                "from": 8,
                "to": 9,
                "weight": 1.000,
                "energy": 8.805,
            },
            "state_1": {
                "w": 11.399,
                "oscillator": 0.00021872,
                "from": 7,
                "to": 9,
                "weight": 1.000,
                "energy": 11.397,
            },
            "state_2": {
                "w": 11.930,
                "oscillator": 0.01051612,
                "from": 6,
                "to": 9,
                "weight": 0.994,
                "energy": 11.749,
            },
            "state_3": {
                "w": 12.758,
                "oscillator": 0.02929646,
                "from": 5,
                "to": 9,
                "weight": 0.991,
                "energy": 12.477,
            },
            "state_4": {
                "w": 14.906,
                "oscillator": 0.00246363,
                "from": 3,
                "to": 9,
                "weight": 0.972,
                "energy": 14.993,
            },
            "state_5": {
                "w": 15.595,
                "oscillator": 0.00090236,
                "from": 8,
                "to": 10,
                "weight": 1.000,
                "energy": 15.584,
            },
            "state_6": {
                "w": 16.824,
                "oscillator": 0.49813771,
                "from": 4,
                "to": 9,
                "weight": 0.958,
                "energy": 14.441,
            },
            "state_7": {
                "w": 18.210,
                "oscillator": 0.00279232,
                "from": 7,
                "to": 10,
                "weight": 1.000,
                "energy": 18.176,
            },
            "state_8": {
                "w": 18.679,
                "oscillator": 0.06038470,
                "from": 6,
                "to": 10,
                "weight": 0.999,
                "energy": 18.529,
            },
            "state_9": {
                "w": 19.459,
                "oscillator": 0.05974118,
                "from": 5,
                "to": 10,
                "weight": 0.999,
                "energy": 19.257,
            },
            "state_10": {
                "w": 20.029,
                "oscillator": 0.01087593,
                "from": 8,
                "to": 11,
                "weight": 1.000,
                "energy": 20.007,
            },
            "state_11": {
                "w": 21.293,
                "oscillator": 0.00620529,
                "from": 4,
                "to": 10,
                "weight": 0.996,
                "energy": 21.221,
            },
            "state_12": {
                "w": 22.523,
                "oscillator": 0.02479502,
                "from": 7,
                "to": 11,
                "weight": 0.701,
                "energy": 22.598,
            },
            "state_13": {
                "w": 22.674,
                "oscillator": 0.04554975,
                "from": 7,
                "to": 11,
                "weight": 0.915,
                "energy": 22.598,
            },
            "state_14": {
                "w": 23.386,
                "oscillator": 0.03078044,
                "from": 5,
                "to": 11,
                "weight": 0.826,
                "energy": 23.679,
            },
        }
        singlet = results["singlet"]

        for (ka, ia), (kb, ib) in zip(singlet.items(), dftbp_results.items()):
            if ka == kb:
                assert np.allclose(ia["w"], ib["w"], atol=2e-3)
                assert np.allclose(ia["oscillator"], ib["oscillator"], atol=1e-5), f"Error in {ka}"
                assert np.allclose(ia["from"], ib["from"], atol=1e-1)
                assert np.allclose(ia["to"], ib["to"], atol=1e-1)
                #assert np.allclose(ia["weight"], ib["weight"], atol=1e-3)
                assert np.allclose(ia["energy"], ib["energy"], atol=1e-3)

    @pytest.mark.references
    def test_scc_water_ldep(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
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
        assert np.allclose(results["energy"]["energy"], -8.06480050, atol=1e-7)
