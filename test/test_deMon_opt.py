import copy
import os

# import configs
import numpy as np
import pytest
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

WORKDIR = ".run/opt/"


class TestOptimization:
    def test_opt_basic(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15563755, atol=1e-7)

    def test_opt_maxiter(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 10,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": True,
                            "SDC": False,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.1487639, atol=1e-7)
        assert len(results["trajectory"]) == 11

    def test_opt_tolerance(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-2,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": True,
                            "SDC": False,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15563755, atol=1e-7)

    def test_opt_gradtol(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 1e-2,
                            "STEP": 0.3,
                            "CGRAD": True,
                            "SDC": False,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results

        assert results["optimization"]["grad_max"] < 1e-2
        assert results["optimization"]["grad_max"] > 1e-5

    def test_opt_cgrad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 1e-4,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": True,
                            "SDC": False,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15563755, atol=1e-7), "Errors in CGRAD"

    def test_opt_steepest_default(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": False,
                            "SDC": True,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15497995, atol=1e-7)
        assert results["errors"][0]["message"] == "optimization not converged"

    def test_opt_steepest(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": False,
                            "SDC": 0.2,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15497675, atol=1e-7)
        assert results["errors"][0]["message"] == "optimization not converged"

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_opt_out(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "OUT": 10,
                            "MAX": 20,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": True,
                            "SDC": False,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.14955388, atol=1e-7)
        assert len(results["trajectory"]) == 3

    def test_opt_noTraj(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "STEP": 0.3,
                            "CGRAD": True,
                            "SDC": False,
                            "OUT": 10,
                            "TRAJECTORY": False,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15563755, atol=1e-7), (
            "Errors TRAJECTORY optimization"
        )
        assert "trajectory" not in results.keys()

    @pytest.mark.beta
    def _test_opt_bfgs(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "BFGS": True,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15563755, atol=1e-7), (
            "Errors TRAJECTORY optimization"
        )

    @pytest.mark.beta
    def _test_opt_lbfgs(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "LBFGS": True,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.15563755, atol=1e-7), (
            "Errors TRAJECTORY optimization"
        )

    @pytest.mark.beta
    def _test_opt_lbfgs_mem(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 999,
                            "TOL": 3e-4,
                            "GRADTOL": 4e-6,
                            "LBFGS": True,
                            "MEMORY": 999,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.14904689, atol=1e-7), (
            "Errors TRAJECTORY optimization"
        )

    @pytest.mark.beta
    def test_opt_sp(self):

        test_force = np.array(
            [
                [0.135530575750, 0.009199940329, 0.011440286997],
                [0.010103858618, -0.000273288920, -0.002630485701],
                [-0.001911798747, 0.001791329037, -0.012982417004],
                [-0.212418203769, 0.109601608000, -0.113561867353],
                [0.026059609077, -0.098807595197, -0.005777818373],
                [0.042635959070, -0.021511993248, 0.123512301433],
            ]
        )

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)
        results = mod.results

        assert np.allclose(results["energy"]["energy"], -8.06209343, atol=1e-7), (
            "Errors TRAJECTORY optimization"
        )
        assert np.allclose(results["forces"], test_force, atol=1e-7)
