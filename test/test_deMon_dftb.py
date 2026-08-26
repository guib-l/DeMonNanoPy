import copy
import os

# import configs
import numpy as np
import pytest
from ase.atoms import Atoms
from conftest import compute_numgrad

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

WORKDIR = ".run/basics/"


class TestDftbBasis:
    
    def test_basic(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "NSC", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": False,
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=".run/basics_2/", **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.12781231

    @pytest.mark.forces
    def test_basic_grad(self):
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "NSC", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": False,
                        },
                    },
                },
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=mod, delta=0.001
        )

        analytic_grad = np.array(
            [
                [0.124071893196, -0.011540525168, -0.018777379600],
                [-0.006380969450, 0.026461971110, -0.002859012651],
                [0.007438210717, -0.006084726858, 0.013443940374],
                [-0.189837064101, 0.092608105513, -0.129732302303],
                [0.021047970469, -0.077415090018, -0.006721464166],
                [0.043659959168, -0.024029734579, 0.144646218347],
            ]
        )
        assert np.allclose(analytic_grad, grad, atol=1e-5)

    def test_scc_bio(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343

    @pytest.mark.forces
    def test_bio_grad(self):
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                    },
                },
            }
        )
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=mod, delta=0.001
        )

        analytic_grad = np.array(
            [
                [0.135530575750, 0.009199940329, 0.011440286997],
                [0.010103858618, -0.000273288920, -0.002630485701],
                [-0.001911798747, 0.001791329037, -0.012982417004],
                [-0.212418203769, 0.109601608000, -0.113561867353],
                [0.026059609077, -0.098807595197, -0.005777818373],
                [0.042635959070, -0.021511993248, 0.123512301433],
            ]
        )
        assert np.allclose(analytic_grad, grad, atol=1e-5)

    def test_scc_mat(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "MAT", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                    },
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.08755742


WORKDIR = ".run/dftb/"


class TestDftb:
    def test_scc(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343

    def test_scc_tol(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-2,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06203571

    def test_scc_memoscc(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": True,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343

    def test_scc_pola(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": True,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                            "ALPHAH": 0.0,
                            "ALPHAO": 0.0,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343

    @pytest.mark.forces
    def test_pola_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "POLA": True, "ALPHAH": 1.0, "ALPHAO": 4.0},
                    },
                }
            }
        )

        copy_parameters_bis = copy.deepcopy(copy_parameters)
        copy_parameters_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters_bis)
        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06614509

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_scc_maxiter(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 5,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        err = results["errors"]
        assert "not converge for geometry in dftb_canonical" in err[0]["message"]

    def test_scc_mixing(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.02,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_scc_simple_mixing(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": True,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.00903688

    def test_scc_ldep(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "L-DEP": True,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.clean_workdir()

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06480023

    def test_scc_fermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 50.00,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343




    def test_scc_disp1(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": 1,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.04913036

    def test_scc_disp2(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": 2,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209886

    @pytest.mark.forces
    def test_disp1_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "DISP": 1,
                        },
                    },
                }
            }
        )

        copy_parameters_bis = copy.deepcopy(copy_parameters)
        copy_parameters_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters_bis)
        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.04913036

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    @pytest.mark.forces
    def test_disp2_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "DISP": 2,
                        },
                    },
                }
            }
        )

        copy_parameters_bis = copy.deepcopy(copy_parameters)
        copy_parameters_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters_bis)
        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209886

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_scc_level_shift(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": 0.0001,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results

        assert np.allclose(results["energy"]["energy"], -8.06209343, atol=1e-7)

    @pytest.mark.beta
    def test_scc_diagDSYGVD(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "DIAG": "DSYGVD",
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results

        assert np.allclose(results["energy"]["energy"], -8.06209343, atol=1e-7)

    @pytest.mark.beta
    def test_scc_dftb3(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD": True,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results

        assert np.allclose(results["energy"]["energy"], -8.06658147, atol=1e-7)

    @pytest.mark.beta
    @pytest.mark.forces
    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_dftb3_grad(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD": True,
                        },
                    },
                }
            }
        )

        copy_parameters_bis = copy.deepcopy(copy_parameters)
        copy_parameters_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters_bis)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results

        assert np.allclose(results["energy"]["energy"], -8.0483818, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    @pytest.mark.beta
    def test_scc_gcor(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "GCOR": 0.3,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions, clean_repository=False)

        results = mod.results

        assert np.allclose(results["energy"]["energy"], -8.22620688, atol=1e-7)

    @pytest.mark.beta
    def test_scc_pola_beta(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {"SCC": True, "POLA": True, "ALPHAH": 1.0, "ALPHAO": 2.0},
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -8.0654775, atol=1e-7)


    def test_scc_ldepFermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "FERMI": 150.00,
                            "L-DEP": True,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.clean_workdir()

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06480023


    def test_scc_disp1Fermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 150,
                            "THRID": False,
                            "DISP": 1,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.04913036

    def test_scc_disp2Fermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 150,
                            "THRID": False,
                            "DISP": 2,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209886

    @pytest.mark.beta
    def test_scc_dftb3Fermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 150,
                            "THIRD": True,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06658147

    @pytest.mark.beta
    def test_scc_dftb3FermiDisp2(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 150,
                            "THIRD": True,
                            "DISP": 2,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06658691

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_scc_ldepMemo(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": True,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": True,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06658691

    def test_scc_memoDisp2(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": True,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": None,
                            "THRID": False,
                            "DISP": 2,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209886

    def test_scc_memoFermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": True,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 100,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343


    @pytest.mark.forces
    def test_scc_fermi_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": False,
                            "FERMI": 50.00,
                            "THRID": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        copy_parameters_bis = copy.deepcopy(copy_parameters)
        copy_parameters_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters_bis)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06209343

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_scc_ldep_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "TOL": 1e-8,
                            "MEMOSCC": False,
                            "POLA": False,
                            "MAX": 100,
                            "MIX": 0.2,
                            "SIMPLE": False,
                            "L-DEP": True,
                            "THIRD": False,
                            "DISP": False,
                            "LEV_S": None,
                        },
                    },
                }
            }
        )

        copy_parameters_bis = copy.deepcopy(copy_parameters)
        copy_parameters_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters_bis)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert results["energy"]["energy"] == -8.06480023

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)










