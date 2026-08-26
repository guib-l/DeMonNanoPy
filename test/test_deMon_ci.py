import os
from copy import deepcopy

import numpy as np
import pytest
import shutil
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano
from conftest import compute_numgrad

deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
            "CHARGE": 1.0,
        },
    },
}

image = Atoms(
    [
        "C", "C", "N", "C", "C", "N", "C", "C", "H", "H", 
        "H", "H", "H", "H", "H", "H", "H", "H", "H", "H", 
        "O", "H", "H"
    ],
    positions=np.array(
        [
            [1.188367820025961, -0.791963696891490, -0.110257297286295],
            [0.760376820025962, -0.020468696891491, 1.213427702713705],
            [-0.116809179974038, 1.033744303108510, 0.816564702713705],
            [-1.312762179974039, 0.526830303108510, 0.224073702713705],
            [-0.884737179974039, -0.244625696891490, -1.099629297286295],
            [0.535658820025961, -0.142239696891490, -1.201197297286295],
            [0.964475820025962, 1.212058303108510, -1.350483297286295],
            [0.536476820025961, 1.983416303108510, -0.027019297286295],
            [2.293157820025963, -0.757119696891491, -0.239642297286295],
            [0.868695820025961, -1.853187696891491, -0.046486297286295],
            [0.238654820025961, -0.717377696891490, 1.902302702713705],
            [1.656376820025962, 0.390792303108509, 1.729724702713705],
            [-1.820598179974038, -0.173774696891491, 0.919554702713705],
            [-2.010428179974038, 1.358783303108510, -0.020114297286295],
            [-1.190464179974038, -1.309529696891490, -1.029229297286295],
            [-1.373583179974038, 0.211010303108510, -1.989537297286295],
            [2.067746820025961, 1.256872303108509, -1.484343297286295],
            [0.483872820025962, 1.675070303108510, -2.240302297286294],
            [-0.152207179974038, 2.821315303108510, -0.273552297286295],
            [1.431709820025961, 2.403197303108510, 0.482362702713705],
            [-1.274156179974038, -2.711962696891490, 1.170045702713705],
            [-1.715265179974038, -2.828611696891490, 2.023954702713705],
            [-1.295279179974038, -3.579516696891490, 0.740793702713705],
        ]
    ),
)

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

WORKDIR = ".run/dftbci/"


class TestDFTBCI:
    def test_ci(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_ci_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)


        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)
        assert np.allclose(results["energy"]["energy"], -23.3449257719042, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_const(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -23.28746978, atol=1e-7)
        assert np.allclose(energy["electronic_energy"], -23.92276564, atol=1e-7)
        assert np.allclose(energy["coulomb_energy"], 0.12999541, atol=1e-7)
        assert np.allclose(energy["repulsive_energy"], 0.63529586, atol=1e-7)


    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_const_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {
                    "SIZECI": 2,
                },
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)


        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -23.34492577, atol=1e-7)
        assert np.allclose(energy["electronic_energy"], -23.92276564, atol=1e-7)
        assert np.allclose(energy["coulomb_energy"], 0.12999541, atol=1e-7)
        assert np.allclose(energy["repulsive_energy"], 0.63529586, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)



    @pytest.mark.beta
    def test_ci_noslat(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"SIZECI": 2, "NOSLAT": True},
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.10000944, atol=1e-7)

    @pytest.mark.beta
    def test_ci_excci(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4},
                "CUTSYS": {
                    "FRAGMENT": [20, 3],
                },
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=image.symbols, positions=image.positions)

        results = dem.results
        conf_1 = results["configuration_1"]
        conf_2 = results["configuration_2"]

        assert np.allclose(conf_1["energy"], -23.34473263, atol=1e-7)
        assert np.allclose(conf_2["energy"], -23.24749332, atol=1e-7)

        assert np.allclose(len(results["states"]), 10.0)


    @pytest.mark.beta
    def test_const_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"CONST": 1},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }} )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.6781923, atol=1e-7)

    @pytest.mark.beta
    def test_ci_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"SIZECI": 2,},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }} )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.71029648, atol=1e-7)

    @pytest.mark.beta
    def test_excci_opt(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4,},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 10, "OUT": 1, "TRAJECTORY": True},
                    },
                }} )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.70495473, atol=1e-7)


    def test_ci_dftb3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "THIRD":True},
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4,},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64602728, atol=1e-7)


    def test_ci_fermi(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "FERMI":50},
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4,},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65375039, atol=1e-7)

    def test_ci_ldep(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "L-DEP":True},
                "CI": {"SIZECI": 2, "NOSLAT": False, "EXCCI": 4,},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.58751914, atol=1e-7)

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_ci_ldep_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "L-DEP":True},
                "CI": {"SIZECI": 2, },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.77607094, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    @pytest.mark.forces
    def test_ci_fermi_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "FERMI":50},
                "CI": {"SIZECI": 2, },
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65119769, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_disp1(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":1},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.63790536, atol=1e-7)

    @pytest.mark.forces
    def test_ci_disp1_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":1},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"],-7.63790536, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)


    def test_ci_disp2(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":2},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65087386, atol=1e-7)

    @pytest.mark.forces
    def test_ci_disp2_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":2},
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.65087386, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)

    def test_ci_freq(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True,},
                "FREQ":True,
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        assert np.allclose(results["zpe"], 0.0458589, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][0]
        assert mode_1["mode"] == 1
        assert np.allclose(mode_1["frequency"], -1323.4, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 54.8, atol=1e-1)

    def test_ci_cm3(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":2},
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64172767, atol=1e-7)

                

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    @pytest.mark.forces
    def test_ci_cm3_grad(self):

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {"SCC": True, "DISP":2},
                "CM3": {
                    "BONDPARAMS": {
                        "O H": 0.08,
                    },
                },
                "CI": {"SIZECI": 2},
                "CUTSYS": {
                    "FRAGMENT": [3, 3],
                },
            }
        )

        parameter_config_bis = deepcopy(parameter_config)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        dem.calculate(symbols=water.symbols, positions=water.positions)

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -7.64172767, atol=1e-7)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)
        grad = compute_numgrad(
            symbols=water.symbols, positions=water.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)




