import copy

# import configs
import numpy as np
import pytest
import shutil
from copy import deepcopy

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

base = Atoms(
    ["O","O","C","N","H","H","C","H","H","H"],
    positions=np.array([
            [2.082633,-2.033395,-0.701809],
            [1.904313,-2.187495, 1.519431],
            [1.720950,-1.513026, 0.440466],
            [1.101965, 0.493364,-0.862487],
            [0.062217,-0.235367, 0.848594],
            [1.629754, 0.475622, 1.230297],
            [1.094008,-0.130999, 0.482069],
            [1.940946,-1.432516,-1.475023],
            [1.882500, 1.148872,-0.953228],
            [0.236814, 1.022592,-1.005798]
        ]
    ),
    charges=np.array([
        -0.382294, -0.868006,0.754458,-0.522439,0.149845,
        0.095409,-0.096701, 0.399571, 0.229906,0.240251
    ])
)



symb = {
    8:"O",1:"H",6:"C",7:"N"
}

WORKDIR = ".run/qmmm/"


class TestDftbQMMM:
    # =========================================================================
    #    MM
    # =========================================================================

    def test_basic(self):

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

    def test_mm(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "MM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":[-0.83,0.417,0.417,-0.83,0.417,0.417],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        
        assert results["energy"]["qmmm-energy"] == 0.268095513


    @pytest.mark.forces
    def test_mm_ff_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "MM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":[-0.83,0.417,0.417,-0.83,0.417,0.417],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        parameter_config_bis = deepcopy(copy_parameters)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        charges = results["output_geometry"].get_initial_charges()

        assert np.allclose(results["energy"]["qmmm-energy"],0.268095513,atol=1e-7)     

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001,
            triger_str="qmmm-energy"
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)


    @pytest.mark.optim
    def test_mm_opt(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "MM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":[-0.83,0.417,0.417,-0.83,0.417,0.417],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 9999, "TRAJECTORY": True},
                    },
                }
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        
        assert results["energy"]["qmmm-energy"] == -0.010737445


    @pytest.mark.optim
    def test_mm_cluster_opt(self):

        table = []
        with open("test/data_test/water-cluster.xyz") as fd:
            
            for line in fd.readlines():
                table.append(list(map(float,line.split())))
        table = np.array(table)

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "MM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )

        symbols = [symb[elm] for elm in table[:,0]]

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=symbols, positions=table[:,1:4])

        results = mod.results
        
        assert results["energy"]["qmmm-energy"] == -4.12929233

    @pytest.mark.dynamics
    def test_mm_cluster_md(self):

        table = []
        with open("test/data_test/water-cluster.xyz") as fd:
            
            for line in fd.readlines():
                table.append(list(map(float,line.split())))
        table = np.array(table)

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "MM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 300,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 150, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "MDTEMP": 300,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )


        symbols = [symb[elm] for elm in table[:,0]]

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=symbols, positions=table[:,1:4])

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote + kine))[1:]) <= 1e-5



    # =========================================================================
    #    QM
    # =========================================================================



    def test_qm_base(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":base.get_initial_charges(),
                            "TYPEMM":{
                                "O":783,
                                "H":780,
                                "C":782,
                                "N":783
                            },
                            "QM":"1-10",
                            "MM":"",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=base.symbols, positions=base.positions,read_charges=True)

        results = mod.results
        
        assert results["energy"]["energy"] == -14.31824344


    def test_qm_base_mmtype(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD":True,
                            "GCOR":4.0
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":base.get_initial_charges(),
                            "TYPEMM":[784,784,782,783,780,780,781,64,785,785],
                            "QM":"1-10",
                            "MM":"",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=base.symbols, positions=base.positions,read_charges=True)

        results = mod.results
        
        assert results["energy"]["energy"] == -14.32804805





    # =========================================================================
    #    QMMM
    # =========================================================================

    def test_qmmm_input(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":[-0.83,0.415,0.415,-0.83,0.415,0.415],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        
        assert results["energy"]["energy"] == -3.81268693

    def test_qmmm_ff(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        
        assert results["energy"]["energy"] == -3.81264238
    
    @pytest.mark.optim
    def test_qmmm_ff_opt(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"MAX": 999, "OUT": 1, "TRAJECTORY": True},
                    },
                }
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        
        assert results["energy"]["energy"] == -4.08200347

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_qmmm_ff_ldep(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "L-DEP":True
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions,read_charges=True)

        results = mod.results
        charges = results["output_geometry"].get_initial_charges()

        ref_charges = np.array([-0.53254, 0.271837, 0.260703, -0.834, 0.417, 0.417])
        assert np.allclose(charges,ref_charges,atol=1e-5)        
        assert results["energy"]["energy"] == -3.82336904

    @pytest.mark.forces
    def test_qmmm_ff_grad(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        parameter_config_bis = deepcopy(copy_parameters)
        parameter_config_bis.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {"SP": True, "TRAJECTORY": True},
                    },
                }
            }
        )
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config_bis)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        charges = results["output_geometry"].get_initial_charges()

        assert np.allclose(results["energy"]["energy"],-3.81264238,atol=1e-7)     

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        grad = compute_numgrad(
            symbols=image.symbols, positions=image.positions, calculator=dem, delta=0.001
        )

        assert np.allclose(results["forces"], grad, atol=1e-5)



    def test_qmmm_ff_fermi(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "FERMI":150
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions,read_charges=True)

        results = mod.results
        charges = results["output_geometry"].get_initial_charges()

        ref_charges = np.array([-0.53254, 0.271837, 0.260703, -0.834, 0.417, 0.417])
        assert np.allclose(charges,ref_charges,atol=1e-5)
        assert results["energy"]["energy"] == -3.81264238

    def test_qmmm_ff_dftb3(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD":True,
                            "GCOR":4.0
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions,read_charges=True)

        results = mod.results
        charges = results["output_geometry"].get_initial_charges()
        
        ref_charges = np.array([-0.626295,0.318292,0.308003,-0.834,0.417,0.417])
        assert np.allclose(charges,ref_charges,atol=1e-5)
        assert results["energy"]["energy"] == -3.82304754

    def test_qmmm_ff_charge(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD":True,
                            "GCOR":4.0
                        },
                        "CHARGE":1.0,
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions,read_charges=True)

        results = mod.results
        charges = results["output_geometry"].get_initial_charges()
         
        ref_charges = np.array([0.196438,0.404447,0.399115, -0.834,0.417,0.417])
        assert np.allclose(charges,ref_charges,atol=1e-5)
        assert results["energy"]["energy"] == -3.30637269

    
    @pytest.mark.freq
    def test_qmmm_ff_freq(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "FREQ":True,
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        assert np.allclose(results["zpe"],0.05456938, atol=1e-4)
        assert len(results["frequency"]) == 18
        mode_1 = results["frequency"][10]
        assert mode_1["mode"] == 11
        assert np.allclose(mode_1["frequency"], 3894.9, atol=1e-1)
        assert np.allclose(mode_1["intensity"], 0.5, atol=1e-1)

    def test_qmmm_ff_tddftb(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "TD-DFTB": True,
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        energy = results["energy"]
        assert np.allclose(energy["energy"],-3.81264238,atol=1e-7)

        assert "triplet" in results.keys()
        assert "singlet" in results.keys()

    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_qmmm_ff_ci(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "CHARGE": 1.0,
                        "CI": {
                            "SIZECI": 2,
                        },
                        "CUTSYS": {
                            "FRAGMENT": [3,3,4],
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":783,
                                "H":780,
                                "C":782,
                                "N":783
                            },
                            "QM":"1-6",
                            "MM":"7-10",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=base.symbols, positions=base.positions)

        results = mod.results
        energy = results["energy"]
        assert np.allclose(energy["energy"],-3.81264238,atol=1e-7)


    @pytest.mark.dynamics
    def test_qmmm_ff_dyn(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4,5,6",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                }
            }
        )
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDYNAMICS": {
                                "ZERO": True,
                                "RANDOM": 300,
                            },
                            "TIMESTEP": 0.4,
                            "MDSTEP": {"MAX": 150, "OUT": 1, "SOUT": 1, "TSIM": None},
                            "MDTEMP": 300,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=image.symbols, positions=image.positions,read_charges=True)

        results = mod.results
        pote = results["potential_energy"]
        kine = results["kinetic_energy"]
        tote = results["total_energy"]

        assert np.sum((tote - (pote + kine))[1:]) <= 1e-5


    

    # =========================================================================
    #    QM/MM verifications
    # =========================================================================


    def test_qmmm_ref1(self,parameters_3ob):

        from pathlib import Path

        if not Path(parameters_3ob).is_dir():
            pytest.skip(f"Folder {parameters_3ob} didn't exists.")
        DEMON_BASIS = parameters_3ob 

        table = []
        with open("test/data_test/mb1") as fd:
            
            for line in fd.readlines():
                table.append(list(map(float,line.split())))
        table = np.array(table)


        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD":True,
                            "GCOR":4.0
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":([[-0.834,0.417,0.417]*212])[0],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"",
                            "MM":"1-636",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )
        symbols = ([["O","H","H"] * 212])[0]

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=symbols, positions=table)

        results = mod.results
        assert results["energy"]["qmmm-energy"] == -4.047883710

    def test_qmmm_ref2(self, parameters_3ob):

        from pathlib import Path

        if not Path(parameters_3ob).is_dir():
            pytest.skip(f"Folder {parameters_3ob} didn't exists.")
        DEMON_BASIS = parameters_3ob

        table = []
        with open("test/data_test/mb2") as fd:
            
            for line in fd.readlines():
                table.append(list(map(float,line.split())))
        table = np.array(table)


        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD":True,
                            "GCOR":4.0
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":table[:,-1],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
                    },
                },
            }
        )
        symbols = [symb[elm] for elm in table[:,0]]

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=["O","H","H"], positions=table[:,:3])

        results = mod.results
        assert results["energy"]["energy"] == -4.06132578


    @pytest.mark.optim
    def test_qmmm_ref3(self, parameters_3ob):

        from pathlib import Path

        if not Path(parameters_3ob).is_dir():
            pytest.skip(f"Folder {parameters_3ob} didn't exists.")
        DEMON_BASIS = parameters_3ob

        table = []
        with open("test/data_test/mb3") as fd:
            
            for line in fd.readlines():
                table.append(list(map(float,line.split())))
        table = np.array(table)

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "BASIS": {"PTYPE": "3OB", "SKFILE": DEMON_BASIS},
                "DEMON_PARAMETERS": {
                    "ACTIVE": {
                        "DFTB": {
                            "SCC": True,
                            "THIRD":True,
                            "GCOR":4.0
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":table[:,3],
                            "TYPEMM":{
                                "O":63,
                                "H":64
                            },
                            "QM":"1-3",
                            "MM":"4-639",
                            "FORCEFIELD":{
                                "FF":"OPLS-AA"
                            }
                        }
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
        symbols = [["O","H","H"] * 213]
        symbols = symbols[0]

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS", f"{WORKDIR}/FFDS")
        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=symbols, positions=table[:,:3])

        results = mod.results
        assert results["energy"]["energy"] == -8.14600091


    def test_qmmm_rtdtdftb(self):

        add_water = Atoms(
            ["O","H","H"],
            positions=np.array([
                [26.250000,30.030001,30.610000],
                [26.370000,29.080000,30.710000],
                [26.570000,30.390000,31.440001],
            ])
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
                        "RTTDDFTB":{
                            "COLL":10,
                            "PROJPT":True
                        },
                        "CUTSYS": {
                            "FRAGMENT": [1]*10,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"FF",
                            "CHARGES":None,
                            "TYPEMM":[
                                5,63,3,1,6,6,2,4,4,64,2001,2002,2002
                            ],
                            "QM":"1-10",
                            "MM":"11-13",
                            "FORCEFIELD":{
                                "FF":"AMBER-FF99SB"
                            }
                        }
                    },
                },
            }
        )

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")

        with open(f"{WORKDIR}/data_col.txt", 'w') as fd:

            fd.write("7\n")
            fd.write("-0.897913 -0.137568 -0.418123 \n")
            fd.write("0.0 0.0 0.0 \n")
            fd.write("1\n")
            fd.write("1\n")
            fd.write("4\n")


        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=_images.symbols,  positions=_images.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"],-14.31818879,atol=1e-7)

    @pytest.mark.xfail(reason="CRITICAL -> TO FIX")
    @pytest.mark.dynamics
    def test_qmmm_rtdtdftb_md(self):

        table = []
        with open("test/data_test/mb4") as fd:
            
            for line in fd.readlines():
                table.append(list(map(float,line.split())))
        table = np.array(table)

        _images = Atoms(
            ["O","O","C","N","H","H","C","H","H","H","O","H","H",],
            positions=table[:,:3]
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
                        "RTTDDFTB":{
                            "COLL":10,
                            "PROJPT":True
                        },
                        "CUTSYS": {
                            "FRAGMENT": [1]*10,
                        },
                        "QMMM":{
                            "COUPLING":"ELECTROSTATIC",
                            "CHR":"INPUT",
                            "CHARGES":None,
                            "TYPEMM":[
                                5,63,3,1,6,6,2,4,4,64,2001,2002,2002
                            ],
                            "QM":"1-10",
                            "MM":"11-13",
                            "FORCEFIELD":{
                                "FF":"AMBER-FF99SB"
                            }
                        }
                    },
                },
            }
        )

        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "MD": {
                            "MDSTEP": {"MAX": 2000, "OUT": 1,},
                            "TIMESTEP": 0.001,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        shutil.copy2("test/data_test/3ord_param", f"{WORKDIR}/3ord_param")
        shutil.copy2("test/data_test/FFDS-AMBER", f"{WORKDIR}/FFDS")

        with open(f"{WORKDIR}/data_col.txt", 'w') as fd:

            fd.write("7\n")
            fd.write("-0.897913 -0.137568 -0.418123 \n")
            fd.write("0.0 0.0 0.0 \n")
            fd.write("1\n")
            fd.write("1\n")
            fd.write("4\n")


        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=_images.symbols,  positions=_images.positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"],0.0,atol=1e-7)












