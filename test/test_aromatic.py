import sys
import os

import pytest
#import configs
import numpy as np

import copy
from copy import deepcopy

from ase.atoms import Atoms
from deMonPy.deMonNano import deMonNano
from deMonPy.deMonNano import Module_DeMonNano

import deMonPy
from deMonPy.deMonNano import deMonNano


deMonPy.configure_from_file(os.path.join("..", "global.json"))

parameters = {
    "DEMON_EXECUTABLE":deMonPy.DEMON_EXECUTABLE,
    "BASIS":{
        "PTYPE":"BIO",
        "SKFILE":deMonPy.DEMON_BASIS
    },
    "DEMON_PARAMETERS":{
        "ACTIVE":{
            "DFTB":{
                "SCC":True,
                "DISP":2,
                "MAX":999,
            },
        },
    }
}

benzen = Atoms(
    ["C","H","C","C","C","C","C","H","H","H","H","H",],
    positions=np.array([
            [ 0.000000, 1.401045, 0.000000],
            [ 0.000000, 2.409045, 0.000000],
            [ 1.212436, 0.701045, 0.000000],
            [ 1.212436,-0.698955, 0.000000],
            [ 0.000000,-1.398955, 0.000000],
            [-1.212436,-0.698955, 0.000000],
            [-1.212436, 0.701045, 0.000000],
            [ 2.155537, 1.245545, 0.000000],
            [ 2.155537,-1.243455, 0.000000],
            [ 0.000000,-2.487955, 0.000000],
            [-2.155537,-1.243455, 0.000000],
            [-2.155537, 1.245545, 0.000000],
        ])
    )

pyrene = Atoms(
    ["C"]*16 + ["H"]*10,
    positions=np.array([
            [ 0.175933, 0.335146, 1.793713],  
            [ 1.360050, 0.325800, 1.110006],  
            [-1.072593, 0.111218, 1.119476],  
            [-2.302066, 0.111541, 1.806122],  
            [ 1.397620, 0.091799,-0.306827],  
            [ 2.606846, 0.072980,-1.028286],  
            [ 0.172008,-0.133217,-1.000407],  
            [ 0.181765,-0.372636,-2.406258],  
            [-1.063252,-0.123546,-0.287152],  
            [-2.288443,-0.353337,-0.979928],  
            [-3.496740,-0.344828,-0.256738],  
            [-3.495810,-0.114190, 1.119551],  
            [ 1.412155,-0.383289,-3.091169],  
            [ 2.606303,-0.162069,-2.403831],  
            [-1.066353,-0.601267,-3.079682],  
            [-2.250464,-0.592049,-2.395959],  
            [-4.444118,-0.521654,-0.784767],  
            [ 1.426319,-0.567696,-4.174401],  
            [-1.053082,-0.786725,-4.163133],  
            [ 3.557977,-0.173091,-2.952575],  
            [-3.197581,-0.770057,-2.924850],  
            [-4.446870,-0.110240, 1.669451],  
            [ 0.163224, 0.514366, 2.878219],  
            [ 2.307746, 0.497406, 1.639983],  
            [ 3.554629, 0.245151,-0.499450],  
            [-2.315834, 0.291268, 2.890140], 
        ])
    )

coronene = Atoms(
    ["C"]*24 + ["H"]*12,
    positions=np.array([
            [ 0.052302,  -0.009515 , 1.641099],
            [ 0.052225,   0.052477 , 3.060236],
            [ 1.288787,  -0.018542 , 0.927144],
            [ 2.518486,   0.034562 , 1.636286],
            [-1.183277,  -0.068207 , 0.928740],
            [-2.413067,  -0.063742 , 1.639357],
            [ 1.288882,  -0.085895 ,-0.498679],
            [ 2.519058,  -0.099369 ,-1.208400],
            [ 0.053279,  -0.144740 ,-1.210209],
            [ 0.053888,  -0.215361 ,-2.628596],
            [-1.182430,  -0.135986 ,-0.496718],
            [-2.411825,  -0.198190 ,-1.204880],
            [ 3.740023,  -0.044293 ,-0.474443],
            [ 3.739905,   0.020302 , 0.901037],
            [-3.633173,  -0.191170 ,-0.469555],
            [-3.633965,  -0.126197 , 0.905835],
            [ 1.301313,  -0.226023 ,-3.318715],
            [ 2.493501,  -0.170214 ,-2.632078],
            [-1.193035,  -0.275139 ,-3.317116],
            [-2.385445,  -0.266953 ,-2.628614],
            [ 1.299489,   0.104668 , 3.748856],
            [ 2.491857,   0.096107 , 3.060435],
            [-1.195211,   0.055974 , 3.750505],
            [-2.387194,   0.000185 , 3.063368],
            [ 4.691379,  -0.054769 ,-1.025283],
            [-4.584085,  -0.239952 ,-1.019148],
            [ 1.300172,  -0.281342 ,-4.416744],
            [-1.191117,  -0.329632 ,-4.415179],
            [-1.193686,   0.103622 , 4.848757],
            [ 1.297489,   0.151248 , 4.847161],
            [ 3.446477,  -0.180799 ,-3.180139],
            [-3.337922,  -0.315062 ,-3.175565],
            [-3.339897,   0.003843 , 3.612013],
            [ 3.444496,   0.135878 , 3.607602],
            [-4.585108,  -0.122647 , 1.457164],
            [ 4.690818,   0.061300 , 1.451086],
        ])
    )


WORKDIR = ".run/aromatic/"



class TestAromatic:

    def test_benzen_mulliken(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99999,
                            "TOL":3e-7,
                            "GRADTOL":1e-5,
                            "OUT":1,
                            "TRAJECTORY":False
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=benzen.symbols,
            positions=benzen.positions,
            read_charges=True
        )

        results = mod.results

        last = results["output_geometry"]

        charge = last.get_initial_charges()
        assert np.allclose(round(charge[0],3),-0.065, atol=2e-3)

        assert np.allclose(results["energy"]["energy"],-12.56863286, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"],-0.00058171, atol=1e-7)

    
    def test_benzen_cm3(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99999,
                            "TOL":3e-7,
                            "GRADTOL":1e-5,
                            "OUT":1,
                            "TRAJECTORY":False
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=benzen.symbols,
            positions=benzen.positions,
            read_charges=True
        )

        results = mod.results

        last = results["output_geometry"]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"CM3":{
                "BONDPARAMS":{
                    "H C":0.14,
                }
            }}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=last.symbols,
            positions=last.positions,
            read_charges=True
        )

        results = mod.results
        last = results["output_geometry"]

        charge = last.get_initial_charges()
        assert np.allclose(charge[0],-0.12, atol=1e-3)

        assert np.allclose(results["energy"]["energy"],-12.57492259, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"],-0.00058228, atol=1e-5)



    def test_pyrene_mulliken(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99999,
                            "TOL":3e-7,
                            "GRADTOL":1e-5,
                            "OUT":1,
                            "TRAJECTORY":False
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
        mod.clean_workdir()
            
        mod.calculate(
            symbols=pyrene.symbols,
            positions=pyrene.positions,
            read_charges=True
        )

        results = mod.results
        last = results["output_geometry"]
        charge = last.get_initial_charges()

        assert np.allclose(results["energy"]["energy"],-31.34507905, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"],-0.00677663, atol=1e-7)

    
    def test_pyrene_cm3(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99999,
                            "TOL":3e-7,
                            "GRADTOL":1e-5,
                            "OUT":1,
                            "TRAJECTORY":False
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=pyrene.symbols,
            positions=pyrene.positions,
            read_charges=True
        )

        results = mod.results

        last = results["output_geometry"]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"CM3":{
                "BONDPARAMS":{
                    "H C":0.06,
                }
            }}
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
        mod.clean_workdir()
            
        mod.calculate(
            symbols=last.symbols,
            positions=last.positions,
            read_charges=True
        )

        results = mod.results
        last = results["output_geometry"]
        charge = last.get_initial_charges()

        assert np.allclose(results["energy"]["energy"],-31.3529546, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"],-0.00677337, atol=1e-5)


    def test_coronene_mulliken(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99999,
                            "TOL":3e-7,
                            "GRADTOL":1e-5,
                            "OUT":1,
                            "TRAJECTORY":False
                        },
                    },
                }
            }
        )

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=coronene.symbols,
            positions=coronene.positions,
            read_charges=True
        )

        results = mod.results

        last = results["output_geometry"]

        idx  = [0,1,12,24]
        chrg = [0.004,0.034,-0.0903,0.0716]

        charge = last.get_initial_charges()
        for i,c in zip(idx,chrg):
            assert np.allclose(round(charge[i],3),c, atol=1e-3)

        assert np.allclose(results["energy"]["energy"],-45.94947883, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"],-0.01330096, atol=1e-7)

    @pytest.mark.optional
    def test_coronene_cm3(self):
        
        copy_parameters = copy.deepcopy(parameters)
        copy_parameters.update(
            {
                "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "MAX":99999,
                            "TOL":3e-7,
                            "GRADTOL":1e-5,
                            "OUT":1,
                            "TRAJECTORY":False
                        },},
                }})

        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=coronene.symbols,
            positions=coronene.positions,
            read_charges=True
        )

        results = mod.results

        last = results["output_geometry"]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"CM3":{
                "BONDPARAMS":{
                    "H C":0.06,
                }
            }}
        )


        mod = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **copy_parameters
        )
            
        mod.calculate(
            symbols=last.symbols,
            positions=last.positions,
            read_charges=True
        )

        results = mod.results

        last = results["output_geometry"]
        charge = last.get_initial_charges()

        idx  = [0,1,12,24]
        chrg = [0.0003,0.0317,-0.148,0.131]

        for i,c in zip(idx,chrg):
            assert np.allclose(round(charge[i],3),c, atol=1e-3)



        assert np.allclose(results["energy"]["energy"],-45.95863134, atol=1e-7)
        assert np.allclose(results["energy"]["london_energy"],-0.01329599, atol=1e-7)



