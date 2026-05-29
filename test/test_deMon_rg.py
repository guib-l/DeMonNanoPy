
import os
import sys

import numpy as np

from copy import deepcopy

import pytest
from ase.atoms import Atoms

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
                "SCC":True
            },
            "CHARGE":0.0,
        },
    }
}

image = Atoms(
    ["C","C","N","C","C","N","C","C","H",
     "H","H","H","H","H","H","H","H","H",
     "H","H","Ar"],
    positions=np.array([
            [ 1.188367820025961, -0.791963696891490, -0.110257297286295],
            [ 0.760376820025962, -0.020468696891491,  1.213427702713705],
            [-0.116809179974038,  1.033744303108510,  0.816564702713705],
            [-1.312762179974039,  0.526830303108510,  0.224073702713705],
            [-0.884737179974039, -0.244625696891490, -1.099629297286295],
            [ 0.535658820025961, -0.142239696891490, -1.201197297286295],
            [ 0.964475820025962,  1.212058303108510, -1.350483297286295],
            [ 0.536476820025961,  1.983416303108510, -0.027019297286295],
            [ 2.293157820025963, -0.757119696891491, -0.239642297286295],
            [ 0.868695820025961, -1.853187696891491, -0.046486297286295],
            [ 0.238654820025961, -0.717377696891490,  1.902302702713705],
            [ 1.656376820025962,  0.390792303108509,  1.729724702713705],
            [-1.820598179974038, -0.173774696891491,  0.919554702713705],
            [-2.010428179974038,  1.358783303108510, -0.020114297286295],
            [-1.190464179974038, -1.309529696891490, -1.029229297286295],
            [-1.373583179974038,  0.211010303108510, -1.989537297286295],
            [ 2.067746820025961,  1.256872303108509, -1.484343297286295],
            [ 0.483872820025962,  1.675070303108510, -2.240302297286294],
            [-0.152207179974038,  2.821315303108510, -0.273552297286295],
            [ 1.431709820025961,  2.403197303108510,  0.482362702713705],
            [-1.274156179974038, -2.711962696891490,  1.170045702713705],
        ])
    )



WORKDIR = ".run/dftbrg/"


class TestDFTB_Rg:

    @pytest.mark.bird
    def test_argon(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "DFTB":{
                    "SCC":True,
                    "DISP":2,
                    "POLA":True,
                    "NOPOLQM":True,
                },
                "RG":{
                    "COUPLING":"ARGON",
                    "ALPHARG":11.07,
                    "FILENAME":None
                }
            }
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        
        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7) 


    @pytest.mark.bird
    def test_argon_derivative(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "DFTB":{
                    "SCC":True,
                    "DISP":2,
                    "POLA":True,
                    "NOPOLQM":True,
                },
                "RG":{
                    "COUPLING":"ARGON",
                    "ALPHARG":11.07,
                    "FILENAME":None
                }
            }
        )
        parameter_config.update(
            {   "DEMON_MODULE":{
                    "ACTIVE":{
                        "OPT":{
                            "SP":True,
                        },
                    }, } } )

        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results

        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7)


    @pytest.mark.bird
    @pytest.mark.xfail(reason="NOT CRITICAL -> TO FIX")
    def test_argon_read(self):

        parameter_config = deepcopy(parameters)
        parameter_config['DEMON_PARAMETERS']['ACTIVE'].update(
            {
                "DFTB":{
                    "SCC":True,
                    "DISP":2,
                    "POLA":True,
                    "NOPOLQM":True,
                },
                "RG":{
                    "COUPLING":"READ",
                    "ALPHARG":11.07,
                    "FILENAME":'./data_test/filename.dat'
                }
            }
        )
        dem = deMonNano(
            title="CALCULATION DEMONANO",
            workdir=WORKDIR,
            **parameter_config
        )

        dem.calculate(
            symbols=image.symbols,
            positions=image.positions
        )

        results = dem.results
        
        assert np.allclose(results["energy"]["energy"], -19.51939925, atol=1e-7) 


















