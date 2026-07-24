import copy
import os

import numpy as np

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.molden import read_XYZ

deMonPy.configure_from_file(os.path.join("..", "global.json"))

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "MAT", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
            },
            "CM3": {
                "BONDPARAMS": {
                    "C H": 0.10
                },
            }
        },
    },
}


WORKDIR = ".run/PAH/"


class TestPAH:

    def _test_pyren_neutral(self):

        LIMITS = 2

        images,ref = read_XYZ("data_test/pyren-neutral.mol")

        copy_parameters = copy.deepcopy(parameters)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        for i,image in enumerate(images[:LIMITS]):
            
            mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
            mod.calculate(symbols=image.symbols, positions=image.positions)

            results = mod.results
            reference = float(ref[i].split()[2])
            assert np.allclose(results['energy']['energy'], reference, atol=1e-5)
        
    def _test_pyren_cation(self):

        LIMITS = 2

        images,ref = read_XYZ("data_test/pyren-cation.mol")

        copy_parameters = copy.deepcopy(parameters)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        for i,image in enumerate(images[:LIMITS]):

            copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
                {
                    "CHARGE":1,
                    "CI": {
                        "SIZECI": len(image) // 26,
                    },
                    "CUTSYS": {
                        "FRAGMENT": [26,] * (len(image) // 26),
                    },
                }
            )
            mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
            mod.calculate(symbols=image.symbols, positions=image.positions)

            results = mod.results
            reference = float(ref[i].split()[2])
            assert np.allclose(results['energy']['energy'], reference, atol=1e-5)
        
    def _test_coronene_opt(self):

        images,ref = read_XYZ("data_test/coronene-neutral.mol")
        image = images[0]

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["BASIS"]  = {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS}
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CM3INTER": {
                    "BONDPARAMS": {
                        "C H": 0.062
                    },
                },
                "CUTSYS": {
                    "FRAGMENT": [36,] * (len(image) // 36),
                },
            }
        )
        copy_parameters.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 500,
                            "OUT": 1,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)
        mod.calculate(symbols=image.symbols, positions=image.positions)

        results = mod.results
        reference = -91.92818485
        assert np.allclose(results['energy']['energy'], reference, atol=1e-5)



    def _test_pyren_dissociation_ci_path(self):
        import shutil

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "FERMI":10
                },
                "CI": {
                    "SIZECI": 2,
                },
                "CHARGE":1.,
                "CM3":{
                    "BONDPARAMS": {
                        "C H": 0.10,
                    },
                },
                "CUTSYS": {
                    "FRAGMENT": [26,26,] 
                },
                "MOLECULES": {"NAMES": ["PYR", "PYR"]},
                "QUATERNION": {
                    "RIGID": True,
                    "COORDS": np.array(
                        [
                            [0.0, 0.0, 0.00, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 3.20, 1.0, 0., 0., 0.0],
                        ]
                    ),
                },
                "PATHS":{
                    "NPTS":20,
                    "NPTHS":1,
                    "CHKPTS":{
                        "PT1":[
                            [0.0, 0.0,  0.0,  1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0,  2.7 , 1.0, 0.0, 0.0, 0.0],
                        ],
                        "PT2":[
                            [0.0, 0.0, 0.0 , 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 15.5, 1.0, 0.0, 0.0, 0.0],
                        ]
                    }
                }
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        shutil.copy2("./data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

        mod.calculate(symbols=[], positions=[], clean_repository=False)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -61.68572519, atol=1e-7)


    def test_pyren_dissociation_ci(self):
        import shutil

        base_parameters = copy.deepcopy(parameters)
        base_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "FERMI":10
                },
                "CHARGE":1,
                "CI": {
                    "SIZECI": 2,
                },
                "CM3":{
                    "BONDPARAMS": {
                        "C H": 0.10,
                    },
                },
                "CUTSYS": {
                    "FRAGMENT": [26,26,] 
                },
                "MOLECULES": {"NAMES": ["PYR", "PYR"]},
                "QUATERNION": {
                    "RIGID": True,
                    "COORDS": np.array(
                        [
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 15.5, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                }
            }
        )

        energies = []

        for l in np.linspace(2.7,15.5,20):

            copy_parameters = copy.deepcopy(base_parameters)
            copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
                {
                    "QUATERNION": {
                        "RIGID": True,
                        "COORDS": np.array(
                            [
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, l, 1.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    }
                }
            )
            mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

            shutil.copy2("./data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

            mod.calculate(symbols=[], positions=[], clean_repository=False)

            results = mod.results
            energies.append(results["energy"]["energy"])

        references = [-61.69352636, -61.71777376, -61.70297177, -61.6950952, -61.69164836, 
                      -61.68979371, -61.68866458, -61.68791885, -61.68740588, -61.6870312, 
                      -61.68674742, -61.68652548, -61.68634855, -61.6862049, -61.68608672, 
                      -61.68598839, -61.68590581, -61.68583591, -61.68577631, -61.68572519]

        assert np.allclose(references,energies, atol=1e-7)


    def _test_pyren_dissociation_noci(self):
        import shutil

        base_parameters = copy.deepcopy(parameters)
        base_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "DFTB": {
                    "SCC": True,
                    "DISP": 2,
                    "FERMI":10
                },
                "CHARGE":1,
                "CM3":{
                    "BONDPARAMS": {
                        "C H": 0.10,
                    },
                },
                "CUTSYS": {
                    "FRAGMENT": [26,26,] 
                },
                "MOLECULES": {"NAMES": ["PYR", "PYR"]},
                "QUATERNION": {
                    "RIGID": True,
                    "COORDS": np.array(
                        [
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 15.5, 1.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                }
            }
        )

        energies = []

        for l in np.linspace(2.7,15.5,20):

            copy_parameters = copy.deepcopy(base_parameters)
            copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
                {
                    "QUATERNION": {
                        "RIGID": True,
                        "COORDS": np.array(
                            [
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, l, 1.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    }
                }
            )

            mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

            shutil.copy2("./data_test/MOLECULES", f"{WORKDIR}/MOLECULES")

            mod.calculate(symbols=[], positions=[], clean_repository=False)

            results = mod.results
            energies.append(results["energy"]["energy"])

        references = [-61.70116668, -61.73017249, -61.71991806, -61.71450774, -61.71288696, 
                      -61.71256938, -61.71278119, -61.71320952, -61.71371839, -61.71424604, 
                      -61.71476405, -61.71525957, -61.71572739, -61.71616608, -61.71657607, 
                      -61.71695871, -61.71731571, -61.71764895, -61.71796025, -61.71825139]
        assert np.allclose(references,energies, atol=1e-7)





