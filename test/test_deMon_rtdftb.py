import copy
import os

import numpy as np
import pytest

import deMonPy
from deMonPy.deMonNano import deMonNano

deMonPy.configure_from_file(os.path.join("..", "global.json"))

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "MAT", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
            }
        },
    },
}


WORKDIR = ".run/rtdftb/"


positions = np.array(
    [
        [-1.73285200000000, -3.36104900000000, 2.96902200000000],
        [-1.69544600000000, 0.288990000000000, 0.277373000000000],
        [-2.23577300000000, -1.30990800000000, -1.83152900000000],
        [-2.36313200000000, -0.184144000000000, -4.18288200000000],
        [0.455692000000000, -2.59211500000000, 1.77269800000000],
        [0.211189000000000, -1.57350600000000, -0.717333000000000],
        [2.38612300000000, -0.613224000000000, -1.99981700000000],
        [-1.39774800000000, -0.782792000000000, 2.73912400000000],
        [-0.583086000000000, 4.34215500000000, 1.98005900000000],
        [0.628119000000000, 1.21781700000000, -1.07590200000000],
        [2.204800000000000e-002, -0.407691000000000, -3.14680300000000],
        [-1.01173700000000, 1.80323000000000, 2.40707500000000],
        [-2.02400400000000, -2.37662200000000, 0.569303000000000],
        [1.24652000000000, 2.70594400000000, 1.09244500000000],
        [0.856043000000000, 6.687200000000000e-002, 1.51589200000000],
        [-1.23263500000000, 2.92205600000000, -0.111302000000000],
        [-1.83022900000000, 1.40252700000000, -2.18021800000000],
        [4.67892100000000, -0.796699000000000, -0.766229000000000],
        [3.01297600000000, 0.980847000000000, 0.170411000000000],
        [2.60817200000000, -1.73131100000000, 0.518719000000000],
    ]
)
symbols = ["Au"] * 20


class TestRTDFTB:
    @pytest.mark.xfail(reason="TO BE CONTINUE")
    def test_tddftb(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["BASIS"] = {"PTYPE": "", "SKFILE": ""}
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"DFTB": {"SCC": True}, "TD-DFTB": {"LRESP": 25, "NO_TRIP": True}}
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=symbols, positions=positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -57.09364137, atol=1e-7)

    @pytest.mark.xfail(reason="TO BE CONTINUE")
    def test_basic_rtdftb(self):

        copy_parameters = copy.deepcopy(parameters)
        copy_parameters["BASIS"] = {"PTYPE": "", "SKFILE": ""}
        copy_parameters["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"DFTB": {"SCC": True}, "RTTDDFTB": {"KICK": 0.003, "KICKAXIS": 1}}
        )
        copy_parameters["DEMON_MODULE"] = {"ACTIVE": {}}
        copy_parameters["DEMON_MODULE"]["ACTIVE"].update(
            {
                "MD": {
                    "TIMESTEP": 0.05,
                    "MDSTEP": {"MAX": 2000, "OUT": 100, "SOUT": 100},
                    "TRAJECTORY": True,
                },
            }
        )

        mod = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **copy_parameters)

        mod.calculate(symbols=symbols, positions=positions)

        results = mod.results
        assert np.allclose(results["energy"]["energy"], -20.7065151, atol=1e-7)
