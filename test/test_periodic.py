import os
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.molden import read_XYZ

deMonPy.configure_from_file(os.path.join("..", "global.json"))

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True, "DISP": 2},
            "PBC": {
                "TYPE": "GENERAL",
                "LATTICE": np.array(
                    [[2.46, 0.0, 0.0], [-1.230000, 2.130422, 0], [0.0, 0.0, 100.0]]
                ),
                "KPTS": [10, 10, 1],
            },
        },
    },
}


WORKDIR = ".run/pbc"


class TestPeriodic:
    def test_periodic_graphen(self):

        cell = np.array(
            [
                [2.460000, 0.000000, 0.000000],
                [-1.230000, 2.130422, 0.000000],
                [0.000000, 0.000000, 100.0000],
            ]
        )
        positions = np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [1.230000, 0.710000, 0.000000],
            ]
        )
        symbols = ["C"] * 2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47255044

    def test_periodic_graphen_bis(self):

        positions = np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [1.230000, 0.710000, 0.000000],
            ]
        )
        symbols = ["C"] * 2

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47255044

    def test_periodic_graphite(self):

        cell = np.array(
            [
                [2.460000, 0.000000, 0.000000],
                [-1.230000, 2.130422, 0.000000],
                [0.000000, 0.000000, 6.700000],
            ]
        )
        positions = np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [1.230000, 0.710000, 0.000000],
            ]
        )
        symbols = ["C"] * 2
        kpts = [10, 10, 10]

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47292185

    def test_periodic_graph_optAtoms(self):

        cell = np.array(
            [
                [2.460000, 0.000000, 0.000000],
                [-1.230000, 2.130422, 0.000000],
                [0.000000, 0.000000, 100.0000],
            ]
        )
        positions = np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [1.230000, 0.710000, 0.000000],
            ]
        )
        symbols = ["C"] * 2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)
        parameter_config.update(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "OPT": {
                            "MAX": 99,
                            "TRAJECTORY": True,
                        },
                    },
                }
            }
        )
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

        results = dem.results
        energy = results["energy"]

        assert energy["energy"] == -3.47255046

    def test_periodic_graph_optCell(self):

        cell = np.array(
            [
                [2.510000, 0.000000, 0.000000],
                [-1.250000, 2.110422, 0.000000],
                [0.000000, 0.000000, 100.0000],
            ]
        )
        positions = np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [1.230000, 0.710000, 0.000000],
            ]
        )
        symbols = ["C"] * 2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)
        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        def to_minimize(factor):
            _cell = cell * factor.T

            dem.calculate(symbols=symbols, positions=positions, cell=_cell, kpts=kpts)

            results = dem.results
            return results["energy"]["energy"]

        res = minimize(
            to_minimize, np.array([1.0, 1.0, 1.0]), method="COBYLA", options={"rhobeg": 0.01}
        )

        ref_cell = np.array(
            [
                [2.460000, 0.000000, 0.000000],
                [-1.230000, 2.130422, 0.000000],
                [0.000000, 0.000000, 100.0000],
            ]
        )
        new_cell = cell * res.x.T

        assert np.allclose(new_cell, ref_cell, rtol=0.01)

    def test_periodic_graphen_layer(self):

        cell = np.array(
            [
                [2.460000, 0.000000, 0.000000],
                [-1.230000, 2.130422, 0.000000],
                [0.000000, 0.000000, 100.0000],
            ]
        )
        positions = np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [1.230000, 0.710000, 0.000000],
            ]
        )
        symbols = ["C"] * 2
        kpts = [10, 10, 1]

        parameter_config = deepcopy(parameters)

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        energies = []

        for kpt in range(1, 25):
            kpts = [kpt, kpt, 1]
            dem.calculate(symbols=symbols, positions=positions, cell=cell, kpts=kpts)

            results = dem.results
            energy = results["energy"]

            energies.append(energy["energy"])

        ref_energies = np.array(
            [
                -3.27894371,
                -3.42075945,
                -3.45312574,
                -3.47430899,
                -3.4721983,
                -3.4703033,
                -3.47284933,
                -3.47246226,
                -3.47177141,
                -3.47255044,
                -3.47244355,
                -3.47213579,
                -3.47246918,
                -3.47242907,
                -3.47226746,
                -3.47243919,
                -3.47242096,
                -3.47232612,
                -3.47242575,
                -3.47241633,
                -3.47235608,
                -3.47241888,
                -3.47241354,
                -3.47237295,
            ]
        )
        assert np.allclose(energies, ref_energies, atol=1e-5)

    def test_periodic_graphen_layer_2(self):

        images, ref = read_XYZ("data_test/graphene.mol")

        image = images[0]
        image.cell = np.array(
            [
                [30.0090007782, 0.0000000000, 0.0000000000],
                [0.0000000000, 34.6514053345, 0.0000000000],
                [0.0000000000, 0.0000000000, 100.0000000000],
            ]
        )
        image.pbc = True

        parameter_config = deepcopy(parameters)
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {"DFTB": {"SCC": True, "DISP": 1, "FERMI": 50, "DIAG": "DSYGVD"}}
        )

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(
            symbols=image.symbols, positions=image.positions, cell=image.cell, kpts=[3, 3, 1]
        )

        results = dem.results
        energy = results["energy"]

        assert np.allclose(energy["energy"], -676.3927574, atol=1e-7)
