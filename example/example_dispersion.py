import copy

import numpy as np
from ase.atoms import Atoms

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.molden import read_XYZ

deMonPy.configure_from_file("global.json")

base_parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
            "CHARGE": 0.0,
        },
    },
}
pyrene = Atoms(
    ["C"] * 16 + ["H"] * 10,
    positions=np.array(
        [
            [-0.683772110668018, -2.464450400481512, 0.000137010366981],
            [0.683758149546043, -2.464452106873366, 0.000144287121293],
            [-1.425886631945578, -1.234520387271399, -0.000003726798454],
            [-2.833921978167532, -1.213463038184029, -0.000076617361486],
            [1.425879792605808, -1.234525151200887, 0.000012543048018],
            [2.833920291494441, -1.213472214880341, -0.000062840843575],
            [0.712731140559170, 0.000001994258728, -0.000012782776036],
            [1.425889306046169, 1.234523765298982, 0.000007511395611],
            [-0.712732208473147, 0.000001730818731, -0.000022083330337],
            [-1.425879331937127, 1.234527658750873, 0.000008368807763],
            [-2.833912360736262, 1.213469603305815, -0.000074995504814],
            [-3.523043716450025, 0.000001220562157, -0.000152729675698],
            [2.833919284775448, 1.213464458665507, -0.000085975197347],
            [3.523040976921263, -0.000004276157801, -0.000155303464147],
            [0.683766324084488, 2.464448346843710, 0.000142623670697],
            [-0.683757080727037, 2.464451024752131, 0.000151871431858],
            [-3.389636837273564, 2.161889472722259, -0.000069042546793],
            [3.389650610638267, 2.161866769716537, -0.000114326869174],
            [1.238166649916377, 3.414116743710609, 0.000297606732403],
            [4.621875922913928, -0.000008425048080, -0.000235659460241],
            [-1.238162092848822, 3.414121931293944, 0.000326067568435],
            [-4.621877572198442, 0.000011763208689, -0.000210634054435],
            [-1.238168594975591, -3.414125341235799, 0.000316803977444],
            [1.238160480221511, -3.414127402491622, 0.000319489819651],
            [3.389638749614898, -2.161895869473155, -0.000061414702835],
            [-3.389645492053683, -2.161876192996518, -0.000058433573312],
        ]
    ),
)


WORKDIR = ".run/examples-disp/"


def exemple_run_dispersion():

    parameter_config = copy.deepcopy(base_parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].update({"DISP": 2})
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "CM3": {
                "BONDPARAMS": {"C H": 0.10},
            }
        }
    )

    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

    dem.calculate(symbols=pyrene.symbols, positions=pyrene.positions)

    results = dem.results

    energy = results["energy"]
    print(" =======================================")
    print(f" > Energy totale     : {energy['energy']}")
    print(f" > Energy electronic : {energy['electronic_energy']}")
    print(f" > Energy repulsion  : {energy['repulsive_energy']}")
    print(f" > Energy dispersion : {energy['london_energy']}")


def exemple_run_cluster_pyrene():

    images, _ = read_XYZ("./example/data_test/pyrene-2_cation.xyz")
    pyrenes = images[-1]

    parameter_config = copy.deepcopy(base_parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].update({"DISP": 2})
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "CM3": {
                "BONDPARAMS": {"C H": 0.10},
            }
        }
    )

    dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

    dem.calculate(symbols=pyrenes.symbols, positions=pyrenes.positions)

    results = dem.results

    energy = results["energy"]
    print(" =======================================")
    print(f" > Energy totale     : {energy['energy']}")
    print(f" > Energy electronic : {energy['electronic_energy']}")
    print(f" > Energy repulsion  : {energy['repulsive_energy']}")
    print(f" > Energy dispersion : {energy['london_energy']}")


def exemple_run_pyrene():
    import os,sys
    print(" =======================================")

    for elm in range(1, 8):
        parameter_config = copy.deepcopy(base_parameters)
        parameter_config.update({"BASIS": {"PTYPE": "MAT", "SKFILE": deMonPy.DEMON_BASIS}})
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"]["DFTB"].update({"DISP": 2})
        parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
            {
                "CHARGE": 1.0,
                "CM3": {
                    "BONDPARAMS": {"C H": 0.10},
                },
                "CI": {
                    "SIZECI": elm,
                },
                "CUTSYS": {
                    "FRAGMENT": [
                        26,
                    ]
                    * elm
                },
            }
        )

        images, _ = read_XYZ(f"./example/data_test/pyrenes/{elm}.mol")
        pyrene = images[0]

        dem = deMonNano(title="CALCULATION DEMONANO", workdir=WORKDIR, **parameter_config)

        dem.calculate(symbols=pyrene.symbols, positions=pyrene.positions, read_charges=True)

        results = dem.results
        energy = results["energy"]

        dem.print_results()
        with open(os.path.join(WORKDIR,"deMon.out"),'r') as fd:
            for line in fd.readlines():
                pass

        charges = results["output_geometry"].get_initial_charges().reshape((elm,26))
        elm_charges = np.sum(charges,axis=1)
        for i,chrg in enumerate(elm_charges):
            print(f"  - Charge n°{i} : {chrg}")

        print(f" Structure with {elm} pyrenes")
        print(f" > Energy totale     : {energy['energy']}")

        if elm==2:
            sys.exit()


if __name__ == "__main__":
    exemple_run_dispersion()

    exemple_run_cluster_pyrene()

    exemple_run_pyrene()
