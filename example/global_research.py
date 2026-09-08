import sys

import configs
import numpy as np
import copy
from ase.atoms import Atoms

import os
from os.path import isfile, join

import deMonPy
from deMonPy.deMonNano import Module_DeMonNano
from deMonPy.molden import read_XYZ,write_XYZ,progressbar


deMonPy.configure_from_file("global.json")

parameters = {
    "DEMON_EXECUTABLE": deMonPy.DEMON_EXECUTABLE,
    "BASIS": {"PTYPE": "BIO", "SKFILE": deMonPy.DEMON_BASIS},
    "DEMON_PARAMETERS": {
        "ACTIVE": {
            "DFTB": {"SCC": True},
            "CHARGE": 0.0,
        },
    },
}

image = Atoms(
    [
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "H",
        "Ar",
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
        ]
    ),
)



WORKDIR = ".run/ptmc/"


def exemple_run_ptmc(max=1000):

    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "POLA": True,
                "NOPOLQM": True,
            },
            "CUTSYS": {"FRAGMENT": [20, 1], "RIGID": True},
            "RG": {
                "COUPLING": "ARGON",
                "ALPHARG": 11.07,
            },
        }
    )

    mod = Module_DeMonNano(
        module="ptmc",
        title="CALCULATION DEMONANO",
        workdir=WORKDIR,
        **parameter_config,
    )

    # Run PTMC
    mod(
        image=image,
        max=max,
        seed=True,
        out=1,
        wall=8.0,
        temperature=None,
        n_temp=10,
        max_temp=300,
        min_temp=30,
        rescale=10,
        distribution_temp="geom",
        temp_list=None,
        restart=False,
    )



def exemple_run_mc():

    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "POLA": True,
                "NOPOLQM": True,
            },
            "CUTSYS": {"FRAGMENT": [20, 1], "RIGID": True},
            "RG": {
                "COUPLING": "ARGON",
                "ALPHARG": 11.07,
            },
        }
    )
    mod = Module_DeMonNano(
        module="ptmc",
        title="CALCULATION DEMONANO",
        workdir=WORKDIR,
        **parameter_config,
    )

    # Run PTMC
    mod(
        method="mc",
        image=image,
        max=30,
        temperature=30
    )



def select_strutures(directory, without=["deMon.mol","deMon.keep.mol"]):

    import random
    method = lambda tot: random.sample(range(tot),k=2)

    if isinstance(directory,list):
        files = directory
    elif os.path.isdir(directory):
        files = [f for f in os.listdir(directory) if isfile(join(directory, f))]
        files = [f for f in files if f.endswith(".xyz") or f.endswith(".mol")]
    else:
        files = directory if isinstance(directory,list) else [directory]

    picked_structures = []

    for f in files:
        if f in without:
            continue
        try:
            img,info = read_XYZ(join(directory, f), is_charges=False)
        except Exception as e:
            continue

        if img is None or len(img) <= 1:
            continue

        index = method(len(img))
        picked_structures += [img[i] for i in index]
        
    return picked_structures


def exemple_run_opt(image):

    WORKDIR = ".run/opt"

    parameter_config = copy.deepcopy(parameters)
    parameter_config["DEMON_PARAMETERS"]["ACTIVE"].update(
        {
            "DFTB": {
                "SCC": True,
                "DISP": 2,
                "POLA": True,
                "NOPOLQM": True,
            },
            "RG": {
                "COUPLING": "ARGON",
                "ALPHARG": 11.07,
            },
        }
    )
    mod = Module_DeMonNano(
        module="opt",
        title="CALCULATION DEMONANO",
        workdir=WORKDIR,
        **parameter_config,
    )

    mod(image=image, max=9999)

    return mod.results



if __name__ == "__main__":

    # ====================================================
    # Monte Carlo optimization
    exemple_run_mc()
    print(" \u2705 MC DABCO-Argon : Done")

    img,info = read_XYZ(join(".run/ptmc", "deMon.01.mol"), is_charges=False)
    energy = [float(_l.split()[2]) for _l in info]
    index = np.argmin(energy)

    print(" The lowest isomer found : ")
    print(" Isomer :", img[index], " at ",energy[index], " Ha.")
    


    # ====================================================
    # Global optimization with PTMC

    exemple_run_ptmc(200)
    print(" \u2705 PTMC DABCO-Argon : Done")

    try:
        os.rmdir("example/optimized.xyz")
    except:
        pass

    for img in progressbar(select_strutures(".run/ptmc"), 
                           size=40,
                           prefix=" \u23F3 Otimization : "):

        res = exemple_run_opt(img)

        write_XYZ("example/optimized.xyz", 
                  intent="a", 
                  images=[res["output_geometry"]],
                  energy=[res["energy"]["energy"]])
    print(" \u2705 OPT DABCO-Argon : Done")
    

    sys.exit()
