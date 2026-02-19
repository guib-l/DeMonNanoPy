#!/usr/bin/env python3
import __future__

import os
import sys

import configs

import deMonPy
from deMonPy.deMonNano import deMonNano


if __name__=='__main__':

    dem = deMonNano(
        title="CALCULATION DEMONANO",
        basis={},
        execut="~/Documents/dev_deMon/deMon.x",
        workdir=".run/",
    )

    print(dem)


    




    sys.exit()





