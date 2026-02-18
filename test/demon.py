#!/usr/bin/env python3
import __future__

import os
import sys

import configs

import deMonPy
from deMonPy import deMonEnviron


if __name__=='__main__':

    env = deMonEnviron(
        workdir='.run',
        config='config.json',
        omp_threads=1,
    )

    print(env)


    




    sys.exit()





