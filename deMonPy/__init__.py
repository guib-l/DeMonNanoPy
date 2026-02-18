import os
import json







class deMonEnviron:

    def __init__(
            self,
            *,
            workdir=".",
            config=None,
            omp_threads=1,
            **kwargs):
        
        self.workdir = workdir
    

    def build(self):
        ...

    def setup_environment(self):
        ...







