#!/usr/bin/env python3
import os
import subprocess

import deMonPy




def read_json(filename=""):
    import json
    data = {}
    try:
        with open(filename, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")
    return data


class ExecuteFailed(Exception):
    """When execution fails"""


class Process:

    def __init__(
            self, 
            executable,
            workdir="",
            omp_threads=1,
            prefix='DEMON',
            system=True):

        self.command = "cd %s && "%workdir + executable 
        self.system  = system
        self.prefix  = prefix
        self.omp_threads = omp_threads


    def execute(self):

        os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)

        if self.command is None:
            raise EnvironmentError("Unknow command")
        
        
        command = self.command

        if 'PREFIX' in command:
            command = command.replace('PREFIX', self.prefix)

        if self.system:
            os.system( command )

        else:
            try:
                proc = subprocess.Popen(command, shell=True, cwd="./")
            except OSError as err:

                msg = f'Failed to execute "{command}"'
                raise EnvironmentError(msg) from err

            errorcode = proc.wait()

            if errorcode:
                path = os.path.abspath("./")
                msg = (
                    'Calculator "{}" failed with command "{}" failed in '
                    '{} with error code {}'.format(
                        self.prefix, command, path, errorcode
                    )
                )
                raise ExecuteFailed(msg)








