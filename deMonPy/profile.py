#!/usr/bin/env python3
import os
import subprocess

import deMonPy



class profile_ASE:

    def __init__(self, command, system=True):

        self.command = command
        self.system  = system
        self.configvars = {}

    def execute(self,calc):

        prefix = 'DEMON'

        if self.command is None:
            raise EnvironmentError(
                'Please set ${} environment variable '.format(
                    'ASE_' + prefix.upper() + '_COMMAND'
                )
                + 'or supply the command keyword'
            )
        command = self.command
        if 'PREFIX' in command:
            command = command.replace('PREFIX', prefix)

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
                        prefix, command, path, errorcode
                    )
                )
                raise CalculationFailed(msg)








