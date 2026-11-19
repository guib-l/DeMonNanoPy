#!/usr/bin/env python3
"""Readers of the optional deMonNano modules (opt, ptmc, ptmd, md, neb)."""

import os

import numpy as np

from deMonPy.ioread import safe_parse
from deMonPy.molden import read_XYZ
from deMonPy.profile import assert_flags


class ModulesMixin:
    """Readers of the optional deMonNano modules (opt, ptmc, ptmd, md, neb).

    Mixed into :class:`~deMonPy.demonnano.output.read_output`; relies on the
    state and helpers of :class:`~deMonPy.demonnano.output_base.OutputBase`.
    """

    @safe_parse()
    @assert_flags("ptmd")
    def _read_ptmd(self):
        """Parse PTMD response"""
        for line in self.lines:
            if "ERROR IN MPI_INIT" in line:
                self._add_error("mpi", "ERROR IN MPI_INIT", line=line.strip())
                return

    @safe_parse("optimization")
    @assert_flags("opt")
    def _read_opt(self):
        """Parse optimization-specific output sections."""

        if "doforces" in self.flags:
            forces = []
            filename = os.path.join(self.workdir, "forces.out")
            with open(filename, "r", encoding="utf-8", errors="replace") as fd:
                # forces.out -- first line is a header, then one force vector per line
                for line in fd.readlines()[1:]:
                    forces.append(list(map(float, line.split())))
            self.complet_results["forces"] = np.array(forces)

        self.complet_results["optimization"] = {}
        for line in self.lines:
            # "Maximum gradient =  1.2e-05" -- last token
            if "Maximum gradient =" in line:
                value = float(line.split()[-1])
                self.complet_results["optimization"]["grad_max"] = value
                break

    @safe_parse("ptmc")
    @assert_flags("ptmc")
    def _read_ptmc(self):
        """Parse PTMC-specific output sections."""
        import re

        text = "\n".join(self.lines)
        data = {}

        seed_match = re.findall(r"SEED\s*=\s*((?:\d+\s+)+)", text)
        if seed_match:
            seeds = list(map(int, seed_match[0].split()))
            data["seeds"] = seeds

        patterns = {
            "nb_step": r"NB STEP\s*=\s*(\d+)",
            "optout": r"OPTOUT\s*=\s*(\d+)",
            "nb_temp": r"NB TEMP\s*=\s*(\d+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = int(match.group(1))

        temps = []
        temp_matches = re.findall(r"\n\s*(\d+)\s+([\d\.]+)\s+([\d\.E\-\+]+)", text)

        for idx, t, val in temp_matches:
            temps.append({"index": int(idx), "temperature": float(t), "value": float(val)})

        data["temperatures"] = temps

        exchange = {}

        # 👉 méthode
        method_match = re.search(r"Method\s*:\s*(.+)", text)
        if method_match:
            exchange["method"] = method_match.group(1).strip()

        start_match = re.search(r"Start after\s+(\d+)\s+steps", text)
        if start_match:
            exchange["start_after"] = int(start_match.group(1))

        each_match = re.search(r"Each\s+(\d+)\s+step", text)
        if each_match:
            exchange["each_step"] = int(each_match.group(1))

        prob_match = re.search(r"probability of swap\s*:\s*([\d\.]+)", text)
        if prob_match:
            exchange["swap_probability"] = float(prob_match.group(1))

        data["exchange"] = exchange

        self.complet_results["ptmc"] = data

        _traj = {}
        if "traj" in self.flags:
            for out in range(len(data["temperatures"])):
                output = f"deMon.{(out + 1):02}.mol"
                filename = os.path.join(self.workdir, output)
                data, info = read_XYZ(filename, is_charges=False, velocities=False, keep=1)

                # deMon.NN.mol frame comment -- value = 3rd token (no in-repo sample; unverified)
                energies = np.array(list(map(lambda x: float(x.split()[2]), info)))

                _traj[output] = {
                    "trajectory": data,
                    "energies": energies,
                }

            self.complet_results["trajectory"] = _traj

    @safe_parse()
    @assert_flags("md")
    def _read_md(self):
        """Parse molecular dynamics summary values."""

        energies_flags = {
            "average_potential_energy": "Average POTENTIAL ENERGY",
            "std_potential_energy": "Std Dev POTENTIAL ENERGY",
            "average_kinetic_energy": "Average KINETIC ENERGY",
            "std_kinetic_energy": "Std Dev KINETIC ENERGY",
            "average_total_energy": "Average TOTAL ENERGY",
            "std_total_energy": "Std Dev TOTAL ENERGY",
            "average_temperature": "Average TEMPERATURE",
            "std_temperature": "Std Dev TEMPERATURE",
            "energy_from_thermostat": "Energy transfer from thermostat",
            "energy_loss": "Energy loss from constraints",
        }

        # "Average TOTAL ENERGY : <value> <unit>" -- value = 2nd-to-last token
        # (no in-repo sample of the MD summary; shape unverified)
        for line in self.lines:
            for it, _flag in energies_flags.items():
                if _flag in line:
                    self.complet_results["energy"].update({it: float(line.split()[-2])})

    @safe_parse()
    @assert_flags("neb")
    def _read_neb(self):
        """Parse nudged elastic band output sections.

        NEB output parsing is not implemented yet.  Rather than returning
        silently (which would leave the caller thinking the run was fully
        parsed), an explicit error entry is recorded so that
        :meth:`has_errors` / :meth:`raise_on_errors` surface it.
        """
        self._add_error(
            "neb",
            "NEB output parsing is not implemented yet.",
        )
