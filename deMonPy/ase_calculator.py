#!/usr/bin/env python3
"""ASE-compatible calculator interface for deMonNano (DFTB).

This module provides :class:`DeMonNano` -- an ASE
:class:`~ase.calculators.calculator.Calculator` subclass that
delegates the actual computation to the :class:`~deMonPy.deMonNano.deMonNano`
engine.

Typical usage::

    from ase.build import molecule
    from deMonPy.ase_calculator import DeMonNano

    atoms = molecule("H2O")
    atoms.calc = DeMonNano(
        execut="/path/to/deMon.x",
        basis={"PTYPE": "BIO", "SKFILE": "/path/to/basis"},
        parameters={"DFTB": {"SCC": True}},
    )

    energy = atoms.get_potential_energy()      # eV
    forces = atoms.get_forces()                # eV / Ang

The calculator converts results from Hartree (deMonNano native unit)
to eV so that they integrate transparently with the rest of ASE.
"""

import os
from copy import deepcopy

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

import deMonPy
from deMonPy.deMonNano import deMonNano
from deMonPy.exceptions import OutputParseError

# Conversion factor: deMonNano reports gradients in Hartree/Bohr.
# ASE expects forces in eV/Angstrom.  F = -gradient.
_FORCE_CONV = Hartree / Bohr


class DeMonNano(Calculator):
    """ASE calculator backed by the deMonNano DFTB engine.

    Supported properties: ``energy``, ``free_energy``, ``forces``,
    ``charges``.

    Attributes:
        implemented_properties: List of properties this calculator can
            provide.
    """

    implemented_properties = ["energy", "free_energy", "forces", "charges"]

    default_parameters = {
        "execut": None,
        "basis": {},
        "omp_threads": 1,
        "prefix": "DEMON",
        "title": "CALCULATION DEMONANO",
    }

    def __init__(
        self,
        restart=None,
        label="ase_demon",
        atoms=None,
        directory=".",
        **kwargs,
    ):
        """Initialise the ASE-compatible deMonNano calculator.

        Args:
            restart: Prefix for restart file.  Currently unused.
            label: Short label used to name the working directory
                under *directory*.  Defaults to ``"demon"``.
            atoms: Optional :class:`~ase.Atoms` object to attach.
            directory: Parent directory for calculation files.
                A sub-directory named *label* is created inside it.
            **kwargs: Calculator-specific options.  Accepted keys:

                * **execut** -- path to the deMonNano executable.
                  Falls back to :data:`deMonPy.DEMON_EXECUTABLE`.
                * **basis** -- dict with ``"PTYPE"`` and ``"SKFILE"``
                  keys.  Falls back to :data:`deMonPy.DEMON_BASIS`.
                * **parameters** -- dict of DFTB parameters placed under
                  ``DEMON_PARAMETERS > ACTIVE`` (e.g.
                  ``{"DFTB": {"SCC": True}}``).
                * **omp_threads** -- OpenMP thread count (default ``1``).
                * **prefix** -- process manager prefix (default
                  ``"DEMON"``).
                * **title** -- input-file title string.
        """
        super().__init__(
            restart=restart,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs,
        )

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=all_changes,
    ):
        """Run a deMonNano calculation and populate :attr:`results`.

        This method is called by ASE every time a property (energy,
        forces, ...) is requested and the cached results are stale.

        Args:
            atoms: The :class:`~ase.Atoms` to compute.  When ``None``
                the previously attached atoms are reused.
            properties: List of property names to compute.
            system_changes: List of changes since the last call.
        """
        super().calculate(atoms, properties, system_changes)

        atoms = self.atoms

        workdir = self._directory

        execut = self.parameters.get("execut") or deMonPy.DEMON_EXECUTABLE
        basis = self.parameters.get("basis") or deMonPy.DEMON_BASIS
        omp_threads = self.parameters.get("omp_threads", 1)
        prefix = self.parameters.get("prefix", "DEMON")
        title = self.parameters.get("title", "CALCULATION DEMONANO")

        # Build the DEMON_PARAMETERS block
        user_params = deepcopy(self.parameters.get("DEMON_PARAMETERS", {}))
        demon_parameters = user_params

        # Determine if forces are requested.  When the user asks for
        # forces deMonNano must write gradients -- this requires the
        # PRINT > GRAD flag.
        need_forces = "forces" in properties
        demon_module = {}
        if need_forces:
            demon_module = {"ACTIVE":{"OPT":{"SP":True}}}

        calc = deMonNano(
            execut=execut,
            workdir=workdir,
            omp_threads=omp_threads,
            prefix=prefix,
            title=title,
            properties=properties,
            basis=basis if isinstance(basis, dict) else {"PTYPE": "BIO", "SKFILE": basis},
            DEMON_PARAMETERS=demon_parameters,
            DEMON_MODULE=demon_module
        )

        calc.calculate(
            symbols=list(atoms.symbols),
            positions=atoms.positions,
        )

        raw = calc.results

        # --- energy (Hartree -> eV) ---
        energy_dict = raw.get("energy", {})
        energy_ha = energy_dict.get("energy", 0.0)
        energy_ev = energy_ha * Hartree

        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        
        # --- charges ---
        out_geom = raw.get("output_geometry", None)
        if out_geom is not None and hasattr(out_geom, "get_initial_charges"):
            charges = out_geom.get_initial_charges()
            if charges is not None:
                self.results["charges"] = np.array(charges)

        # --- forces (Hartree/Bohr -> eV/Ang) ---
        if need_forces:
            forces = raw.get("forces", {})
            if forces is None:
                raise OutputParseError(
                    f"Could not parse the CARTESIAN GRADIENT block for "
                    f"{len(atoms)} atoms in {os.path.join(workdir, 'deMon.out')}. "
                    f"Forces are unavailable; check that the calculation "
                    f"completed and that the 'PRINT GRAD' directive was active."
                )
            self.results["forces"] = -forces
