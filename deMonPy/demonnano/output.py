#!/usr/bin/env python3
"""deMonNano output reader (deMon.out and deMon.mol).

The reader is split over several modules of this package:

- ``output_formats``: line formats (``LineFormat``) and format constants;
- ``output_base``:    ``OutputBase``, file loading and parse helpers;
- ``output_coef``:    the deMon.coef reader;
- ``output_modules``: readers of the optional modules (opt, ptmc, ptmd, md, neb).
"""

import os
import re

import numpy as np

from deMonPy.demonnano.output_base import OutputBase, _is_int, convert_float
from deMonPy.demonnano.output_coef import read_coef
from deMonPy.demonnano.output_formats import (
    AOM_DATA_START,
    BASICS_COUNT,
    CI_ATOM_CHARGE,
    CI_CONF_WEIGHT,
    CI_STATE,
    FREQ_ATOMS,
    FREQ_FREQUENCY,
    FREQ_INTENSITY,
    FREQ_MODE,
    FREQ_MODE_HEADER_LINES,
    FREQ_VECT,
    FREQ_ZPE,
    GAMMA_DATA_START,
    PBC_EIG_SCAN_WINDOW_FACTOR,
    PBC_EIG_TOKEN_COUNT,
    PBC_KCOORD_LOOKBACK,
    TDDFTB_ARROW,
    TDDFTB_ARROW_TOKEN,
    TDDFTB_NTRANSITIONS,
    TDDFTB_SUMMARY_HEADER_LINES,
    TDDFTB_TRANSITION,
    XYZ_INFO_KINETIC_COL,
    XYZ_INFO_POTENTIAL_COL,
    XYZ_INFO_TIME_COL,
    XYZ_INFO_TOTAL_COL,
)
from deMonPy.demonnano.output_modules import ModulesMixin
from deMonPy.exceptions import OutputParseError
from deMonPy.ioread import safe_parse
from deMonPy.molden import read_XYZ
from deMonPy.profile import assert_flags, exclude_flags


class read_output(ModulesMixin, OutputBase):
    """Parser for deMonNano output files and derived results."""

    _criteria_energy_str = {
        "DFTB total energy": "energy",
        "DFTB electronic energy": "electronic_energy",
        "DFTB band energy": "band_energy",
        "DFTB repulsive energy": "repulsive_energy",
        "DFTB Coulomb energy": "coulomb_energy",
        "DFTB London energy": "london_energy",
        "DFTB MM+Mechanical Coupl.": "MM_coupl.",
        "DFTB electronic entropy": "electronic_entropy",
        "DFTB Polarisation energy": "polarisation_energy",
        "DFTB HOMO-LUMO gap": "HOMO-LUMO_gap",
        "DFTB (HOMO)-(HOMO-1) gap": "HOMO_gap",
        "DFTB Fermi energy level": "fermi_energy",
        "DFTB third order Coulomb energy": "3d_coulomb_energy",
    }
    _criteria_mm = {
        "TOTAL ENERGY                =": "qmmm-energy",
        "VAN DER WAALS ENERGY": "qmmm-VdW_energy",
        "ELECTROSTATIC CHARGE ENERGY": "qmmm-electrostatic_energy",
        "UREY BRADLEY ENERGY": "qmmm-UB_energy",
        "IMPROPER TORSION ENERGY": "qmmm-improper_torsion_energy",
        "TORSION ANGLE ENERGY": "qmmm-torsion_angle_energy",
        "BOND ANGLE ENERGY": "qmmm-bond_angle_energy",
        "BOND ENERGY": "qmmm-bond_energy",
    }

    # read_basics: "marker" -> (key, regex, match index, converter)
    _basics_scc = {
        # "Requested SCC tolerance :  0.10000E-07"
        "Requested SCC tolerance": ("scc_tolerance", r"[-+]?\d+\.\d+E[-+]?\d+", 0, float),
        # "Maximum number of SCC cycles :  200"
        "Maximum number of SCC cycles": ("max_scc_cycles", r"\d+", -1, int),
        # "SCC mixing paramter :   0.30"
        "SCC mixing": ("scc_mixing", r"[-+]?\d*\.\d+", 0, float),
    }
    # read_basics: "marker : <int>" -> key
    _basics_counts = {
        "Number of DFTB atoms": "n_atoms",
        "Number of DFTB atom types": "n_atom_types",
        "Dimension of DFTB matrices": "matrix_dim",
        "Number of DFTB electrons": "n_electrons",
        "Number of DFTB shells": "n_shells",
    }

    # =================================
    # READ ENERGY (basics)

    @safe_parse("properties")
    def read_basics(self):
        """Read basics things"""

        data = {
            "scc_tolerance": None,
            "max_scc_cycles": None,
            "scc_mixing": None,
            "n_atoms": None,
            "n_atom_types": None,
            "matrix_dim": None,
            "n_electrons": None,
            "n_shells": None,
        }

        for line in self.lines:
            line = line.strip()

            # -------- SCC PARAMS --------
            for marker, (key, pattern, pick, conv) in self._basics_scc.items():
                if marker in line:
                    val = re.findall(pattern, line)
                    if val:
                        data[key] = conv(val[pick])
                    break

            # -------- ELECTRONIC INFO --------
            else:
                for marker, key in self._basics_counts.items():
                    if marker in line:
                        row = self._parse_line(line, BASICS_COUNT, f"read_basics: '{marker}'")
                        data[key] = row["value"]
                        break

        self.complet_results["properties"] = data

    @safe_parse("energy")
    def read_energy(self):
        """Parse energy values from the loaded output lines."""

        start_search = False

        _state = {}
        self.complet_results["energy"] = {}

        for line in self.lines:
            _energy = self.get_energies(line, start_search=start_search)
            _state.update(_energy)

            if "energy" in self.complet_results.keys():
                start_search = True
        self.complet_results["energy"].update(_state)

    def get_energies(self, line, criteria="DFTB total energy", start_search=False):
        """Extract energy terms from a single output line.

        Args:
            line: Line to inspect.
            criteria: Primary energy marker used to seed parsing.
            start_search: Whether extended energy parsing is active.

        Returns:
            dict: Parsed energy terms found on the line.
        """

        _energy = {}

        _criteria_energy_str_bis = self._criteria_energy_str

        if "mm" in self.flags:
            _criteria_energy_str_bis = self._criteria_mm

        if "qmmm" in self.flags:
            # copy: updating the class-level dict would leak the MM markers
            # into every later read_output instance
            _criteria_energy_str_bis = {**self._criteria_energy_str, **self._criteria_mm}

        # "DFTB total energy         [Hartree]       =        -8.06209343" -- last token
        if self.is_inside(criteria, line):
            label = _criteria_energy_str_bis[criteria]
            _energy[label] = self.get_float(line, index=-1)

        if start_search:
            for cs, it in _criteria_energy_str_bis.items():
                if self.is_inside(cs, line):
                    _energy[it] = self.get_float(line, index=-1)
        return _energy

    # =================================
    # READ GEOMETRY (basics)

    @safe_parse("input_geometry", "output_geometry")
    @exclude_flags(["ptmc", "freq", "ptmd", "rttddftb"])
    def read_geometry(self, output="deMon.mol", is_charges=False, velocities=False, keep=1):
        """Read input, output, or trajectory geometries.

        Args:
            output: Geometry file name.
            is_charges: Whether the XYZ reader should parse charges.
            keep: Geometry sampling interval.

        Raises:
            OutputParseError: If the geometry file is missing (typically
                when the calculation crashed before emitting it).
        """

        filename = os.path.join(self.workdir, output)
        if not os.path.isfile(filename):
            raise OutputParseError(
                f"Geometry file not found at {filename}. The calculation likely did not complete."
            )
        try:
            data, info = read_XYZ(
                filename,
                is_charges=is_charges,
                velocities=velocities,
                keep=keep,
                ase_obj=self._ase_obj,
            )
        except (OSError, ValueError) as err:
            raise OutputParseError(f"Failed to parse geometry file {filename}: {err}") from err

        if len(data) == 1:
            if "opt" in self.flags:
                self.complet_results["input_geometry"] = data[0]
                self.complet_results["output_geometry"] = data[0]

        if len(data) == 2:
            self.complet_results["input_geometry"] = data[0]
            self.complet_results["output_geometry"] = data[-1]

        if len(data) > 2:
            self.complet_results["input_geometry"] = data[0]
            if "traj" in self.flags:
                self.complet_results["trajectory"] = data
            self.complet_results["output_geometry"] = data[-1]

        if "md" in self.flags:
            self.complet_results["time"] = info[:, XYZ_INFO_TIME_COL]
            self.complet_results["potential_energy"] = info[:, XYZ_INFO_POTENTIAL_COL]
            self.complet_results["kinetic_energy"] = info[:, XYZ_INFO_KINETIC_COL]
            self.complet_results["total_energy"] = info[:, XYZ_INFO_TOTAL_COL]

    # =================================
    # READ PARAMETERS (basics)

    @safe_parse("states", "charges")
    @assert_flags("ci")
    def read_ci(self):
        """Parse configuration interaction states and configurations."""
        start_search = True
        state_search = False

        _conf = {}
        count = 0
        state = {}

        self.complet_results["states"] = {}

        for line in self.lines:
            _energy = self.get_energies(line, start_search=start_search)
            _conf.update(_energy)

            if "*********   CONFIGURATIONS   *********" in line:
                if count > 0:
                    self.complet_results[f"configuration_{count}"] = _conf
                count += 1
                _conf = {}
                self.complet_results["energy"] = {"energy": None}

            if self.is_inside("************    STATES    ************", line):
                state_search = True

            if self.is_inside("for state", line) and state_search:
                row = self._parse_line(line, CI_STATE, "read_ci")
                num = row["num"]
                wgh, chrg_mulliken, chrg_cm3 = [], [], []
                state = {f"state {num}": {"energy": row["energy"]}}

            if self.is_inside("weight of conf", line) and state_search:
                row = self._parse_line(line, CI_CONF_WEIGHT, "read_ci", strict=False)
                wgh.append(None if row is None else row["weight"])
                state[f"state {num}"].update({"weight": wgh})

            if self.is_inside("molecule", line) and state_search:
                row = self._parse_line(line, CI_ATOM_CHARGE, "read_ci", strict=False)
                chrg = chrg_mulliken
                label = "charges_mulliken"
                if len(chrg_mulliken) == len(wgh):
                    chrg = chrg_cm3
                    label = "charges_cm3"
                chrg.append(None if row is None else row["charge"])

                state[f"state {num}"].update({label: chrg})

            self.complet_results["states"].update(state)

        self.complet_results[f"configuration_{count}"] = _conf
        self.complet_results["charges"] = {}

        if "states" in self.complet_results.keys():
            if len(self.complet_results["states"].keys()) > 0:
                self.complet_results.pop("energy")

                # deMon numbers states from 1; np.argmin is 0-based
                idx = (
                    np.argmin(
                        [state["energy"] for k, state in self.complet_results["states"].items()]
                    )
                    + 1
                )

                energies = self.complet_results["states"][f"state {idx}"]["energy"]
                self.complet_results["energy"] = {"energy": energies}

                charges = self.complet_results["states"][f"state {idx}"]["charges_mulliken"]
                self.complet_results["charges"].update({"charges mulliken": charges})

                if "charges_cm3" in self.complet_results["states"][f"state {idx}"]:
                    charges = self.complet_results["states"][f"state {idx}"]["charges_cm3"]
                    self.complet_results["charges"].update({"charges cm3": charges})

    @safe_parse("triplet", "singlet")
    @assert_flags("td-dftb")
    def read_tddftb(self):
        """Parse TD-DFTB singlet and triplet transitions."""

        msg = "LINEAR RESPONSE FOR CLOSED SHELL MOLECULES ONLY."
        for line in self.lines:
            if msg in line:
                self._add_error("tddftb", msg, line=line.strip())
                return

        N = None
        for line in self.lines:
            if self.is_inside("requested transitions to calculate", line):
                N = self._parse_line(line, TDDFTB_NTRANSITIONS, "read_tddftb")["value"]
                break
        if N is None:
            raise OutputParseError(
                "read_tddftb: 'requested transitions to calculate' line not found; "
                "cannot size the SUMMARY blocks"
            )

        text = "\n".join(self.lines)
        for key, signature in zip(["triplet", "singlet"], ["SUMMARY TRIPLET:", "SUMMARY SINGLET"]):
            # triplet and singlet states are each numbered from state_0
            count = 0
            args = {}
            if signature not in text:
                continue
            for i, line in enumerate(self.lines):
                self.compute_block(line, signature, "", nb_line=N + TDDFTB_SUMMARY_HEADER_LINES)
                if not self._tocken_block:
                    continue
                # every block line is tried (headers fail silently); a parse
                # error is recorded only for lines that look like transitions
                values = line.split()
                row = self._parse_line(
                    line,
                    TDDFTB_TRANSITION,
                    f"read_tddftb: '{signature}'",
                    strict=False,
                    record=len(values) > TDDFTB_ARROW_TOKEN
                    and values[TDDFTB_ARROW_TOKEN] == TDDFTB_ARROW,
                )
                if row is not None:
                    args[f"state_{count}"] = row
                    count += 1
            self.complet_results[key] = args

    @safe_parse("frequency", "zpe")
    @assert_flags("freq")
    def read_freq(self):
        """Parse vibrational frequency results."""
        freq, temp = [], []
        zpe, Nbatm = 0.00, 0

        for line in self.lines:
            if "NUMBER OF ATOMS:" in line:
                Nbatm = self._parse_line(line, FREQ_ATOMS, "read_freq")["value"]
            if "ZERO-POINT ENERGY =" in line:
                zpe = self._parse_line(line, FREQ_ZPE, "read_freq")["value"]

            self.compute_block(line, "MODE:", "", FREQ_MODE_HEADER_LINES + Nbatm)

            if self.counter == 0 and self._block != "":
                temp.append(self._block)
                self._block = ""

        for frq in temp:
            mode, frequency, intensity = 0, 0.0, 0.0

            for line in frq.split("\n"):
                if self.is_inside(control="MODE:", line=line):
                    mode = self._parse_line(line, FREQ_MODE, "read_freq")["value"]
                if self.is_inside(control="FREQUENCY:", line=line):
                    frequency = self._parse_line(line, FREQ_FREQUENCY, "read_freq")["value"]
                if self.is_inside(control="INTENSITY:", line=line):
                    intensity = self._parse_line(line, FREQ_INTENSITY, "read_freq")["value"]

            table = []
            rows = frq.split("\n")[FREQ_MODE_HEADER_LINES:-1]
            self._check(
                len(rows) >= Nbatm,
                f"read_freq: MODE block expects {Nbatm} atom-displacement rows "
                f"({FREQ_MODE_HEADER_LINES} header lines + {Nbatm} atom rows), "
                f"found {len(rows)}",
            )
            for lin in rows:
                row = self._parse_line(lin, FREQ_VECT, "read_freq")
                table.append([row["x"], row["y"], row["z"]])

            table = np.array(table, dtype=float).reshape(-1, 3)

            freq.append(
                {"mode": mode, "frequency": frequency, "intensity": intensity, "vect": table}
            )

        self.complet_results["frequency"] = freq

        self.complet_results["zpe"] = zpe

    def read_AOM_matrix(
        self,
        trigger_line="  S",
        keep_matrix="all",
    ):
        """Read the AOM matrix from a DFTB output file."""

        lines = self.lines.copy()
        if isinstance(keep_matrix, int):
            keep_matrix = [keep_matrix]

        start_idx = None
        for i, line in enumerate(lines):
            if trigger_line in line:
                start_idx = i + 1
                break

        if start_idx is None:
            raise ValueError(f"Section '{trigger_line}' not found in file.")

        data = []
        block_num = 0
        authorized = True
        symb_save = trigger_line.strip()
        i = start_idx

        size = self.complet_results["properties"]["matrix_dim"]
        tab = np.zeros((size, size))

        while i < len(lines):
            line = lines[i].strip()
            if symb_save == lines[i].strip():
                authorized = True
            if (not line or not line.startswith("Orbit")) or not authorized:
                i += 1
                continue

            # "Orbit El  N  L   1  2 ..."
            cols_tokens = line.split()[AOM_DATA_START:]
            self._check(
                cols_tokens and all(_is_int(x) for x in cols_tokens),
                f"read_AOM_matrix: 'Orbit' header expects integer column numbers "
                f"from token {AOM_DATA_START}, found: {line!r}",
                line=line,
            )
            cols = [int(x) for x in cols_tokens]

            min_cols = min(cols)
            max_cols = max(cols)

            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("Orbit") or line.startswith("&&"):
                    break

                # "1  O  2  0  1.000000 ..."
                parts = line.split()
                try:
                    row = int(parts[0])
                except (ValueError, IndexError):
                    break

                values = parts[AOM_DATA_START:]
                if not self._check(
                    len(values) == len(cols),
                    f"read_AOM_matrix: data row expects {len(cols)} values (one per "
                    f"column {min_cols}..{max_cols}), found {len(values)}: {line!r}",
                    line=line,
                ):
                    # Fail with a diagnosable error instead of letting the
                    # assignment below raise an opaque numpy broadcast error.
                    raise OutputParseError(
                        f"read_AOM_matrix: section '{symb_save}' row {row} has "
                        f"{len(values)} values, header declares {len(cols)} columns: {line!r}"
                    )

                # deMon rows and columns are numbered from 1
                tab[min_cols - 1 : max_cols, row - 1] = [
                    convert_float(x, safe=True) for x in values
                ]

                i += 1

            if cols[-1] == size:
                tab = tab.T
                if keep_matrix == "all":
                    data.append(tab.copy())
                    tab = np.zeros((size, size))
                elif keep_matrix == [-1]:
                    data = [tab.copy()]
                    tab = np.zeros((size, size))
                else:
                    if block_num in keep_matrix:
                        data.append(tab.copy())
                        tab = np.zeros((size, size))

                block_num += 1
                authorized = False
        return data

    @safe_parse()
    @assert_flags("print")
    def read_debug(self, extract_data=False, keep_matrix=-1):

        if extract_data:
            matrix = {}
            labels = {}
            current_cols = None

            in_section = False
            text = "\n".join(self.lines)

            for line in text.splitlines():
                if "SCC ATOM-DEPENDENT GAMMA MATRIX" in line:
                    in_section = True
                    continue

                if in_section:
                    if not line:
                        continue
                    if "&&" in line:
                        break

                    parts = line.split()

                    # "Shell El         1  2 ..."
                    if line.startswith("Shell El"):
                        current_cols = parts[GAMMA_DATA_START:]
                        self._check(
                            current_cols and all(_is_int(c) for c in current_cols),
                            f"read_debug: 'Shell El' header expects integer column numbers "
                            f"from token {GAMMA_DATA_START}, found: {line.strip()!r}",
                            line=line,
                        )
                        current_cols = list(map(int, current_cols))
                        continue

                    # "1 O  0.52220 ..." -- row index, element, values
                    if len(parts) <= GAMMA_DATA_START:
                        self._check(
                            not line.strip(),
                            f"read_debug: Gamma row expects at least "
                            f"{GAMMA_DATA_START + 1} tokens (row, element, values), "
                            f"found {len(parts)}: {line.strip()!r}",
                            line=line,
                        )
                        continue

                    if current_cols is None:
                        raise OutputParseError(
                            f"read_debug: Gamma row before any 'Shell El' header: "
                            f"{line.strip()!r}"
                        )

                    if self._check(
                        _is_int(parts[0]),
                        f"read_debug: Gamma row expects an integer row index, found: "
                        f"{line.strip()!r}",
                        line=line,
                    ):
                        self._check(
                            len(parts[GAMMA_DATA_START:]) == len(current_cols),
                            f"read_debug: Gamma row expects {len(current_cols)} values "
                            f"(one per column), found {len(parts[GAMMA_DATA_START:])}: "
                            f"{line.strip()!r}",
                            line=line,
                        )

                    row_idx = int(parts[0])
                    labels[row_idx] = parts[1]
                    values = list(map(float, parts[GAMMA_DATA_START:]))
                    matrix.setdefault(row_idx, {}).update(zip(current_cols, values))

            size = max(matrix.keys())
            full_matrix = np.zeros((size, size))

            for i in matrix:
                for j in matrix[i]:
                    full_matrix[i - 1][j - 1] = matrix[i][j]

            self.complet_results["Gamma"] = full_matrix.copy()

            for key, symbol in (
                ("Overlaps", "S "),
                ("Ham0", "F0"),
                ("Hamiltonian", "F "),
                ("Density", "P "),
                ("Coefficients", "C "),
            ):
                tab = self.read_AOM_matrix(
                    trigger_line=" " * 20 + symbol,
                    keep_matrix=keep_matrix,
                )
                self.complet_results[key] = tab.copy()

    @safe_parse("k-points", "eigenvalues", "kpts_coords")
    @assert_flags("pbc")
    def read_pbc(
        self,
    ):
        text = "\n".join(self.lines)

        energyK = []
        kcoords = []
        lines = text.splitlines()
        nband = self.complet_results["properties"]["matrix_dim"]

        count, nkpt, limit = False, 0, 0
        for i, line in enumerate(lines):

            if "kpoint number" in line:
                count = True
                limit = i + nband * PBC_EIG_SCAN_WINDOW_FACTOR
                nkpt += 1
                # Fractional coordinates are on the previous non-empty line.
                # They are optional: a missing or non-numeric line must not
                # discard the eigenvalues.
                window = lines[max(0, i - PBC_KCOORD_LOOKBACK):i]
                prev = next((li for li in reversed(window) if li.strip()), "")
                try:
                    kcoords.append([float(x) for x in prev.split()[1:4]])
                except ValueError:
                    kcoords.append([])

            # " -0.790839405237485        2.00000000000000" -- energy, occupation
            if count and i <= limit:
                li = line.split()
                try:
                    energyK.append([float(li[0]), float(li[1])])
                except (ValueError, IndexError):
                    pass

        if nkpt > 0 and nband is not None:
            self._check(
                len(energyK) == nkpt * nband,
                f"read_pbc: expected {nkpt * nband} (energy, occupation) entries "
                f"({nkpt} k-points x {nband} bands x {PBC_EIG_TOKEN_COUNT} floats), "
                f"found {len(energyK)}",
            )

        energyK = np.array(energyK)
        energyK = np.reshape(energyK, (nkpt, nband, PBC_EIG_TOKEN_COUNT))
        eigenv = energyK[:, :, :1].reshape((nkpt, nband))

        self.complet_results["k-points"] = energyK
        self.complet_results["eigenvalues"] = eigenv
        self.complet_results["kpts_coords"] = np.array(kcoords)

    @safe_parse("moe", "mos")
    @assert_flags("print")
    def read_print(self):
        """Parse print output sections."""

        text = "\n".join(self.lines)

        occupied = []
        virtual = []

        in_section = False

        for line in text.splitlines():
            if "DFTB Eigen values" in line:
                in_section = True
                occupied = []
                virtual = []
                continue

            if in_section:
                if "Occupied Eigen values" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+", line)
                    occupied.extend(map(float, nums))

                elif "Virtual Eigen values" in line:
                    nums = re.findall(r"[-+]?\d*\.\d+", line)
                    virtual.extend(map(float, nums))

            if "DFTB total energy" in line:
                in_section = False

        self.complet_results["moe"] = {"occupied": occupied, "virtual": virtual}

        filename = os.path.join(self.workdir, "deMon.coef")
        if os.path.exists(filename):
            self.complet_results["mos"] = read_coef(filename, self._parse_line)

    @safe_parse()
    def read_errors(
        self,
    ):
        """Scan for textual errors emitted by deMonNano and record them.

        Appends structured entries produced by :meth:`_add_error`.
        """
        for line in self.lines:
            # "ERROR : loop over charges not converged"
            if "ERROR :" in line:
                _str = line.split(":")[-1].strip()
                self._add_error("demon", _str, line=line.strip())

        msg = ["optimization not converged"]
        for line in self.lines:
            for m in msg:
                if m in line:
                    self._add_error("optimization", m, line=line.strip())

    @safe_parse("tensors")
    @exclude_flags(["ptmc", "freq"])
    def parse_tensors(self):

        text = "\n".join(self.lines)
        data = {}

        # Helpers
        def extract_vector(pattern):
            match = re.search(pattern, text)
            if match:
                return [float(x) for x in match.groups()]
            return None

        # Scalars / vectors
        data["mass_center"] = extract_vector(
            r"mass center\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        )

        data["charge_center"] = extract_vector(
            r"charge center\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        )

        monopole = re.search(r"charge monopole\s*=\s*([-\d\.]+)", text)
        if monopole:
            data["charge_monopole"] = float(monopole.group(1))

        data["charge_dipole"] = extract_vector(
            r"charge dipole\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)"
        )

        norme = re.search(r"norme dipole\s*=\s*([-\d\.]+)", text)
        if norme:
            data["dipole_norm"] = float(norme.group(1))

        # Matrices
        inertia = [[0] * 3 for _ in range(3)]
        quadrupole = [[0] * 3 for _ in range(3)]

        matrix_pattern = re.findall(
            r"\(\s*(\d)\s*,\s*(\d)\s*\)\s*=\s*([-\d\.]+)\s+([-\d\.]+)", text
        )

        # "(1, 2) = ..." -- matrix labels are 1-based
        for i, j, val1, val2 in matrix_pattern:
            i, j = int(i) - 1, int(j) - 1
            inertia[i][j] = float(val1)
            quadrupole[i][j] = float(val2)

        data["inertia_matrix"] = inertia
        data["charge_quadrupole"] = quadrupole

        # Eigenvalues
        eig_inertia = re.search(
            r"Inertia eigenvalues\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", text
        )
        if eig_inertia:
            data["inertia_eigenvalues"] = [float(x) for x in eig_inertia.groups()]

        eig_charge = re.search(
            r"Charges eigenvalues\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)", text
        )
        if eig_charge:
            data["charge_eigenvalues"] = [float(x) for x in eig_charge.groups()]

        self.complet_results["tensors"] = data
