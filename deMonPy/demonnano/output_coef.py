#!/usr/bin/env python3
"""Reader of the deMon.coef file written by deMonNano with the PRINT keyword."""

import re

import numpy as np

from deMonPy.demonnano.output_formats import (
    COEF_ATOM,
    COEF_INT_VALUE,
    COEF_LUMO_RANGE,
    COEF_MOE,
)


def read_coef(filename, parse_line):
    """Parse a deMon.coef file (MO energies, occupations and coefficients).

    Args:
        filename: Path of the deMon.coef file.
        parse_line: The reader's ``_parse_line`` method, used to convert and
            check the raw-position fields (errors land in the reader's results).

    Returns:
        dict: Header values, atoms, MO energies, occupations and coefficients.
    """
    data = {
        "n_atoms": None,
        "n_basis": None,
        "parameter_type": None,
        "homo": None,
        "lumo_range": None,
        "atoms": [],
        "mo_energies": [],
        "occupations": [],
        "mo_coefficients": [],
    }

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # deMon.coef header: "<label>" line followed by its value line
    int_headers = {"NAtoms": "n_atoms", "N basis": "n_basis", "homo": "homo"}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --------- HEADER ----------
        if line in int_headers:
            row = parse_line(
                lines[i + 1], COEF_INT_VALUE, f"read_print: deMon.coef '{line}'"
            )
            data[int_headers[line]] = row["value"]
            i += 2
            continue

        elif line == "PArameter type":
            data["parameter_type"] = lines[i + 1].strip()
            i += 2
            continue

        elif "llmos" in line:
            parse_line(
                lines[i + 1], COEF_LUMO_RANGE, "read_print: deMon.coef", strict=False
            )
            data["lumo_range"] = tuple(map(int, lines[i + 1].split()))
            i += 2
            continue

        # --------- ATOMS ----------
        elif line.startswith("atomic position"):
            i += 1
            data["atoms"] = []
            while i < len(lines) and lines[i].strip():
                if lines[i].split()[0] == "occupation":
                    break
                data["atoms"].append(
                    parse_line(lines[i], COEF_ATOM, "read_print: deMon.coef")
                )
                i += 1
            continue

        # --------- MO ENERGIES ----------
        elif line.startswith("occupation and MOE"):
            i += 1
            occupations, mo_energies = [], []
            while i < len(lines) and lines[i].strip():
                if lines[i].split()[1] == "Coefficients":
                    break
                row = parse_line(lines[i], COEF_MOE, "read_print: deMon.coef")
                occupations.append(row["occupation"])
                mo_energies.append(row["energy"])
                i += 1
            data["occupations"] = np.array(occupations)
            data["mo_energies"] = np.array(mo_energies)
            continue

        # --------- MO COEFFICIENTS ----------
        elif line.startswith("MO Coefficients"):
            i += 1
            coeffs = []
            data["mo_coefficients"] = []
            while i < len(lines) and lines[i].strip():
                if lines[i].split()[0] == "NAtoms":
                    break
                nums = re.findall(r"[-+]?\d*\.\d+", lines[i])
                coeffs.extend(map(float, nums))
                i += 1

            data["mo_coefficients"] = np.reshape(
                np.array(coeffs), (data["n_basis"], data["n_basis"])
            )
            continue

        i += 1

    return data
