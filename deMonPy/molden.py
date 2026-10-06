import importlib
import sys


def optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


np = optional_import("numpy")
ase = optional_import("ase")


def convert_float(val, safe=False):
    try:
        return float(val)
    except ValueError:
        return val if not safe else None


def _read_xyz_ext(fileobj, is_charges=False, velocities=False, keep=1, ase_obj=True):
    info = []
    images = []

    lines = fileobj.readlines()
    if lines and lines[0] == "\n":
        lines = lines[1:]

    nbmol = 0
    i = 0
    nlines = len(lines)

    while i < nlines:
        natoms = int(lines[i])
        i += 1

        comment = lines[i]
        i += 1

        if velocities:
            info.append([float(x) for x in comment.split()[2:6]])
        else:
            info.append(comment)

        symbols = []
        positions = []
        charges = []
        veloc = [] if velocities else None

        for _ in range(natoms):
            line = lines[i]

            if len(line.split()) == 1:
                break

            i += 1
            fields = line.split()

            symbols.append(fields[0].capitalize())
            positions.append(
                [
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                ]
            )

            if is_charges:
                charges.append(float(fields[4]) if len(fields) > 4 else 0.0)

                if velocities:
                    veloc.append(
                        [
                            float(fields[5]),
                            float(fields[6]),
                            float(fields[7]),
                        ]
                    )
            else:
                charges.append(0.0)

        mask = [symbol != "Xx" for symbol in symbols]
        has_xx = not all(mask)

        if has_xx:
            periodic = True

            # xx_positions = [
            #    pos for pos, is_real in zip(positions, mask)
            #    if not is_real
            # ]

            # lattice = xx_positions[:-1]
            cell = np.zeros((3, 3))
            periodic = True
        else:
            # lattice = None
            cell = None
            periodic = False

        # Filter Xx atoms
        symbols = [s for s, keep_atom in zip(symbols, mask) if keep_atom]
        positions = [p for p, keep_atom in zip(positions, mask) if keep_atom]
        charges = [c for c, keep_atom in zip(charges, mask) if keep_atom]

        if velocities:
            veloc = [v for v, keep_atom in zip(veloc, mask) if keep_atom]

        if ase and ase_obj:
            img = ase.Atoms(
                symbols,
                positions=positions,
                charges=charges,
                velocities=(np.asarray(veloc) / ase.units.fs if velocities else None),
                cell=cell,
                pbc=periodic,
            )

        elif np:
            img = {
                "symbols": np.asarray(symbols),
                "positions": np.asarray(positions),
                "charges": np.asarray(charges),
                "velocities": (np.asarray(veloc) if velocities else None),
                "cell": cell,
                "pbc": periodic,
            }

        else:
            img = {
                "symbols": symbols,
                "positions": positions,
                "charges": charges,
                "velocities": veloc if velocities else None,
                "cell": cell,
                "pbc": periodic,
            }

        nbmol += 1

        if nbmol % keep == 0:
            images.append(img)

    # print(" \u2705 Loaded {} elements from XYZ file.".format(nbmol,))
    return images, np.asarray(info)


def read_XYZ(filename, **kwargs):
    with open(filename, "r") as fd:
        temp = _read_xyz_ext(fd, **kwargs)
    return temp


def write_xyz_ext(
    fileobj, images, charges=None, energy=None, speed=None, comment="", fmt="%22.15f"
):
    """
    Write XYZ file with additional information.
    Parameters
    ----------
    fileobj : file object
        File object to write to.
    images : list of Atoms
        List of Atoms objects to write.
    charges : list of float
        List of charges for each atom in the images.
    energy : list of float, optional
        List of energies for each image. If None, no energy will be written.
    comment : str, optional
        Comment to be written in the file. Default is ''.
    fmt : str, optional
        Format string for the coordinates. Default is '%22.15f'.
    """

    comment = comment.rstrip()

    if "\n" in comment:
        raise ValueError("Comment line should not have line breaks.")

    nImg = len(images)
    if charges is None:
        charges = [
            None,
        ] * nImg
    if energy is None:
        energy = [
            None,
        ] * nImg

    for atoms, charge, energie in zip(images, charges, energy):
        natoms = len(atoms)

        if charge is None:
            charge = [
                0.0,
            ] * natoms

        fileobj.write("%s \n" % natoms)
        fileobj.write("energy (Ha) : %s | %s\n" % (energie, comment))
        for s, (x, y, z), c in zip(atoms.symbols, atoms.positions, charge):
            fileobj.write("%-2s %s %s %s %s\n" % (s, fmt % x, fmt % y, fmt % z, fmt % c))


def write_XYZ(filename, images, intent="w", **kwargs):
    if not isinstance(images, list):
        images = [images]
    with open(filename, intent) as fd:
        write_xyz_ext(fd, images, **kwargs)
    return None


# *************************** \
def progressbar(it, prefix="", size=80, out=sys.stdout):  # Python3.3+
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            "{}[{}{}] {}/{}".format(prefix, "█" * x, "." * (size - x), j, count),
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)
