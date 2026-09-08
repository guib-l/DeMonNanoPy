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
    lines = fileobj.readlines()
    if lines[0] == "\n":
        lines = lines[1:]

    images = []

    nbmol = 0
    periodic = False

    while len(lines) > 0:
        symbols = []
        positions, charges, veloc = [], [], []
        natoms = int(lines.pop(0))
        comment = lines[0]

        if not velocities:
            info.append(comment)
            lines.pop(0)  # Comment line; ignored
        else:
            comment = lines.pop(0)  # Comment line; ignored
            # print(comment.split()[2:])

            _values = list(map(float, comment.split()[2:]))
            # print(_values)
            info.append([*_values[:4]])

        nread = natoms
        while nread > 0:
            if len(lines[0].split()) == 1:
                break
            line = lines.pop(0)

            if is_charges:
                try:
                    symbol, x, y, z, c = line.split()[:5]
                except ValueError:
                    symbol, x, y, z = line.split()[:4]
                    c = 0.0
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                charges.append(float(c))
                if velocities:
                    vx, vy, vz = line.split()[5:8]
                    veloc.append([float(vx), float(vy), float(vz)])
            else:
                symbol, x, y, z = line.split()[:4]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                charges.append(0.00)

            nread -= 1

        true_atoms = np.array(symbols) != "Xx"
        _positions = np.array(positions)[true_atoms]
        _symbols = np.array(symbols)[true_atoms]
        charges = np.array(charges)[true_atoms]

        if velocities:
            veloc = np.array(veloc)[true_atoms]

        lattice = (np.array(positions)[np.array(symbols) == "Xx"])[:-1]

        if len(lattice) > 0:
            periodic = True
            cell = np.zeros((3, 3))
            positions = _positions
            symbols = _symbols

            print("[WARNING] : Not implemented reading cell in molden files.", file=sys.stdout)
        else:
            cell = None
            periodic = False
            lattice = None

        if nread == 0:
            if ase and ase_obj:
                img = ase.Atoms(
                    symbols,
                    positions=positions,
                    charges=charges,
                    velocities=np.array(veloc) / ase.units.fs if velocities else None,
                    cell=cell,
                    pbc=periodic,
                )
            elif np:
                # No ASE available: keep raw velocities (cannot convert to
                # ASE internal units without ``ase.units.fs``).
                img = {
                    "symbols": np.array(symbols),
                    "positions": np.array(positions),
                    "charges": np.array(charges),
                    "velocities": np.array(veloc) if velocities else None,
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
    return images, np.array(info)


def read_XYZ(filename, **kwargs):
    with open(filename, "r") as fd:
        temp = _read_xyz_ext(fd, **kwargs)
    return temp


def write_xyz_ext(fileobj, images, charges=None, energy=None, speed=None, comment='', fmt='%22.15f'):
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

    if '\n' in comment:
        raise ValueError('Comment line should not have line breaks.')
    
    nImg = len(images)
    if charges is None:
        charges = [None,] * nImg
    if energy is None:
        energy = [None,] * nImg

    for atoms,charge,energie in zip(images,charges,energy):
        natoms = len(atoms)
    
        if charge is None:
            charge = [0.,] * natoms

        fileobj.write('%s \n'%natoms)
        fileobj.write('energy (Ha) : %s | %s\n' % (energie, comment))
        for s, (x, y, z), c in zip(atoms.symbols, atoms.positions, charge):
            fileobj.write('%-2s %s %s %s %s\n' % (s, fmt % x, fmt % y, fmt % z, fmt % c ))


def write_XYZ(filename, images, intent='w',**kwargs):
    if not isinstance(images,list):
        images = [images]
    with open(filename,intent) as fd:
        write_xyz_ext(fd, images,**kwargs)
    return None




# *************************** \
def progressbar(it, prefix="", size=80, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, u'█'*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

