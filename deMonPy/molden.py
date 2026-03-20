
import importlib

def optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None

np  = optional_import("numpy")
ase = optional_import("ase")


def _read_xyz_ext(fileobj, is_charges=False, velocities=False, keep=1):
    info = []
    lines = fileobj.readlines()
    if lines[0] == "\n":
        lines = lines[1:]
        
    images = []

    nbmol = 0 
    
    while len(lines) > 0:

        symbols = []
        positions,charges,veloc = [],[],[]
        natoms = int(lines.pop(0))
        comment = lines[0]
        info.append(  comment  )
        
        lines.pop(0)  # Comment line; ignored
        
        nread = natoms
        while nread>0:

            if len(lines[0].split())==1:
                break
            line = lines.pop(0)

            if is_charges:
                symbol, x, y, z, c = line.split()[:5]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                charges.append( float(c) )
                if velocities:
                    vx, vy, vz = line.split()[5:8]
                    veloc.append([float(vx), float(vy), float(vz)])
            else:
                symbol, x, y, z = line.split()[:4]
                symbol = symbol.lower().capitalize()
                symbols.append(symbol)
                positions.append([float(x), float(y), float(z)])
                charges.append( 0.00 )

            
            nread -= 1

        if nread==0:
            if ase:
                img = ase.Atoms(
                    symbols, 
                    positions=positions,
                    charges=charges,
                    velocities=veloc if velocities else None
                )
            elif np:
                img = {'symbols':np.array(symbols), 
                       'positions':np.array(positions), 
                       'charges':np.array(charges),
                       'velocities':np.array(veloc) if velocities else None}
            else:
                img = {'symbols':symbols, 
                       'positions':positions, 
                       'charges':charges,
                       'velocities':veloc if velocities else None}

            

        nbmol += 1
        
        if nbmol % keep == 0:
            images.append(img)

    print(" \u2705 Loaded {} elements from XYZ file.".format(nbmol,))
    return images,np.array(info)

def read_XYZ(filename, **kwargs):
    with open(filename,"r") as fd:
        temp = _read_xyz_ext(fd, **kwargs)
    return temp














