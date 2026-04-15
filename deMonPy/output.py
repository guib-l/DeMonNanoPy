#!/usr/bin/env python3

# Import standard de python3
import os
import re
import numpy as np


import deMonPy
from deMonPy.molden import read_XYZ
from deMonPy.profile import assert_flags,exclude_flags

def convert_float(val, safe=False):
    try:
        return float(val)
    except ValueError:
        return val if not safe else None

class IOread(object):
    """Base helper providing low-level parsing utilities for output files."""

    def __init__(self,):
        """Initialize parsing state."""

        self._block = ""
        self._tocken_block = False

        self.counter = 0

    def compute_block(self, line, 
            ctrl_in="", ctrl_out="", nb_line=None, add=1 ):
        """Track and accumulate a fixed-size text block.

        Args:
            line: Current line being parsed.
            ctrl_in: Marker indicating the beginning of the block.
            ctrl_out: Unused end-of-block marker placeholder.
            nb_line: Number of lines to collect once the block starts.
            add: Unused compatibility argument.
        """

        if self.is_inside(control=ctrl_in, line=line ):
            self._tocken_block = True
            self.counter = nb_line

        if self.counter > 0:
            self._block  += line
            self.counter -= 1
        else:
            self._tocken_block = False

    def get_float(self, line, index=-1):
        """Extract a float from a split line.

        Args:
            line: Input line.
            index: Token index to convert.

        Returns:
            float: Parsed floating-point value.
        """
        assert index!=None, ValueError("index need to ba an integer")
        return float(line.split()[index])

    def get_int(self, line, index=-1):
        """Extract an integer from a split line.

        Args:
            line: Input line.
            index: Token index to convert.

        Returns:
            int: Parsed integer value.
        """
        assert index!=None, ValueError("index need to ba an integer")
        return int(line.split()[index])

    def get_list(self, line, index_in=None, index_out=None, ctype=np.float64):
        """Extract a typed slice of tokens from a line.

        Args:
            line: Input line.
            index_in: First token index to include.
            index_out: First token index to exclude.
            ctype: Target NumPy-compatible type constructor.

        Returns:
            Any: Converted token slice.
        """
        if index_in==None:
            return ctype( line.split()[:index_out] )
        if index_out==None:
            return ctype( line.split()[index_in:] )
        return ctype( line.split()[index_in:index_out] )
        
    def get_dict(self,):
        """Return a dictionary representation of parsed content.

        Raises:
            NotImplementedError: Always raised in the base class.
        """
        raise NotImplementedError

    def get_string(self, line, index=None):
        """Extract a string token or return the raw line.

        Args:
            line: Input line.
            index: Optional token index.

        Returns:
            str: Extracted token or original line.
        """
        if index==None:
            return line
        return str(line.split()[index])


    def is_inside(self, control="", line="" ): 
        """Check whether a control substring is present in a line.

        Args:
            control: Substring to search for.
            line: Line to inspect.

        Returns:
            bool: True when the substring is found.
        """
        if control in line: return  True
        else: return False





class read_output(IOread):
    """Parser for deMonNano output files and derived results."""

    _criteria_energy_str = {
        "DFTB total energy":"energy",
        "DFTB electronic energy": "electronic_energy",
        "DFTB band energy"      : "band_energy",
        "DFTB repulsive energy" : "repulsive_energy",
        "DFTB Coulomb energy"   : "coulomb_energy",
        "DFTB London energy"        : "london_energy",
        "DFTB MM+Mechanical Coupl." : "MM_coupl.",
        "DFTB electronic entropy"   : "electronic_entropy",
        "DFTB Polarisation energy"  : "polarisation_energy",
        "DFTB HOMO-LUMO gap"        : "HOMO-LUMO_gap",
        "DFTB (HOMO)-(HOMO-1) gap"  : "HOMO_gap",
        "DFTB Fermi energy level"   : "fermi_energy", 
        "DFTB third order Coulomb energy" : "3d_coulomb_energy",
    }

    
    def __init__(self, 
            properties=["energy"], 
            workdir="./",
            output="deMon.out", 
            flags=set(),
            type_calculation={}, ): 
        """Initialize the output reader.

        Args:
            properties: Requested properties to parse.
            workdir: Directory containing output files.
            output: Main output file name.
            flags: Active parser flags.
            type_calculation: Reserved calculation metadata.
        """
        
        IOread.__init__(self, )


        self.workdir = workdir
        self.output = output

        self.flags = flags

        self.properties = properties
        self.complet_results = {}
        
        self.lines = []



    def read_file(self,):
        """Load the main output file into memory."""

        filename = os.path.join(self.workdir,self.output)
        with open(filename,'r') as fd:

            for line in fd.readlines():
                self.lines.append(line)
    

    # =================================
    # READ ENERGY (basics)

    def read_basics(self):
        """Read basics things"""

        text = '\n'.join(self.lines)

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

        for line in text.splitlines():
            line = line.strip()

            # -------- SCC PARAMS --------
            if "Requested SCC tolerance" in line:
                val = re.findall(r"[-+]?\d+\.\d+E[-+]?\d+", line)
                if val:
                    data["scc_tolerance"] = float(val[0])

            elif "Maximum number of SCC cycles" in line:
                val = re.findall(r"\d+", line)
                if val:
                    data["max_scc_cycles"] = int(val[-1])

            elif "SCC mixing" in line:
                val = re.findall(r"[-+]?\d*\.\d+", line)
                if val:
                    data["scc_mixing"] = float(val[0])

            # -------- ELECTRONIC INFO --------
            elif "Number of DFTB atoms" in line:
                data["n_atoms"] = int(line.split(":")[-1])

            elif "Number of DFTB atom types" in line:
                data["n_atom_types"] = int(line.split(":")[-1])

            elif "Dimension of DFTB matrices" in line:
                data["matrix_dim"] = int(line.split(":")[-1])

            elif "Number of DFTB electrons" in line:
                data["n_electrons"] = int(line.split(":")[-1])

            elif "Number of DFTB shells" in line:
                data["n_shells"] = int(line.split(":")[-1])

        self.complet_results['properties'] = data

    def read_energy(self):
        """Parse energy values from the loaded output lines."""

        start_search = False
        
        _state = {}
        self.complet_results[f"energy"] = {}

        for line in self.lines:
                        
            _energy = self.get_energies(line,start_search=start_search)
            _state.update(_energy)

            if 'energy' in self.complet_results.keys():
                start_search = True
        self.complet_results[f"energy"].update(_state)

    def get_energies(self, 
                     line, 
                     criteria="DFTB total energy",
                     start_search=False):
        """Extract energy terms from a single output line.

        Args:
            line: Line to inspect.
            criteria: Primary energy marker used to seed parsing.
            start_search: Whether extended energy parsing is active.

        Returns:
            dict: Parsed energy terms found on the line.
        """
        
        _energy = {}

        if self.is_inside(criteria, line):
            label = self._criteria_energy_str[criteria]
            _energy[label] = self.get_float( line, index=-1)
            
        if start_search:
            
            for cs,it in self._criteria_energy_str.items():
                if self.is_inside(cs, line):
                    _energy[it] = self.get_float(line, index=-1) 
        return _energy


    # =================================
    # READ GEOMETRY (basics)

    @exclude_flags(["ptmc","freq"])
    def read_geometry(self, output='deMon.mol',is_charges=False, velocities=False, keep=1):       
        """Read input, output, or trajectory geometries.

        Args:
            output: Geometry file name.
            is_charges: Whether the XYZ reader should parse charges.
            keep: Geometry sampling interval.
        """
        
        filename = os.path.join(self.workdir,output)
        data,info = read_XYZ(filename,is_charges=is_charges, velocities=velocities, keep=keep)
        
        
        if len(data)==2:
            self.complet_results["input_geometry"]  = data[0]
            self.complet_results["output_geometry"] = data[-1]

        if  len(data)>2:
            self.complet_results["input_geometry"]  = data[0]
            if "traj" in self.flags:
                self.complet_results["trajectory"] = data
            self.complet_results["output_geometry"] = data[-1]

        if "md" in self.flags:
            self.complet_results["time"] = info[:,3]
            self.complet_results["potential_energy"] = info[:,0]
            self.complet_results["kinetic_energy"] = info[:,1]
            self.complet_results["total_energy"] = info[:,2]
            











    # =================================
    # READ PARAMETERS (basics)

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
                        
            _energy = self.get_energies(line,start_search=start_search)
            _conf.update(_energy)

            if "*********   CONFIGURATIONS   *********" in line:
                if count>0:
                    self.complet_results[f"configuration_{count}"] = _conf
                count += 1
                _conf = {}

            if self.is_inside("************    STATES    ************",line):
                state_search = True

            if self.is_inside("for state", line) and state_search:
                sl = line.split()
                num = int(sl[2])
                sub = float(sl[4])
                wgh = []
                state = {f"state {num}":{"energy":sub}}

            if self.is_inside("weight of conf", line) and state_search:
                sl = line.split()
                try:
                    wgh.append(float(sl[5]))
                except:
                    wgh.append(None)
                state[f"state {num}"].update({"weight":wgh})

            self.complet_results["states"].update(state)

        self.complet_results[f"configuration_{count}"] = _conf

        
        if "states" in self.complet_results.keys():
            if len(self.complet_results["states"].keys())>0:
                self.complet_results.pop("energy")

                energies = [ state["energy"] for k,state in self.complet_results["states"].items()]       
                self.complet_results["energy"] = {"energy":min(energies)}
        

    @assert_flags("td-dftb")
    def read_tddftb(self):
        """Parse TD-DFTB singlet and triplet transitions."""

        msg = "LINEAR RESPONSE FOR CLOSED SHELL MOLECULES ONLY."
        for line in self.lines:
            if msg in line:
                self.complet_results["errors"] = [msg.lower()]
                return 
        
        args = {}
        for line in self.lines:
            
            if self.is_inside("requested transitions to calculate",line):
                N = self.get_int(line,-1)
                break

        count = 0

        for key,signature in zip(["triplet","singlet"],["SUMMARY TRIPLET:","SUMMARY SINGLET"]):

            for line in self.lines:

                self.compute_block(line,signature,"",nb_line=N+4)
                if self._tocken_block:
                    
                    values = line.split()
                    try:
                        args.update(
                            {f"state_{count}":{
                                "w":float(values[0]),
                                "ocillator":float(values[1]),
                                "from":int(values[2]),
                                "to":int(values[4]),
                                "weight":float(values[5]),
                                "energy":float(values[6]),
                            }}
                        )
                        count += 1
                    except:
                        pass
            self.complet_results[key] = args


    @assert_flags("freq")
    def read_freq(self):
        """Parse vibrational frequency results."""
        freq,temp = [],[]
        zpe,Nbatm = 0.00,0

        for line in self.lines:
            if 'NUMBER OF ATOMS:' in line:
                Nbatm = int(line.split()[-1])
            if 'ZERO-POINT ENERGY =' in line:
                zpe = float(line.split()[3])

            self.compute_block(
                line, "MODE:", "", 6 + Nbatm
            )

            if self.counter==0 and self._block!="":
                temp.append(self._block)
                self._block = ""

        for frq in temp:
            mode,frequency,intensity,vect = 0,0.0,0.0,np.zeros((Nbatm,3))

            for line in frq.split('\n'):
                if self.is_inside(control='MODE:', line=line ):
                    mode = int(line.split()[1])
                if self.is_inside(control='FREQUENCY:', line=line ):
                    frequency = float(line.split()[1])
                if self.is_inside(control='INTENSITY:', line=line ):
                    intensity = float(line.split()[1])

            table = []
            for lin in frq.split('\n')[6:-1]:
                valeurs = lin.strip().split()
                nombres = [convert_float(v, False) for v in valeurs]
                table.append(nombres)
            
            table = np.array(table)[:,2:]

            freq.append( {
                    "mode":mode,
                    "frequency":frequency,
                    "intensity":intensity,
                    "vect":table
                } )

        self.complet_results["frequency"] = freq

        self.complet_results["zpe"] = zpe



    def read_AOM_matrix(self,trigger_line = "  S"):
        """ Read the AOM matrix from a DFTB output file. """
        
        lines = self.lines.copy()

        # --- 1. Trouver le début de la section S ---
        start_idx = None
        for i, line in enumerate(lines):
            if trigger_line in line:
                print(trigger_line, line)
                start_idx = i + 1
                break

        if start_idx is None:
            raise ValueError(f"Section '{trigger_line}' not found in file.")

        # --- 2. Lire les blocs ---
        data = []
        i = start_idx
        

        while i < len(lines):
            line = lines[i].strip()
            filled_rows = 0

            # Stop si fin de section
            if not line or not line.startswith("Orbit"):
                i += 1
                continue
            
            line = lines[i].strip()

            if not line.startswith("Orbit"):
                i += 1
                continue

            # Colonnes du bloc
            parts = line.split()
            cols = [int(x) for x in parts[4:]]

            size = self.complet_results["properties"]["matrix_dim"]
            tab = np.zeros((size, size))


            i += 1

            # Lire les lignes du bloc
            while i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith("&&"):
                    break
                
                if not line or line.startswith("Orbit"):
                    break        
                
                parts = line.split()

                try:
                    row_idx = int(parts[0])
                except:
                    break

                row = int(parts[0])
                values = [float(x) for x in parts[4:]]


                for c, v in zip(cols, values):

                    tab[row-1, c-1] = v

                filled_rows += 1

                i += 1

            #print(data)

        # --- 3. Déterminer la taille ---
        max_index = max(row for row, _, _ in data)

        tab = np.zeros((max_index, max_index))

        # --- 4. Remplir la matrice ---
        for row_idx, cols, values in data:
            for col, val in zip(cols, values):
                tab[row_idx - 1, col - 1] = val
        return tab

    @assert_flags("print")
    def read_debug(self, extract_data=False):
        
        if extract_data:
            
            matrix = {}
            labels = {}
            
            in_section = False
            text = '\n'.join(self.lines)

            for line in text.splitlines():
                if "SCC ATOM-DEPENDENT GAMMA MATRIX" in line:
                    in_section = True
                    continue

                if in_section:
                    if not line:
                        continue
                    if "&&" in line:
                        break
                    
                    # -------- HEADER (colonnes) --------
                    if line.startswith("Shell El"):
                        parts = line.split()
                        # colonnes = indices après "Shell El"
                        current_cols = list(map(int, parts[2:]))
                        continue
                    
                    # -------- LIGNES MATRICE --------
                    parts = line.split()
                    
                    # sécurité : ignorer lignes invalides
                    if len(parts) < 3:
                        continue
                    
                    row_idx = int(parts[0])
                    element = parts[1]
                    values = list(map(float, parts[2:]))

                    labels[row_idx] = element
                    
                    if row_idx not in matrix:
                        matrix[row_idx] = {}
                    
                    for col_idx, val in zip(current_cols, values):
                        matrix[row_idx][col_idx] = val


            # -------- CONVERSION EN MATRICE 2D --------
            size = max(matrix.keys())
            full_matrix = np.zeros((size,size))
            
            for i in matrix:
                for j in matrix[i]:
                    full_matrix[i-1][j-1] = matrix[i][j]

            print(full_matrix)

            tab = self.read_AOM_matrix("           F ")
            print(tab)
            print(tab.shape)
            tab = self.read_AOM_matrix("           S ")
            print(tab)
            print(tab.shape)



    @assert_flags("print")
    def read_print(self):
        """Parse print output sections."""
        
        text = '\n'.join(self.lines)
        
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


        self.complet_results['moe'] =  {
            "occupied": occupied,
            "virtual": virtual
        }

        filename = os.path.join(self.workdir, "deMon.coef")
        if os.path.exists(filename):
            
            data = {
                "n_atoms": None,
                "n_basis": None,
                "parameter_type": None,
                "homo": None,
                "lumo_range": None,
                "atoms": [],
                "mo_energies": [],
                "occupations": [],
                "mo_coefficients": []
            }
        
            section = None
            with open(filename, "r") as f:
                lines = f.readlines()
        
            i = 0
            while i < len(lines):
                line = lines[i].strip()
        
                # --------- HEADER ----------
                if line == "NAtoms":
                    data["n_atoms"] = int(lines[i+1].strip())
                    i += 2
                    continue
                
                elif line == "N basis":
                    data["n_basis"] = int(lines[i+1].strip())
                    i += 2
                    continue
                
                elif line == "PArameter type":
                    data["parameter_type"] = lines[i+1].strip()
                    i += 2
                    continue
                
                elif line == "homo":
                    data["homo"] = int(lines[i+1].strip())
                    i += 2
                    continue
                
                elif "llmos" in line:
                    parts = lines[i+1].split()
                    data["lumo_range"] = tuple(map(int, parts))
                    i += 2
                    continue
                
                
                # --------- ATOMS ----------
                elif line.startswith("atomic position"):
                    i += 1
                    data["atoms"] = []
                    while i < len(lines) and lines[i].strip():
                        parts = lines[i].split()
                        if parts[0] == 'occupation':
                            break
                        atom = {
                            "Z": int(parts[0]),
                            "symbol": parts[1],
                            "x": float(parts[2]),
                            "y": float(parts[3]),
                            "z": float(parts[4]),
                            "basis_start": int(parts[5]),
                            "basis_end": int(parts[6]),
                        }
                        data["atoms"].append(atom)
                        i += 1
                    continue
                
                # --------- MO ENERGIES ----------
                elif line.startswith("occupation and MOE"):
                    i += 1
                    data["occupations"] = []
                    data["mo_energies"] = []
                    while i < len(lines) and lines[i].strip():
                        parts = lines[i].split()
                        if parts[1] == 'Coefficients':
                            break
                        data["occupations"].append(float(parts[1]))
                        data["mo_energies"].append(float(parts[2]))
                        i += 1
                    continue

                    data["occupations"] = np.array(data["occupations"])
                    data["mo_energies"] = np.array(data["mo_energies"])
                
                # --------- MO COEFFICIENTS ----------
                elif line.startswith("MO Coefficients"):
                    i += 1
                    coeffs = []
                    data["mo_coefficients"] = []
                    while i < len(lines) and lines[i].strip():
                        parts = lines[i].split()
                        if parts[0] == 'NAtoms':
                            break
                        nums = re.findall(r"[-+]?\d*\.\d+", lines[i])
                        coeffs.extend(map(float, nums))
                        i += 1
                    
                    data["mo_coefficients"] = np.reshape(
                        np.array(coeffs),
                        (data["n_basis"],data["n_basis"])
                    )
                    continue
                
                i += 1

            self.complet_results['mos'] =  data
        
        
            
    def read_errors(self,):
        err = self.complet_results["errors"] if "errors" in self.complet_results.keys()  else  []
        for line in self.lines:
            if 'ERROR :' in line:
                _str = line.split(':')[-1]
                err.append(_str.strip())

        msg = ["optimization not converged",]
        for line in self.lines:
            for m in msg:
                if m in line:
                    err.append(m)

        self.complet_results["errors"] = err

    @exclude_flags(["ptmc","freq"])
    def parse_tensors(self):

        text = '\n'.join(self.lines)
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
        inertia = [[0]*3 for _ in range(3)]
        quadrupole = [[0]*3 for _ in range(3)]

        matrix_pattern = re.findall(
            r"\(\s*(\d)\s*,\s*(\d)\s*\)\s*=\s*([-\d\.]+)\s+([-\d\.]+)",
            text
        )

        for i, j, val1, val2 in matrix_pattern:
            i, j = int(i)-1, int(j)-1
            inertia[i][j] = float(val1)
            quadrupole[i][j] = float(val2)

        data["inertia_matrix"] = inertia
        data["charge_quadrupole"] = quadrupole

        # Eigenvalues
        eig_inertia = re.search(
            r"Inertia eigenvalues\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)",
            text
        )
        if eig_inertia:
            data["inertia_eigenvalues"] = [float(x) for x in eig_inertia.groups()]

        eig_charge = re.search(
            r"Charges eigenvalues\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)",
            text
        )
        if eig_charge:
            data["charge_eigenvalues"] = [float(x) for x in eig_charge.groups()]

        self.complet_results['tensors'] = data

    # =================================
    # READ MODULES (basics)

    @assert_flags("opt")
    def _read_opt(self):
        """Parse optimization-specific output sections."""

    @assert_flags("ptmc")
    def _read_ptmc(self):
        """Parse PTMC-specific output sections."""
        import re
        text = '\n'.join(self.lines)
        data = {}

        seed_match = re.findall(r"SEED\s*=\s*((?:\d+\s+)+)", text)
        if seed_match:
            seeds = list(map(int, seed_match[0].split()))
            data["seeds"] = seeds

        patterns = {
            "nb_step": r"NB STEP\s*=\s*(\d+)",
            "optout": r"OPTOUT\s*=\s*(\d+)",
            "nb_temp": r"NB TEMP\s*=\s*(\d+)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = int(match.group(1))

        temps = []
        temp_matches = re.findall(r"\n\s*(\d+)\s+([\d\.]+)\s+([\d\.E\-\+]+)", text)

        for idx, t, val in temp_matches:
            temps.append({
                "index": int(idx),
                "temperature": float(t),
                "value": float(val)
            })

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

        self.complet_results['ptmc'] = data

        _traj = {}
        if 'traj' in self.flags:

            for out in range(len(data["temperatures"])):
                output = f"deMon.{(out+1):02}.mol"
                filename = os.path.join(self.workdir,output)
                data,info = read_XYZ(filename,is_charges=False, velocities=False, keep=1)

                energies = np.array(list(map(lambda x:float(x.split()[2]),info)))

                _traj[output] = {
                    "trajectory":data,
                    "energies":energies,
                }

            self.complet_results["trajectory"] = _traj


    @assert_flags("md")
    def _read_md(self):
        """Parse molecular dynamics summary values."""
        
        energies_flags = {
            "average_potential_energy":"Average POTENTIAL ENERGY",
            "std_potential_energy":"Std Dev POTENTIAL ENERGY",
            "average_kinetic_energy":"Average KINETIC ENERGY",
            "std_kinetic_energy":"Std Dev KINETIC ENERGY",
            "average_total_energy":"Average TOTAL ENERGY",
            "std_total_energy":"Std Dev TOTAL ENERGY",
            "average_temperature":"Average TEMPERATURE",
            "std_temperature":"Std Dev TEMPERATURE",
            "energy_from_thermostat":"Energy transfer from thermostat",
            "energy_loss":"Energy loss from constraints"
        }

        for line in self.lines:
            for it,_flag in energies_flags.items():
                if _flag in line:
                    self.complet_results["energy"].update(
                        {
                            it:float(line.split()[-2])
                        }
                    )
                

    @assert_flags("neb")
    def _read_neb(self):
        """Parse nudged elastic band output sections."""
        pass







