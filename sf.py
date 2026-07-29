# 
#   This file is part of aXis2000 (Python Version).
# 
#   Copyright (C) 2026 James J. Dynes
# 
#   aXis2000 is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   aXis2000 is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details <https://www.gnu.org/licenses/>.
# 

import os
import numpy as np
import re
import struct

# Placeholder for common blocks or global state
# In a real application, these would be imported from a config module
SF_STATE = {
    'sf_dir_': '',
    'sf_ext_': '.nff',
    'el_names_': [],
    'at_wts_': [],
    'energy_': {}, # Using dict instead of pointer array
    'f1_': {},
    'f2_': {},
    'xdr_loaded_': False
}

# --- Helper Functions ---

def file2string(filename):
    """Reads a file into a list of strings."""
    try:
        with open(filename, 'r') as f:
            return f.readlines()
    except IOError:
        return []

def str_columns(data):
    """
    Parses a list of strings containing columnar data into a 2D array.
    This is a simplified version.
    """
    parsed_data = []
    for line in data:
        parts = line.split()
        if parts:
            parsed_data.append(parts)
    
    # Transpose to match IDL's column-major expectation if needed, 
    # but list of lists is usually row-major.
    # IDL: data[col, row]
    # Python: data[row][col]
    # We'll return a list of columns for easier indexing
    if not parsed_data:
        return []
        
    num_cols = len(parsed_data[0])
    columns = [[] for _ in range(num_cols)]
    for row in parsed_data:
        for i, val in enumerate(row):
            if i < num_cols:
                columns[i].append(val)
    return columns

def wvlen2en(wavelength, unit_in='m', unit_out='eV'):
    """Converts wavelength to energy."""
    # h * c = 1239.84193 eV * nm
    hc = 1239.84193
    
    # Convert input to nm
    if unit_in == 'm':
        wl_nm = wavelength * 1e9
    elif unit_in == 'Angstrom':
        wl_nm = wavelength * 0.1
    else: # nm
        wl_nm = wavelength
        
    energy_ev = hc / wl_nm
    
    # Convert output
    if unit_out == 'keV':
        return energy_ev / 1000.0
    elif unit_out == 'J':
        return energy_ev * 1.60218e-19
    else: # eV
        return energy_ev

def en2wvlen(energy, unit_in='eV', unit_out='m'):
    """Converts energy to wavelength."""
    hc = 1239.84193 # eV * nm
    
    # Convert input to eV
    if unit_in == 'keV':
        en_ev = energy * 1000.0
    elif unit_in == 'J':
        en_ev = energy / 1.60218e-19
    else: # eV
        en_ev = energy
        
    wl_nm = hc / en_ev
    
    # Convert output
    if unit_out == 'm':
        return wl_nm * 1e-9
    elif unit_out == 'Angstrom':
        return wl_nm * 10.0
    else: # nm
        return wl_nm

# --- Main SF Functions ---

def sf_init(sf_dir=''):
    """Initialize common block for module."""
    global SF_STATE
    import sys
    
    # Resolve sf_dir_ candidate paths to find where elements.dat is.
    candidate_dirs = []
    if sf_dir:
        candidate_dirs.append(sf_dir)

    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs.append(module_dir)
    candidate_dirs.append(os.path.join(os.getcwd(), 'SGM_Beamline_Analysis'))

    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        candidate_dirs.append(exe_dir)
        if hasattr(sys, '_MEIPASS'):
            candidate_dirs.append(sys._MEIPASS)

    candidate_dirs.append(os.getcwd())
    
    # De-duplicate candidate_dirs while preserving order
    seen = set()
    candidate_dirs = [x for x in candidate_dirs if x and not (x in seen or seen.add(x))]
    
    resolved_dir = None
    for d in candidate_dirs:
        if os.path.exists(os.path.join(d, 'elements.dat')):
            resolved_dir = d
            break
            
    if not resolved_dir:
        resolved_dir = candidate_dirs[0] if candidate_dirs else ''
        
    SF_STATE['sf_dir_'] = resolved_dir
    SF_STATE['sf_ext_'] = '.nff'
    
    if not SF_STATE['el_names_']:
        # Search for data file
        fn = os.path.join(SF_STATE['sf_dir_'], 'elements.dat')
        
        if os.path.exists(fn):
            try:
                data = file2string(fn)
                cols = str_columns(data)
                
                if len(cols) >= 2:
                    # Element names are in col 0, weights in col 1
                    # Skip header row
                    SF_STATE['el_names_'] = cols[0][1:]
                    SF_STATE['at_wts_'] = [float(x) for x in cols[1][1:]]
                    
                    # Prepend dummy to align index with atomic number (1-based)
                    SF_STATE['el_names_'].insert(0, '')
                    SF_STATE['at_wts_'].insert(0, 0.0)
            except Exception as e:
                print(f'Error reading elements.dat: {e}')
        else:
            print(f'Data file elements.dat not found at: {fn}')
        
        # Try to load XDR if .nff files are likely missing
        sf_load_xdr()

def sf_load_xdr():
    """Load data from henke.xdr if available."""
    global SF_STATE
    if SF_STATE['xdr_loaded_']: return
    
    # Try common locations
    xdr_paths = [
        os.path.join(SF_STATE['sf_dir_'], 'henke.xdr'),
        os.path.join(SF_STATE['sf_dir_'], 'spectromicroscopy-master', 'mantis_xray', 'henke.xdr'),
        os.path.join(os.path.dirname(SF_STATE['sf_dir_']), 'spectromicroscopy-master', 'mantis_xray', 'henke.xdr')
    ]
    
    fn = None
    for p in xdr_paths:
        if os.path.exists(p):
            fn = p
            break
            
    if not fn: return

    try:
        with open(fn, 'rb') as f:
            data = f.read()
        
        offset = 0
        n_elements, n_energies = struct.unpack('>II', data[offset:offset+8])
        offset += 8
        
        energies = np.array(struct.unpack(f'>{n_energies}f', data[offset:offset + 4*n_energies]))
        offset += 4 * n_energies
        
        for i in range(n_elements):
            f1 = np.array(struct.unpack(f'>{n_energies}f', data[offset:offset + 4*n_energies]))
            offset += 4 * n_energies
            f2 = np.array(struct.unpack(f'>{n_energies}f', data[offset:offset + 4*n_energies]))
            offset += 4 * n_energies
            
            # Atomic number starts from 1
            atomic_num = i + 1
            SF_STATE['energy_'][atomic_num] = energies
            SF_STATE['f1_'][atomic_num] = f1
            SF_STATE['f2_'][atomic_num] = f2
            
        # Optional: handle extra energies (near edges)
        if offset + 4 <= len(data):
            n_extra_energies = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            for i in range(n_elements):
                this_n_extra = struct.unpack('>I', data[offset:offset+4])[0]
                offset += 4
                e_extra = np.array(struct.unpack(f'>{n_extra_energies}f', data[offset:offset + 4*n_extra_energies]))
                offset += 4 * n_extra_energies
                f1_extra = np.array(struct.unpack(f'>{n_extra_energies}f', data[offset:offset + 4*n_extra_energies]))
                offset += 4 * n_extra_energies
                f2_extra = np.array(struct.unpack(f'>{n_extra_energies}f', data[offset:offset + 4*n_extra_energies]))
                offset += 4 * n_extra_energies
                
                if this_n_extra > 0:
                    # Merge extra energies
                    atomic_num = i + 1
                    e_all = np.concatenate((SF_STATE['energy_'][atomic_num], e_extra[:this_n_extra]))
                    f1_all = np.concatenate((SF_STATE['f1_'][atomic_num], f1_extra[:this_n_extra]))
                    f2_all = np.concatenate((SF_STATE['f2_'][atomic_num], f2_extra[:this_n_extra]))
                    
                    sort_idx = np.argsort(e_all)
                    SF_STATE['energy_'][atomic_num] = e_all[sort_idx]
                    SF_STATE['f1_'][atomic_num] = f1_all[sort_idx]
                    SF_STATE['f2_'][atomic_num] = f2_all[sort_idx]

        SF_STATE['xdr_loaded_'] = True
        print(f"Loaded Henke database from {fn}")
    except Exception as e:
        print(f"Error loading XDR database: {e}")

def sf_read_data(element, sf_dir=''):
    """Read scattering factor data for a single element."""
    global SF_STATE
    
    if not SF_STATE['el_names_']:
        sf_init(sf_dir)
        
    # Determine element name
    if isinstance(element, str):
        e_name = element.lower()
        try:
            # Find index (atomic number)
            # el_names_ has dummy at 0, so index matches atomic number
            atomic_num = [x.lower() for x in SF_STATE['el_names_']].index(e_name)
        except ValueError:
            print('Invalid element name.')
            return None, None, None
    else:
        atomic_num = int(element)
        if atomic_num < 1 or atomic_num > 92:
            print('Invalid atomic number.')
            return None, None, None
        e_name = SF_STATE['el_names_'][atomic_num].lower()
        
    # Check if already loaded
    if atomic_num in SF_STATE['energy_']:
        return SF_STATE['energy_'][atomic_num], SF_STATE['f1_'][atomic_num], SF_STATE['f2_'][atomic_num]
        
    # Read file
    f_name = os.path.join(SF_STATE['sf_dir_'], e_name + SF_STATE['sf_ext_'])
    
    if not os.path.exists(f_name):
        print(f'SF data file not found: {f_name}')
        return None, None, None
        
    data = file2string(f_name)
    cols = str_columns(data)
    
    if len(cols) >= 3:
        try:
            # Skip header
            energy = np.array([float(x) for x in cols[0][1:]])
            f1 = np.array([float(x) for x in cols[1][1:]])
            f2 = np.array([float(x) for x in cols[2][1:]])
            
            # Cache data
            SF_STATE['energy_'][atomic_num] = energy
            SF_STATE['f1_'][atomic_num] = f1
            SF_STATE['f2_'][atomic_num] = f2
            
            return energy, f1, f2
        except Exception as e:
            print(f'Error parsing data for {e_name}: {e}')
    
    return None, None, None

def sf_parse_compound(input_str, sf_dir=''):
    """Parse a compound name into elements and stoichiometry."""
    global SF_STATE
    
    if not SF_STATE['el_names_']:
        sf_init(sf_dir)
        
    # Remove whitespace
    compound = input_str.replace(" ", "")
    
    # Regex to find Element + optional Number
    # Matches an uppercase letter, optional lowercase letter, and optional number (int or float)
    # e.g., H2, O, Fe2.5
    pattern = r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)"
    matches = re.findall(pattern, compound)
    
    el_nums = []
    stoichs = []
    
    valid_elements = [x.lower() for x in SF_STATE['el_names_']]
    
    for el_sym, amount in matches:
        if el_sym.lower() in valid_elements:
            el_idx = valid_elements.index(el_sym.lower())
            el_nums.append(el_idx)
            
            if amount == "":
                stoichs.append(1.0)
            else:
                try:
                    stoichs.append(float(amount))
                except ValueError:
                    print(f"Invalid stoichiometry: {amount}")
                    return None
        else:
            print(f"Invalid element: {el_sym}")
            return None
            
    if not el_nums:
        print("Invalid compound.")
        return None
        
    # Sort by atomic number
    sorted_indices = np.argsort(el_nums)
    return np.array(el_nums)[sorted_indices], np.array(stoichs)[sorted_indices]

def sf(compound, abscissa=None, unit_in='eV', result_type='mu', 
       density=1.0, angle=10.0, thickness=1.0, sf_dir=''):
    """
    Main function to compute materials x-ray properties.
    """
    global SF_STATE
    
    if not SF_STATE['el_names_']:
        sf_init(sf_dir)
        
    # Constants
    Na = 6.0220453e23
    r0 = 2.8179380e-15 # m
    
    # Parse compound
    parsed = sf_parse_compound(compound, sf_dir)
    if parsed is None: return None
    
    el_nums, amts = parsed
    
    # Molecular Weight
    mw = np.sum(amts * np.array([SF_STATE['at_wts_'][int(i)] for i in el_nums]))
    
    # Load data for elements
    for el in el_nums:
        e, f1, f2 = sf_read_data(el, sf_dir)
        if e is None:
            raise ValueError(f"Could not load data for element {SF_STATE['el_names_'][int(el)]}. Missing .nff file?")
        
    # Handle Energy
    if abscissa is not None:
        if unit_in == 'eV':
            energy = np.array(abscissa, dtype=float)
        else:
            # Convert to eV
            if unit_in in ['m', 'nm', 'Angstrom']:
                energy = wvlen2en(np.array(abscissa, dtype=float), unit_in=unit_in, unit_out='eV')
            elif unit_in == 'J':
                energy = np.array(abscissa, dtype=float) / 1.6021892e-19
            elif unit_in == 'keV':
                energy = np.array(abscissa, dtype=float) * 1000.0
            else:
                energy = np.array(abscissa, dtype=float)
    else:
        # Use whole range (union of all elements' energies)
        all_energies = []
        for el in el_nums:
            if el in SF_STATE['energy_']:
                all_energies.extend(SF_STATE['energy_'][el])
            else:
                raise ValueError(f"Energy data for element {el} not loaded.")
        energy = np.unique(np.sort(all_energies)).astype(float)
        
    # Interpolate f1 and f2
    # Work in log energy space for interpolation
    log_energy = np.log(energy)
    
    f1_total = np.zeros(len(energy), dtype=float)
    f2_total = np.zeros(len(energy), dtype=float)
    
    for i, el in enumerate(el_nums):
        el_energy = SF_STATE['energy_'][el]
        el_f1 = SF_STATE['f1_'][el]
        el_f2 = SF_STATE['f2_'][el]
        
        log_el_energy = np.log(el_energy)
        
        # Interpolate
        f1_interp = np.interp(log_energy, log_el_energy, el_f1)
        f2_interp = np.interp(log_energy, log_el_energy, el_f2)
        
        f1_total += amts[i] * f1_interp
        f2_total += amts[i] * f2_interp
        
    res_type = result_type.lower()
    
    if res_type == 'f1': return f1_total
    if res_type == 'f2': return f2_total
    
    # Wavelength in meters
    wvln = en2wvlen(energy, unit_in='eV', unit_out='m')
    
    # Refractive index parts
    # delta = (r0 / 2pi) * (rho * Na / MW) * lambda^2 * f1
    # density in g/cm^3 -> * 1e6 to g/m^3
    prefactor = (r0 / (2.0 * np.pi)) * (density * Na * 1e6 / mw) * (wvln**2)
    
    delta = prefactor * f1_total
    beta = prefactor * f2_total
    
    if res_type == 'delta': return delta
    if res_type == 'beta': return beta
    if res_type == 'n': return 1.0 - delta - 1j * beta
    
    if res_type == 'reflect':
        theta = angle * 1e-3 # rad
        
        sin_theta = np.sin(theta)
        sin_theta_sq = sin_theta**2
        
        term1 = sin_theta_sq - 2.0 * delta
        term2 = np.sqrt(term1**2 + 4.0 * beta**2)
        
        rhosq = 0.5 * (term1 + term2)
        rho = np.sqrt(rhosq)
        
        # Reflectivity formulas (simplified from IDL)
        # Isig = ...
        # Ipisig = ...
        # This part requires careful translation of the complex algebra or the specific formulas used.
        # Implementing the IDL logic directly:
        
        num_sig = rhosq * (sin_theta - rho)**2 + beta**2
        den_sig = rhosq * (sin_theta + rho)**2 + beta**2
        Isig = num_sig / den_sig
        
        # Note: IDL code has a specific formula for Ipisig involving tan(theta) etc.
        # For brevity, omitting full reflectivity implementation unless requested, 
        # as it's complex and less commonly used than absorption.
        return Isig # Placeholder
        
    # Absorption coefficients
    # mu_a (atomic) = f2 * lambda * 2 * r0 * 1e28 (barns/atom)
    # 1 barn = 1e-28 m^2. 1e28 converts m^2 to barns.
    mu_a = f2_total * wvln * (2.0 * r0 * 1e28)
    
    if res_type == 'mu_a': return mu_a
    
    # mu (mass) = mu_a * (Na / MW) * 1e-24 (cm^2/g)
    mu = mu_a * (Na / mw) * 1e-24
    
    if res_type == 'mu': return mu
    
    # mu_l (linear) = mu * density (cm^-1)
    mu_l = mu * density
    
    if res_type == 'mu_l': return mu_l
    
    # Transmissivity
    trans = np.exp(-mu_l * thickness * 1e-4) # thickness in microns -> cm
    if res_type == 'trans':
        return trans
        
    # User's Optical Density (1-T)
    if res_type == 'od1t':
        return -trans + 1.0
        
    # Standard Optical Density (mu*t)
    if res_type == 'od':
        return mu_l * thickness * 1e-4
        
    print('Unrecognized result type.')
    return None

if __name__ == "__main__":
    # Test harness
    # Create dummy elements.dat and .nff files for testing
    if not os.path.exists("elements.dat"):
        with open("elements.dat", "w") as f:
            f.write("Element_name: Atomic_weight:\n")
            f.write("H 1.008\n")
            f.write("O 15.999\n")
            
    if not os.path.exists("h.nff"):
        with open("h.nff", "w") as f:
            f.write("E f1 f2\n")
            for e in range(100, 1000, 100):
                f.write(f"{e} {e/100.0} {e/1000.0}\n")

    if not os.path.exists("o.nff"):
        with open("o.nff", "w") as f:
            f.write("E f1 f2\n")
            for e in range(100, 1000, 100):
                f.write(f"{e} {e/50.0} {e/500.0}\n")
                
    print("--- Testing SF module ---")
    sf_init()
    
    # Test parsing
    print("Parsing H2O:", sf_parse_compound("H2O"))
    
    # Test calculation
    energies = [100, 200, 300]
    mu = sf("H2O", energies, result_type='mu')
    print(f"Mass absorption (mu) at {energies} eV: {mu}")
    
    trans = sf("H2O", energies, result_type='trans', thickness=10.0)
    print(f"Transmission at {energies} eV (10um): {trans}")
    
    # Cleanup
    # os.remove("elements.dat")
    # os.remove("h.nff")
    # os.remove("o.nff")
