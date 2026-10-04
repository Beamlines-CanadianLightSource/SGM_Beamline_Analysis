import h5py
import numpy as np
import os
import re
import glob
import sys
import json
import tkinter as tk
from tkinter import filedialog

from alignment_utils import format_num_val, safe_filedialog_call

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_dir.json")

def get_last_dir():
    """Reads the last accessed directory from a config file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return data.get("last_dir", os.getcwd())
        except Exception:
            pass
    return os.getcwd()

def save_last_dir(directory):
    """Saves the last accessed directory to a config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"last_dir": directory}, f)
    except Exception:
        pass

def browse_for_file():
    """Opens a file dialog to select an HDF5 file safely via subprocess to avoid Jupyter hangs."""
    last_dir = get_last_dir()
    file_path = safe_filedialog_call(
        filedialog.askopenfilename,
        title="Select HDF5 Stack File",
        initialdir=last_dir,
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    return file_path

def extract_sample_name(identifier):
    """
    Extracts the clean sample name from a scan identifier, directory name, or filename.
    
    Examples:
        '032433_TiON_sheet_30min_oK_fine_stack' -> 'TiON_sheet_30min'
        '032433_TiON_sheet_30min_o_k_fine_stack' -> 'TiON_sheet_30min'
        'TiON_sheet_30min_o_k_fine' -> 'TiON_sheet_30min'
        '083357_LMFP82_nocarbon_fe_l3l2_fine_stack' -> 'LMFP82_nocarbon'
        '123159_Na_molybdate_Infoil_mo_m32_semifine_stack' -> 'Na_molybdate_Infoil'
        '141438_Ca_silicate_Ctape_si_k_fine_stack' -> 'Ca_silicate_Ctape'
    """
    if not identifier:
        return ""
    s = str(identifier).strip()
    if s.endswith('.h5') or s.endswith('.hdf5'):
        s = s.rsplit('.', 1)[0]
    
    # 1. Strip leading timestamp or scan number (e.g., '032433_' or '2026-07-20_032434_')
    s = re.sub(r'^(?:\d{4}[-_]\d{2}[-_]\d{2}_)?\d{4,8}_', '', s)
    
    # 2. Strip absorption edge & scan mode suffixes
    edge_pattern = r'_(?:[a-zA-Z]{1,3}_(?:[kKLMNOklmno]\d*(?:[kKLMNOklmno]\d*)*)|[a-zA-Z]{1,3}(?:[kKLMNOklmno]\d*(?:[kKLMNOklmno]\d*)*))(?i:_(?:semi)?fine|_coarse|_fast|_slow)?(?i:_(?:fine|stack|map)(?:_data)?)*$'
    m = re.search(edge_pattern, s)
    if m:
        s = s[:m.start()]
    else:
        # Fallback: remove general trailing scan mode keywords
        s = re.sub(r'(?i:_(?:semi)?fine|_coarse|_fast|_slow)?(?i:_(?:fine|stack|map)(?:_data)?)+$', '', s)
        
    return s.strip('_')


def resolve_sample_name(file_path=None, h5_obj=None, scan_name=None):
    """
    Robustly resolves the human-meaningful sample name from HDF5 metadata,
    parent directory name, or filename.
    """
    # 1. Check explicit HDF5 attributes
    if h5_obj is not None:
        search_groups = [
            h5_obj, h5_obj.get('scan_metadata'), h5_obj.get('stack_metadata'),
            h5_obj.get('entry'), h5_obj.get('map_data')
        ]
        for grp in search_groups:
            if grp is not None and hasattr(grp, 'attrs'):
                for key in ['sample_name', 'sample']:
                    val = grp.attrs.get(key)
                    if val and str(val).strip() not in ('N/A', 'None', ''):
                        return str(val).strip()

    # 2. Try h5 stack_metadata/scan_metadata scan_name attribute
    if h5_obj is not None:
        for grp_name in ['stack_metadata', 'scan_metadata']:
            if grp_name in h5_obj and 'scan_name' in h5_obj[grp_name].attrs:
                cand = h5_obj[grp_name].attrs['scan_name']
                s = extract_sample_name(cand)
                if s and s.lower() not in ('stack', 'map', 'stack_data', 'map_data'):
                    return s

    # 3. Try parent directory of file_path (e.g. 032433_TiON_sheet_30min_o_k_fine_stack)
    if file_path:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
        s = extract_sample_name(parent_dir)
        if s and s.lower() not in ('stack', 'map', 'stack_data', 'map_data'):
            return s

    # 4. Try scan_name parameter
    if scan_name:
        s = extract_sample_name(scan_name)
        if s and s.lower() not in ('stack', 'map', 'stack_data', 'map_data'):
            return s

    # 5. Try file stem
    if file_path:
        fname = os.path.basename(file_path)
        s = extract_sample_name(fname)
        if s and s.lower() not in ('stack', 'map', 'stack_data', 'map_data'):
            return s

    return scan_name or "Sample"


def detect_energy_regions(energies):
    """
    Identifies continuous energy regions with constant spacing.
    Returns a formatted string like '280.0-281.5: 0.5 eV, 281.6-282.0: 0.1 eV'.
    """
    if energies is None or len(energies) == 0: 
        return "N/A"
    if len(energies) == 1: 
        return f"{energies[0]:.2f}: 0.0 eV"
    
    # Ensure unique and sorted energies
    en_sorted = np.sort(np.unique(energies))
    regions = []
    
    start_idx = 0
    while start_idx < len(en_sorted):
        # Case for the very last point if it wasn't part of a previous group
        if start_idx == len(en_sorted) - 1:
            regions.append(f"{en_sorted[start_idx]:.2f}: 0.0 eV")
            break
            
        # Determine the spacing for the current potential region
        # We need at least two points to have a spacing
        spacing = round(en_sorted[start_idx+1] - en_sorted[start_idx], 4)
        end_idx = start_idx + 1
        
        # Look ahead to find all points with this same spacing
        while end_idx + 1 < len(en_sorted):
            next_spacing = round(en_sorted[end_idx+1] - en_sorted[end_idx], 4)
            # Use 0.001 as tolerance for floating point comparisons
            if abs(next_spacing - spacing) < 0.001:
                end_idx += 1
            else:
                break
        
        if end_idx == start_idx:
             # Should not happen as we checked start_idx == len-1
             regions.append(f"{en_sorted[start_idx]:.2f}: 0.0 eV")
             start_idx = end_idx + 1
        else:
            regions.append(f"{en_sorted[start_idx]:.2f}-{en_sorted[end_idx]:.2f}: {spacing} eV")
            start_idx = end_idx + 1
        
    return ", ".join(regions)

def analyze_sgm_bsky_data(file_path=None, verbose=True):
    """
    Scans data in the given HDF5 file's directory and returns a dictionary
    of file paths and metadata, organized by energy and detector.

    Args:
        file_path (str, optional): The path to the HDF5 file from a stack scan.
                                   If None, opens a file browser.
        
    Returns:
        dict: A dictionary containing energies, coordinates, metadata, and data file paths.
    """
    if file_path is None:
        file_path = browse_for_file()
        
    if not file_path:
        if verbose:
            print("No file selected.", file=sys.stderr)
        return None

    if verbose:
        print(f"\nAnalyzing File: {os.path.abspath(file_path)}")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return None

    # Save the directory for next time
    save_last_dir(os.path.dirname(os.path.abspath(file_path)))

    data_pack = {
        "energies": np.array([]),
        "Number of images": 0,
        "Energy Regions": "N/A",
        "x": np.array([]),
        "y": np.array([]),
        "nx": "N/A",
        "ny": "N/A",
        "date": "N/A",
        "scan_name": "N/A",
        "sample_name": "N/A",
        "project": "N/A",
        "grating": "N/A",
        "harmonic": "N/A",
        "strip": "N/A",
        "command": "N/A",
        "coordinates": "N/A",
        "beamline": "N/A",
        "polarization": "N/A",
        "exit_slit_gap": "N/A",
        "xps_z": "N/A",
        "time_per_map": "N/A",
        "mcc_files": {},
        "mcc_data": {},
        "mcc_channel_names": [],
        "sdd_files": {},
        "h5_dir": "N/A",
        "h5_file_path": os.path.abspath(file_path),
    }

    def robust_extract_date(f_path, f_obj=None, attrs=None):
        # 1. Try passed metadata attrs
        if attrs:
            for key in ['session', 'date', 'start_time', 'time', 'timestamp', 'datetime', 'end_time']:
                val = attrs.get(key)
                if val and str(val).strip() not in ('N/A', 'None', ''):
                    return str(val).strip()
        
        # 2. Try searching all common HDF5 groups if f_obj is available
        if f_obj is not None:
            groups = [f_obj, f_obj.get('scan_metadata'), f_obj.get('stack_metadata'),
                      f_obj.get('entry'), f_obj.get('map_data'),
                      f_obj.get('initial_motor_positions/all_beamline_motors_snapshot')]
            for grp in groups:
                if grp is not None and hasattr(grp, 'attrs'):
                    for key in ['session', 'date', 'start_time', 'time', 'timestamp', 'datetime', 'end_time']:
                        val = grp.attrs.get(key)
                        if val and str(val).strip() not in ('N/A', 'None', ''):
                            return str(val).strip()

        # 3. Try filename regex (YYYY-MM-DD or YYYY_MM_DD)
        fname = os.path.basename(f_path)
        match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', fname)
        if match: return match.group(1).replace('_', '-')
            
        # 4. Try directory names (search from leaf to root)
        path_parts = os.path.abspath(f_path).split(os.sep)
        for part in reversed(path_parts):
            match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', part)
            if match: return match.group(1).replace('_', '-')
            
        # 5. File modification date fallback from disk
        try:
            import datetime
            mtime = os.path.getmtime(f_path)
            return datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        except Exception:
            pass
            
        return "N/A"

    # Extract scan_name from filename stem as initial fallback
    try:
        data_pack['scan_name'] = os.path.splitext(os.path.basename(file_path))[0]
    except Exception:
        pass

    # Use swmr=True for robustness if file is being written to
    try:
        f = h5py.File(file_path, 'r', swmr=True)
    except OSError:
        # Fallback if SWMR fails or file is locked differently
        try:
            f = h5py.File(file_path, 'r')
        except Exception as e:
            print(f"Error opening HDF5 file: {e}", file=sys.stderr)
            return None

    with f:
        stack_dir = os.path.dirname(file_path)
        data_pack['h5_dir'] = stack_dir

        # --- Robust Metadata Extraction ---
        for key in ['project', 'scan_type', 'grating', 'harmonic', 'strip', 'command', 
                    'coordinates', 'beamline', 'polarization', 'exit_slit_gap', 
                    'xps_z', 'time_per_map', 'number_of_points', 'scan_name', 'sample_name']:
            if key not in data_pack:
                data_pack[key] = 'N/A'

        search_groups = [
            f, f.get('scan_metadata'), f.get('stack_metadata'),
            f.get('entry'), f.get('entry/measurement'), f.get('entry/xanes_measurement'),
            f.get('map_data'), f.get('initial_motor_positions/all_beamline_motors_snapshot'),
            f.get('stitching_metadata')
        ]
        
        for grp in search_groups:
            if grp is not None and hasattr(grp, 'attrs'):
                attrs = grp.attrs
                if data_pack['project'] == 'N/A': data_pack['project'] = attrs.get('project', 'N/A')
                if data_pack['scan_type'] == 'N/A': data_pack['scan_type'] = attrs.get('plan_name', 'N/A')
                if data_pack['grating'] == 'N/A': data_pack['grating'] = attrs.get('grating', attrs.get('grating_selection', 'N/A'))
                if data_pack['harmonic'] == 'N/A': data_pack['harmonic'] = attrs.get('harmonic', 'N/A')
                if data_pack['strip'] == 'N/A': data_pack['strip'] = attrs.get('stripe', attrs.get('strip', attrs.get('mirror_stripe', attrs.get('mirror_strip', 'N/A'))))
                if data_pack['command'] == 'N/A': data_pack['command'] = attrs.get('command', 'N/A')
                if data_pack['coordinates'] == 'N/A': data_pack['coordinates'] = attrs.get('coordinates', 'N/A')
                if data_pack['beamline'] == 'N/A': data_pack['beamline'] = attrs.get('beamline', 'N/A')
                if data_pack['polarization'] == 'N/A': data_pack['polarization'] = attrs.get('polarization', 'N/A')
                if data_pack['exit_slit_gap'] == 'N/A': data_pack['exit_slit_gap'] = attrs.get('exit_slit_gap', 'N/A')
                if data_pack['xps_z'] == 'N/A': data_pack['xps_z'] = attrs.get('vaz', attrs.get('xps_z', 'N/A'))
                if data_pack['time_per_map'] == 'N/A': data_pack['time_per_map'] = attrs.get('time_per_map', attrs.get('time_per_image', 'N/A'))
                if data_pack['number_of_points'] == 'N/A': data_pack['number_of_points'] = attrs.get('number_of_points', attrs.get('num_points', 'N/A'))
                if data_pack['scan_name'] in ('N/A', '') and 'scan_name' in attrs: data_pack['scan_name'] = attrs['scan_name']
                if data_pack['sample_name'] in ('N/A', ''):
                    if 'sample_name' in attrs: data_pack['sample_name'] = attrs['sample_name']
                    elif 'sample' in attrs: data_pack['sample_name'] = attrs['sample']

        if 'stitching_metadata' in f and 'stitched_tag' in f['stitching_metadata'].attrs:
            data_pack['scan_name'] = f"Stitched_{f['stitching_metadata'].attrs['stitched_tag']}"

        # Resolve sample name robustly (from attributes, parent folder, or scan name)
        resolved_sample = resolve_sample_name(file_path, f, data_pack.get('scan_name'))
        if resolved_sample and resolved_sample != 'N/A':
            data_pack['sample_name'] = resolved_sample

        for k in data_pack:
            if isinstance(data_pack[k], (bytes, np.bytes_)):
                data_pack[k] = data_pack[k].decode('utf-8')

        data_pack['date'] = robust_extract_date(file_path, f, None)
        
        # Determine fallback energy from metadata groups if not in map_data
        metadata_energy = -1.0
        for grp in search_groups:
            if grp is not None and hasattr(grp, 'attrs') and 'energy' in grp.attrs:
                try:
                    metadata_energy = float(grp.attrs['energy'])
                    break
                except (ValueError, TypeError):
                    pass
        
        if 'map_data/energy' in f and len(f['map_data/energy']) > 0:
            # Round energies to 2 decimal places immediately upon extraction
            data_pack['energies'] = np.round(f['map_data/energy'][:], 2)
        elif metadata_energy != -1.0:
            data_pack['energies'] = np.array([np.round(metadata_energy, 2)])
        
        # Final fallback: Try to extract energy from the filename or parent dir (e.g. ..._1195.00eV.h5 or 1195eV_Maps) if still missing or -1.0
        if len(data_pack.get('energies', [])) == 0 or (len(data_pack['energies']) == 1 and data_pack['energies'][0] == -1.0):
            fname = os.path.basename(file_path)
            match = re.search(r'_(\d+\.\d+)eV', fname) or re.search(r'_(\d+)eV', fname)
            
            extracted_energy = None
            if match:
                extracted_energy = float(match.group(1))
            else:
                # Try parent directory name (e.g. 1195eV_Maps or 1195_00eV)
                parent_dir = os.path.basename(os.path.dirname(os.path.abspath(file_path)))
                p_match = re.search(r'(\d+\.\d+)eV', parent_dir) or re.search(r'(\d+)eV', parent_dir)
                if p_match:
                    extracted_energy = float(p_match.group(1))
                else:
                    p_match2 = re.search(r'(\d+)_(\d+)eV', parent_dir)
                    if p_match2:
                        extracted_energy = float(f"{p_match2.group(1)}.{p_match2.group(2)}")
                        
            if extracted_energy is not None:
                data_pack['energies'] = np.array([np.round(extracted_energy, 2)])
                if verbose:
                    print(f"  [Fallback] Extracted energy {extracted_energy:.2f} eV from filename/directory.")
            elif len(data_pack['energies']) == 0:
                print("Warning: Energy data not found in HDF5 file or filename.", file=sys.stderr)
                return data_pack

        if 'hexapod_waves/x' in f and 'hexapod_waves/y' in f:
            data_pack['x'] = f['hexapod_waves/x'][:]
            data_pack['y'] = f['hexapod_waves/y'][:]
            
            # Infer grid dimensions
            if data_pack['x'].size > 0 and data_pack['y'].size > 0:
                data_pack['nx'] = len(np.unique(np.round(data_pack['x'], 4)))
                data_pack['ny'] = len(np.unique(np.round(data_pack['y'], 4)))
            
            if data_pack['coordinates'] == 'N/A':
                 data_pack['coordinates'] = f"X: {np.min(data_pack['x']):.2f} to {np.max(data_pack['x']):.2f}, Y: {np.min(data_pack['y']):.2f} to {np.max(data_pack['y']):.2f}"

        else:
            print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

        # --- Pre-scan Subdirectories for Fuzzy Matching ---
        subdirs = [d for d in os.listdir(stack_dir) if os.path.isdir(os.path.join(stack_dir, d))]
        
        dir_energy_map = {}
        for d in subdirs:
            # Match subdirectories ending in _1195_00eV or _1195eV
            match = re.search(r'_(\d+)_(\d+)eV$', d)
            if match:
                try:
                    energy_val = float(f"{match.group(1)}.{match.group(2)}")
                    # Prefer subdirectory matching current scan_name prefix
                    if data_pack['scan_name'] != 'N/A' and d.startswith(data_pack['scan_name']):
                        dir_energy_map[energy_val] = os.path.join(stack_dir, d)
                    elif energy_val not in dir_energy_map:
                        dir_energy_map[energy_val] = os.path.join(stack_dir, d)
                except ValueError:
                    continue

        # --- Find Raw Data Files ---
        got_mcc_header = False
        
        for energy in data_pack['energies']:
            en_dir_path = None
            
            if energy in dir_energy_map:
                en_dir_path = dir_energy_map[energy]
            else:
                closest_energy = None
                min_diff = 0.05 
                
                for dir_en in dir_energy_map.keys():
                    diff = abs(dir_en - energy)
                    if diff < min_diff:
                        min_diff = diff
                        closest_energy = dir_en
                
                if closest_energy is not None:
                    en_dir_path = dir_energy_map[closest_energy]
            
            if not en_dir_path:
                energy_str = f"{energy:.2f}".replace('.', '_')
                expected_subdir_name = f"{data_pack['scan_name']}_{energy_str}eV"
                fallback_path = os.path.join(stack_dir, expected_subdir_name)
                
                if os.path.isdir(fallback_path):
                    en_dir_path = fallback_path
                else:
                    # Fallback to the root directory if no subdirectory is found (common for single maps)
                    en_dir_path = stack_dir

            # MCC Data File
            mcc_files = glob.glob(os.path.join(en_dir_path, 'mcc*.csv'))
            if mcc_files:
                mcc_file_path = mcc_files[0]
                data_pack['mcc_files'][energy] = mcc_file_path
                
                try:
                    if not got_mcc_header:
                        with open(mcc_file_path, 'r') as mcc_f:
                            header = mcc_f.readline().strip()
                            if header.startswith('#'):
                                header = header[1:]
                            data_pack['mcc_channel_names'] = [name.strip() for name in header.split(',')]
                        got_mcc_header = True
                    
                    data_pack['mcc_data'][energy] = np.genfromtxt(mcc_file_path, delimiter=',', skip_header=1)
                except Exception as e:
                    print(f"Warning: Failed to load MCC data from {mcc_file_path}: {e}", file=sys.stderr)

            # SDD Data Files
            sdd_out_files = glob.glob(os.path.join(en_dir_path, 'sdd*.out'))
            sdd_bin_files = glob.glob(os.path.join(en_dir_path, 'sdd*_*.bin'))
            sdd_files = sdd_out_files + sdd_bin_files
            if not sdd_files:
                continue

            for sdd_file_path in sdd_files:
                match = re.match(r'(sdd\d+)', os.path.basename(sdd_file_path))
                if not match:
                    continue
                detector_name = match.group(1)

                if detector_name not in data_pack['sdd_files']:
                    data_pack['sdd_files'][detector_name] = {}
                
                data_pack['sdd_files'][detector_name][energy] = sdd_file_path

    # Set total image count and detect energy regions
    data_pack['Number of images'] = len(data_pack['energies'])
    data_pack['Energy Regions'] = detect_energy_regions(data_pack['energies'])
    if len(data_pack['energies']) > 0:
        data_pack['representative_energy'] = data_pack['energies'][len(data_pack['energies']) // 2]
    else:
        data_pack['representative_energy'] = -1.0

    data_pack['exit_slit_gap'] = format_num_val(data_pack.get('exit_slit_gap'))
    data_pack['xps_z'] = format_num_val(data_pack.get('xps_z'))

    # --- Print Summary if requested ---
    if verbose:
        print("\n--- Scan Analysis Summary ---")
        print(f"Energies ({data_pack['energies'].shape[0]} points): {data_pack['energies']}")
        print(f"Number of Images:      {data_pack['Number of images']}")
        print(f"Energy Regions:        {data_pack['Energy Regions']}")
        print(f"X array (shape):       {data_pack['x'].shape} (Nx: {data_pack['nx']})")
        print(f"Y array (shape):       {data_pack['y'].shape} (Ny: {data_pack['ny']})")
        pts = data_pack['nx'] * data_pack['ny']
        print(f"Grid Dimensions:       {data_pack['nx']} x {data_pack['ny']} ({pts} points)")
        print("----------------------------")
        print(f"Date:                  {data_pack['date']}")
        print(f"Scan Name:             {data_pack['scan_name']}")
        if data_pack.get('sample_name') and data_pack['sample_name'] != 'N/A':
            print(f"Sample Name:           {data_pack['sample_name']}")
        print(f"Project:               {data_pack['project']}")
        print(f"Scan Type:             {data_pack['scan_type']}")
        print(f"Endstation:            {data_pack['beamline']}")
        print(f"Polarization:          {data_pack['polarization']}")
        print(f"Grating:               {data_pack['grating']}")
        print(f"Harmonic:              {data_pack['harmonic']}")
        print(f"Strip:                 {data_pack['strip']}")
        print(f"Coordinates:           {data_pack['coordinates']}")
        print(f"Exit Slit Gap:         {data_pack['exit_slit_gap']}")
        print(f"XPS Z:                 {data_pack['xps_z']}")
        t_per_img = data_pack.get('time_per_map') or data_pack.get('time_per_image')
        if t_per_img and str(t_per_img).strip() not in ('N/A', 'None', ''):
            print(f"Time Per Image:        {t_per_img}")
        print(f"\nMCC Files Found:       {len(data_pack['mcc_files'])}")
        print("\nSDD Files Found:")
        for detector, sdd_dict in data_pack['sdd_files'].items():
            print(f"  Detector {detector}: {len(sdd_dict)} files")
        print("-----------------------------------")

    return data_pack

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = None # Triggers file browser
        
    paths_data = analyze_sgm_bsky_data(file_path, verbose=True)