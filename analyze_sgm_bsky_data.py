import h5py
import numpy as np
import os
import re
import glob
import sys
import json
import tkinter as tk
from tkinter import filedialog

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
    """Opens a file dialog to select an HDF5 file."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    last_dir = get_last_dir()
    file_path = filedialog.askopenfilename(
        title="Select HDF5 Stack File",
        initialdir=last_dir,
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path

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
        "project": "N/A",
        "grating": "N/A",
        "harmonic": "N/A",
        "strip": "N/A",
        "command": "N/A",
        "coordinates": "N/A",
        "beamline": "N/A",
        "polarization": "N/A",
        "exit_slit_gap": "N/A",
        "mcc_files": {},
        "mcc_data": {},
        "mcc_channel_names": [],
        "sdd_files": {},
        "h5_dir": "N/A",
        "h5_file_path": os.path.abspath(file_path),
    }

    def robust_extract_date(f_path, attrs):
        # 1. Try HDF5 metadata
        for key in ['date', 'start_time', 'time']:
            val = attrs.get(key)
            if val: return str(val)
        
        # 2. Try filename regex (YYYY-MM-DD)
        fname = os.path.basename(f_path)
        match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if match: return match.group(1)
            
        # 3. Try directory names (search from leaf to root)
        path_parts = f_path.split(os.sep)
        for part in reversed(path_parts):
            match = re.search(r'(\d{4}-\d{2}-\d{2})', part)
            if match: return match.group(1)
        return "N/A"

    # Extract scan_name from direct parent as fallback
    try:
        data_pack['scan_name'] = os.path.basename(os.path.dirname(file_path))
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

        # --- Extract Metadata ---
        metadata_group = None
        if 'stack_metadata' in f:
            metadata_group = f['stack_metadata']
        elif 'scan_metadata' in f:
            metadata_group = f['scan_metadata']

        if metadata_group is not None:
            metadata_attrs = metadata_group.attrs
            data_pack['project'] = metadata_attrs.get('project', 'N/A')
            data_pack['grating'] = metadata_attrs.get('grating', 'N/A')
            data_pack['harmonic'] = metadata_attrs.get('harmonic', 'N/A')
            data_pack['strip'] = metadata_attrs.get('strip', 'N/A')
            data_pack['command'] = metadata_attrs.get('command', 'N/A')
            data_pack['coordinates'] = metadata_attrs.get('coordinates', 'N/A')
            data_pack['beamline'] = metadata_attrs.get('beamline', 'N/A')
            data_pack['polarization'] = metadata_attrs.get('polarization', 'N/A')
            data_pack['exit_slit_gap'] = metadata_attrs.get('exit_slit_gap', 'N/A')
            
            # Robust scan_name extraction (common in map metadata)
            if 'scan_name' in metadata_attrs:
                data_pack['scan_name'] = metadata_attrs['scan_name']
            
            # Robust Date extraction
            data_pack['date'] = robust_extract_date(file_path, metadata_attrs)
            
        else:
            print("Warning: Neither /stack_metadata nor /scan_metadata group found in HDF5 file.", file=sys.stderr)
            return None

        # --- Extract Energies and Coordinates ---
        if 'map_data/energy' in f:
            # Round energies to 2 decimal places immediately upon extraction
            data_pack['energies'] = np.round(f['map_data/energy'][:], 2)
        elif 'initial_motor_positions/all_beamline_motors_snapshot' in f:
            # Fallback for single map scans
            motors_snapshot_attrs = f['initial_motor_positions/all_beamline_motors_snapshot'].attrs
            energy = motors_snapshot_attrs.get('energy', -1.0)
            data_pack['energies'] = np.array([np.round(float(energy), 2)])
        elif metadata_group is not None and 'energy' in metadata_group.attrs:
            # Another fallback for energy in metadata attributes
            data_pack['energies'] = np.array([np.round(float(metadata_group.attrs['energy']), 2)])
        
        # Final fallback: Try to extract energy from the filename (e.g. ..._0.00eV.h5) if still missing or -1.0
        if len(data_pack['energies']) == 0 or (len(data_pack['energies']) == 1 and data_pack['energies'][0] == -1.0):
            fname = os.path.basename(file_path)
            match = re.search(r'_(\d+\.\d+)eV', fname)
            if not match:
                match = re.search(r'_(\d+)eV', fname)
                
            if match:
                extracted_energy = float(match.group(1))
                data_pack['energies'] = np.array([np.round(extracted_energy, 2)])
                if verbose:
                    print(f"  [Fallback] Extracted energy {extracted_energy:.2f} eV from filename.")
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
            match = re.search(r'_(\d+)_(\d+)eV$', d)
            if match:
                try:
                    energy_val = float(f"{match.group(1)}.{match.group(2)}")
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

    # --- Print Summary if requested ---
    if verbose:
        print("\n--- Scan Analysis Summary ---")
        print(f"Energies ({data_pack['energies'].shape[0]} points): {data_pack['energies']}")
        print(f"Number of Images:      {data_pack['Number of images']}")
        print(f"Energy Regions:        {data_pack['Energy Regions']}")
        print(f"X array (shape):       {data_pack['x'].shape} (Nx: {data_pack['nx']})")
        print(f"Y array (shape):       {data_pack['y'].shape} (Ny: {data_pack['ny']})")
        print(f"Grid Dimensions:       {data_pack['nx']} x {data_pack['ny']}")
        print("----------------------------")
        print(f"Date:                  {data_pack['date']}")
        print(f"Scan Name:             {data_pack['scan_name']}")
        print(f"Project:               {data_pack['project']}")
        print(f"Beamline:              {data_pack['beamline']}")
        print(f"Polarization:          {data_pack['polarization']}")
        print(f"Grating:               {data_pack['grating']}")
        print(f"Harmonic:              {data_pack['harmonic']}")
        print(f"Strip:                 {data_pack['strip']}")
        print(f"Command:               {data_pack['command']}")
        print(f"Coordinates:           {data_pack['coordinates']}")
        print(f"Exit Slit Gap:         {data_pack['exit_slit_gap']}")
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