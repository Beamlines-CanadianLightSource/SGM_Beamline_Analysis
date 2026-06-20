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
        title="Select HDF5 Map File",
        initialdir=last_dir,
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path

def analyze_map(file_path=None, verbose=True):
    """
    Scans data in the given HDF5 file's directory and returns a dictionary
    of file paths and metadata from a single map scan.

    Args:
        file_path (str, optional): The path to the HDF5 file from a map scan.
                                   If None, opens a file browser.
        
    Returns:
        dict: A dictionary containing coordinates, metadata, and data file paths.
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
        "energy": -1.0,
        "Number of images": 1,
        "Energy Regions": "N/A",
        "x": np.array([]),
        "y": np.array([]),
        "nx": 0,
        "ny": 0,
        "scan_name": "N/A",
        "project": "N/A",
        "date": "N/A",
        "exit_slit_gap": "N/A",
        "mcc_file": None,
        "sdd_files": {},
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

    with h5py.File(file_path, 'r') as f:
        directory = os.path.dirname(file_path)

        # --- Extract Metadata ---
        if 'scan_metadata' in f:
            metadata_attrs = f['scan_metadata'].attrs
            data_pack['scan_name'] = metadata_attrs.get('scan_name', 'N/A')
            data_pack['project'] = metadata_attrs.get('project', 'N/A')
            
            # Robust Date extraction
            data_pack['date'] = robust_extract_date(file_path, metadata_attrs)
            data_pack['exit_slit_gap'] = metadata_attrs.get('exit_slit_gap', 'N/A')

            # Energy extraction
            energy = metadata_attrs.get('energy', None)
            if energy is None:
                if 'initial_motor_positions' in f and 'all_beamline_motors_snapshot' in f['initial_motor_positions']:
                    motors_snapshot_attrs = f['initial_motor_positions/all_beamline_motors_snapshot'].attrs
                    energy = motors_snapshot_attrs.get('energy', -1.0)
            
            
            data_pack['energy'] = float(energy) if energy is not None else -1.0
        else:
            print("Warning: /scan_metadata group not found in HDF5 file.", file=sys.stderr)

        # Final fallback: Try to extract energy from the filename if it's still -1.0
        if data_pack['energy'] == -1.0:
            fname = os.path.basename(file_path)
            match = re.search(r'_(\d+\.\d+)eV', fname)
            if not match:
                match = re.search(r'_(\d+)eV', fname)
            if match:
                data_pack['energy'] = float(match.group(1))
                if verbose:
                    print(f"  [Fallback] Extracted energy {data_pack['energy']:.2f} eV from filename.")

        # --- Extract Coordinates ---
        if 'hexapod_waves/x' in f and 'hexapod_waves/y' in f:
            data_pack['x'] = f['hexapod_waves/x'][:]
            data_pack['y'] = f['hexapod_waves/y'][:]
            
            # Inferred Grid Calculation
            if data_pack['x'].size > 0 and data_pack['y'].size > 0:
                data_pack['nx'] = len(np.unique(np.round(data_pack['x'], 4)))
                data_pack['ny'] = len(np.unique(np.round(data_pack['y'], 4)))
        else:
            print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

        # --- Find Raw Data Files ---
        # MCC Data File
        mcc_files = glob.glob(os.path.join(directory, 'mcc*.csv'))
        if not mcc_files:
            mcc_files = glob.glob(os.path.join(directory, '*', 'mcc*.csv'))
        if mcc_files:
            data_pack['mcc_file'] = mcc_files[0]

        # SDD Data Files
        sdd_out_files = glob.glob(os.path.join(directory, 'sdd*.out'))
        sdd_bin_files = glob.glob(os.path.join(directory, 'sdd*_*.bin'))
        if not sdd_out_files and not sdd_bin_files:
            sdd_out_files = glob.glob(os.path.join(directory, '*', 'sdd*.out'))
            sdd_bin_files = glob.glob(os.path.join(directory, '*', 'sdd*_*.bin'))
        sdd_files = sdd_out_files + sdd_bin_files
        for sdd_file_path in sdd_files:
            match = re.match(r'(sdd\d+)', os.path.basename(sdd_file_path))
            if match:
                detector_name = match.group(1)
                data_pack['sdd_files'][detector_name] = sdd_file_path

    # --- Print Summary ---
    if verbose:
        print("\n--- Scan Analysis Summary ---")
        print(f"Energy:                {data_pack['energy']} eV")
        data_pack['Energy Regions'] = f"{data_pack['energy']:.2f}: 0.0 eV"
        data_pack['representative_energy'] = data_pack['energy']
        print(f"Number of Images:      1")
        print(f"Energy Regions:        {data_pack['Energy Regions']}")
        print(f"X array (shape):       {data_pack['x'].shape} (Nx: {data_pack['nx']})")
        print(f"Y array (shape):       {data_pack['y'].shape} (Ny: {data_pack['ny']})")
        print(f"Grid Dimensions:       {data_pack['nx']} x {data_pack['ny']}")
        print("----------------------------")
        print(f"Date:                  {data_pack['date']}")
        print(f"Scan Name:             {data_pack['scan_name']}")
        print(f"Project:               {data_pack['project']}")
        print(f"Exit Slit Gap:         {data_pack.get('exit_slit_gap', 'N/A')}")
        print(f"XPS Z:                 {data_pack.get('xps_z', 'N/A')}")
        print(f"Time Per Map:          {data_pack.get('time_per_map', 'N/A')}")
        
        if data_pack['mcc_file']:
            print(f"\nMCC File Found:       {data_pack['mcc_file']}")
        
        print("\nSDD Files Found:")
        if data_pack['sdd_files']:
            for detector, sdd_path in data_pack['sdd_files'].items():
                print(f"  Detector {detector}: {sdd_path}")
        print("-----------------------------------")

    return data_pack

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = None # Triggers file browser
        
    paths_data = analyze_map(file_path, verbose=True)