import h5py
import numpy as np
import os
import re
import glob
import sys
import json
import tkinter as tk
from tkinter import filedialog
from alignment_utils import format_num_val

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

    with h5py.File(file_path, 'r') as f:
        directory = os.path.dirname(file_path)

        # --- Extract Metadata ---
        # --- Robust Metadata Extraction ---
        for key in ['project', 'scan_type', 'grating', 'harmonic', 'strip', 'command', 
                    'coordinates', 'beamline', 'polarization', 'exit_slit_gap', 
                    'xps_z', 'time_per_map', 'number_of_points', 'scan_name']:
            if key not in data_pack:
                data_pack[key] = 'N/A'

        search_groups = [
            f, f.get('scan_metadata'), f.get('stack_metadata'),
            f.get('entry'), f.get('entry/measurement'), f.get('entry/xanes_measurement'),
            f.get('map_data'), f.get('initial_motor_positions/all_beamline_motors_snapshot')
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
                if data_pack['time_per_map'] == 'N/A': data_pack['time_per_map'] = attrs.get('time_per_map', 'N/A')
                if data_pack['number_of_points'] == 'N/A': data_pack['number_of_points'] = attrs.get('number_of_points', attrs.get('num_points', 'N/A'))
                if data_pack['scan_name'] == 'N/A' and 'scan_name' in attrs: data_pack['scan_name'] = attrs['scan_name']

        for k in data_pack:
            if isinstance(data_pack[k], (bytes, np.bytes_)):
                data_pack[k] = data_pack[k].decode('utf-8')

        data_pack['date'] = robust_extract_date(file_path, f, None)
        
        # Determine fallback energy from metadata groups if not in map_data
        metadata_energy = -1.0
        for grp in search_groups:
            if grp is not None and hasattr(grp, 'attrs') and 'energy' in grp.attrs:
                metadata_energy = float(grp.attrs['energy'])
                break
                
        data_pack['energy'] = metadata_energy



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
        data_pack['exit_slit_gap'] = format_num_val(data_pack.get('exit_slit_gap'))
        data_pack['xps_z'] = format_num_val(data_pack.get('xps_z'))
        print(f"Number of Images:      1")
        print(f"Energy Regions:        {data_pack['Energy Regions']}")
        print(f"X array (shape):       {data_pack['x'].shape} (Nx: {data_pack['nx']})")
        print(f"Y array (shape):       {data_pack['y'].shape} (Ny: {data_pack['ny']})")
        pts = data_pack['nx'] * data_pack['ny']
        print(f"Grid Dimensions:       {data_pack['nx']} x {data_pack['ny']} ({pts} points)")
        print("----------------------------")
        print(f"Date:                  {data_pack.get('date', 'N/A')}")
        print(f"Scan Name:             {data_pack.get('scan_name', 'N/A')}")
        print(f"Project:               {data_pack.get('project', 'N/A')}")
        print(f"Scan Type:             {data_pack.get('scan_type', 'N/A')}")
        print(f"Endstation:            {data_pack.get('beamline', 'N/A')}")
        print(f"Polarization:          {data_pack.get('polarization', 'N/A')}")
        print(f"Grating:               {data_pack.get('grating', 'N/A')}")
        print(f"Harmonic:              {data_pack.get('harmonic', 'N/A')}")
        print(f"Strip:                 {data_pack.get('strip', 'N/A')}")
        print(f"Coordinates:           {data_pack.get('coordinates', 'N/A')}")
        print(f"Exit Slit Gap:         {data_pack.get('exit_slit_gap', 'N/A')}")
        print(f"XPS Z:                 {data_pack.get('xps_z', 'N/A')}")
        t_per_img = data_pack.get('time_per_map') or data_pack.get('time_per_image')
        if t_per_img and str(t_per_img).strip() not in ('N/A', 'None', ''):
            print(f"Time Per Image:        {t_per_img}")
        
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