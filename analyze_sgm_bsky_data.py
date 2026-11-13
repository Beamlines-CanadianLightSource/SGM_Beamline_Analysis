import h5py
import numpy as np
import os
import re
import glob
import sys
import json
try:
    import tkinter as tk
    from tkinter import filedialog
except (ImportError, ModuleNotFoundError):
    tk = None
    filedialog = None

from alignment_utils import format_num_val, safe_filedialog_call, HAS_GUI
import datetime

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".last_dir.json")

# Detector physical position configurations before/on and after August 12, 2026
SDD_POSITIONS_CUTOFF_DATE = datetime.date(2026, 8, 12)

SDD_NAMES_PRE_2026_08_12 = {
    1: "sdd1 (90, top)",
    2: "sdd2 (45, front, in-plane)",
    3: "sdd3 (90, side)",
    4: "sdd4 (45, front, out-plane)"
}

SDD_NAMES_POST_2026_08_12 = {
    1: "sdd1 (90, side)",
    2: "sdd2 (45, front, in-plane)",
    3: "sdd3 (45, front, out of plane)",
    4: "sdd4 (90, top)"
}


def _extract_date_from_text(text):
    """Attempts to parse a datetime.date from a string (YYYY-MM-DD, YYYY_MM_DD, or YYMMDD folder prefix)."""
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    if s in ('N/A', 'None', ''):
        return None
    # 1. Standard YYYY-MM-DD or YYYY_MM_DD or YYYY/MM/DD
    m = re.search(r'\b(20\d{2})[-_/](\d{2})[-_/](\d{2})\b', s)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # 2. Compact 8-digit YYYYMMDD (e.g. 20261028)
    m8 = re.search(r'(?<!\d)(20[2-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)', s)
    if m8:
        try:
            return datetime.date(int(m8.group(1)), int(m8.group(2)), int(m8.group(3)))
        except ValueError:
            pass
    # 3. 6-digit YYMMDD prefix in directory/scan names (e.g. '260728-Butterfly_wing')
    for part in re.split(r'[\\/]', s):
        m6 = re.match(r'^([2-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:[-_A-Za-z]|$)', part)
        if m6:
            try:
                return datetime.date(2000 + int(m6.group(1)), int(m6.group(2)), int(m6.group(3)))
            except ValueError:
                pass
    return None


def parse_scan_date(date_or_source=None):
    """
    Resolves a scan's acquisition date (as a datetime.date object) from a date string,
    datetime/date object, data_pack/context dict, or HDF5 file path.
    """
    if date_or_source is None:
        return None
    if isinstance(date_or_source, datetime.datetime):
        return date_or_source.date()
    if isinstance(date_or_source, datetime.date):
        return date_or_source

    if isinstance(date_or_source, dict):
        # Check direct 'date' key or nested 'path_pack'
        candidates = [
            date_or_source.get('date'),
            (date_or_source.get('path_pack') or {}).get('date') if isinstance(date_or_source.get('path_pack'), dict) else None,
            date_or_source.get('scan_name'),
            (date_or_source.get('path_pack') or {}).get('scan_name') if isinstance(date_or_source.get('path_pack'), dict) else None,
            date_or_source.get('h5_file_path'),
            (date_or_source.get('path_pack') or {}).get('h5_file_path') if isinstance(date_or_source.get('path_pack'), dict) else None,
            date_or_source.get('h5_dir'),
            date_or_source.get('save_dir'),
        ]
        for cand in candidates:
            dt = _extract_date_from_text(str(cand)) if cand else None
            if dt is not None:
                return dt
        # If h5_file_path exists on disk, inspect HDF5 attributes / source_files
        h5_p = date_or_source.get('h5_file_path') or (
            date_or_source.get('path_pack', {}).get('h5_file_path')
            if isinstance(date_or_source.get('path_pack'), dict) else None
        )
        if h5_p and isinstance(h5_p, str) and os.path.isfile(h5_p):
            return parse_scan_date(h5_p)
        return None

    if isinstance(date_or_source, str):
        dt = _extract_date_from_text(date_or_source)
        if dt is not None:
            return dt
        if os.path.isfile(date_or_source) and date_or_source.lower().endswith(('.h5', '.hdf5', '.nxs')):
            try:
                with h5py.File(date_or_source, 'r') as f:
                    for grp_name in ['', 'scan_metadata', 'stack_metadata', 'entry', 'entry/measurement', 'stitching_metadata']:
                        grp = f if grp_name == '' else f.get(grp_name)
                        if grp is not None and hasattr(grp, 'attrs'):
                            for key in ['date', 'session', 'start_time', 'time', 'timestamp', 'datetime']:
                                val = grp.attrs.get(key)
                                if isinstance(val, (bytes, np.bytes_)):
                                    val = val.decode('utf-8', errors='ignore')
                                dt = _extract_date_from_text(str(val)) if val is not None else None
                                if dt is not None:
                                    return dt
                            if 'source_files' in grp.attrs:
                                for sf in grp.attrs['source_files']:
                                    if isinstance(sf, (bytes, np.bytes_)):
                                        sf = sf.decode('utf-8', errors='ignore')
                                    dt = _extract_date_from_text(str(sf))
                                    if dt is not None:
                                        return dt
            except Exception:
                pass
            try:
                return datetime.datetime.fromtimestamp(os.path.getmtime(date_or_source)).date()
            except Exception:
                pass
    return None


def is_post_aug_12_2026(date_or_source=None):
    """
    Returns True if the scan date is strictly after August 12, 2026 (> 2026-08-12),
    or if no date can be determined (defaulting to the current post-Aug 12, 2026 layout).
    Returns False if the scan date is on or before August 12, 2026 (<= 2026-08-12).
    """
    dt = parse_scan_date(date_or_source)
    if dt is None:
        return True
    return dt > SDD_POSITIONS_CUTOFF_DATE


def get_sdd_names_map(date_or_source=None):
    """
    Returns the dictionary mapping SDD detector integer ID (1..4) to its human-readable
    position string based on whether the scan was acquired after Aug 12, 2026.
    """
    if is_post_aug_12_2026(date_or_source):
        return dict(SDD_NAMES_POST_2026_08_12)
    return dict(SDD_NAMES_PRE_2026_08_12)


def get_sdd_label(det, date_or_source=None, uppercase=False):
    """
    Given a detector identifier (e.g., 1, 'sdd1', 'SDD1'), returns its date-appropriate
    position label (e.g., 'sdd1 (90, side)' after Aug 12, 2026 or 'sdd1 (90, top)' on/before Aug 12, 2026).
    If uppercase=True, capitalizes the detector prefix (e.g., 'SDD1 (90, side)').
    """
    sdd_map = get_sdd_names_map(date_or_source)
    det_id = None
    if isinstance(det, (int, np.integer)):
        det_id = int(det)
    elif isinstance(det, str):
        m = re.match(r'^sdd(\d+)$', det.strip(), re.IGNORECASE)
        if m:
            det_id = int(m.group(1))
    if det_id in sdd_map:
        lbl = sdd_map[det_id]
        if uppercase:
            lbl = re.sub(r'^sdd(\d+)', lambda m_obj: f"SDD{m_obj.group(1)}", lbl)
        return lbl
    if isinstance(det, str):
        return det.upper() if uppercase else det
    return str(det)

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
    func = getattr(filedialog, 'askopenfilename', 'askopenfilename') if filedialog else 'askopenfilename'
    file_path = safe_filedialog_call(
        func,
        title="Select HDF5 Stack File",
        initialdir=last_dir,
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    return file_path

def resolve_preprocessed_save_dir(raw_dir):
    r"""
    If the directory path contains a 'raw' folder component (e.g. from beamline storage like
    V:\projects\prj41S15109\raw\2026-08-13\042855_CuO_cu_l3l2_fine_stack),
    maps it to the corresponding 'preprocessed' directory:
    V:\projects\prj41S15109\preprocessed\2026-08-13\042855_CuO_cu_l3l2_fine_stack.
    Otherwise, returns the raw_dir unchanged.
    """
    if not raw_dir or raw_dir == 'N/A':
        return raw_dir
    norm_dir = os.path.abspath(raw_dir)
    is_unc = norm_dir.startswith('\\\\') or norm_dir.startswith('//')
    parts = norm_dir.split(os.sep)
    raw_indices = [i for i, part in enumerate(parts) if part.lower() == 'raw']
    if raw_indices:
        idx = raw_indices[-1]
        parts[idx] = 'preprocessed'
        res = os.sep.join(parts)
        if is_unc and not res.startswith('\\\\'):
            res = '\\\\' + res.lstrip('\\')
        return res
    return norm_dir

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


def build_scan_filename_base(scan_name, sample_name=None):
    """
    Constructs a descriptive filename base containing the date/identifier,
    sample name, and scan descriptor.
    
    Examples:
        scan_name='2026-07-21-154341_stack_data', sample_name='BN_ctape'
        -> '2026-07-21-154341_BN_ctape_stack_data'
        
        scan_name='2026-07-19_133033_stack_data', sample_name='TiON_sheet_30min'
        -> '2026-07-19_133033_TiON_sheet_30min_stack_data'
        
        scan_name='154339_BN_ctape_o_k_fine_stack', sample_name='BN_ctape'
        -> '154339_BN_ctape_o_k_fine_stack' (already contains sample name)
        
        scan_name='scan1', sample_name='SampleA'
        -> 'scan1_SampleA'
    """
    if not scan_name:
        return sample_name or "scan"
    if not sample_name or str(sample_name).strip() in ('N/A', 'None', '', 'Sample'):
        return scan_name

    s_clean = re.sub(r'[^\w\-.]', '_', str(sample_name).strip())
    s_clean = re.sub(r'_+', '_', s_clean).strip('_')
    if not s_clean or s_clean.lower() in scan_name.lower():
        return scan_name

    # If scan_name starts with date/identifier (e.g. '2026-07-21-154341_' or '2026-07-21_154341_' or '154339_')
    m = re.match(r'^((?:\d{4}[-_]\d{2}[-_]\d{2}[-_])?\d{4,8})(?:_(.*))?$', scan_name)
    if m:
        prefix = m.group(1)
        rest = m.group(2)
        if rest:
            return f"{prefix}_{s_clean}_{rest}"
        else:
            return f"{prefix}_{s_clean}"

    return f"{scan_name}_{s_clean}"


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
        "pixel_size": "N/A",
        "pixel_size_x_um": "N/A",
        "pixel_size_y_um": "N/A",
        "image_size_x_mm": "N/A",
        "image_size_y_mm": "N/A",
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
        "save_dir": "N/A",
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
                      f_obj.get('initial_motor_positions/all_beamline_motors_snapshot'),
                      f_obj.get('stitching_metadata')]
            for grp in groups:
                if grp is not None and hasattr(grp, 'attrs'):
                    for key in ['session', 'date', 'start_time', 'time', 'timestamp', 'datetime', 'end_time']:
                        val = grp.attrs.get(key)
                        if isinstance(val, (bytes, np.bytes_)):
                            val = val.decode('utf-8', errors='ignore')
                        if val and str(val).strip() not in ('N/A', 'None', ''):
                            return str(val).strip()
                    if 'source_files' in grp.attrs:
                        for sf in grp.attrs['source_files']:
                            if isinstance(sf, (bytes, np.bytes_)):
                                sf = sf.decode('utf-8', errors='ignore')
                            dt_sf = _extract_date_from_text(str(sf))
                            if dt_sf is not None:
                                return dt_sf.strftime('%Y-%m-%d')

        # 3. Try filename and directory path via _extract_date_from_text (YYYY-MM-DD, YYYYMMDD, or YYMMDD)
        dt_path = _extract_date_from_text(os.path.abspath(f_path))
        if dt_path is not None:
            return dt_path.strftime('%Y-%m-%d')
            
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
        data_pack['save_dir'] = resolve_preprocessed_save_dir(stack_dir)

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
                nx = len(np.unique(np.round(data_pack['x'], 4)))
                ny = len(np.unique(np.round(data_pack['y'], 4)))
                data_pack['nx'] = nx
                data_pack['ny'] = ny

                min_x = float(np.min(data_pack['x']))
                max_x = float(np.max(data_pack['x']))
                size_x = abs(max_x - min_x)

                min_y = float(np.min(data_pack['y']))
                max_y = float(np.max(data_pack['y']))
                size_y = abs(max_y - min_y)

                data_pack['image_size_x_mm'] = round(size_x, 4)
                data_pack['image_size_y_mm'] = round(size_y, 4)

                data_pack['coordinates'] = (
                    f"X: {min_x:.2f} to {max_x:.2f} mm ({size_x:.2f} mm), "
                    f"Y: {min_y:.2f} to {max_y:.2f} mm ({size_y:.2f} mm)"
                )

                if nx > 0 and ny > 0:
                    px_x_um = (size_x / float(nx)) * 1000.0
                    px_y_um = (size_y / float(ny)) * 1000.0
                    data_pack['pixel_size_x_um'] = round(px_x_um, 2)
                    data_pack['pixel_size_y_um'] = round(px_y_um, 2)
                    data_pack['pixel_size'] = f"{px_x_um:.2f} µm x {px_y_um:.2f} µm"

        else:
            print("Warning: Coordinate data (hexapod_waves/x or y) not found.", file=sys.stderr)

        # Fallback if coordinates were loaded from metadata group without image size
        if data_pack['coordinates'] != 'N/A' and data_pack['pixel_size'] == 'N/A':
            coords_str = str(data_pack['coordinates']).strip()
            match = re.search(r'X:\s*([-\d.]+)\s*to\s*([-\d.]+),\s*Y:\s*([-\d.]+)\s*to\s*([-\d.]+)', coords_str)
            if match:
                min_x, max_x = float(match.group(1)), float(match.group(2))
                min_y, max_y = float(match.group(3)), float(match.group(4))
                size_x = abs(max_x - min_x)
                size_y = abs(max_y - min_y)
                data_pack['image_size_x_mm'] = round(size_x, 4)
                data_pack['image_size_y_mm'] = round(size_y, 4)
                data_pack['coordinates'] = (
                    f"X: {min_x:.2f} to {max_x:.2f} mm ({size_x:.2f} mm), "
                    f"Y: {min_y:.2f} to {max_y:.2f} mm ({size_y:.2f} mm)"
                )
                nx, ny = data_pack.get('nx'), data_pack.get('ny')
                if isinstance(nx, (int, float, np.integer)) and isinstance(ny, (int, float, np.integer)) and nx > 0 and ny > 0:
                    px_x_um = (size_x / float(nx)) * 1000.0
                    px_y_um = (size_y / float(ny)) * 1000.0
                    data_pack['pixel_size_x_um'] = round(px_x_um, 2)
                    data_pack['pixel_size_y_um'] = round(px_y_um, 2)
                    data_pack['pixel_size'] = f"{px_x_um:.2f} µm x {px_y_um:.2f} µm"

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

    raw_gap = data_pack.get('exit_slit_gap')
    gap_val = format_num_val(raw_gap)
    if gap_val != 'N/A':
        if not any(u in str(gap_val).lower() for u in ['µm', 'um', 'micron', 'microns']):
            gap_val = f"{gap_val} µm"
    data_pack['exit_slit_gap'] = gap_val
    data_pack['xps_z'] = format_num_val(data_pack.get('xps_z'))

    t_per_img = data_pack.get('time_per_map') or data_pack.get('time_per_image')
    if t_per_img and str(t_per_img).strip() not in ('N/A', 'None', ''):
        val_str = str(t_per_img).strip()
        if not any(u in val_str.lower() for u in ['s', 'sec', 'second', 'seconds']):
            try:
                f_val = float(val_str)
                if f_val.is_integer():
                    formatted_t = f"{int(f_val)} s"
                else:
                    formatted_t = f"{f_val:.2f} s"
            except (ValueError, TypeError):
                formatted_t = f"{val_str} s"
        else:
            formatted_t = val_str
        data_pack['time_per_map'] = formatted_t
        data_pack['time_per_image'] = formatted_t

    # --- Print Summary if requested ---
    if verbose:
        print("\n--- Scan Analysis Summary ---")
        print(f"Energies ({data_pack['energies'].shape[0]} points): {data_pack['energies']}")
        print(f"Number of Images:      {data_pack['Number of images']}")
        print(f"Energy Regions:        {data_pack['Energy Regions']}")
        print(f"X array (shape):       {data_pack['x'].shape} (Nx: {data_pack['nx']})")
        print(f"Y array (shape):       {data_pack['y'].shape} (Ny: {data_pack['ny']})")
        if isinstance(data_pack['nx'], (int, np.integer)) and isinstance(data_pack['ny'], (int, np.integer)):
            pts = data_pack['nx'] * data_pack['ny']
            print(f"Grid Dimensions:       {data_pack['nx']} x {data_pack['ny']} ({pts} points)")
        else:
            print(f"Grid Dimensions:       {data_pack['nx']} x {data_pack['ny']}")
        print(f"Pixel Size:            {data_pack['pixel_size']}")
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
        if data_pack.get('save_dir') and data_pack['save_dir'] != 'N/A':
            if data_pack['save_dir'] != data_pack.get('h5_dir'):
                print(f"\n[OUTPUT REDIRECTION] Preprocessed Save Dir:\n  {data_pack['save_dir']}")
        print("-----------------------------------")

    return data_pack


def browse_beamline_data(start_dir=None, on_load=None, file_filter=".h5", title="CLS Beamline Data Explorer", auto_display=True, on_select_folder=None, file_button_label="Select File"):
    """
    Launches an interactive, non-blocking Jupyter GUI browser for CLS Loki beamline data.
    Allows navigating through subfolders and selecting .h5 stack files or .csv files directly inside JupyterLab.
    If on_select_folder is provided, an additional button passes the current folder to it.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("[Warning] ipywidgets / IPython not available. Use file path directly.")
        return None

    candidates = [
        start_dir,
        "/gpfs/Primary/beamlinedata/sgm",
        "/gpfs/Primary/beamlinedata/SGM",
        "/gpfs/Primary/beamlinedata",
        "/gpfs/Primary",
        "/projects",
        os.path.expanduser("~/scratch"),
        get_last_dir(),
        os.getcwd()
    ]
    base_dir = next((p for p in candidates if p and os.path.isdir(p)), os.getcwd())

    # Support multiple extensions (e.g. '.h5' or ('.csv', '.txt') or '.csv')
    if isinstance(file_filter, str):
        filters = (file_filter.lower(),)
    else:
        filters = tuple(f.lower() for f in file_filter)

    path_box = widgets.Text(value=base_dir, description="Folder:", layout=widgets.Layout(width="650px"))
    btn_up = widgets.Button(description="Up ⬆", layout=widgets.Layout(width="80px"))
    sort_dropdown = widgets.Dropdown(
        options=[("Most Recent Projects First (prj43..., Date)", "project_desc"), ("Alphabetical (A-Z)", "name_asc"), ("Reverse (Z-A)", "name_desc")],
        value="project_desc",
        description="Sort by:",
        layout=widgets.Layout(width="380px")
    )
    folder_select = widgets.Select(description="Subfolders:", rows=8, layout=widgets.Layout(width="650px"))
    lbl_files = f"Files ({', '.join(filters)}):"
    file_select = widgets.Select(description=lbl_files, rows=6, layout=widgets.Layout(width="650px"))
    btn_open = widgets.Button(description="Open Selected Folder", button_style="info", layout=widgets.Layout(width="200px"))
    btn_load = widgets.Button(description=file_button_label, button_style="success", icon="folder-open", layout=widgets.Layout(width="200px"))
    status_msg = widgets.HTML(value=f"<i>Select a folder to navigate, or select a file to load</i>")

    def refresh_view():
        current = path_box.value.strip()
        if not os.path.isdir(current):
            folder_select.options = [("Directory not found", "")]
            file_select.options = []
            return
        try:
            with os.scandir(current) as it:
                raw_entries = list(it)

            dir_entries = [e for e in raw_entries if e.is_dir() and not e.name.startswith('.')]
            file_entries = [e for e in raw_entries if e.is_file() and any(e.name.lower().endswith(ext) for ext in filters)]

            sort_mode = sort_dropdown.value
            def natural_project_key(entry):
                s = entry.name.strip()
                # Match CLS project patterns like prj43G15689, prj40S15086, 41S15109
                m = re.match(r'^(prj)?(\d+)([a-zA-Z]?)(\d+)?(.*)$', s, re.IGNORECASE)
                if m:
                    has_prj = 1 if m.group(1) else 0
                    cycle = int(m.group(2))
                    sub_num = int(m.group(4)) if m.group(4) else 0
                    letter = (m.group(3) or '').lower()
                    rest = (m.group(5) or '').lower()
                    return (1, has_prj, cycle, sub_num, letter, rest)
                # General natural sort for dates (e.g. 2026-08-13) and other alphanumeric names
                parts = re.split(r'(\d+)', s.lower())
                parsed = [int(p) if p.isdigit() else p for p in parts]
                return (0, 0, 0, 0, '', parsed)

            if sort_mode == "project_desc":
                # Groups by cycle number descending (prj43..., prj42..., prj41...), ignoring the letter between
                dirs = [e.name for e in sorted(dir_entries, key=natural_project_key, reverse=True)]
                matched_files = [e.name for e in sorted(file_entries, key=natural_project_key, reverse=True)]
            elif sort_mode == "name_desc":
                dirs = [e.name for e in sorted(dir_entries, key=lambda e: e.name.lower(), reverse=True)]
                matched_files = [e.name for e in sorted(file_entries, key=lambda e: e.name.lower(), reverse=True)]
            else: # "name_asc"
                dirs = [e.name for e in sorted(dir_entries, key=lambda e: e.name.lower())]
                matched_files = [e.name for e in sorted(file_entries, key=lambda e: e.name.lower())]

            folder_select.options = dirs if dirs else ["(No subfolders)"]
            file_select.options = matched_files if matched_files else [f"(No matching files in this folder)"]
            status_msg.value = f"Current path: <code>{current}</code>"
        except Exception as e:
            status_msg.value = f"<b style='color:red;'>Error reading folder: {e}</b>"

    def on_open_clicked(b):
        chosen = folder_select.value
        if chosen and chosen != "(No subfolders)":
            new_path = os.path.join(path_box.value.strip(), chosen)
            if os.path.isdir(new_path):
                path_box.value = new_path
                refresh_view()

    def on_up_clicked(b):
        parent = os.path.dirname(path_box.value.strip().rstrip('/'))
        if parent and os.path.isdir(parent):
            path_box.value = parent
            refresh_view()

    def on_load_clicked(b):
        chosen_file = file_select.value
        if chosen_file and any(chosen_file.lower().endswith(ext) for ext in filters):
            full_path = os.path.join(path_box.value.strip(), chosen_file)
            status_msg.value = f"<b>Selected:</b> {full_path} ..."
            if on_load and callable(on_load):
                on_load(full_path)
            else:
                if chosen_file.lower().endswith('.h5'):
                    res = analyze_sgm_bsky_data(full_path)
                    import __main__
                    setattr(__main__, 'sgm_data', res)
                    status_msg.value = f"<b style='color:green;'>✓ Successfully loaded:</b> {chosen_file} (stored in <code>sgm_data</code>)"
                else:
                    status_msg.value = f"<b style='color:green;'>✓ Selected:</b> {chosen_file}"
        else:
            status_msg.value = f"<b style='color:red;'>Please select a valid file ({', '.join(filters)}) from the list</b>"

    btn_open.on_click(on_open_clicked)
    btn_up.on_click(on_up_clicked)
    btn_load.on_click(on_load_clicked)
    path_box.observe(lambda c: refresh_view(), names="value")
    sort_dropdown.observe(lambda c: refresh_view(), names="value")

    browser_children = [
        widgets.HTML(f"<h3>{title}</h3>"),
        widgets.HBox([path_box, btn_up]),
        sort_dropdown,
        folder_select,
        btn_open,
        file_select,
        btn_load
    ]
    if on_select_folder is not None:
        btn_folder = widgets.Button(description="Add Current Folder", layout=widgets.Layout(width="200px"))

        def on_folder_clicked(b):
            current = path_box.value.strip()
            if not os.path.isdir(current):
                status_msg.value = "<b style='color:red;'>Please select a valid folder.</b>"
                return
            on_select_folder(current)

        btn_folder.on_click(on_folder_clicked)
        browser_children.append(btn_folder)
    browser_children.append(status_msg)
    ui = widgets.VBox(browser_children)
    refresh_view()
    if auto_display:
        display(ui)
        return None
    return ui


if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = None # Triggers file browser
        
    paths_data = analyze_sgm_bsky_data(file_path, verbose=True)