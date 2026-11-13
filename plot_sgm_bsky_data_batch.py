import os
import sys
import traceback
import io
from html import escape
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import Button, SpanSelector
try:
    from tkinter import filedialog
except (ImportError, ModuleNotFoundError):
    filedialog = None
import mplcursors
import sdd_element_roi_utils as elem_roi_utils

from alignment_utils import (
    safe_filedialog_call,
    show_custom_dialog,
    interactive_channel_selector,
    get_safe_save_path,
    format_num_val,
    HAS_GUI
)
from sdd_calibration_utils import (
    load_calibration,
    channel_to_energy,
    get_calibrated_bounds
)
from plot_sgm_bsky_data import (
    read_csv_with_comments, ExternalI0PreviewDialog, get_tk_root,
    _i0_preview_widget, compute_modified_i0
)
from analyze_sgm_bsky_data import (
    build_scan_filename_base, resolve_sample_name, get_sdd_label, browse_beamline_data,
    resolve_preprocessed_save_dir
)

def _batch_roi_label(channel_roi, energy_min, energy_max, calibrated):
    if calibrated and energy_min is not None and energy_max is not None:
        return f"SDD Energy ROI: {energy_min:.1f}-{energy_max:.1f} eV"
    return f"SDD Channel ROI: {channel_roi[0]}-{channel_roi[1]} (energy unavailable without calibration)"


def _batch_spectrum_label(path_pack, scan_name):
    sample = path_pack.get('sample_name')
    if not sample or sample == 'N/A':
        sample = resolve_sample_name(path_pack.get('h5_file_path'), scan_name=scan_name)
    return sample


def get_mcc_val(path_pack, energy, channel, spatial_mask, strict=False):
    """
    Extracts the average value of a given MCC channel inside the spatial mask
    at a specific energy step.
    """
    mf = path_pack.get('mcc_files', {}).get(energy)
    if mf and os.path.exists(mf):
        try:
            df = pd.read_csv(mf) if mf.endswith('.csv') else pd.read_table(mf)
            df.columns = [c.replace('#', '').strip() for c in df.columns]
            num_pts = min(len(df), spatial_mask.size)
            col = next((v for v in [f'ch{channel}', f'mcc{channel}', str(channel)] if v in df.columns), None)
            if col is not None:
                col_data = df[col].values[:num_pts]
                val = col_data[spatial_mask[:num_pts]].mean()
                if not np.isnan(val):
                    return val
        except Exception as error:
            if strict:
                raise ValueError(f"Cannot read MCC channel {channel} at {energy} eV: {error}") from error
    if strict:
        raise ValueError(f"MCC channel {channel} is unavailable at {energy} eV in {path_pack.get('scan_name', 'scan')}.")
    return 1.0 if channel == 1 else 0.0


def _batch_spatial_mask(path_pack, map_roi):
    x_raw = path_pack.get('x', np.array([]))
    y_raw = path_pack.get('y', np.array([]))
    count = min(x_raw.size, y_raw.size)
    try:
        first_det = list(path_pack['sdd_files'].keys())[0]
        first_path = path_pack['sdd_files'][first_det].get(path_pack['energies'][0])
        if first_path and os.path.exists(first_path):
            count = min(os.path.getsize(first_path) // 1024, count)
    except (IndexError, KeyError, TypeError, OSError):
        pass
    x, y = x_raw[:count], y_raw[:count]
    if map_roi is None:
        if len(x) and len(y):
            map_roi = [np.min(x), np.max(x), np.min(y), np.max(y)]
        else:
            map_roi = [0.0, 1.0, 0.0, 1.0]
    x1, x2 = sorted(map_roi[0:2])
    y1, y2 = sorted(map_roi[2:4])
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)


def _plot_sgm_bsky_data_batch(data_packs, channel_roi=None, xrf_roi=None, map_roi=None, show_markers=True, use_color=True, auto_element_roi=False, element=None, n_sigma=2.0, hpc_config=None):
    """
    Plots the XANES spectra (energy dependence) of multiple loaded scans together,
    supporting interactive ROI selection, live ROI refresh, I0 normalization,
    SDD energy calibration, and vertical offsets (waterfall style).

    Args:
        data_packs (list of dict): A list of data packs returned by analyze_sgm_bsky_data_batch.
        channel_roi (tuple of int, optional): (start_channel, end_channel). Pre-loads initial channel ROI.
        xrf_roi (tuple of float, optional): (min_eV, max_eV). Pre-loads initial energy ROI in eV (overrides channel_roi).
        map_roi (tuple/list of float, optional): [x1, x2, y1, y2]. Defaults to the full map.
        show_markers (bool): Whether to draw markers on the spectral lines.
        use_color (bool): Whether to use a color palette.
        auto_element_roi (bool): Fit the first multi-energy stack and share its energy ROI
            across the batch. Requires applying SDD calibration. Overrides xrf_roi on success.
        element (str, optional): Element override for automatic fitting (also enables fitting).
        n_sigma (float): Gaussian ROI half-width in standard deviations.
    """
    if not data_packs:
        print("No scans to plot.", file=sys.stderr)
        return

    # Disable auto-display temporarily to prevent duplicate plotting in Jupyter
    plt.ioff()

    # 1. Prompt for SDD Calibration FIRST
    use_calibration = (hpc_config['use_calibration'] if hpc_config is not None else
                       show_custom_dialog(
                           title="SDD Calibration",
                           message="Apply SDD Energy Calibration (using sdd_calibration.json)?",
                           dialog_type="yesno"
                       ))
    calib_data = {}
    if use_calibration is True:
        calib_data = hpc_config['calib_data'] if hpc_config is not None else load_calibration()
        if not calib_data:
            print("Warning: sdd_calibration.json not found. Falling back to fixed channel ROI.", file=sys.stderr)
            use_calibration = False

    if auto_element_roi or element is not None:
        if not use_calibration or not calib_data:
            print("[Batch Auto Element ROI] SDD calibration must be applied for automatic fitting. "
                  "Using the supplied ROI or default channel ROI.", file=sys.stderr)
        else:
            reference = next(
                (pack for pack in data_packs if len(pack.get('energies', [])) > 2), None
            )
            fit_res = None
            if reference is not None:
                print(f"[Batch Auto Element ROI] Reference: {reference.get('scan_name', 'first spectrum stack')}")
                fit_res = elem_roi_utils.auto_detect_and_fit_stack_roi(
                    reference, map_roi=map_roi, element_override=element,
                    n_sigma=n_sigma, verbose=True
                )
            if fit_res and 'xrf_roi' in fit_res:
                xrf_roi = fit_res['xrf_roi']
                reference['auto_element_roi_info'] = fit_res
                print(f"[Batch Auto Element ROI] Shared ROI for all scans: "
                      f"{min(xrf_roi):.2f}-{max(xrf_roi):.2f} eV")
            else:
                print("[Batch Auto Element ROI] No ROI could be fitted from the first spectrum stack. "
                      "Using the supplied ROI or default channel ROI.", file=sys.stderr)

    # 2. Set Initial ROI bounds for batch plotting
    ref_det_gain, ref_det_offset = 1.0, 0.0
    if xrf_roi is not None:
        energy_min, energy_max = float(min(xrf_roi)), float(max(xrf_roi))
        if use_calibration and calib_data:
            ref_det = 'sdd3' if 'sdd3' in calib_data else ('sdd1' if 'sdd1' in calib_data else next(iter(calib_data.keys()), None))
            if ref_det and ref_det in calib_data:
                ref_det_gain = calib_data[ref_det].get("gain", 1.0)
                ref_det_offset = calib_data[ref_det].get("offset", 0.0)
            ch_s = int(max(0, (energy_min - ref_det_offset) / (ref_det_gain if ref_det_gain != 0 else 1.0)))
            ch_e = int(min(255, (energy_max - ref_det_offset) / (ref_det_gain if ref_det_gain != 0 else 1.0)))
            channel_roi = (ch_s, ch_e)
        elif channel_roi is None:
            channel_roi = (20, 40)
        print(f"Initial Batch ROI (xrf_roi): {energy_min:.2f} eV to {energy_max:.2f} eV (channels {channel_roi[0]} to {channel_roi[1]})")
    else:
        channel_roi = channel_roi if channel_roi is not None else (20, 40)
        energy_min, energy_max = None, None
        if use_calibration and calib_data:
            ref_det = 'sdd3' if 'sdd3' in calib_data else ('sdd1' if 'sdd1' in calib_data else next(iter(calib_data.keys()), None))
            if ref_det and ref_det in calib_data:
                ref_det_gain = calib_data[ref_det].get("gain", 1.0)
                ref_det_offset = calib_data[ref_det].get("offset", 0.0)
                
            energy_min = channel_to_energy(channel_roi[0], ref_det_gain, ref_det_offset)
            energy_max = channel_to_energy(channel_roi[1], ref_det_gain, ref_det_offset)
            print(f"Initial Batch ROI (channel_roi): channels {channel_roi[0]} to {channel_roi[1]} ({energy_min:.2f} eV to {energy_max:.2f} eV)")
        else:
            print(f"Initial Batch ROI: channels {channel_roi[0]} to {channel_roi[1]}")

    # 3. Prompt for I0 normalization source
    use_internal = (hpc_config['use_internal'] if hpc_config is not None else
                    show_custom_dialog(
                        title="I0 Selection",
                        message="Use INTERNAL Au Mesh I0 (mcc1) for I0 normalization?\n\n(Select 'No' to load an EXTERNAL I0 CSV)",
                        dialog_type="yesno"
                    ))
    if use_internal is None:
        print("I0 selection cancelled. Exiting plot.", file=sys.stderr)
        plt.ion()
        return

    x_sorted, y_sorted = None, None
    i0_source_label = "Au Mesh I0 (mcc1)"
    i0_settings = hpc_config.get('i0_settings') if hpc_config is not None else None
    if hpc_config is not None:
        x_sorted, y_sorted = hpc_config['external_xy']
        i0_source_label = hpc_config['i0_source_label']
    elif not use_internal:
        func = getattr(filedialog, 'askopenfilename', 'askopenfilename') if filedialog else 'askopenfilename'
        ext_path = safe_filedialog_call(
            func,
            title="Select External I0 CSV for Batch Normalization",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if ext_path:
            try:
                ext_df = read_csv_with_comments(ext_path)
                ext_df.columns = [c.strip() for c in ext_df.columns]
                e_col = next((c for c in ext_df.columns if 'energy' in c.lower()), ext_df.columns[0])
                i_col = next((c for c in ext_df.columns if any(k in c.lower() for k in ['i0', 'intensity', 'norm', 'tey'])), ext_df.columns[1])
                
                root_i0 = get_tk_root()
                root_i0.withdraw()
                dialog = ExternalI0PreviewDialog(root_i0, ext_df, e_col, i_col)
                dialog.title("External I0 Selection & Processing (Batch)")
                dialog.mainloop()
                
                if dialog.result:
                    selected_e_col, selected_i_col, x_s, y_s, extra_str, cal_en, cal_val = dialog.result[:7]
                    const_offset_val = getattr(dialog, 'const_val', 0.0)
                    shift = cal_val if cal_en else 0.0
                    x_sorted = x_s + shift
                    y_sorted = y_s
                    i0_source_label = f"External: {os.path.basename(ext_path)} [{selected_i_col}]{extra_str}"
                    print(f"Loaded external I0 calibration from {ext_path} ({i0_source_label})")
                    if const_offset_val != 0.0:
                        print(f"  [Batch Normalization] I0 was modified by adding constant {const_offset_val:+.4g} (I0' = I0 + {const_offset_val:+.4g})")
                else:
                    print("External I0 selection cancelled in dialog. Falling back to internal I0.", file=sys.stderr)
                    use_internal = True
            except Exception as e:
                print(f"Error loading external I0: {e}. Falling back to internal I0.", file=sys.stderr)
                use_internal = True
        else:
            print("No external I0 file selected. Falling back to internal I0.", file=sys.stderr)
            use_internal = True

    # 4. Prompt for Waterfall/Vertical Offset
    apply_waterfall = (hpc_config['apply_waterfall'] if hpc_config is not None else
                       show_custom_dialog(
                           title="Waterfall Offset",
                           message="Apply a vertical offset (waterfall plot) to separate the spectra?\n\n(Select 'No' to overlay them directly)",
                           dialog_type="yesno"
                       ))
    if apply_waterfall is None:
        apply_waterfall = False

    # State tracking for live ROI refresh
    current_state = {
        'channel_roi': channel_roi,
        'energy_min': energy_min,
        'energy_max': energy_max,
        'plot_data': None
    }

    # Helper function to compute batch spectra for a given ROI
    def compute_batch_spectra(curr_ch_roi, curr_e_min, curr_e_max):
        data_out = {
            'sdd1_raw': [], 'sdd1_norm': [],
            'sdd2_raw': [], 'sdd2_norm': [],
            'sdd3_raw': [], 'sdd3_norm': [],
            'sdd4_raw': [], 'sdd4_norm': [],
            'mcc1_raw': [],
            'mcc1_used': [],
            'mcc4_raw': [], 'mcc4_norm': []
        }

        for scan_idx, path_pack in enumerate(data_packs):
            scan_name = path_pack.get('scan_name', f"Scan_{scan_idx + 1}")
            spectrum_label = _batch_spectrum_label(path_pack, scan_name)
            all_energies = np.array(sorted(path_pack.get('energies', [])))
            if len(all_energies) <= 2:
                print(f"  [Batch Loader] Skipping spatial map / survey scan '{scan_name}' ({len(all_energies)} energy image(s); batch mode processes multi-energy spectrum stacks).")
                continue

            spatial_mask = _batch_spatial_mask(path_pack, map_roi)

            all_energies = np.array(sorted(path_pack['energies']))
            i0_values = np.ones(len(all_energies))
            if not use_internal and x_sorted is not None and y_sorted is not None:
                i0_values = np.interp(all_energies, x_sorted, y_sorted)
            elif use_internal and i0_settings is not None:
                original_i0 = np.array([
                    get_mcc_val(path_pack, energy, 1, spatial_mask, strict=True)
                    for energy in all_energies
                ])
                i0_values, od_vals, descriptions, _ = compute_modified_i0(all_energies, original_i0, **i0_settings)
                if ((i0_settings['do_od'] and od_vals is None)
                        or (i0_settings['do_smooth'] and not any(d.startswith("Smoothed w=") for d in descriptions))
                        or not np.all(np.isfinite(i0_values))):
                    raise ValueError(f"I0 processing failed for {scan_name}.")
            if hpc_config is not None and (not use_internal or i0_settings is not None):
                if not np.all(np.isfinite(i0_values)) or np.any(i0_values <= 0):
                    raise ValueError(f"The selected I0 for {scan_name} must be finite and positive for normalization.")

            for det_name in ['sdd1', 'sdd2', 'sdd3', 'sdd4']:
                if det_name not in path_pack['sdd_files']:
                    continue

                if use_calibration and calib_data and curr_e_min is not None and curr_e_max is not None:
                    ch_start, ch_end = get_calibrated_bounds(curr_e_min, curr_e_max, det_name, calib_data)
                else:
                    ch_start, ch_end = curr_ch_roi[0], curr_ch_roi[1]

                raw_intensities = []
                norm_intensities = []
                for e_idx, energy in enumerate(all_energies):
                    sdd_filepath = path_pack['sdd_files'][det_name].get(energy)
                    val_raw = np.nan
                    if sdd_filepath and os.path.exists(sdd_filepath):
                        try:
                            data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
                            num_spectra = len(data_1d) // 256
                            if num_spectra >= spatial_mask.size:
                                spectra_2d = data_1d[:num_spectra * 256].reshape((num_spectra, 256))
                                selected_spectra = spectra_2d[spatial_mask]
                                val_raw = np.sum(selected_spectra[:, ch_start:ch_end + 1])
                        except Exception:
                            pass
                    
                    raw_intensities.append(val_raw)
                    val_norm = np.nan
                    if not np.isnan(val_raw):
                        mcc_val = get_mcc_val(path_pack, energy, 1, spatial_mask)
                        norm_factor = mcc_val if use_internal and i0_settings is None else i0_values[e_idx]
                        if norm_factor <= 0: norm_factor = 1.0
                        val_norm = val_raw / norm_factor
                    norm_intensities.append(val_norm)

                data_out[f"{det_name}_raw"].append({"scan_name": scan_name, "spectrum_label": spectrum_label, "energies": all_energies, "intensities": np.array(raw_intensities)})
                data_out[f"{det_name}_norm"].append({"scan_name": scan_name, "spectrum_label": spectrum_label, "energies": all_energies, "intensities": np.array(norm_intensities)})

            mcc1_raw_list, mcc4_raw_list, mcc4_norm_list = [], [], []
            for e_idx, energy in enumerate(all_energies):
                mcc1_val = get_mcc_val(path_pack, energy, 1, spatial_mask)
                mcc4_val = get_mcc_val(path_pack, energy, 4, spatial_mask)
                mcc1_raw_list.append(mcc1_val)
                mcc4_raw_list.append(mcc4_val)
                norm_factor = mcc1_val if use_internal and i0_settings is None else i0_values[e_idx]
                if norm_factor <= 0: norm_factor = 1.0
                mcc4_norm_list.append(mcc4_val / norm_factor)

            data_out['mcc1_raw'].append({"scan_name": scan_name, "spectrum_label": spectrum_label, "energies": all_energies, "intensities": np.array(mcc1_raw_list)})
            if use_internal and i0_settings is not None:
                data_out['mcc1_used'].append({"scan_name": scan_name, "energies": all_energies, "intensities": i0_values})
            data_out['mcc4_raw'].append({"scan_name": scan_name, "spectrum_label": spectrum_label, "energies": all_energies, "intensities": np.array(mcc4_raw_list)})
            data_out['mcc4_norm'].append({"scan_name": scan_name, "spectrum_label": spectrum_label, "energies": all_energies, "intensities": np.array(mcc4_norm_list)})

        return data_out

    print("\nCalculating batch XANES spectra...")
    plot_data = compute_batch_spectra(current_state['channel_roi'], current_state['energy_min'], current_state['energy_max'])
    current_state['plot_data'] = plot_data

    figs_to_display = []
    lines_for_cursor = []

    lines_by_key = {
        'sdd1_raw': [], 'sdd1_norm': [],
        'sdd2_raw': [], 'sdd2_norm': [],
        'sdd3_raw': [], 'sdd3_norm': [],
        'sdd4_raw': [], 'sdd4_norm': [],
        'mcc1_raw': [],
        'mcc4_raw': [], 'mcc4_norm': []
    }

    # Function to save CSVs on demand via yellow button
    def execute_save_csvs(b=None):
        print("\nSaving XANES spectra CSV files...")
        saved_count = 0
        active_plot_data = current_state['plot_data'] or plot_data
        curr_ch_roi = current_state['channel_roi']
        curr_e_min = current_state['energy_min']
        curr_e_max = current_state['energy_max']

        for scan_idx, path_pack in enumerate(data_packs):
            h5_path = path_pack.get('h5_file_path') or path_pack.get('h5_dir')
            scan_name = path_pack.get('scan_name', f"Scan_{scan_idx + 1}")
            save_dir = path_pack.get('save_dir')
            if not save_dir or save_dir == 'N/A':
                if h5_path and os.path.isfile(h5_path):
                    save_dir = os.path.dirname(os.path.abspath(h5_path))
                elif h5_path and os.path.isdir(h5_path):
                    save_dir = os.path.abspath(h5_path)
                elif hpc_config is not None:
                    print(f"  [ERROR] No output directory is available for {scan_name}; skipped.", file=sys.stderr)
                    continue
                else:
                    save_dir = os.getcwd()
            if hpc_config is not None:
                try:
                    save_dir = resolve_preprocessed_save_dir(save_dir)
                    os.makedirs(save_dir, exist_ok=True)
                except OSError as error:
                    print(f"  [ERROR] Cannot create output directory for {scan_name}: {error}", file=sys.stderr)
                    continue
            sample_name = path_pack.get('sample_name')
            if not sample_name or str(sample_name).strip() in ('N/A', 'None', '', 'Sample'):
                sample_name = resolve_sample_name(h5_path, scan_name=scan_name)
            base_filename = build_scan_filename_base(scan_name, sample_name)
            scan_energies = np.array(sorted(path_pack['energies']))
            if i0_settings is not None and len(scan_energies) <= 2:
                print(f"  [SKIP] {scan_name} is not a multi-energy spectrum stack.")
                continue
            
            if use_calibration and curr_e_min is not None and curr_e_max is not None:
                roi_str = f"{curr_e_min:.1f}-{curr_e_max:.1f}eV"
            else:
                roi_str = f"Ch{curr_ch_roi[0]}-{curr_ch_roi[1]}"
                
            csv_filename = f"{base_filename}_Rect_{roi_str}_summary.csv"
            csv_filepath = os.path.join(save_dir, csv_filename)
            if hpc_config is not None:
                try:
                    csv_filepath = get_safe_save_path(save_dir, csv_filename)
                except OSError as error:
                    print(f"  [ERROR] Cannot prepare output file for {scan_name}: {error}", file=sys.stderr)
                    continue
                if not csv_filepath:
                    print(f"  [SKIP] Skipped saving CSV for {scan_name}.", file=sys.stderr)
                    continue
            elif os.path.exists(csv_filepath):
                overwrite = show_custom_dialog(
                    title="File Exists",
                    message=f"The CSV file '{csv_filename}' already exists in:\n{save_dir}\n\nDo you want to overwrite it?",
                    dialog_type="yesno"
                )
                if overwrite is False:
                    csv_filepath = get_safe_save_path(save_dir, csv_filename)
                    if not csv_filepath:
                        print(f"  [SKIP] Skipped saving CSV for {scan_name}.")
                        continue
                elif overwrite is None:
                    print(f"  [CANCEL] Cancelled saving CSV for {scan_name}.")
                    continue
            
            if use_calibration:
                meta_calib = calib_data.get("_metadata", {}) if calib_data else {}
                scan_used = meta_calib.get("scan_used", "N/A")
                edges_used = meta_calib.get("edges_used", "N/A")
                if isinstance(edges_used, list):
                    edges_used = ", ".join(edges_used)
                calib_date = meta_calib.get("last_updated", "N/A")
                calib_file = "sdd_calibration.json"

                calib_meta_rows = [
                    "# SDD Calibration: Active",
                    f"# SDD Calibration File: {calib_file}",
                    f"# SDD Calibration Scan Used: {scan_used}",
                    f"# SDD Calibration Edges Used: {edges_used}",
                ]
                if calib_date and calib_date != "N/A":
                    calib_meta_rows.append(f"# SDD Calibration Date: {calib_date}")
                
                calib_params = []
                for det in ['sdd1', 'sdd2', 'sdd3', 'sdd4']:
                    if det in calib_data and isinstance(calib_data[det], dict):
                        g = calib_data[det].get('gain', 1.0)
                        o = calib_data[det].get('offset', 0.0)
                        calib_params.append(f"{det}: Gain={g:.4f}, Offset={o:.4f} eV")
                if calib_params:
                    calib_meta_rows.append(f"# SDD Calibration Parameters: {'; '.join(calib_params)}")
            else:
                calib_meta_rows = [
                    "# SDD Calibration: Disabled"
                ]

            rows = []
            rows += [
                f"# Scan Name: {scan_name}",
                f"# Scan Type: {path_pack.get('scan_type', 'N/A')}",
                f"# Project: {path_pack.get('project', 'N/A')}",
                f"# Date: {path_pack.get('date', 'N/A')}",
                f"# Number of Images: {len(scan_energies)}",
                f"# Energy Regions: {path_pack.get('Energy Regions', 'N/A')}",
                f"# Grid Dimensions: {path_pack.get('nx', 'N/A')} x {path_pack.get('ny', 'N/A')} ({path_pack.get('x', np.array([])).size} points)",
                f"# Pixel Size: {path_pack.get('pixel_size', 'N/A')}",
                f"# Coordinates: {path_pack.get('coordinates', 'N/A')}",
                f"# Grating: {path_pack.get('grating', 'N/A')}",
                f"# Harmonic: {path_pack.get('harmonic', 'N/A')}",
                f"# Strip: {path_pack.get('strip', 'N/A')}",
                f"# Polarization: {path_pack.get('polarization', 'N/A')}",
                f"# Exit Slit Gap: {format_num_val(path_pack.get('exit_slit_gap'))}",
                f"# XPS Z: {format_num_val(path_pack.get('xps_z'))}",
            ]
            t_per_img = path_pack.get('time_per_map') or path_pack.get('time_per_image')
            if t_per_img and str(t_per_img).strip() not in ('N/A', 'None', ''):
                rows.append(f"# Time Per Image: {t_per_img}")
            sdd_pos_str = "; ".join([get_sdd_label(d, path_pack) for d in ['sdd1', 'sdd2', 'sdd3', 'sdd4']])
            rows += [
                f"# ROI Selection: Rect",
                f"# {'Energy ROI' if use_calibration else 'Channels'}: {roi_str}",
                f"# Normalization: {i0_source_label}",
                f"# SDD Detector Positions: {sdd_pos_str}",
            ]
            rows += calib_meta_rows
            rows.append("#")
            
            ext_i0_col_values = np.interp(scan_energies, x_sorted, y_sorted) if (not use_internal and x_sorted is not None and y_sorted is not None) else None

            rows.append("# Column 1: Calibrated Energy (eV)")
            rows.append("# Column 2: Original Energy (eV)")
            c_idx = 3
            for det in ['sdd1', 'sdd2', 'sdd3', 'sdd4']:
                rows.append(f"# Column {c_idx}: RAW_{get_sdd_label(det, path_pack)}"); c_idx += 1
            rows.append(f"# Column {c_idx}: RAW_Average_SDD"); c_idx += 1
            
            for det in ['sdd1', 'sdd2', 'sdd3', 'sdd4']:
                rows.append(f"# Column {c_idx}: NORM_{get_sdd_label(det, path_pack)} (by {i0_source_label})"); c_idx += 1
            rows.append(f"# Column {c_idx}: NORM_Average_SDD"); c_idx += 1
            
            rows.append(f"# Column {c_idx}: RAW_Au Mesh I0 (mcc1)"); c_idx += 1
            if use_internal and i0_settings is not None:
                rows.append(f"# Column {c_idx}: Modified_Au_Mesh_I0 (used for normalization)"); c_idx += 1
            if ext_i0_col_values is not None:
                label = "External_I0_used_for_normalization" if hpc_config is not None else "RAW_External_I0"
                rows.append(f"# Column {c_idx}: {label} (from {i0_source_label})"); c_idx += 1
            rows.append(f"# Column {c_idx}: RAW_TEY of Sample (mcc4)"); c_idx += 1
            rows.append(f"# Column {c_idx}: NORM_TEY of Sample (mcc4) (by {i0_source_label})"); c_idx += 1
            rows.append("#")
            
            sdd_raw_data = []
            for d in [1, 2, 3, 4]:
                found = next((ds['intensities'] for ds in active_plot_data[f'sdd{d}_raw'] if ds['scan_name'] == scan_name), None)
                if found is None:
                    found = np.zeros(len(scan_energies))
                sdd_raw_data.append(found)

            sdd_norm_data = []
            for d in [1, 2, 3, 4]:
                found = next((ds['intensities'] for ds in active_plot_data[f'sdd{d}_norm'] if ds['scan_name'] == scan_name), None)
                if found is None:
                    found = np.zeros(len(scan_energies))
                sdd_norm_data.append(found)

            raw_avg = np.nanmean(sdd_raw_data, axis=0)
            norm_avg = np.nanmean(sdd_norm_data, axis=0)
            
            mcc1_raw_data = next((ds['intensities'] for ds in active_plot_data['mcc1_raw'] if ds['scan_name'] == scan_name), np.zeros(len(scan_energies)))
            mcc1_used_data = next((ds['intensities'] for ds in active_plot_data['mcc1_used'] if ds['scan_name'] == scan_name), None)
            mcc4_raw_data = next((ds['intensities'] for ds in active_plot_data['mcc4_raw'] if ds['scan_name'] == scan_name), np.zeros(len(scan_energies)))
            mcc4_norm_data = next((ds['intensities'] for ds in active_plot_data['mcc4_norm'] if ds['scan_name'] == scan_name), np.zeros(len(scan_energies)))
            
            for i, energy in enumerate(scan_energies):
                row_vals = [f"{energy:.2f}", f"{energy:.2f}"]
                for d_idx in range(4):
                    row_vals.append(f"{sdd_raw_data[d_idx][i]:.2f}")
                row_vals.append(f"{raw_avg[i]:.2f}")
                for d_idx in range(4):
                    row_vals.append(f"{sdd_norm_data[d_idx][i]:.6f}")
                row_vals.append(f"{norm_avg[i]:.6f}")
                row_vals.append(f"{mcc1_raw_data[i]:.6f}")
                if use_internal and i0_settings is not None:
                    if mcc1_used_data is None:
                        raise ValueError(f"Modified I0 data unavailable for {scan_name}.")
                    row_vals.append(f"{mcc1_used_data[i]:.6f}")
                if ext_i0_col_values is not None:
                    row_vals.append(f"{ext_i0_col_values[i]:.6f}")
                row_vals.append(f"{mcc4_raw_data[i]:.6f}")
                row_vals.append(f"{mcc4_norm_data[i]:.6f}")
                rows.append(",".join(row_vals))
                
            try:
                with open(csv_filepath, 'w', encoding='utf-8-sig') as f_out:
                    f_out.write("\n".join(rows) + "\n")
                print(f"  Saved XANES CSV: {csv_filepath}")
                saved_count += 1
            except Exception as e:
                print(f"  Error saving CSV for {scan_name} at {csv_filepath}: {e}", file=sys.stderr)
                
        if saved_count:
            show_custom_dialog(
                title="Save Complete",
                message=f"Saved XANES spectra CSV files for {saved_count} scans. See the paths above.",
                dialog_type="info"
            )
        else:
            print("No XANES spectra CSV files were saved.", file=sys.stderr)

    # ------------------ Figure 0: XRF Emission Spectrum (Tab 0 Inspector) ------------------
    figure_scale = 0.7 if hpc_config is not None else 1.0
    fig_xrf, ax_xrf = plt.subplots(figsize=(10 * figure_scale, 4.5 * figure_scale))
    ref_pack = data_packs[0]
    ref_det = 'sdd3' if 'sdd3' in ref_pack.get('sdd_files', {}) else next(iter(ref_pack.get('sdd_files', {}).keys()), 'sdd1')
    
    xrf_spec = np.zeros(256, dtype=float)
    try:
        for en, fpath in ref_pack.get('sdd_files', {}).get(ref_det, {}).items():
            if fpath and os.path.exists(fpath):
                d1d = np.fromfile(fpath, dtype=np.uint32)
                if len(d1d) >= 256:
                    num_s = len(d1d) // 256
                    d2d = d1d[:num_s*256].reshape((num_s, 256))
                    xrf_spec += d2d.sum(axis=0)
    except Exception:
        pass

    channels = np.arange(256)
    if use_calibration and calib_data and ref_det in calib_data:
        gain = calib_data[ref_det].get("gain", 1.0)
        offset = calib_data[ref_det].get("offset", 0.0)
        xrf_x = gain * channels + offset
        xlabel_str = "Calibrated Energy (eV)"
        span_s = energy_min if energy_min is not None else (gain * channel_roi[0] + offset)
        span_e = energy_max if energy_max is not None else (gain * channel_roi[1] + offset)
    else:
        gain, offset = 1.0, 0.0
        xrf_x = channels
        xlabel_str = "Channel Index"
        span_s, span_e = channel_roi[0], channel_roi[1]

    ref_det_label = get_sdd_label(ref_det, ref_pack, uppercase=True)
    line_xrf, = ax_xrf.plot(xrf_x, xrf_spec, color='blue', lw=1.5, label=f"XRF Spectrum ({ref_det_label})")
    ax_xrf.set_xlabel(xlabel_str, fontsize=10, fontweight='bold')
    ax_xrf.set_ylabel("PFY Counts", fontsize=10, fontweight='bold')
    reference_label = _batch_spectrum_label(ref_pack, ref_pack.get('scan_name', 'Scan 1'))
    ax_xrf.set_title(f"Interactive XRF Spectrum ROI Inspector ({reference_label} - {ref_det_label})\n(Drag mouse left/right on peak to change ROI dynamically)", fontsize=11, fontweight='bold')
    ax_xrf.grid(True, linestyle=':', alpha=0.6)
    ax_xrf.legend(loc='upper right')

    def on_xrf_span_select(xmin, xmax):
        try:
            if use_calibration and calib_data:
                s_val = round(min(xmin, xmax), 1)
                e_val = round(max(xmin, xmax), 1)
            else:
                s_val = int(max(0, np.floor(min(xmin, xmax))))
                e_val = int(min(255, np.ceil(max(xmin, xmax))))
            if 'roi_start_w' in locals() or 'roi_start_w' in globals():
                roi_start_w.value = s_val
                roi_end_w.value = e_val
                on_refresh_click(None)
        except Exception:
            pass

    ax_xrf._span = SpanSelector(ax_xrf, on_xrf_span_select, 'horizontal', useblit=True,
                                props=dict(alpha=0.3, facecolor='red'), interactive=True)
    try:
        ax_xrf._span.extents = (span_s, span_e)
    except Exception:
        pass

    fig_xrf.tight_layout()
    figs_to_display.append(fig_xrf)

    raw_keys = ['sdd1_raw', 'sdd2_raw', 'sdd3_raw', 'sdd4_raw', 'mcc1_raw', 'mcc4_raw']
    norm_keys = ['sdd1_norm', 'sdd2_norm', 'sdd3_norm', 'sdd4_norm', 'mcc1_raw', 'mcc4_norm']

    # ------------------ Figure 1: Raw Spectra (3x2 Subplots) ------------------
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    scan_colors = {
        ds['scan_name']: colors[index % len(colors)]
        for index, ds in enumerate(plot_data['mcc1_raw'])
    }
    figure_height = max(13.0, 2.0 + 0.25 * len(plot_data['mcc1_raw']))
    fig_raw, axes_raw = plt.subplots(3, 2, figsize=(15 * figure_scale, figure_height * figure_scale))
    roi_label = _batch_roi_label(channel_roi, energy_min, energy_max, use_calibration)
    fig_raw.suptitle(f"Batch RAW Spectra Comparison\n{roi_label}", fontsize=14, fontweight='bold')
    ax_list_raw = axes_raw.flatten()
    
    for idx, key in enumerate(raw_keys):
        ax = ax_list_raw[idx]
        if 'sdd' in key:
            det_lbl_raw = get_sdd_label(key.split('_')[0], ref_pack, uppercase=True)
            ax.set_title(f"Detector {det_lbl_raw} (Raw)", fontsize=10.5, fontweight='bold')
            ax.set_ylabel("Raw Counts", fontsize=9)
        elif key == 'mcc1_raw':
            ax.set_title("Raw Au Mesh I0 (mcc1) Intensity", fontsize=11, fontweight='bold')
            ax.set_ylabel("Raw Counts / Intensity", fontsize=9)
        elif key == 'mcc4_raw':
            ax.set_title("Raw TEY of Sample (mcc4) Intensity", fontsize=11, fontweight='bold')
            ax.set_ylabel("Raw Counts / Intensity", fontsize=9)
            
        ax.set_xlabel("Energy (eV)", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        datasets = plot_data[key]
        if not datasets:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', va='center')
            continue
            
        max_amp = 0.0
        for ds in datasets:
            if len(ds["intensities"]) > 0:
                valid_data = ds["intensities"][~np.isnan(ds["intensities"])]
                if len(valid_data) > 0:
                    max_amp = max(max_amp, np.max(valid_data) - np.min(valid_data))
        offset_step = 0.5 * max_amp if max_amp > 0 else 1.0
        
        for ds_idx, ds in enumerate(datasets):
            y_plot = ds["intensities"]
            if apply_waterfall:
                offset_val = ds_idx * offset_step
                y_plot = y_plot + offset_val
            
            fmt = 'o-' if show_markers else '-'
            line, = ax.plot(ds["energies"], y_plot, fmt, label=ds['spectrum_label'],
                            color=scan_colors[ds['scan_name']], alpha=0.85)
            lines_for_cursor.append(line)
            lines_by_key[key].append(line)
        
    fig_raw.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.07, right=0.70)
    handles, labels = ax_list_raw[4].get_legend_handles_labels()
    fig_raw.legend(handles, labels, title="Sample Name", loc='upper left',
                   bbox_to_anchor=(0.72, 0.90), fontsize='small')
    figs_to_display.append(fig_raw)

    # ------------------ Figure 2: Normalized Spectra (3x2 Subplots) ------------------
    fig_norm, axes_norm = plt.subplots(3, 2, figsize=(15 * figure_scale, figure_height * figure_scale))
    # Format suptitle with multi-line wrap if i0_source_label is long
    i0_fmt_str = i0_source_label
    if len(i0_fmt_str) > 60 and " (Divided by OD:" in i0_fmt_str:
        i0_fmt_str = i0_fmt_str.replace(" (Divided by OD:", "\nOD: ")
    elif len(i0_fmt_str) > 60 and " (Smoothed" in i0_fmt_str:
        i0_fmt_str = i0_fmt_str.replace(" (Smoothed", "\n(Smoothed")

    fig_norm.suptitle(f"Batch NORMALIZED XANES Comparison\n(I0 Source: {i0_fmt_str})\n{roi_label}", fontsize=12, fontweight='bold')
    ax_list_norm = axes_norm.flatten()
    
    for idx, key in enumerate(norm_keys):
        ax = ax_list_norm[idx]
        if 'sdd' in key:
            det_lbl_norm = get_sdd_label(key.split('_')[0], ref_pack, uppercase=True)
            ax.set_title(f"Detector {det_lbl_norm} (Normalized)", fontsize=10.5, fontweight='bold')
            ax.set_ylabel("Intensity (Normalized)", fontsize=9)
        elif key == 'mcc1_raw':
            ax.set_title("Raw Au Mesh I0 (mcc1) Intensity (Reference)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Raw Counts / Intensity", fontsize=9)
        elif key == 'mcc4_norm':
            ax.set_title("Normalized TEY of Sample (mcc4) XANES", fontsize=11, fontweight='bold')
            ax.set_ylabel("Normalized Intensity", fontsize=9)
            
        ax.set_xlabel("Energy (eV)", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        datasets = plot_data[key]
        if not datasets:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center', va='center')
            continue
            
        max_amp = 0.0
        for ds in datasets:
            if len(ds["intensities"]) > 0:
                valid_data = ds["intensities"][~np.isnan(ds["intensities"])]
                if len(valid_data) > 0:
                    max_amp = max(max_amp, np.max(valid_data) - np.min(valid_data))
        offset_step = 0.5 * max_amp if max_amp > 0 else 1.0
        
        for ds_idx, ds in enumerate(datasets):
            y_plot = ds["intensities"]
            if apply_waterfall:
                offset_val = ds_idx * offset_step
                y_plot = y_plot + offset_val
            
            fmt = 'o-' if show_markers else '-'
            line, = ax.plot(ds["energies"], y_plot, fmt, label=ds['spectrum_label'],
                            color=scan_colors[ds['scan_name']], alpha=0.85)
            lines_for_cursor.append(line)
            lines_by_key[key].append(line)
        
    fig_norm.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.07, right=0.70)
    handles, labels = ax_list_norm[4].get_legend_handles_labels()
    fig_norm.legend(handles, labels, title="Sample Name", loc='upper left',
                    bbox_to_anchor=(0.72, 0.90), fontsize='small')
    figs_to_display.append(fig_norm)
    
    if hpc_config is not None:
        from matplotlib.text import Text
        for fig in figs_to_display:
            for text in fig.findobj(match=Text):
                text.set_fontsize(text.get_fontsize() * figure_scale)
        fig_xrf.tight_layout()

    # Add hover interactive cursors to all lines
    if lines_for_cursor:
        try:
            cursor = mplcursors.cursor(lines_for_cursor, hover=True)
            @cursor.connect("add")
            def on_add(sel):
                sel.annotation.set_text(f"Scan: {sel.artist.get_label().split(' (')[0]}\nE: {sel.target[0]:.2f} eV\nI: {sel.target[1]:.2f}")
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9, boxstyle="round,pad=0.3")
        except Exception:
            pass

    # Use decoupled display layout with professional tabs & live ROI controls in Jupyter notebook
    try:
        from IPython import get_ipython
        in_jupyter = (get_ipython() is not None)
    except Exception:
        in_jupyter = False

    if in_jupyter:
        import ipywidgets as widgets
        from IPython.display import display
        
        # Dynamic ROI inputs & REFRESH button for Jupyter Notebook
        if use_calibration and energy_min is not None and energy_max is not None:
            init_s = round(energy_min, 1)
            init_e = round(energy_max, 1)
            lbl_desc = "ROI Start (eV):"
            lbl_desc_e = "ROI End (eV):"
        else:
            init_s = channel_roi[0]
            init_e = channel_roi[1]
            lbl_desc = "ROI Start (Ch):"
            lbl_desc_e = "ROI End (Ch):"

        roi_start_w = widgets.FloatText(value=init_s, description=lbl_desc, layout=widgets.Layout(width='180px'))
        roi_end_w = widgets.FloatText(value=init_e, description=lbl_desc_e, layout=widgets.Layout(width='180px'))
        
        btn_refresh = widgets.Button(
            description="REFRESH BATCH PLOTS",
            button_style='success',
            icon='refresh',
            layout=widgets.Layout(width='220px', height='34px')
        )

        status_label = widgets.Label(value="Ready. Adjust ROI and click REFRESH to update spectra live.")

        def on_refresh_click(b):
            s_val = roi_start_w.value
            e_val = roi_end_w.value
            if s_val is None or e_val is None: return

            status_label.value = "Recalculating batch spectra..."
            try:
                if use_calibration and calib_data:
                    current_state['energy_min'] = min(s_val, e_val)
                    current_state['energy_max'] = max(s_val, e_val)
                    ch_s = int(max(0, (current_state['energy_min'] - ref_det_offset) / (ref_det_gain if ref_det_gain != 0 else 1.0)))
                    ch_e = int(min(255, (current_state['energy_max'] - ref_det_offset) / (ref_det_gain if ref_det_gain != 0 else 1.0)))
                    current_state['channel_roi'] = (ch_s, ch_e)
                    span_start_plot = current_state['energy_min']
                    span_end_plot = current_state['energy_max']
                else:
                    ch_s = int(max(0, min(s_val, e_val)))
                    ch_e = int(min(255, max(s_val, e_val)))
                    current_state['channel_roi'] = (ch_s, ch_e)
                    span_start_plot = ch_s
                    span_end_plot = ch_e

                roi_str = _batch_roi_label(
                    current_state['channel_roi'], current_state['energy_min'],
                    current_state['energy_max'], use_calibration
                )
                new_plot_data = compute_batch_spectra(current_state['channel_roi'], current_state['energy_min'], current_state['energy_max'])
                current_state['plot_data'] = new_plot_data

                # Update XRF ROI Span on fig_xrf
                if hasattr(ax_xrf, '_span') and ax_xrf._span is not None:
                    try:
                        ax_xrf._span.extents = (span_start_plot, span_end_plot)
                        fig_xrf.canvas.draw_idle()
                    except Exception:
                        pass

                # Update lines on raw figure
                for key in raw_keys:
                    ds_list = new_plot_data[key]
                    lines = lines_by_key[key]
                    max_amp = 0.0
                    for d in ds_list:
                        v = d["intensities"][~np.isnan(d["intensities"])]
                        if len(v) > 0: max_amp = max(max_amp, np.max(v) - np.min(v))
                    step = 0.5 * max_amp if max_amp > 0 else 1.0

                    for ds_idx, ds in enumerate(ds_list):
                        if ds_idx < len(lines):
                            y_plot = ds["intensities"]
                            if apply_waterfall:
                                y_plot = y_plot + (ds_idx * step)
                            lines[ds_idx].set_ydata(y_plot)

                # Update lines on norm figure
                for key in norm_keys:
                    ds_list = new_plot_data[key]
                    lines = lines_by_key[key]
                    max_amp = 0.0
                    for d in ds_list:
                        v = d["intensities"][~np.isnan(d["intensities"])]
                        if len(v) > 0: max_amp = max(max_amp, np.max(v) - np.min(v))
                    step = 0.5 * max_amp if max_amp > 0 else 1.0

                    for ds_idx, ds in enumerate(ds_list):
                        if ds_idx < len(lines):
                            y_plot = ds["intensities"]
                            if apply_waterfall:
                                y_plot = y_plot + (ds_idx * step)
                            lines[ds_idx].set_ydata(y_plot)

                fig_raw.suptitle(f"Batch RAW Spectra Comparison\n{roi_str}", fontsize=14, fontweight='bold')
                fig_norm.suptitle(f"Batch NORMALIZED XANES Comparison\n(I0 Source: {i0_fmt_str})\n{roi_str}", fontsize=12, fontweight='bold')

                for ax in ax_list_raw: ax.relim(); ax.autoscale_view()
                for ax in ax_list_norm: ax.relim(); ax.autoscale_view()

                fig_raw.canvas.draw_idle()
                fig_norm.canvas.draw_idle()

                status_label.value = f"Updated spectra for {roi_str}!"
            except Exception as ex:
                status_label.value = f"Error refreshing: {ex}"

        btn_refresh.on_click(on_refresh_click)

        controls_box = widgets.HBox([roi_start_w, roi_end_w, btn_refresh, status_label], layout=widgets.Layout(align_items='center', margin='0px 0px 10px 0px'))

        scroll_boxes = []
        for fig in figs_to_display:
            if hasattr(fig.canvas, 'layout'):
                fig.canvas.layout.min_width = '700px' if hpc_config is not None else '1000px'
            scroll_boxes.append(widgets.VBox([fig.canvas], layout=widgets.Layout(width='100%', overflow_x='auto')))
        
        tab_widget = widgets.Tab()
        tab_widget.children = scroll_boxes
        tab_widget.set_title(0, 'XRF Spectrum (ROI Inspector)')
        tab_widget.set_title(1, 'Raw Spectra')
        tab_widget.set_title(2, 'Normalized XANES')
        
        # SINGLE clean Yellow Save button for Jupyter
        btn_save_widget = widgets.Button(
            description="Save XANES Spectra CSVs",
            button_style='', # Custom gold/yellow styling
            icon='download',
            layout=widgets.Layout(width='240px', height='36px', margin='10px 0px 0px 5px'),
            style={'button_color': '#FFD700', 'font_weight': 'bold'}
        )
        if hpc_config is not None:
            save_output = widgets.Output()

            def save_with_visible_output(button):
                with save_output:
                    save_output.clear_output(wait=True)
                    execute_save_csvs(button)

            btn_save_widget.on_click(save_with_visible_output)
            display(widgets.VBox([controls_box, tab_widget, btn_save_widget, save_output]))
        else:
            btn_save_widget.on_click(execute_save_csvs)
            display(widgets.VBox([controls_box, tab_widget, btn_save_widget]))
    else:
        # CLI Mode: add single Matplotlib yellow button to fig_norm canvas
        ax_save_norm = fig_norm.add_axes([0.38, 0.012, 0.24, 0.035])
        btn_save_norm = Button(ax_save_norm, 'Save XANES Spectra CSVs', color='yellow', hovercolor='khaki')
        btn_save_norm.on_clicked(lambda event: execute_save_csvs())
        btn_save_norm.label.set_fontsize(9)
        btn_save_norm.label.set_weight('bold')
        fig_norm._btn_save_csv = btn_save_norm
        plt.show()

    plt.ion() # Restore interactive mode


def _batch_hpc_setup(data_packs, map_roi, on_start):
    """Choose calibration and preview batch I0 without blocking the notebook kernel."""
    import ipywidgets as widgets
    from IPython.display import display

    calibration = load_calibration()
    metadata = calibration.get('_metadata', {}) if calibration else {}
    details = (
        f"Scan: {escape(str(metadata.get('scan_used', 'N/A')))}; "
        f"Edges: {escape(str(metadata.get('edges_used', 'N/A')))}; "
        f"Updated: {escape(str(metadata.get('last_updated', 'N/A')))}"
    )
    spectrum_options = []
    for pack in data_packs:
        for detector, files in sorted(pack.get('sdd_files', {}).items()):
            paths = [(energy, path) for energy, path in files.items() if path and os.path.isfile(path)]
            if paths:
                representative = pack.get('representative_energy')
                if representative is not None:
                    energy, path = min(paths, key=lambda pair: abs(float(pair[0]) - float(representative)))
                else:
                    energy, path = paths[0]
                scan = pack.get('scan_name', 'Scan')
                sample = _batch_spectrum_label(pack, scan)
                spectrum_options.append((f"{sample} | {scan} | {detector}", (pack, detector, energy, path)))

    spectrum_select = widgets.Dropdown(
        options=[(label, index) for index, (label, _) in enumerate(spectrum_options)],
        description="Spectrum:", layout=widgets.Layout(width="900px")
    )
    spectrum_image = widgets.Image(format="png")
    spectrum_status = widgets.HTML()
    detectors = {det for pack in data_packs if len(pack.get('energies', [])) > 2
                 for det in pack.get('sdd_files', {})}
    missing = []
    for detector in sorted(detectors):
        params = calibration.get(detector)
        try:
            gain, offset = float(params['gain']), float(params['offset'])
            if not np.isfinite(gain) or not np.isfinite(offset) or gain == 0:
                missing.append(detector)
        except (TypeError, KeyError, ValueError):
            missing.append(detector)
    use_calibration = widgets.Checkbox(value=bool(calibration) and not missing,
                                       description="Apply SDD calibration", indent=False)
    if not calibration or missing:
        use_calibration.disabled = True
        reason = "No saved calibration available." if not calibration else f"Missing or invalid calibration: {', '.join(missing)}."
        details += f" <b style='color:red;'>{escape(reason)} Continue without calibration or update the calibration file.</b>"
    source = widgets.RadioButtons(options=[("Internal Au Mesh I0 (mcc1)", "internal"),
                                           ("External I0 CSV", "external")],
                                  value="internal", description="I0 source:")
    waterfall = widgets.Checkbox(value=False, description="Waterfall offsets", indent=False)
    next_button = widgets.Button(description="Continue", button_style="success")
    cancel_button = widgets.Button(description="Cancel")
    status = widgets.HTML()
    stage = widgets.VBox()
    output = widgets.Output()

    def update_spectrum(change=None):
        if not spectrum_options:
            spectrum_image.value = b""
            spectrum_status.value = "<b style='color:red;'>No SDD spectrum file is available for preview.</b>"
            return
        _, (pack, detector, energy, path) = spectrum_options[spectrum_select.value]
        try:
            raw = np.fromfile(path, dtype=np.uint32)
            mask = _batch_spatial_mask(pack, map_roi)
            count = len(raw) // 256
            if count < len(mask) or not np.any(mask):
                raise ValueError("Spectrum file does not match the selected map area.")
            spectrum = raw[:count * 256].reshape(count, 256)[:len(mask)][mask].sum(axis=0)
            channels = np.arange(256)
            fig = Figure(figsize=(9, 3.5))
            FigureCanvasAgg(fig)
            left, right = fig.subplots(1, 2)
            left.plot(channels, spectrum)
            left.set_xlabel("Channel")
            left.set_title("Uncalibrated")
            params = calibration.get(detector)
            if isinstance(params, dict) and 'gain' in params and 'offset' in params:
                gain, offset = float(params['gain']), float(params['offset'])
                if not np.isfinite(gain) or not np.isfinite(offset) or gain == 0:
                    raise ValueError(f"Invalid saved calibration for {detector}.")
                right.plot(gain * channels + offset, spectrum)
                right.set_xlabel("Emission energy (eV)")
                right.set_title("Saved calibration")
                sample = _batch_spectrum_label(pack, pack.get('scan_name', 'Scan'))
                spectrum_status.value = (
                    f"Sample: {escape(str(sample))}; {escape(detector)}; incident energy {energy:.2f} eV. "
                    f"gain = {gain:.4g} eV/channel; offset = {offset:.4g} eV."
                )
            else:
                right.text(0.5, 0.5, "No saved calibration for this detector", ha="center", va="center",
                           transform=right.transAxes)
                spectrum_status.value = f"<b style='color:red;'>No calibration for {escape(detector)}.</b>"
            for ax in (left, right):
                ax.set_ylabel("Counts")
                ax.grid(True, linestyle="--", alpha=0.4)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=100)
            spectrum_image.value = buffer.getvalue()
        except (OSError, ValueError, TypeError) as error:
            spectrum_image.value = b""
            spectrum_status.value = f"<b style='color:red;'>Cannot preview spectrum: {escape(str(error))}</b>"

    def start_plot(config):
        stage.children = (widgets.HTML("Preparing batch plots..."),)
        with output:
            try:
                on_start(config)
                stage.children = (widgets.HTML("Batch plots ready."),)
            except Exception:
                plt.ion()
                stage.children = (widgets.HTML("<b style='color:red;'>Batch plotting failed; see error below.</b>"),)
                traceback.print_exc()

    def next_step(button):
        config = {
            'use_calibration': use_calibration.value,
            'calib_data': calibration,
            'use_internal': source.value == 'internal',
            'apply_waterfall': waterfall.value,
            'external_xy': (None, None),
            'i0_settings': None,
            'i0_source_label': "Au Mesh I0 (mcc1)",
        }
        status.value = ""
        if config['use_internal']:
            try:
                curves = []
                for pack in data_packs:
                    energies = np.array(sorted(pack.get('energies', [])), dtype=float)
                    if len(energies) <= 2:
                        continue
                    mask = _batch_spatial_mask(pack, map_roi)
                    values = np.array([get_mcc_val(pack, en, 1, mask, strict=True) for en in energies])
                    scan = pack.get('scan_name', 'Scan')
                    sample = _batch_spectrum_label(pack, scan)
                    curves.append((f"{sample} | {scan}", energies, values))
                if not curves:
                    raise ValueError("No multi-energy scans are available for the internal I0 preview.")
            except (ValueError, OSError, TypeError) as error:
                status.value = f"<b style='color:red;'>{escape(str(error))}</b>"
                return

            def finish_internal(result):
                config['i0_settings'] = result[6]
                config['i0_source_label'] += result[1]
                start_plot(config)

            preview = _i0_preview_widget(
                curves[0][1], curves[0][2], "Internal Au Mesh I0", False,
                finish_internal, curves=curves, include_settings=True
            )
            stage.children = (
                widgets.HTML("Select a scan to preview. The same settings apply to each scan's own I0."),
                preview,
            )
        else:
            def external_selected(path):
                try:
                    df = read_csv_with_comments(path)
                    df.columns = [c.strip() for c in df.columns]
                    if len(df.columns) < 2:
                        raise ValueError("The I0 file needs energy and intensity columns.")
                    e_col = next((c for c in df.columns if 'energy' in c.lower()), df.columns[0])
                    i_col = next((c for c in df.columns if any(k in c.lower() for k in ('i0', 'intensity', 'norm', 'tey'))), df.columns[1])
                    x = pd.to_numeric(df[e_col], errors='coerce').to_numpy()
                    y = pd.to_numeric(df[i_col], errors='coerce').to_numpy()
                    valid = np.isfinite(x) & np.isfinite(y)
                    order = np.argsort(x[valid])
                    x, y = x[valid][order], y[valid][order]
                    if len(x) == 0:
                        raise ValueError("No valid energy and intensity pairs found.")

                    def finish_external(result):
                        processed, extra, _, _, shift_enabled, shift = result
                        config['external_xy'] = (x + (shift if shift_enabled else 0.0), processed)
                        config['i0_source_label'] = f"External: {os.path.basename(path)} [{i_col}]{extra}"
                        start_plot(config)

                    preview = _i0_preview_widget(x, y, f"{os.path.basename(path)} [{i_col}]", True, finish_external)
                    stage.children = (widgets.HTML(f"Energy column: {escape(e_col)}; I0 column: {escape(i_col)}"), preview)
                except (OSError, ValueError, TypeError, IndexError) as error:
                    status.value = f"<b style='color:red;'>Cannot load external I0: {escape(str(error))}</b>"

            browser = browse_beamline_data(
                on_load=external_selected, file_filter=('.csv', '.txt'),
                title="Select external I0 for the batch", auto_display=False
            )
            stage.children = (browser, status, cancel_button)

    def cancel(button):
        stage.children = (widgets.HTML("Batch plotting cancelled."),)

    spectrum_select.observe(update_spectrum, names="value")
    next_button.on_click(next_step)
    cancel_button.on_click(cancel)
    update_spectrum()
    stage.children = (
        widgets.HTML("<b>SDD calibration preview</b><br>Preview shows a spectrum from a loaded sample, "
                     "not necessarily the calibration scan.<br>" + details),
        spectrum_select, spectrum_image, spectrum_status,
        use_calibration, source, waterfall, widgets.HBox([next_button, cancel_button]), status
    )
    display(widgets.VBox([stage, output]))


def plot_sgm_bsky_data_batch(data_packs, channel_roi=None, xrf_roi=None, map_roi=None, show_markers=True, use_color=True, auto_element_roi=False, element=None, n_sigma=2.0):
    """Plot batch spectra, with interactive configuration in headless notebooks."""
    if not data_packs:
        print("No scans to plot.", file=sys.stderr)
        return
    if not HAS_GUI:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            def start(config):
                return _plot_sgm_bsky_data_batch(
                    data_packs, channel_roi, xrf_roi, map_roi, show_markers, use_color,
                    auto_element_roi, element, n_sigma, hpc_config=config
                )
            _batch_hpc_setup(data_packs, map_roi, start)
            return
    return _plot_sgm_bsky_data_batch(
        data_packs, channel_roi, xrf_roi, map_roi, show_markers, use_color,
        auto_element_roi, element, n_sigma
    )

