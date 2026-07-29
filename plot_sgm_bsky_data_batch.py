import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
from tkinter import filedialog
import mplcursors

from alignment_utils import (
    safe_filedialog_call,
    show_custom_dialog,
    interactive_channel_selector,
    get_safe_save_path,
    format_num_val
)
from sdd_calibration_utils import (
    load_calibration,
    channel_to_energy,
    get_calibrated_bounds
)
from plot_sgm_bsky_data import read_csv_with_comments, ExternalI0PreviewDialog, get_tk_root

def get_mcc_val(path_pack, energy, channel, spatial_mask):
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
        except Exception:
            pass
    return 1.0 if channel == 1 else 0.0

def plot_sgm_bsky_data_batch(data_packs, channel_roi=None, xrf_roi=None, map_roi=None, show_markers=True, use_color=True):
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
    """
    if not data_packs:
        print("No scans to plot.", file=sys.stderr)
        return

    # Disable auto-display temporarily to prevent duplicate plotting in Jupyter
    plt.ioff()

    # 1. Prompt for SDD Calibration FIRST
    use_calibration = show_custom_dialog(
        title="SDD Calibration",
        message="Apply SDD Energy Calibration (using sdd_calibration.json)?",
        dialog_type="yesno"
    )
    calib_data = {}
    if use_calibration is True:
        calib_data = load_calibration()
        if not calib_data:
            print("Warning: sdd_calibration.json not found. Falling back to fixed channel ROI.", file=sys.stderr)
            use_calibration = False

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
    use_internal = show_custom_dialog(
        title="I0 Selection",
        message="Use INTERNAL mcc1 for I0 normalization?\n\n(Select 'No' to load an EXTERNAL I0 CSV)",
        dialog_type="yesno"
    )
    if use_internal is None:
        print("I0 selection cancelled. Exiting plot.", file=sys.stderr)
        plt.ion()
        return

    x_sorted, y_sorted = None, None
    i0_source_label = "mcc1"
    if not use_internal:
        ext_path = safe_filedialog_call(
            filedialog.askopenfilename,
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
                    selected_e_col, selected_i_col, x_s, y_s, extra_str, cal_en, cal_val = dialog.result
                    shift = cal_val if cal_en else 0.0
                    x_sorted = x_s + shift
                    y_sorted = y_s
                    i0_source_label = f"External: {os.path.basename(ext_path)} [{selected_i_col}]{extra_str}"
                    print(f"Loaded external I0 calibration from {ext_path} ({i0_source_label})")
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
    apply_waterfall = show_custom_dialog(
        title="Waterfall Offset",
        message="Apply a vertical offset (waterfall plot) to separate the spectra?\n\n(Select 'No' to overlay them directly)",
        dialog_type="yesno"
    )
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
            'mcc4_raw': [], 'mcc4_norm': []
        }

        for scan_idx, path_pack in enumerate(data_packs):
            scan_name = path_pack.get('scan_name', f"Scan_{scan_idx + 1}")
            x_coords_raw = path_pack.get('x', np.array([]))
            y_coords_raw = path_pack.get('y', np.array([]))
            actual_num_s = min(x_coords_raw.size, y_coords_raw.size)
            
            try:
                first_det = list(path_pack['sdd_files'].keys())[0]
                first_en = path_pack['energies'][0]
                first_path = path_pack['sdd_files'][first_det].get(first_en)
                if first_path and os.path.exists(first_path):
                    actual_num_s = min(os.path.getsize(first_path) // 1024, actual_num_s)
            except Exception:
                pass

            x_coords = x_coords_raw[:actual_num_s]
            y_coords = y_coords_raw[:actual_num_s]

            if map_roi is None:
                curr_map_roi = [np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)]
            else:
                curr_map_roi = map_roi

            x1, x2 = sorted(curr_map_roi[0:2])
            y1, y2 = sorted(curr_map_roi[2:4])
            spatial_mask = (x_coords >= x1) & (x_coords <= x2) & (y_coords >= y1) & (y_coords <= y2)

            all_energies = np.array(sorted(path_pack['energies']))
            i0_values = np.ones(len(all_energies))
            if not use_internal and x_sorted is not None and y_sorted is not None:
                i0_values = np.interp(all_energies, x_sorted, y_sorted)

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
                        norm_factor = mcc_val if use_internal else i0_values[e_idx]
                        if norm_factor <= 0: norm_factor = 1.0
                        val_norm = val_raw / norm_factor
                    norm_intensities.append(val_norm)

                data_out[f"{det_name}_raw"].append({"scan_name": scan_name, "energies": all_energies, "intensities": np.array(raw_intensities)})
                data_out[f"{det_name}_norm"].append({"scan_name": scan_name, "energies": all_energies, "intensities": np.array(norm_intensities)})

            mcc1_raw_list, mcc4_raw_list, mcc4_norm_list = [], [], []
            for e_idx, energy in enumerate(all_energies):
                mcc1_val = get_mcc_val(path_pack, energy, 1, spatial_mask)
                mcc4_val = get_mcc_val(path_pack, energy, 4, spatial_mask)
                mcc1_raw_list.append(mcc1_val)
                mcc4_raw_list.append(mcc4_val)
                norm_factor = mcc1_val if use_internal else i0_values[e_idx]
                if norm_factor <= 0: norm_factor = 1.0
                mcc4_norm_list.append(mcc4_val / norm_factor)

            data_out['mcc1_raw'].append({"scan_name": scan_name, "energies": all_energies, "intensities": np.array(mcc1_raw_list)})
            data_out['mcc4_raw'].append({"scan_name": scan_name, "energies": all_energies, "intensities": np.array(mcc4_raw_list)})
            data_out['mcc4_norm'].append({"scan_name": scan_name, "energies": all_energies, "intensities": np.array(mcc4_norm_list)})

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
            save_dir = os.path.dirname(os.path.abspath(h5_path)) if h5_path and os.path.isfile(h5_path) else (os.path.abspath(h5_path) if h5_path and os.path.isdir(h5_path) else os.getcwd())
            scan_name = path_pack.get('scan_name', f"Scan_{scan_idx + 1}")
            scan_energies = np.array(sorted(path_pack['energies']))
            
            if use_calibration and curr_e_min is not None and curr_e_max is not None:
                roi_str = f"{curr_e_min:.1f}-{curr_e_max:.1f}eV"
            else:
                roi_str = f"Ch{curr_ch_roi[0]}-{curr_ch_roi[1]}"
                
            csv_filename = f"{scan_name}_Rect_{roi_str}_summary.csv"
            csv_filepath = os.path.join(save_dir, csv_filename)
            
            if os.path.exists(csv_filepath):
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
                calib_str = f"Active (Scan: {scan_used}, Edges: {edges_used})"
            else:
                calib_str = "Disabled"

            rows = []
            rows += [
                f"# Scan Name: {scan_name}",
                f"# Scan Type: {path_pack.get('scan_type', 'N/A')}",
                f"# Project: {path_pack.get('project', 'N/A')}",
                f"# Date: {path_pack.get('date', 'N/A')}",
                f"# Number of Images: {len(scan_energies)}",
                f"# Energy Regions: {path_pack.get('Energy Regions', 'N/A')}",
                f"# Grid Dimensions: {path_pack.get('nx', 'N/A')} x {path_pack.get('ny', 'N/A')} ({path_pack.get('x', np.array([])).size} points)",
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
            rows += [
                f"# ROI Selection: Rect",
                f"# {'Energy ROI' if use_calibration else 'Channels'}: {roi_str}",
                f"# Normalization: {i0_source_label}",
                f"# SDD Calibration: {calib_str}",
                "#"
            ]
            
            ext_i0_col_values = np.interp(scan_energies, x_sorted, y_sorted) if (not use_internal and x_sorted is not None and y_sorted is not None) else None

            rows.append("# Column 1: Calibrated Energy (eV)")
            rows.append("# Column 2: Original Energy (eV)")
            c_idx = 3
            for det in ['sdd1', 'sdd2', 'sdd3', 'sdd4']:
                rows.append(f"# Column {c_idx}: RAW_{det.upper()}"); c_idx += 1
            rows.append(f"# Column {c_idx}: RAW_Average_SDD"); c_idx += 1
            
            for det in ['sdd1', 'sdd2', 'sdd3', 'sdd4']:
                rows.append(f"# Column {c_idx}: NORM_{det.upper()} (by {i0_source_label})"); c_idx += 1
            rows.append(f"# Column {c_idx}: NORM_Average_SDD"); c_idx += 1
            
            rows.append(f"# Column {c_idx}: RAW_I0 (MCC1)"); c_idx += 1
            if ext_i0_col_values is not None:
                rows.append(f"# Column {c_idx}: RAW_External_I0 (from {i0_source_label})"); c_idx += 1
            rows.append(f"# Column {c_idx}: RAW_TEY (MCC4)"); c_idx += 1
            rows.append(f"# Column {c_idx}: NORM_TEY (MCC4) (by {i0_source_label})"); c_idx += 1
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
                if ext_i0_col_values is not None:
                    row_vals.append(f"{ext_i0_col_values[i]:.6f}")
                row_vals.append(f"{mcc4_raw_data[i]:.6f}")
                row_vals.append(f"{mcc4_norm_data[i]:.6f}")
                rows.append(",".join(row_vals))
                
            try:
                with open(csv_filepath, 'w', encoding='utf-8') as f_out:
                    f_out.write("\n".join(rows) + "\n")
                print(f"  Saved XANES CSV: {csv_filename}")
                saved_count += 1
            except Exception as e:
                print(f"  Error saving CSV for {scan_name}: {e}", file=sys.stderr)
                
        show_custom_dialog(
            title="Save Complete",
            message=f"Successfully saved XANES spectra CSV files for {saved_count} scans in their original folders.",
            dialog_type="info"
        )

    # ------------------ Figure 0: XRF Emission Spectrum (Tab 0 Inspector) ------------------
    fig_xrf, ax_xrf = plt.subplots(figsize=(10, 4.5))
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

    line_xrf, = ax_xrf.plot(xrf_x, xrf_spec, color='blue', lw=1.5, label=f"XRF Spectrum ({ref_det.upper()})")
    ax_xrf.set_xlabel(xlabel_str, fontsize=10, fontweight='bold')
    ax_xrf.set_ylabel("PFY Counts", fontsize=10, fontweight='bold')
    ax_xrf.set_title(f"Interactive XRF Spectrum ROI Inspector ({ref_pack.get('scan_name', 'Scan 1')} - {ref_det.upper()})\n(Drag mouse left/right on peak to change ROI dynamically)", fontsize=11, fontweight='bold')
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
    fig_raw, axes_raw = plt.subplots(3, 2, figsize=(11, 13.0))
    fig_raw.suptitle(f"Batch RAW Spectra Comparison\nSDD Channel ROI: {channel_roi[0]}-{channel_roi[1]}", fontsize=14, fontweight='bold')
    ax_list_raw = axes_raw.flatten()
    
    for idx, key in enumerate(raw_keys):
        ax = ax_list_raw[idx]
        if 'sdd' in key:
            ax.set_title(f"Detector {key.split('_')[0].upper()} (Raw)", fontsize=11, fontweight='semibold')
            ax.set_ylabel("Raw Counts", fontsize=9)
        elif key == 'mcc1_raw':
            ax.set_title("Raw I0 (MCC1) Intensity", fontsize=11, fontweight='semibold')
            ax.set_ylabel("Raw Counts / Intensity", fontsize=9)
        elif key == 'mcc4_raw':
            ax.set_title("Raw TEY (MCC4) Intensity", fontsize=11, fontweight='semibold')
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
            label_suffix = ""
            if apply_waterfall:
                offset_val = ds_idx * offset_step
                y_plot = y_plot + offset_val
                label_suffix = f" (+{offset_val:.1f})"
            
            fmt = 'o-' if show_markers else '-'
            line, = ax.plot(ds["energies"], y_plot, fmt, label=f"{ds['scan_name']}{label_suffix}", alpha=0.85)
            lines_for_cursor.append(line)
            lines_by_key[key].append(line)
        ax.legend(fontsize='x-small', loc='best')
        
    fig_raw.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.07)
    figs_to_display.append(fig_raw)

    # ------------------ Figure 2: Normalized Spectra (3x2 Subplots) ------------------
    fig_norm, axes_norm = plt.subplots(3, 2, figsize=(11, 13.0))
    # Format suptitle with multi-line wrap if i0_source_label is long
    i0_fmt_str = i0_source_label
    if len(i0_fmt_str) > 60 and " (Divided by OD:" in i0_fmt_str:
        i0_fmt_str = i0_fmt_str.replace(" (Divided by OD:", "\nOD: ")
    elif len(i0_fmt_str) > 60 and " (Smoothed" in i0_fmt_str:
        i0_fmt_str = i0_fmt_str.replace(" (Smoothed", "\n(Smoothed")

    fig_norm.suptitle(f"Batch NORMALIZED XANES Comparison\n(I0 Source: {i0_fmt_str})\nSDD Channel ROI: {channel_roi[0]}-{channel_roi[1]}", fontsize=12, fontweight='bold')
    ax_list_norm = axes_norm.flatten()
    
    for idx, key in enumerate(norm_keys):
        ax = ax_list_norm[idx]
        if 'sdd' in key:
            ax.set_title(f"Detector {key.split('_')[0].upper()} (Normalized)", fontsize=11, fontweight='semibold')
            ax.set_ylabel("Intensity (Normalized)", fontsize=9)
        elif key == 'mcc1_raw':
            ax.set_title("Raw I0 (MCC1) Intensity (Reference)", fontsize=11, fontweight='semibold')
            ax.set_ylabel("Raw Counts / Intensity", fontsize=9)
        elif key == 'mcc4_norm':
            ax.set_title("Normalized TEY (MCC4) XANES", fontsize=11, fontweight='semibold')
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
            label_suffix = ""
            if apply_waterfall:
                offset_val = ds_idx * offset_step
                y_plot = y_plot + offset_val
                label_suffix = f" (+{offset_val:.1f})"
            
            fmt = 'o-' if show_markers else '-'
            line, = ax.plot(ds["energies"], y_plot, fmt, label=f"{ds['scan_name']}{label_suffix}", alpha=0.85)
            lines_for_cursor.append(line)
            lines_by_key[key].append(line)
        ax.legend(fontsize='x-small', loc='best')
        
    fig_norm.subplots_adjust(hspace=0.35, wspace=0.25, bottom=0.07)
    figs_to_display.append(fig_norm)
    
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
                    roi_str = f"SDD Energy ROI: {current_state['energy_min']:.1f}-{current_state['energy_max']:.1f} eV (Ch {ch_s}-{ch_e})"
                    span_start_plot = current_state['energy_min']
                    span_end_plot = current_state['energy_max']
                else:
                    ch_s = int(max(0, min(s_val, e_val)))
                    ch_e = int(min(255, max(s_val, e_val)))
                    current_state['channel_roi'] = (ch_s, ch_e)
                    roi_str = f"SDD Channel ROI: {ch_s}-{ch_e}"
                    span_start_plot = ch_s
                    span_end_plot = ch_e

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
                fig.canvas.layout.min_width = '1000px'
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

