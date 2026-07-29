import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import filedialog
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
import sdd_calibration_utils as sdd_calib
from alignment_utils import safe_filedialog_call

def plot_selected_emission_spectra(
    data_pack=None,
    selected_energies=None,
    detector='sdd1',
    use_energy_calib=True,
    apply_waterfall=False,
    waterfall_offset=None,
    show_markers=True,
    map_roi=None,
    save_plot=False,
    output_dir=None,
    file_format='png',
    dpi=300
):
    """
    Plots overlaid 1D Emission / XRD Spectra (256 channels or calibrated eV)
    for up to 5 selected incident energy images to observe scatter peak shifts.

    Args:
        data_pack (dict, optional): Data dictionary returned by analyze_sgm_bsky_data().
                                    If None, opens a file browser to select an H5 stack file.
        selected_energies (list or array, optional): List of up to 5 incident energy values (eV)
                                                    or energy indices. If None, auto-selects 5 spaced energies.
        detector (str): Detector name ('sdd1', 'sdd2', 'sdd3', 'sdd4', or 'average').
        use_energy_calib (bool): If True, converts emission channels to calibrated eV.
        apply_waterfall (bool): If True, applies vertical offset between spectra.
        waterfall_offset (float, optional): Custom vertical offset. If None, auto-calculated.
        show_markers (bool): If True, displays markers on spectral lines.
        map_roi (list, optional): [x1, x2, y1, y2] spatial ROI for filtering.
        save_plot (bool): If True, opens save dialog after display.
        output_dir (str, optional): Target folder for save dialog.
        file_format (str): Default image format ('png', 'pdf', 'svg', 'jpg').
        dpi (int): Image resolution (DPI).

    Returns:
        list of str or None: List of saved file paths if saved, else None.
    """
    if data_pack is None:
        data_pack = analyze_sgm_bsky_data(verbose=True)

    if not data_pack or not data_pack.get('sdd_files'):
        print("Error: No valid SDD stack data found.", file=sys.stderr)
        return None

    all_energies = np.array(sorted(data_pack['energies']))
    if len(all_energies) == 0:
        print("Error: Stack contains no incident energies.", file=sys.stderr)
        return None

    # Handle energy selection (max 5)
    if selected_energies is None:
        if len(all_energies) <= 5:
            target_energies = list(all_energies)
        else:
            # Pick 5 evenly spaced energies across range by default
            idx_choices = np.linspace(0, len(all_energies) - 1, 5, dtype=int)
            target_energies = list(all_energies[idx_choices])
    else:
        # Convert user inputs to closest available energies
        target_energies = []
        for item in selected_energies:
            if isinstance(item, (int, np.integer)) and 0 <= item < len(all_energies):
                target_energies.append(all_energies[item])
            else:
                try:
                    val = float(item)
                    closest_idx = np.abs(all_energies - val).argmin()
                    target_energies.append(all_energies[closest_idx])
                except (ValueError, TypeError):
                    continue

        # Unique & limit to max 5
        target_energies = sorted(list(dict.fromkeys(target_energies)))[:5]

    if not target_energies:
        print("Error: No valid incident energies selected.", file=sys.stderr)
        return None

    scan_title = data_pack.get('scan_name', 'Stack Scan')
    available_dets = sorted(data_pack['sdd_files'].keys())

    # Determine detectors to read
    if detector == 'average':
        target_dets = available_dets
    elif detector in available_dets:
        target_dets = [detector]
    else:
        print(f"Warning: Detector '{detector}' not found. Available: {available_dets}")
        target_dets = available_dets[:1]

    # Spatial mask setup
    x_raw = data_pack.get('x', np.array([]))
    y_raw = data_pack.get('y', np.array([]))
    spatial_mask = None
    if map_roi and x_raw.size > 0 and y_raw.size > 0:
        x1, x2 = sorted(map_roi[0:2])
        y1, y2 = sorted(map_roi[2:4])
        spatial_mask = (x_raw >= x1) & (x_raw <= x2) & (y_raw >= y1) & (y_raw <= y2)

    # Calibration setup
    calib_data = sdd_calib.load_calibration() if use_energy_calib else None

    # Emission X-axis
    ref_det = target_dets[0]
    if use_energy_calib:
        gain = 1.0
        offset = 0.0
        if calib_data and ref_det in calib_data:
            gain = calib_data[ref_det].get('gain', 1.0)
            offset = calib_data[ref_det].get('offset', 0.0)

        if gain != 1.0 or offset != 0.0:
            emission_axis = sdd_calib.channel_to_energy(np.arange(256), gain, offset)
        else:
            emission_axis = np.arange(256) * 10.0
        xlabel = "Emission Energy (eV)"
    else:
        emission_axis = np.arange(256)
        xlabel = "Emission Channel"

    # Extract 1D spectra for selected energies
    spectra_data = []
    for en in target_energies:
        det_spectra = []
        for det_name in target_dets:
            p = data_pack['sdd_files'][det_name].get(en)
            if p and os.path.exists(p):
                d1d = np.fromfile(p, dtype=np.uint32)
                num_s = len(d1d) // 256
                if num_s > 0:
                    s2d = d1d[:num_s * 256].reshape((num_s, 256))
                    if spatial_mask is not None and spatial_mask.size >= num_s:
                        m = spatial_mask[:num_s]
                        spec = np.sum(s2d[m], axis=0) if np.any(m) else np.sum(s2d, axis=0)
                    else:
                        spec = np.sum(s2d, axis=0)
                    det_spectra.append(spec)

        if det_spectra:
            avg_spec = np.mean(det_spectra, axis=0)
            spectra_data.append({'energy': en, 'spectrum': avg_spec})

    if not spectra_data:
        print("Error: Could not extract spectral data for selected energies.", file=sys.stderr)
        return None

    # Calculate Waterfall offset
    if apply_waterfall:
        if waterfall_offset is None:
            max_amp = max(np.ptp(item['spectrum']) for item in spectra_data)
            offset_step = 0.5 * max_amp if max_amp > 0 else 1.0
        else:
            offset_step = waterfall_offset
    else:
        offset_step = 0.0

    # Colors for distinct overlay (up to 5 colors)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Prepare Figure
    fig, ax = plt.subplots(figsize=(9.5, 6))
    plt.subplots_adjust(bottom=0.15)

    lines = []
    for idx, item in enumerate(spectra_data):
        en_val = item['energy']
        raw_y = item['spectrum']
        off_val = idx * offset_step
        y_plot = raw_y + off_val

        color = palette[idx % len(palette)]
        fmt = 'o-' if show_markers else '-'
        lbl = f"E_inc = {en_val:.2f} eV" + (f" (+{off_val:.1f})" if apply_waterfall and off_val > 0 else "")

        line, = ax.plot(emission_axis, y_plot, fmt, color=color, label=lbl, alpha=0.85, linewidth=1.5, markersize=3)
        lines.append(line)

    ax.set_xlabel(xlabel, fontsize=10, fontweight='semibold')
    ax.set_ylabel('Counts / Intensity' + (' (Waterfall Offset)' if apply_waterfall else ''), fontsize=10, fontweight='semibold')
    ax.set_title(f'Emission / XRD Spectra Comparison ({len(spectra_data)} Energies)\n{scan_title} ({detector})', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize='small')

    saved_files = []

    # Add interactive "Save Spectra Plot" button
    ax_save_btn = fig.add_axes([0.38, 0.02, 0.24, 0.05])
    btn_save = Button(ax_save_btn, 'Save Spectra Plot', color='yellow', hovercolor='khaki')
    btn_save.label.set_fontsize(9)
    btn_save.label.set_fontweight('bold')

    def trigger_save(event=None):
        if output_dir:
            save_directory = os.path.abspath(output_dir)
        else:
            h5_path = data_pack.get('h5_file_path') or data_pack.get('h5_dir')
            if h5_path and os.path.isfile(h5_path):
                save_directory = os.path.dirname(os.path.abspath(h5_path))
            elif h5_path and os.path.isdir(h5_path):
                save_directory = os.path.abspath(h5_path)
            else:
                save_directory = os.getcwd()

        os.makedirs(save_directory, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in scan_title)
        ext = file_format.lstrip('.')
        initial_name = f"{safe_title}_Emission_Spectra_Overlay_{detector}.{ext}"

        ax_save_btn.set_visible(False)
        fig.canvas.draw_idle()

        out_filename = safe_filedialog_call(
            filedialog.asksaveasfilename,
            title="Save Emission / XRD Spectra Plot",
            initialdir=save_directory,
            initialfile=initial_name,
            defaultextension=f".{ext}",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("SVG Vector Image", "*.svg"),
                ("JPEG Image", "*.jpg;*.jpeg"),
                ("All Files", "*.*")
            ]
        )

        if out_filename:
            fig.savefig(out_filename, dpi=dpi, bbox_inches='tight')
            print(f"Saved emission spectra plot to: {out_filename}")
            saved_files.append(out_filename)
        else:
            print("Save operation cancelled.")

        ax_save_btn.set_visible(True)
        fig.canvas.draw_idle()

    btn_save.on_clicked(trigger_save)
    fig._btn_save_ref = btn_save

    if save_plot:
        trigger_save()

    plt.show()
    return saved_files if saved_files else None

if __name__ == '__main__':
    plot_selected_emission_spectra()
