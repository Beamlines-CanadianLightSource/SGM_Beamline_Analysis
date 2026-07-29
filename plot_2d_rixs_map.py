import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Button
from tkinter import filedialog
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
import sdd_calibration_utils as sdd_calib
from alignment_utils import safe_filedialog_call

def plot_incident_vs_emission_2d(
    data_pack=None,
    detector='sdd1',
    use_energy_calib=True,
    log_scale=False,
    cmap='viridis',
    save_plot=False,
    output_dir=None,
    file_format='png',
    dpi=300
):
    """
    Plots a 2D Heatmap of Incident Energy vs Emission Energy (RIXS/2D XRF Map).

    Displays the plots on screen first arranged compactly (2 side-by-side).
    An interactive 'Save 2D Map' button is available on the figure window so the user
    can view the maps first before choosing to save.

    Args:
        data_pack (dict, optional): Data dictionary returned by analyze_sgm_bsky_data().
                                    If None, opens a file browser to select an H5 stack file.
        detector (str): Detector to plot ('sdd1', 'sdd2', 'sdd3', 'sdd4', or 'all').
        use_energy_calib (bool): If True, converts emission channels to calibrated eV.
        log_scale (bool): If True, applies log scaling to the intensity colorbar.
        cmap (str): Matplotlib colormap name.
        save_plot (bool or str): If True, opens save file dialog after displaying plot.
        output_dir (str, optional): Default directory path for save dialog. Defaults to original data folder.
        file_format (str): Default image format for save dialog ('png', 'pdf', 'svg', 'jpg'). Defaults to 'png'.
        dpi (int): Output image resolution (DPI). Defaults to 300.

    Returns:
        list of str or None: List of saved image filepaths if saved, else None.
    """
    if data_pack is None:
        data_pack = analyze_sgm_bsky_data(verbose=True)

    if not data_pack or not data_pack.get('sdd_files'):
        print("Error: No valid SDD stack data found.", file=sys.stderr)
        return None

    energies = data_pack['energies']
    if len(energies) == 0:
        print("Error: Stack contains no incident energies.", file=sys.stderr)
        return None

    available_dets = sorted(data_pack['sdd_files'].keys())
    if detector == 'all':
        target_dets = available_dets
    elif detector in available_dets:
        target_dets = [detector]
    else:
        print(f"Warning: Detector '{detector}' not found. Available: {available_dets}")
        target_dets = available_dets[:1]

    calib_data = sdd_calib.load_calibration() if use_energy_calib else None
    saved_files = []
    scan_title = data_pack.get('scan_name', 'Stack Scan')

    # Pre-extract matrices for target detectors
    matrices = {}
    for det_name in target_dets:
        sdd_dict = data_pack['sdd_files'][det_name]
        matrix = np.zeros((len(energies), 256), dtype=np.float64)

        for i, e in enumerate(energies):
            p = sdd_dict.get(e)
            if p and os.path.exists(p):
                d1d = np.fromfile(p, dtype=np.uint32)
                num_s = len(d1d) // 256
                if num_s > 0:
                    s2d = d1d[:num_s * 256].reshape((num_s, 256))
                    matrix[i, :] = np.sum(s2d, axis=0)
        matrices[det_name] = matrix

    num_dets = len(target_dets)

    # Determine Grid Layout (2 side by side)
    if num_dets == 1:
        fig, ax_arr = plt.subplots(1, 1, figsize=(7.5, 5.2))
        axes = [ax_arr]
    elif num_dets == 2:
        fig, ax_arr = plt.subplots(1, 2, figsize=(11.5, 4.8))
        axes = list(ax_arr)
    else:
        nrows = int(np.ceil(num_dets / 2))
        fig, ax_arr = plt.subplots(nrows, 2, figsize=(11.5, 4.2 * nrows))
        axes = list(ax_arr.flatten())

    fig.suptitle(f'2D Incident vs. Emission Map - {scan_title}', fontsize=12, fontweight='bold')

    for idx, det_name in enumerate(target_dets):
        ax = axes[idx]
        matrix = matrices[det_name]

        # Build emission energy / channel axis
        if use_energy_calib:
            gain = 1.0
            offset = 0.0
            if calib_data and det_name in calib_data:
                gain = calib_data[det_name].get('gain', 1.0)
                offset = calib_data[det_name].get('offset', 0.0)

            if gain != 1.0 or offset != 0.0:
                emission_axis = sdd_calib.channel_to_energy(np.arange(256), gain, offset)
            else:
                emission_axis = np.arange(256) * 10.0
            ylabel = "Emission Energy (eV)"
        else:
            emission_axis = np.arange(256)
            ylabel = "Emission Channel"

        norm = LogNorm(vmin=max(1.0, np.min(matrix[matrix > 0])), vmax=np.max(matrix)) if log_scale else None

        mesh = ax.pcolormesh(
            energies,
            emission_axis,
            matrix.T,
            shading='auto',
            cmap=cmap,
            norm=norm
        )

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Counts / Intensity' + (' (Log Scale)' if log_scale else ''), fontsize=8)

        ax.set_xlabel('Incident Energy (eV)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f'Detector {det_name.upper()}', fontsize=10, fontweight='semibold')

    # Hide unused subplots if num_dets is odd and > 2
    for unused_idx in range(num_dets, len(axes)):
        axes[unused_idx].axis('off')

    plt.subplots_adjust(bottom=0.12, hspace=0.32, wspace=0.28)

    # Add interactive "Save 2D Map" button on figure layout
    ax_save_btn = fig.add_axes([0.40, 0.02, 0.20, 0.04])
    btn_save = Button(ax_save_btn, 'Save 2D Map', color='yellow', hovercolor='khaki')
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
                save_directory = None
                for d_name, en_dict in data_pack.get('sdd_files', {}).items():
                    for en, fpath in en_dict.items():
                        if fpath and os.path.exists(fpath):
                            parent_dir = os.path.dirname(os.path.abspath(fpath))
                            if any(parent_dir.endswith(suffix) for suffix in ['eV', 'ev', 'EV']):
                                save_directory = os.path.dirname(parent_dir)
                            else:
                                save_directory = parent_dir
                            break
                    if save_directory:
                        break

                if not save_directory:
                    save_directory = os.getcwd()

        os.makedirs(save_directory, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in scan_title)
        ext = file_format.lstrip('.')
        det_suffix = "_".join(target_dets) if num_dets <= 2 else "all_dets"
        initial_name = f"{safe_title}_2D_RIXS_Map_{det_suffix}.{ext}"

        # Hide save button temporarily during image render so button isn't baked into saved file
        ax_save_btn.set_visible(False)
        fig.canvas.draw_idle()

        out_filename = safe_filedialog_call(
            filedialog.asksaveasfilename,
            title="Save 2D Emission Map",
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
            print(f"Saved 2D emission plot to: {out_filename}")
            saved_files.append(out_filename)
        else:
            print("Save operation skipped/cancelled.")

        # Restore button on interactive figure window
        ax_save_btn.set_visible(True)
        fig.canvas.draw_idle()

    btn_save.on_clicked(trigger_save)
    fig._btn_save_ref = btn_save

    plt.show()

    return saved_files if saved_files else None

if __name__ == '__main__':
    plot_incident_vs_emission_2d()
