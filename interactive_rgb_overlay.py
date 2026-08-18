import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ipywidgets as widgets
from IPython.display import display, clear_output
import tkinter as tk
from tkinter import filedialog
import matplotlib.patches as mpatches

# Ensure script directory is in sys.path for robust imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from analyze_map import analyze_map
import sdd_calibration_utils as sdd_calib

def interactive_rgb_overlay(h5_file=None):
    """
    Creates an interactive IPyWidgets UI to select 3 ROIs on an XRF spectrum (in Energy eV)
    and plot an RGB composite map of those regions.
    """
    if h5_file is None:
        root = tk.Tk()
        root.attributes("-alpha", 0.0)
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        h5_file = filedialog.askopenfilename(
            title="Select HDF5 Map File for RGB Overlay",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            parent=root
        )
        root.destroy()
        
        if not h5_file:
            print("No file selected.")
            return

    path_pack = analyze_map(h5_file)
    if not path_pack or not path_pack.get('sdd_files'):
        print("Error: Could not retrieve data paths or no SDD files found.")
        return

    sdd_files = path_pack['sdd_files']
    detectors = sorted(list(sdd_files.keys()))
    if not detectors:
        print("No detectors found.")
        return

    x = path_pack.get('x', np.array([]))
    y = path_pack.get('y', np.array([]))
    
    pixels_per_spectrum = 256
    
    # Load SDD calibration data if available
    calib_data = sdd_calib.load_calibration()

    def get_energy_axis(detector):
        if calib_data and detector in calib_data:
            gain = calib_data[detector].get('gain', 1.0)
            offset = calib_data[detector].get('offset', 0.0)
            if gain != 1.0 or offset != 0.0:
                return sdd_calib.channel_to_energy(np.arange(pixels_per_spectrum), gain, offset)
        return np.arange(pixels_per_spectrum) * 10.0

    def energy_to_channel_bounds(e_min, e_max, detector):
        if calib_data and detector in calib_data:
            gain = calib_data[detector].get('gain', 1.0)
            offset = calib_data[detector].get('offset', 0.0)
            if gain != 1.0 or offset != 0.0:
                ch_min = sdd_calib.energy_to_channel(e_min, gain, offset)
                ch_max = sdd_calib.energy_to_channel(e_max, gain, offset)
                ch_start = int(max(0, np.floor(min(ch_min, ch_max))))
                ch_end = int(min(pixels_per_spectrum, np.ceil(max(ch_min, ch_max))))
                if ch_start >= ch_end and ch_start < pixels_per_spectrum:
                    ch_end = ch_start + 1
                return ch_start, ch_end
        
        # Default uncalibrated scaling: 10 eV per channel
        ch_min = e_min / 10.0
        ch_max = e_max / 10.0
        ch_start = int(max(0, np.floor(min(ch_min, ch_max))))
        ch_end = int(min(pixels_per_spectrum, np.ceil(max(ch_min, ch_max))))
        if ch_start >= ch_end and ch_start < pixels_per_spectrum:
            ch_end = ch_start + 1
        return ch_start, ch_end

    # Determine initial energy axis limits from first detector
    init_det = detectors[0]
    init_energy_axis = get_energy_axis(init_det)
    min_energy = float(np.floor(np.min(init_energy_axis)))
    max_energy = float(np.ceil(np.max(init_energy_axis)))

    # Initial default energy ROI values derived from default channel ranges (20-40, 50-70, 80-100)
    init_r_roi = [round(float(init_energy_axis[20]), 1), round(float(init_energy_axis[40]), 1)]
    init_g_roi = [round(float(init_energy_axis[50]), 1), round(float(init_energy_axis[70]), 1)]
    init_b_roi = [round(float(init_energy_axis[80]), 1), round(float(init_energy_axis[100]), 1)]

    # UI Elements
    det_dropdown = widgets.Dropdown(options=detectors, value=detectors[0], description='Detector:')
    
    btn_save_rgb = widgets.Button(
        description="Save RGB Image (PNG & TIFF)", 
        button_style='success', 
        tooltip='Export current RGB map to Images folder in PNG and TIFF formats',
        layout=widgets.Layout(width='230px')
    )
    save_status = widgets.Label(value="")

    r_name = widgets.Text(value='Element 1', description='R Name:', layout=widgets.Layout(width='180px'))
    g_name = widgets.Text(value='Element 2', description='G Name:', layout=widgets.Layout(width='180px'))
    b_name = widgets.Text(value='Element 3', description='B Name:', layout=widgets.Layout(width='180px'))

    # Numeric FloatText controls for ROI Start & End instead of Sliders
    r_start = widgets.FloatText(value=init_r_roi[0], description='R Start (eV):', layout=widgets.Layout(width='170px'))
    r_end = widgets.FloatText(value=init_r_roi[1], description='R End (eV):', layout=widgets.Layout(width='170px'))
    
    g_start = widgets.FloatText(value=init_g_roi[0], description='G Start (eV):', layout=widgets.Layout(width='170px'))
    g_end = widgets.FloatText(value=init_g_roi[1], description='G End (eV):', layout=widgets.Layout(width='170px'))
    
    b_start = widgets.FloatText(value=init_b_roi[0], description='B Start (eV):', layout=widgets.Layout(width='170px'))
    b_end = widgets.FloatText(value=init_b_roi[1], description='B End (eV):', layout=widgets.Layout(width='170px'))

    r_contrast = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, description='R Contrast%:')
    g_contrast = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, description='G Contrast%:')
    b_contrast = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, description='B Contrast%:')
    
    # Compact figure default sizing for smaller screens
    fig_width = widgets.IntSlider(value=10, min=4, max=24, description='Fig Width:')
    fig_height = widgets.IntSlider(value=4.5, min=3, max=16, description='Fig Height:')
    initial_marker_size = max(0.5, 20000 / max(1, x.size)) if x.size > 0 else 5.0
    marker_size_slider = widgets.FloatSlider(value=initial_marker_size, min=0.1, max=100.0, step=0.1, description='Dot Size:')
    
    plot_output = widgets.Output()
    scroll_plot_container = widgets.VBox([plot_output], layout=widgets.Layout(width='100%', overflow_x='auto', overflow_y='auto'))

    current_fig = [None]

    # Pre-load data cache to keep sliders fast
    data_cache = {}

    def get_data(detector):
        if detector not in data_cache:
            filepath = sdd_files[detector]
            if not os.path.exists(filepath):
                return None
            data_1d = np.fromfile(filepath, dtype=np.uint32)
            num_spectra = len(data_1d) // pixels_per_spectrum
            
            # Align with x, y lengths
            limit = min(num_spectra, x.size)
            clean_size = limit * pixels_per_spectrum
            spectra_2d = data_1d[:clean_size].reshape((limit, pixels_per_spectrum))
            
            total_spectrum = np.sum(spectra_2d, axis=0)
            data_cache[detector] = {
                'spectra_2d': spectra_2d,
                'total_spectrum': total_spectrum,
                'limit': limit
            }
        return data_cache[detector]

    def update_plot(change=None):
        with plot_output:
            clear_output(wait=True)
            
            det = det_dropdown.value
            data = get_data(det)
            
            if data is None:
                print(f"Error loading data for {det}")
                return
                
            spectra_2d = data['spectra_2d']
            total_spectrum = data['total_spectrum']
            limit = data['limit']
            
            curr_x = x[:limit]
            curr_y = y[:limit]
            
            energy_axis = get_energy_axis(det)
            
            fig, (ax_spec, ax_map) = plt.subplots(1, 2, figsize=(fig_width.value, fig_height.value))
            current_fig[0] = fig
            
            # --- Spectrum Plot ---
            ax_spec.plot(energy_axis, np.maximum(total_spectrum, 1e-1), color='black', lw=1)
            ax_spec.set_title(f"Total Spectrum ({det})")
            ax_spec.set_xlabel("Energy (eV)")
            ax_spec.set_ylabel("Total Intensity")
            ax_spec.set_yscale('log') # Use log scale to see smaller peaks easily
            
            # Add highlights (using Energy in eV on X-axis)
            r_val_start, r_val_end = min(r_start.value, r_end.value), max(r_start.value, r_end.value)
            g_val_start, g_val_end = min(g_start.value, g_end.value), max(g_start.value, g_end.value)
            b_val_start, b_val_end = min(b_start.value, b_end.value), max(b_start.value, b_end.value)

            r_lbl = f"{r_name.value} ({r_val_start:.1f}-{r_val_end:.1f} eV)"
            g_lbl = f"{g_name.value} ({g_val_start:.1f}-{g_val_end:.1f} eV)"
            b_lbl = f"{b_name.value} ({b_val_start:.1f}-{b_val_end:.1f} eV)"
            
            ax_spec.axvspan(r_val_start, r_val_end, color='red', alpha=0.3, label=r_lbl)
            ax_spec.axvspan(g_val_start, g_val_end, color='green', alpha=0.3, label=g_lbl)
            ax_spec.axvspan(b_val_start, b_val_end, color='blue', alpha=0.3, label=b_lbl)
            ax_spec.legend(loc='upper right')
            
            # --- Map Plot ---
            # Extract ROIs by converting energy limits to detector channel bounds
            r_ch_start, r_ch_end = energy_to_channel_bounds(r_val_start, r_val_end, det)
            g_ch_start, g_ch_end = energy_to_channel_bounds(g_val_start, g_val_end, det)
            b_ch_start, b_ch_end = energy_to_channel_bounds(b_val_start, b_val_end, det)

            R_inten = np.sum(spectra_2d[:, r_ch_start:r_ch_end], axis=1)
            G_inten = np.sum(spectra_2d[:, g_ch_start:g_ch_end], axis=1)
            B_inten = np.sum(spectra_2d[:, b_ch_start:b_ch_end], axis=1)
            
            # Normalize with contrast
            def normalize(arr, p_range):
                p_low = np.percentile(arr, p_range[0])
                p_high = np.percentile(arr, p_range[1])
                if p_high == p_low: p_high = p_low + 1e-6
                norm = (arr - p_low) / (p_high - p_low)
                return np.clip(norm, 0, 1)
                
            R_norm = normalize(R_inten, r_contrast.value)
            G_norm = normalize(G_inten, g_contrast.value)
            B_norm = normalize(B_inten, b_contrast.value)
            
            rgb_array = np.column_stack((R_norm, G_norm, B_norm))
            
            # Scatter is the only way to plot true RGB colors for non-gridded scattered data
            ax_map.scatter(curr_x, curr_y, c=rgb_array, marker='s', s=marker_size_slider.value, edgecolors='none')
                
            ax_map.set_aspect('equal')
            ax_map.set_title(f"RGB Composite Map - {os.path.basename(h5_file)}")
            ax_map.set_xlabel("Hexapod X")
            ax_map.set_ylabel("Hexapod Y")
            
            # Custom legend for the map
            r_patch = mpatches.Patch(color='red', label=r_lbl)
            g_patch = mpatches.Patch(color='green', label=g_lbl)
            b_patch = mpatches.Patch(color='blue', label=b_lbl)
            ax_map.legend(handles=[r_patch, g_patch, b_patch], loc='upper right', bbox_to_anchor=(1.35, 1))
            
            plt.tight_layout()
            plt.show()

    def on_save_clicked(b):
        if current_fig[0] is None:
            save_status.value = "No figure available to save."
            return
        try:
            h5_dir = os.path.dirname(os.path.abspath(h5_file))
            target_dir = os.path.join(h5_dir, "Images")
            os.makedirs(target_dir, exist_ok=True)
            
            scan_name = path_pack.get('scan_name', 'scan')
            det_name = det_dropdown.value
            base_name = f"{scan_name}_RGB_Overlay_{det_name}"
            
            png_path = os.path.join(target_dir, f"{base_name}.png")
            tif_path = os.path.join(target_dir, f"{base_name}.tiff")
            
            current_fig[0].savefig(png_path, bbox_inches='tight', dpi=300)
            try:
                current_fig[0].savefig(tif_path, bbox_inches='tight', dpi=300)
            except Exception as e:
                print(f"    ! Warning saving TIFF: {e}")
                
            save_status.value = f"Saved: {base_name}.png & .tiff to Images/"
            with plot_output:
                print(f"\n  -> [SAVE SUCCESS] Saved RGB Overlay images to:\n     PNG:  {png_path}\n     TIFF: {tif_path}")
        except Exception as e:
            save_status.value = f"Error: {e}"
            with plot_output:
                print(f"  ! Error saving RGB image: {e}")

    btn_save_rgb.on_click(on_save_clicked)

    # Link events
    det_dropdown.observe(update_plot, names='value')
    for w in [r_name, g_name, b_name, r_start, r_end, g_start, g_end, b_start, b_end, r_contrast, g_contrast, b_contrast, fig_width, fig_height, marker_size_slider]:
        w.observe(update_plot, names='value')

    accordion = widgets.Accordion(children=[
        widgets.VBox([
            widgets.HBox([r_name, r_start, r_end, r_contrast]),
            widgets.HBox([g_name, g_start, g_end, g_contrast]),
            widgets.HBox([b_name, b_start, b_end, b_contrast])
        ]),
        widgets.VBox([
            widgets.HBox([fig_width, fig_height]),
            marker_size_slider
        ])
    ])
    accordion.set_title(0, 'RGB Overlay Controls (Energy ROIs in eV)')
    accordion.set_title(1, 'Display & Scaling Controls')
    
    ui = widgets.VBox([
        widgets.HBox([det_dropdown, btn_save_rgb, save_status]), 
        accordion, 
        scroll_plot_container
    ])
    display(ui)
    
    update_plot()

if __name__ == "__main__":
    interactive_rgb_overlay()

