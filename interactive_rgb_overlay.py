import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ipywidgets as widgets
from IPython.display import display, clear_output
import tkinter as tk
from tkinter import filedialog
from analyze_map import analyze_map
import matplotlib.patches as mpatches

def interactive_rgb_overlay(h5_file=None):
    """
    Creates an interactive IPyWidgets UI to select 3 ROIs on an XRF spectrum
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
    
    # UI Elements
    det_dropdown = widgets.Dropdown(options=detectors, value=detectors[0], description='Detector:')
    
    r_name = widgets.Text(value='Element 1', description='R Name:')
    g_name = widgets.Text(value='Element 2', description='G Name:')
    b_name = widgets.Text(value='Element 3', description='B Name:')
    
    r_roi = widgets.IntRangeSlider(value=[20, 40], min=0, max=pixels_per_spectrum-1, description='Red ROI:')
    g_roi = widgets.IntRangeSlider(value=[50, 70], min=0, max=pixels_per_spectrum-1, description='Green ROI:')
    b_roi = widgets.IntRangeSlider(value=[80, 100], min=0, max=pixels_per_spectrum-1, description='Blue ROI:')
    
    r_contrast = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, description='R Contrast%:')
    g_contrast = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, description='G Contrast%:')
    b_contrast = widgets.FloatRangeSlider(value=[0, 100], min=0, max=100, description='B Contrast%:')
    
    fig_width = widgets.IntSlider(value=14, min=6, max=30, description='Fig Width:')
    fig_height = widgets.IntSlider(value=6, min=4, max=24, description='Fig Height:')
    initial_marker_size = max(0.5, 20000 / max(1, x.size)) if x.size > 0 else 5.0
    marker_size_slider = widgets.FloatSlider(value=initial_marker_size, min=0.1, max=100.0, step=0.1, description='Dot Size:')
    
    plot_output = widgets.Output()

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
            
            fig, (ax_spec, ax_map) = plt.subplots(1, 2, figsize=(fig_width.value, fig_height.value))
            
            # --- Spectrum Plot ---
            ax_spec.plot(total_spectrum, color='black', lw=1)
            ax_spec.set_title(f"Total Spectrum ({det})")
            ax_spec.set_xlabel("Channel")
            ax_spec.set_ylabel("Total Intensity")
            ax_spec.set_yscale('log') # Use log scale to see smaller peaks easily
            
            # Add highlights
            ax_spec.axvspan(r_roi.value[0], r_roi.value[1], color='red', alpha=0.3, label=r_name.value)
            ax_spec.axvspan(g_roi.value[0], g_roi.value[1], color='green', alpha=0.3, label=g_name.value)
            ax_spec.axvspan(b_roi.value[0], b_roi.value[1], color='blue', alpha=0.3, label=b_name.value)
            ax_spec.legend()
            
            # --- Map Plot ---
            # Extract ROIs
            R_inten = np.sum(spectra_2d[:, r_roi.value[0]:r_roi.value[1]], axis=1)
            G_inten = np.sum(spectra_2d[:, g_roi.value[0]:g_roi.value[1]], axis=1)
            B_inten = np.sum(spectra_2d[:, b_roi.value[0]:b_roi.value[1]], axis=1)
            
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
            r_patch = mpatches.Patch(color='red', label=r_name.value)
            g_patch = mpatches.Patch(color='green', label=g_name.value)
            b_patch = mpatches.Patch(color='blue', label=b_name.value)
            ax_map.legend(handles=[r_patch, g_patch, b_patch], loc='upper right', bbox_to_anchor=(1.35, 1))
            
            plt.tight_layout()
            plt.show()

    # Link events - using 'continuous_update=False' if sliders are laggy, but standard observe is fine
    det_dropdown.observe(update_plot, names='value')
    for w in [r_name, g_name, b_name, r_roi, g_roi, b_roi, r_contrast, g_contrast, b_contrast, fig_width, fig_height, marker_size_slider]:
        w.observe(update_plot, names='value')

    accordion = widgets.Accordion(children=[
        widgets.VBox([
            widgets.HBox([r_name, r_roi, r_contrast]),
            widgets.HBox([g_name, g_roi, g_contrast]),
            widgets.HBox([b_name, b_roi, b_contrast])
        ]),
        widgets.VBox([
            widgets.HBox([fig_width, fig_height]),
            marker_size_slider
        ])
    ])
    accordion.set_title(0, 'RGB Overlay Controls')
    accordion.set_title(1, 'Display & Scaling Controls')
    
    ui = widgets.VBox([det_dropdown, accordion])
    display(ui, plot_output)
    
    update_plot()

if __name__ == "__main__":
    interactive_rgb_overlay()
