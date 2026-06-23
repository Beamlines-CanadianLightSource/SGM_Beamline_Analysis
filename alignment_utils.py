import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ipywidgets as widgets
from IPython.display import display
import traceback
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import messagebox, simpledialog
from matplotlib.figure import Figure

# --- Jupyter/macOS Tkinter Stability Patch ---
_orig_showinfo = messagebox.showinfo
_orig_showerror = messagebox.showerror
_orig_showwarning = messagebox.showwarning
_orig_askyesno = messagebox.askyesno
_orig_askokcancel = messagebox.askokcancel

def _in_jupyter():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except:
        return False

def _safe_showinfo(title, message, **kwargs):
    if _in_jupyter():
        print(f"\n[{title}] {message}\n")
        try:
            import sys
            for mod_name in ['plot_sgm_bsky_data', 'plot_sdd_stack']:
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    sync = getattr(mod, '_GLOBAL_SYNC_OBJ', None)
                    if sync and sync.status_widget:
                        sync.status_widget.value = f"[{title}] {message.replace(chr(10), ' ')}"
                        break
        except:
            pass
        return "ok"
    try:
        parent = kwargs.get('parent')
        if parent:
            try: parent.attributes("-topmost", True)
            except: pass
        res = _orig_showinfo(title, message, **kwargs)
        if parent:
            try: parent.update()
            except: pass
        return res
    except Exception as e:
        print(f"[{title}] {message} (Tkinter dialog error: {e})")
        return "ok"

def _safe_showerror(title, message, **kwargs):
    if _in_jupyter():
        print(f"\n[ERROR] [{title}] {message}\n")
        try:
            import sys
            for mod_name in ['plot_sgm_bsky_data', 'plot_sdd_stack']:
                if mod_name in sys.modules:
                    mod = sys.modules[mod_name]
                    sync = getattr(mod, '_GLOBAL_SYNC_OBJ', None)
                    if sync and sync.status_widget:
                        sync.status_widget.value = f"[ERROR] {message.replace(chr(10), ' ')}"
                        break
        except:
            pass
        return "ok"
    try:
        parent = kwargs.get('parent')
        if parent:
            try: parent.attributes("-topmost", True)
            except: pass
        res = _orig_showerror(title, message, **kwargs)
        if parent:
            try: parent.update()
            except: pass
        return res
    except Exception as e:
        print(f"[ERROR] [{title}] {message} (Tkinter dialog error: {e})")
        return "ok"

def _safe_showwarning(title, message, **kwargs):
    if _in_jupyter():
        print(f"\n[WARNING] [{title}] {message}\n")
        return "ok"
    try:
        parent = kwargs.get('parent')
        if parent:
            try: parent.attributes("-topmost", True)
            except: pass
        res = _orig_showwarning(title, message, **kwargs)
        if parent:
            try: parent.update()
            except: pass
        return res
    except Exception as e:
        print(f"[WARNING] [{title}] {message} (Tkinter dialog error: {e})")
        return "ok"

def _safe_askyesno(title, message, **kwargs):
    if _in_jupyter():
        print(f"\n[PROMPT] {message}")
        try:
            val = input("Enter 'y' for Yes, 'n' for No: ").strip().lower()
            return val.startswith('y')
        except Exception as e:
            print(f"Non-interactive environment. Defaulting to Yes. (Error: {e})")
            return True
    try:
        parent = kwargs.get('parent')
        if parent:
            try: parent.attributes("-topmost", True)
            except: pass
        res = _orig_askyesno(title, message, **kwargs)
        if parent:
            try: parent.update()
            except: pass
        return res
    except Exception as e:
        print(f"Tkinter dialog error: {e}. Defaulting to Yes.")
        return True

def _safe_askokcancel(title, message, **kwargs):
    if _in_jupyter():
        print(f"\n[Auto-Confirm] {message} -> Proceeding.")
        return True
    try:
        parent = kwargs.get('parent')
        if parent:
            try: parent.attributes("-topmost", True)
            except: pass
        res = _orig_askokcancel(title, message, **kwargs)
        if parent:
            try: parent.update()
            except: pass
        return res
    except Exception as e:
        return True

# Apply monkeypatching globally to Tkinter modules
import tkinter.messagebox as tk_messagebox
tk_messagebox.showinfo = _safe_showinfo
tk_messagebox.showerror = _safe_showerror
tk_messagebox.showwarning = _safe_showwarning
tk_messagebox.askyesno = _safe_askyesno
tk_messagebox.askokcancel = _safe_askokcancel
messagebox.showinfo = _safe_showinfo
messagebox.showerror = _safe_showerror
messagebox.showwarning = _safe_showwarning
messagebox.askyesno = _safe_askyesno
messagebox.askokcancel = _safe_askokcancel
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import SpanSelector, Button
import matplotlib.tri as tri

def get_masked_triangulation(x, y, max_edge=None):
    """
    Creates a Matplotlib Triangulation object and masks triangles with long edges.
    Helps prevent 'smearing' across missing quadrants in stitched maps.
    """
    try:
        if len(x) < 3:
            return None
            
        triang = tri.Triangulation(x, y)
        
        # Calculate edge lengths
        x_tri = x[triang.triangles]
        y_tri = y[triang.triangles]
        
        e1 = np.sqrt((x_tri[:, 0] - x_tri[:, 1])**2 + (y_tri[:, 0] - y_tri[:, 1])**2)
        e2 = np.sqrt((x_tri[:, 1] - x_tri[:, 2])**2 + (y_tri[:, 1] - y_tri[:, 2])**2)
        e3 = np.sqrt((x_tri[:, 2] - x_tri[:, 0])**2 + (y_tri[:, 2] - y_tri[:, 0])**2)
        
        if max_edge is None:
            # More robust heuristic: use 5x the median edge length
            # This handles both high-res and low-res maps correctly.
            all_edges = np.concatenate([e1, e2, e3])
            max_edge = np.median(all_edges) * 5.0
            
            # Absolute sanity check: don't mask if max_edge is too small relative to total range
            range_max = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
            if max_edge < range_max * 0.01:
                max_edge = range_max * 0.5 # Effectively disable masking if it seems wrong
        
        # Mask triangles where any edge is too long
        mask = (e1 > max_edge) | (e2 > max_edge) | (e3 > max_edge)
        triang.set_mask(mask)
        return triang
    except Exception as e:
        print(f"  [Warning] Masked triangulation failed: {e}")
        return None



_TK_ROOT = None
def get_tk_root():
    global _TK_ROOT
    if _TK_ROOT is None:
        _TK_ROOT = tk.Tk()
        _TK_ROOT.withdraw()
    return _TK_ROOT

def get_safe_save_path(save_dir, default_name):
    """
    Auto-increments the file suffix if the file already exists to prevent Jupyter freezes
    caused by Tkinter simpledialog.
    """
    save_path = os.path.join(save_dir, default_name)
    if not os.path.exists(save_path):
        return save_path
    
    # If in Jupyter, auto-increment filename to prevent blocking/freezing the kernel
    if _in_jupyter():
        base, ext = os.path.splitext(default_name)
        counter = 1
        while True:
            new_name = f"{base}_{counter}{ext}"
            new_path = os.path.join(save_dir, new_name)
            if not os.path.exists(new_path):
                print(f"  [Auto-Increment] '{default_name}' already exists. Auto-saved to: {new_name}")
                return new_path
            counter += 1

    base, ext = os.path.splitext(default_name)
    root = get_tk_root()
    try: root.attributes("-topmost", True)
    except: pass
    while True:
        suffix = simpledialog.askstring("File Exists", 
                                        f"'{default_name}' already exists in the folder.\n\n"
                                        "Please enter a suffix to append (e.g., '_v2', '_new'), or leave blank to overwrite:",
                                        parent=root)
        
        # User clicked Cancel
        if suffix is None:
            return None
            
        # User left blank -> Overwrite
        if suffix.strip() == "":
            return save_path
            
        # Try new name
        new_name = f"{base}{suffix}{ext}"
        new_path = os.path.join(save_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        
        # If new name also exists, loop continues
        default_name = new_name
        base, ext = os.path.splitext(default_name)

def apply_spatial_trim(x, y, data, x_trim=0.0, y_trim=0.0):
    """
    Filters coordinates and data based on a distance (in mm) from the min/max of each axis.
    """
    if x_trim <= 0 and y_trim <= 0:
        return x, y, data
        
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    mask = (x >= x_min + x_trim) & (x <= x_max - x_trim) & \
           (y >= y_min + y_trim) & (y <= y_max - y_trim)
           
    return x[mask], y[mask], data[mask]

def get_sdd_intensity_map(file_path, x_coords, y_coords, channel_roi, roll_shift=0, x_trim=0.0, y_trim=0.0):
    """
    Loads raw SDD binary data, applies roll shift and spatial trim, and returns the 2D intensity map.
    """
    if not os.path.exists(file_path):
        return None, None
        
    try:
        pixels_per_spectrum = 256
        data_1d = np.fromfile(file_path, dtype=np.uint32)
        
        num_spectra = len(data_1d) // pixels_per_spectrum
        
        # Truncate to match coordinates
        num_points = min(num_spectra, x_coords.size)
        intensity = data_1d[:num_points * pixels_per_spectrum].reshape((num_points, pixels_per_spectrum))
        
        # Apply roll shift
        if roll_shift != 0:
            intensity = np.roll(intensity, shift=roll_shift, axis=0)
            
        # Sum ROI
        roi_sum = np.sum(intensity[:, channel_roi[0]:channel_roi[1]], axis=1)
        
        # Apply spatial trim
        trimmed_x, trimmed_y, trimmed_intensity = apply_spatial_trim(
            x_coords[:num_points], y_coords[:num_points], roi_sum, x_trim, y_trim
        )
        
        return trimmed_intensity, (trimmed_x, trimmed_y)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def auto_align_shift(target_img, ref_img, max_shift=50):
    """
    Finds the roll_shift that maximizes correlation between target and reference images.
    Input images are expected to be 1D flattened ROI sums.
    """
    best_shift = 0
    max_corr = -1.0
    
    # We test shifts in the range [-max_shift, max_shift]
    for s in range(-max_shift, max_shift + 1):
        shifted = np.roll(target_img, shift=s)
        corr = np.corrcoef(shifted, ref_img)[0, 1]
        if corr > max_corr:
            max_corr = corr
            best_shift = s
            
    return best_shift, max_corr

def interactive_roll_align(path_pack, channel_roi=(80, 120), max_shift=100, use_color=True):
    """
    Creates an interactive widget to adjust roll_shift and spatial trim in a Jupyter Notebook.
    Handles both 'analyze_map' and 'analyze_stack' data packets.
    """
    if path_pack is None:
        print("Error: path_pack is None. Please check if the analysis function (analyze_stack or analyze_map) succeeded.")
        return

    x = path_pack.get('x', np.array([]))
    y = path_pack.get('y', np.array([]))
    sdd_files = path_pack.get('sdd_files', {})
    
    if not sdd_files:
        print("No SDD files found in path_pack.")
        return

    # Determine if it's a stack or a map
    is_stack = isinstance(next(iter(sdd_files.values())), dict)
    
    detectors = sorted(sdd_files.keys())
    energies = []
    if is_stack:
        # Get energies from the first detector's dict
        energies = sorted(sdd_files[detectors[0]].keys())
    
    # --- UI Components ---
    det_dropdown = widgets.Dropdown(options=detectors, description='Detector:')
    en_dropdown = widgets.Dropdown(options=energies, description='Energy (eV):') if is_stack else None
    ref_en_dropdown = widgets.Dropdown(options=energies, description='Ref Energy:') if is_stack else None
    shift_slider = widgets.IntSlider(value=0, min=-max_shift, max=max_shift, step=1, description='Roll Shift:', layout=widgets.Layout(width='50%'))
    
    # Spatial Trim Sliders - expanded to allow full scan range
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    x_trim_slider = widgets.FloatSlider(value=0.0, min=0.0, max=x_range*0.99, step=x_range/200, description='X-Trim (mm):', layout=widgets.Layout(width='50%'))
    y_trim_slider = widgets.FloatSlider(value=0.0, min=0.0, max=y_range*0.99, step=y_range/200, description='Y-Trim (mm):', layout=widgets.Layout(width='50%'))
    
    color_toggle = widgets.Checkbox(value=use_color, description='Use Color', indent=False)
    
    # Contrast Slider
    contrast_slider = widgets.FloatRangeSlider(
        value=[0, 100], min=0, max=100, step=0.1,
        description='Contrast %:', layout=widgets.Layout(width='80%')
    )
    
    auto_btn = widgets.Button(description="Auto-Align", button_style='info')
    save_btn = widgets.Button(description="Save Current Map", button_style='success')
    output = widgets.Output()

    fig_id = f"align_fig_{id(path_pack)}"
    
    def update_plot(change=None):
        with output:
            det = det_dropdown.value
            shift = shift_slider.value
            xt = x_trim_slider.value
            yt = y_trim_slider.value
            use_color = color_toggle.value
            
            if is_stack:
                en = en_dropdown.value
                f_path = sdd_files[det].get(en)
                title = f"{det} at {en:.2f} eV\nShift: {shift} | Trim: X={xt}, Y={yt}"
            else:
                f_path = sdd_files[det]
                title = f"{det}\nShift: {shift} | Trim: X={xt}, Y={yt}"
                
            intensity, coords = get_sdd_intensity_map(f_path, x, y, channel_roi, roll_shift=shift, x_trim=xt, y_trim=yt)
            
            path_pack['roll_shift'] = shift
            path_pack['x_trim'] = xt
            path_pack['y_trim'] = yt
            
            if intensity is not None:
                # Reuse or create figure
                if not plt.fignum_exists(fig_id):
                    output.clear_output(wait=True)
                    fig, ax = plt.subplots(1, 2, figsize=(9, 4), num=fig_id)
                    update_plot.fig = fig
                    update_plot.ax = ax
                else:
                    fig = update_plot.fig
                    ax = update_plot.ax
                    for a in ax: a.clear()

                # --- Left: 2D Intensity Map ---
                map_ax = ax[0]
                cmap_name = 'viridis' if use_color else 'gray'
                
                # Calculate contrast limits
                p_low, p_high = contrast_slider.value
                vmin = np.percentile(intensity, p_low)
                vmax = np.percentile(intensity, p_high)
                if vmin == vmax: vmax = vmin + 1
                
                try:
                    triang = get_masked_triangulation(coords[0], coords[1])
                    if triang is not None:
                        sc = map_ax.tripcolor(triang, intensity, shading='gouraud', 
                                             edgecolors='none', rasterized=True, cmap=cmap_name,
                                             vmin=vmin, vmax=vmax)
                    else:
                        sc = map_ax.tripcolor(coords[0], coords[1], intensity, shading='gouraud', 
                                             edgecolors='none', rasterized=True, cmap=cmap_name,
                                             vmin=vmin, vmax=vmax)
                    # Clear existing colorbars for this axis
                    if hasattr(update_plot, 'cbar') and update_plot.cbar is not None:
                        try:
                            update_plot.cbar.remove()
                        except:
                            pass
                    update_plot.cbar = plt.colorbar(sc, ax=map_ax, label='Counts')
                except Exception as e:
                    map_ax.text(0.5, 0.5, f"Plot error: {e}", transform=map_ax.transAxes)

                map_ax.set_title(title, fontsize='small')
                map_ax.set_xlabel('X')
                map_ax.set_ylabel('Y')
                map_ax.set_aspect('equal')
                
                map_roi = path_pack.get('map_roi')
                if map_roi is not None:
                    x1, x2 = sorted(map_roi[0:2])
                    y1, y2 = sorted(map_roi[2:4])
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5, ec='r', fc='none', ls='--')
                    map_ax.add_patch(rect)
                
                # --- Right: Count Distribution Histogram ---
                hist_ax = ax[1]
                hist_ax.hist(intensity, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                hist_ax.set_title(f"Intensity Distribution", fontsize='small')
                hist_ax.set_xlabel('Total Counts')
                hist_ax.set_ylabel('Pixel Frequency')
                hist_ax.grid(True, linestyle=':', alpha=0.6)
                
                plt.tight_layout()
                fig.canvas.draw_idle()
            else:
                print("Failed to load map data.")

    def run_auto(b):
        if not is_stack:
            print("Auto-align requires multiple images (Stack file required).")
            return
            
        with output:
            det = det_dropdown.value
            target_en = en_dropdown.value
            ref_en = ref_en_dropdown.value
            xt = x_trim_slider.value
            yt = y_trim_slider.value
            
            if target_en == ref_en:
                print("Target and Reference must be different energies.")
                return
            
            print(f"Finding best shift relative to {ref_en:.2f} eV (using current trim)...")
            
            target_path = sdd_files[det].get(target_en)
            ref_path = sdd_files[det].get(ref_en)
            
            # Load images with 0 shift for correlation, but KEEP the spatial trim
            target_data, _ = get_sdd_intensity_map(target_path, x, y, channel_roi, roll_shift=0, x_trim=xt, y_trim=yt)
            ref_data, _ = get_sdd_intensity_map(ref_path, x, y, channel_roi, roll_shift=0, x_trim=xt, y_trim=yt)
            
            if target_data is not None and ref_data is not None:
                best_s, corr = auto_align_shift(target_data, ref_data, max_shift=max_shift)
                print(f"Found Optimal Shift: {best_s} (Correlation: {corr:.3f})")
                shift_slider.value = best_s # Automatically update the slider
            else:
                print("Error loading data for auto-alignment.")
                
    def run_save(b):
        """Saves current interactive view (clean PNG)."""
        det = det_dropdown.value
        shift = shift_slider.value
        xt = x_trim_slider.value
        yt = y_trim_slider.value
        use_color = color_toggle.value
        
        if is_stack:
            en = en_dropdown.value
            f_path = sdd_files[det].get(en)
            prefix = f"aligned_{det}_{en:.2f}eV"
        else:
            f_path = sdd_files[det]
            prefix = f"aligned_{det}"
            
        intensity, coords = get_sdd_intensity_map(f_path, x, y, channel_roi, roll_shift=shift, x_trim=xt, y_trim=yt)
        
        if intensity is not None:
             # Find save directory
            h5_path = path_pack.get('h5_file_path')
            save_dir = os.path.abspath(os.path.dirname(h5_path)) if h5_path else os.getcwd()
            default_name = f"{prefix}_ROI_{channel_roi[0]}-{channel_roi[1]}.png"
            save_filename = get_safe_save_path(save_dir, default_name)
            
            if not save_filename:
                print("    [CANCEL] Save cancelled by user.")
                return

            # ... Save logic ...
            clean_fig = Figure(figsize=(6, 6))
            canvas = FigureCanvasAgg(clean_fig)
            clean_ax = clean_fig.add_subplot(111)
            triang = get_masked_triangulation(coords[0], coords[1])
            if triang is not None:
                clean_ax.tripcolor(triang, intensity, shading='gouraud', 
                                 edgecolors='none', rasterized=True, cmap='viridis')
            else:
                clean_ax.tripcolor(coords[0], coords[1], intensity, shading='gouraud', 
                                 edgecolors='none', rasterized=True, cmap='viridis')
            clean_ax.set_aspect('equal')
            clean_ax.axis('off')
            clean_fig.savefig(save_filename, bbox_inches='tight', pad_inches=0, transparent=True)
            print(f"    -> [SAVE] Image saved to: {save_filename}")
            
            # Show visible popup
            root_fin = get_tk_root()
            root_fin.attributes("-topmost", True)
            messagebox.showinfo("Save Successful", f"Image saved to:\n{save_filename}", parent=root_fin)
        else:
            with output:
                print("Error: Could not load data for saving.")

    # Observers
    det_dropdown.observe(update_plot, names='value')
    shift_slider.observe(update_plot, names='value')
    x_trim_slider.observe(update_plot, names='value')
    y_trim_slider.observe(update_plot, names='value')
    color_toggle.observe(update_plot, names='value')
    contrast_slider.observe(update_plot, names='value')
    if is_stack:
        en_dropdown.observe(update_plot, names='value')
    auto_btn.on_click(run_auto)
    save_btn.on_click(run_save)

    # Layout
    header = widgets.HBox([det_dropdown, en_dropdown]) if is_stack else widgets.HBox([det_dropdown])
    trim_controls = widgets.VBox([x_trim_slider, y_trim_slider])
    align_controls = widgets.HBox([ref_en_dropdown, auto_btn, save_btn]) if is_stack else widgets.HBox([save_btn])
    
    display(widgets.VBox([header, shift_slider, trim_controls, color_toggle, contrast_slider, align_controls, output]))
    update_plot() # Initial plot

def interactive_channel_selector(path_pack, initial_roi=(20, 40)):
    """
    Opens an interactive spectrum plot to select the channel ROI.
    Returns the selected (start, end) channel tuple.
    """
    if not path_pack:
        print("Error: path_pack is None.")
        return initial_roi

    # 1. Load a representative spectrum
    rep_e = path_pack.get('representative_energy')
    sdd_files = path_pack.get('sdd_files', {})
    if not sdd_files:
        print("Error: No SDD files found in path_pack.")
        return initial_roi
    
    # Pick the first detector available
    det = sorted(sdd_files.keys())[0]
    f_path = sdd_files[det].get(rep_e) if isinstance(sdd_files[det], dict) else sdd_files[det]
    
    if not f_path or not os.path.exists(f_path):
        # Fallback to any energy if representative is missing
        if isinstance(sdd_files[det], dict) and sdd_files[det]:
            rep_e = next(iter(sdd_files[det]))
            f_path = sdd_files[det][rep_e]
        else:
            print(f"Error: Could not find valid data for {det}.")
            return initial_roi

    try:
        # Load and sum across pixels for the chosen energy
        data_1d = np.fromfile(f_path, dtype=np.uint32)
        num_s = min(len(data_1d) // 256, path_pack.get('x', np.array([])).size)
        s2d = data_1d[:num_s * 256].reshape((num_s, 256))
        total_spec = np.sum(s2d, axis=0)
    except Exception as e:
        print(f"Error loading spectrum for ROI selection: {e}")
        return initial_roi

    # 2. UI Setup
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot(np.arange(256), total_spec, color='blue', lw=1.5)
    ax.set_title(f"Channel ROI Selector: {path_pack.get('scan_name')}\n(Drag to select range, click 'Confirm' to finish)", fontsize='medium')
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Total Counts")
    ax.grid(True, linestyle=':', alpha=0.6)

    selected_roi = [initial_roi[0], initial_roi[1]]
    # Initial visual indicator
    region = ax.axvspan(selected_roi[0], selected_roi[1], color='red', alpha=0.15)

    def onselect(xmin, xmax):
        selected_roi[0] = int(max(0, np.floor(xmin)))
        selected_roi[1] = int(min(255, np.ceil(xmax)))
        # Update the visual span manually (Rectangle vertex update)
        verts = region.get_xy()
        verts[0:2, 0] = selected_roi[0]
        verts[2:4, 0] = selected_roi[1]
        verts[4, 0] = selected_roi[0]
        region.set_xy(verts)
        fig.canvas.draw_idle()

    # Keep a reference to SpanSelector so it's not garbage collected
    ax._span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.3, facecolor='red'), interactive=True)

    # 3. Confirm Button
    def confirm(event):
        plt.close(fig)

    ax_btn = fig.add_axes([0.82, 0.015, 0.12, 0.07])
    btn = Button(ax_btn, 'Confirm ROI', color='lightgreen', hovercolor='palegreen')
    btn.on_clicked(confirm)
    ax._btn = btn # Keep reference

    plt.tight_layout()
    plt.show()

    print(f"  [ROI Selector] Final Selection: Channels {selected_roi[0]} to {selected_roi[1]}")
    return (selected_roi[0], selected_roi[1])

def grid_interpolate_map(x, y, z, resolution=200, map_roi=None):
    """
    Interpolates scattered (x, y, z) data onto a regular grid for smooth rendering.
    Returns (grid_x, grid_y, grid_z).
    """
    if map_roi:
        x1, x2 = sorted(map_roi[0:2])
        y1, y2 = sorted(map_roi[2:4])
    else:
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)
        
    xi = np.linspace(x1, x2, resolution)
    yi = np.linspace(y1, y2, resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)
    
    # Grid the data
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
    
    return xi, yi, grid_z
def visualize_stitching_overlap(data_packs):
    """
    Plots the coordinates of multiple data packs on a single axis to visualize overlap.
    """
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, dp in enumerate(data_packs):
        c = colors[i % len(colors)]
        plt.scatter(dp['x'], dp['y'], s=1, color=c, alpha=0.5, label=dp['scan_name'])
        
        # Draw bounding box
        x1, x2 = np.min(dp['x']), np.max(dp['x'])
        y1, y2 = np.min(dp['y']), np.max(dp['y'])
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec=c, fc='none')
        plt.gca().add_patch(rect)
        
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Map Overlap Visualization')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
