import numpy as np
print("\n[VERSION] Dashboard Engine v2.5 (ROI FIXED)\n")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
import os
import csv
import traceback
import sys

# GLOBAL REGISTRY to prevent reference loss in Jupyter
_GLOBAL_SUMMARY_DASH = None
_GLOBAL_SYNC_OBJ = None

import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
from alignment_utils import grid_interpolate_map, get_safe_save_path, get_tk_root, get_masked_triangulation
import sdd_calibration_utils as sdd_calib
import mplcursors
from matplotlib.widgets import RectangleSelector, Button, PolygonSelector, Slider, SpanSelector, CheckButtons
from matplotlib.path import Path
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk
from scipy.signal import savgol_filter
import ipywidgets as widgets
from IPython.display import display
import io
import time

# Global to track double-click timing for backends that don't support event.dblclick
_LAST_CLICK_TIME = 0
_CLICK_THRESHOLD = 0.4 # seconds

try:
    import win32clipboard
    from PIL import Image
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

def _show_clipboard_error(msg):
    """Shows a popup error if clipboard fails, as prints can be missed in notebooks."""
    try:
        from tkinter import messagebox
        import tkinter as tk
        root = get_tk_root(); root.attributes("-topmost", True)
        messagebox.showerror("Clipboard Error", msg)
    except:
        print(f"\n[Clipboard Error] {msg}")

def _copy_fig_to_clipboard(fig, ax=None):
    """Internal helper to copy a figure or specific axis to the Windows clipboard."""
    if not HAS_CLIPBOARD:
        _show_clipboard_error("The 'pywin32' library is missing.\n\nPlease run the following in a terminal or notebook cell to enable copying:\n\npip install pywin32")
        return
    
    try:
        buf = io.BytesIO()
        if ax is not None:
            # Re-draw to ensure renderer is available
            fig.canvas.draw()
            # Calculate the bounding box of the specific subplot
            # Convert display coordinates to figure-inch coordinates
            extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            # Save just that extent
            fig.savefig(buf, format='png', bbox_inches=extent, dpi=150)
        else:
            # Save the entire figure
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        
        buf.seek(0)
        img = Image.open(buf)
        
        # Convert to Device Independent Bitmap (DIB) for Windows clipboard
        output = io.BytesIO()
        img.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:] # Remove BMP file header
        output.close()
        
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        label = "Subplot" if ax else "Full Figure"
        print(f"  [Clipboard] Successfully copied {label} to clipboard.")
    except Exception as e:
        print(f"  [Clipboard] Error during copy: {e}")

def _on_dashboard_click(event):
    """Global double-click handler for all dashboard figures."""
    global _LAST_CLICK_TIME
    curr_time = time.time()
    
    # Check both the standard dblclick flag and our manual timer
    if event.dblclick or (curr_time - _LAST_CLICK_TIME < _CLICK_THRESHOLD):
        # Reset timer to prevent triple-clicks from triggering double-clicks twice
        _LAST_CLICK_TIME = 0
        _copy_fig_to_clipboard(event.canvas.figure, ax=event.inaxes)
    else:
        _LAST_CLICK_TIME = curr_time

# --- CRITICAL STABILITY PATCH ---
# Fixes AttributeError: 'NoneType' object has no attribute 'xdata' 
# occurring in some Matplotlib versions during interactive events in Jupyter.
import matplotlib.widgets as mwidgets
if hasattr(mwidgets, '_SelectorWidget'):
    _orig_get_data = mwidgets._SelectorWidget._get_data
    def _safe_get_data(self, event):
        if event is None: return None, None
        return _orig_get_data(self, event)
    mwidgets._SelectorWidget._get_data = _safe_get_data
# --------------------------------

# Increase the figure limit to avoid warnings in notebooks with many plots
plt.rcParams['figure.max_open_warning'] = 50

# Human-readable names for SDD and MCC channels
SDD_NAMES = {
    1: "sdd1 (90, top)",
    2: "sdd2 (45, front, in-plane)",
    3: "sdd3 (90, side)",
    4: "sdd4 (45, front, out-plane)"
}

MCC_NAMES = {
    1: "mcc1 (I0 from Au Mesh, before KBs)",
    2: "mcc2 (Photodiode)",
    3: "mcc3 (Auxillary)",
    4: "mcc4 (TEY of Sample)"
}

# --- GUI Helpers ---
# Removed local get_safe_save_path - now imported from alignment_utils

class MetadataDialog(simpledialog.Dialog):
    def __init__(self, parent, title, initial_data=None):
        self.initial_data = initial_data or {}
        super().__init__(parent, title)

    def body(self, master):
        fields = [
            ("Sample Name", "Name"),
            ("Sample Formula", "Formula"),
            ("Authors", "Authors"),
            ("Affiliation", "Affiliation"),
            ("Element", "Element"),
            ("Edge", "Edge"),
            ("Preparation Method", "Prep"),
            ("Calibrated To", "Calib"),
            ("Temperature", "Temp"),
            ("Scan Mode", "Mode"),
            ("Chamber Conditions", "Chamber"),
            ("Comments", "Comments")
        ]
        self.entries = {}
        for i, (label_text, key) in enumerate(fields):
            tk.Label(master, text=f"{label_text}:").grid(row=i, column=0, sticky='w', padx=5, pady=2)
            entry = tk.Entry(master, width=40)
            entry.grid(row=i, column=1, padx=5, pady=2)
            if self.initial_data.get(key):
                entry.insert(0, self.initial_data[key])
            self.entries[key] = entry
        return self.entries["Name"]

    def apply(self):
        self.result = {key: (entry.get() or "N/A") for key, entry in self.entries.items()}

class ExternalI0PreviewDialog(tk.Toplevel):
    def __init__(self, parent, dataframe, default_e_col, default_i_col):
        super().__init__(parent)
        self.title("External I0 Selection")
        self.df = dataframe
        self.result = None
        
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.attributes("-topmost", True)
        
        self.i0_calib_enabled = tk.BooleanVar(value=False)
        self.i0_energy_shift = tk.DoubleVar(value=0.0)
        
        # UI Setup
        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        columns = list(self.df.columns)
        
        tk.Label(ctrl_frame, text="Energy Column (X):").grid(row=0, column=0, padx=5, pady=5)
        self.cb_e = ttk.Combobox(ctrl_frame, values=columns, state="readonly", width=30)
        if default_e_col in columns: self.cb_e.set(default_e_col)
        self.cb_e.grid(row=0, column=1, padx=5, pady=5)
        self.cb_e.bind("<<ComboboxSelected>>", self.update_plot)
        
        tk.Label(ctrl_frame, text="Intensity Column (Y):").grid(row=1, column=0, padx=5, pady=5)
        self.cb_i = ttk.Combobox(ctrl_frame, values=columns, state="readonly", width=30)
        if default_i_col in columns: self.cb_i.set(default_i_col)
        self.cb_i.grid(row=1, column=1, padx=5, pady=5)
        self.cb_i.bind("<<ComboboxSelected>>", self.update_plot)
        
        # New Smoothing UI
        smooth_frame = tk.LabelFrame(ctrl_frame, text="Smoothing (Savitzky-Golay)")
        smooth_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.do_smooth = tk.BooleanVar(value=False)
        self.chk_smooth = tk.Checkbutton(smooth_frame, text="Enable Smoothing", variable=self.do_smooth, command=self.update_plot)
        self.chk_smooth.grid(row=0, column=0, padx=5, pady=5)
        
        tk.Label(smooth_frame, text="Window Size (odd):").grid(row=0, column=1, padx=5, pady=5)
        self.spin_window = ttk.Spinbox(smooth_frame, from_=3, to=1001, increment=2, width=5, command=self.update_plot)
        self.spin_window.set(11)
        self.spin_window.grid(row=0, column=2, padx=5, pady=5)
        self.spin_window.bind("<Return>", self.update_plot)
        
        # New Energy Calibration UI (External I0 Only)
        calib_frame = tk.LabelFrame(ctrl_frame, text="I0 Energy Calibration")
        calib_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.chk_calib = tk.Checkbutton(calib_frame, text="Enable Energy Calibration", variable=self.i0_calib_enabled, command=self.update_plot)
        self.chk_calib.grid(row=0, column=0, padx=5, pady=5)
        
        tk.Label(calib_frame, text="Shift (eV):").grid(row=0, column=1, padx=5, pady=5)
        self.ent_calib = ttk.Entry(calib_frame, textvariable=self.i0_energy_shift, width=10)
        self.ent_calib.grid(row=0, column=2, padx=5, pady=5)
        self.ent_calib.bind("<Return>", self.update_plot)
        self.ent_calib.bind("<FocusOut>", self.update_plot)
        
        btn_frame = tk.Frame(ctrl_frame)
        btn_frame.grid(row=0, column=2, rowspan=3, padx=20)
        
        tk.Button(btn_frame, text="Apply", command=self.on_apply, width=15, bg='lightgreen').pack(pady=2)
        tk.Button(btn_frame, text="Cancel", command=self.on_cancel, width=15, bg='lightcoral').pack(pady=2)
        
        # Plot Setup
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.update_plot()
        
    def update_plot(self, event=None):
        e_col = self.cb_e.get()
        i_col = self.cb_i.get()
        
        self.ax.clear()
        
        if e_col and i_col:
            x = self.df[e_col].values
            y = self.df[i_col].values
            
            idx = np.argsort(x)
            x_sorted = x[idx]
            y_sorted = y[idx]
            
            self.x_final = x_sorted
            self.y_final = y_sorted
            
            if self.do_smooth.get():
                try:
                    w = int(self.spin_window.get())
                    if w % 2 == 0: w += 1
                    if w < 3: w = 3
                    
                    if len(y_sorted) >= w:
                        y_smooth = savgol_filter(y_sorted, window_length=w, polyorder=2)
                        self.ax.plot(x_sorted, y_sorted, color='gray', alpha=0.5, label='Raw')
                        self.ax.plot(x_sorted, y_smooth, 'b.-', label=f'Smoothed (w={w})')
                        self.ax.legend()
                        self.y_final = y_smooth
                    else:
                        self.ax.plot(x_sorted, y_sorted, 'b.-')
                except Exception as e:
                    print(f"Smoothing error: {e}")
                    self.ax.plot(x_sorted, y_sorted, 'b.-')
            else:
                self.ax.plot(x_sorted, y_sorted, 'b.-')
                
            self.ax.set_xlabel(e_col)
            self.ax.set_ylabel(i_col)
            self.ax.set_title(f"Preview: {i_col} vs {e_col}")
            self.ax.grid(True)
            
            if self.i0_calib_enabled.get():
                try:
                    shift = float(self.i0_energy_shift.get())
                    self.ax.plot(x_sorted + shift, y_sorted, 'r--', alpha=0.7, label=f'Shifted ({shift:+.2f} eV)')
                    self.ax.legend()
                    self.ax.set_title(f"Preview: {i_col} vs {e_col} (Shift: {shift:+.2f} eV)")
                except: pass
            
        self.fig.tight_layout()
        self.canvas.draw()
        
    def on_apply(self):
        smoothed_str = f" (Smoothed w={self.spin_window.get()})" if self.do_smooth.get() else ""
        shift_str = f" (Energy Shift: {self.i0_energy_shift.get():+.2f} eV)" if self.i0_calib_enabled.get() else ""
        self.result = (self.cb_e.get(), self.cb_i.get(), self.x_final, self.y_final, smoothed_str + shift_str, self.i0_calib_enabled.get(), self.i0_energy_shift.get())
        try: self.grab_release()
        except: pass
        self.withdraw()
        self.update_idletasks()
        try: self.quit()  # Safely exit mainloop
        except: pass
        self.destroy()
        
    def on_cancel(self):
        self.result = None
        try: self.grab_release()
        except: pass
        self.withdraw()
        self.update_idletasks()
        try: self.quit()  # Safely exit mainloop
        except: pass
        self.destroy()

def save_csv_idl(path, rows):
    """Reliable binary write for IDL compatibility (no trailing newline)."""
    try:
        with open(path, 'wb') as f:
            f.write("\r\n".join(rows).encode('ascii'))
        print(f"  -> [SAVE] CSV Exported: {path}")
    except Exception as e:
        print(f"Error saving {path}: {e}")

def get_dynamic_mask(cx, cy, xt, yt, roi=None, poly=None):
    xmin, xmax, ymin, ymax = np.min(cx), np.max(cx), np.min(cy), np.max(cy)
    mask = (cx >= xmin + xt) & (cx <= xmax - xt) & (cy >= ymin + yt) & (cy <= ymax - yt)
    if poly is not None and len(poly) > 2:
        path = Path(poly)
        points = np.column_stack((cx, cy))
        mask &= path.contains_points(points)
    elif roi:
        x1, x2 = sorted(roi[0:2]); y1, y2 = sorted(roi[2:4])
        mask &= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
    return mask

# --- Core Dashboard State & Synchronization ---

class Synchronizer:
    def __init__(self, all_energies, representative_energy, map_roi=None, use_color=True, use_full_metadata=False, calibrated_energies=None):
        self.all_energies = all_energies
        self.calibrated_energies = calibrated_energies if calibrated_energies is not None else all_energies
        self.energy_idx = self._find_nearest_idx(representative_energy)
        self.map_roi = map_roi
        self.use_color = use_color
        self.use_full_metadata = use_full_metadata
        self.dashboards = []
        self.summary_dash = None
        self.is_syncing = False
        self.channel_roi = (0, 255)
        self.contrast_percentiles = (0.0, 100.0)
        self.use_log = False 
        self.use_log_spec = False # New: Log scale for spectra
        self.current_roi = map_roi
        self.current_poly = []
        self.mode = 'rect'
        self.full_metadata = use_full_metadata
        self.status_widget = None
        self.user_metadata = None
        self.i0_calib_enabled = False
        self.i0_energy_shift = 0.0
        self.ipfy_mode = False
        # SDD Energy Calibration
        self.sdd_calib_data = sdd_calib.load_calibration()
        self.use_sdd_calib = False
        self.energy_roi = (1470.0, 1500.0) # Default Energy ROI (Al region)

    def _find_nearest_idx(self, value):
        if value is None or len(self.all_energies) == 0: return 0
        return np.argmin(np.abs(self.all_energies - value))

    def broadcast_energy(self, new_idx):
        if self.is_syncing: return
        self.is_syncing = True
        try:
            self.energy_idx = int(new_idx)
            print(f"  [Energy Select] Switching all plots to {self.calibrated_energies[self.energy_idx]:.2f} eV...")
            for d in self.dashboards:
                try:
                    d.update_energy(self.energy_idx)
                except Exception as e:
                    print(f"    ! Error updating {d.name} energy: {e}")
            print("  [Energy Select] Update complete.")
        finally:
            self.is_syncing = False

    def broadcast_i0_calib(self, enabled=None, shift=None):
        """Updates I0 energy calibration settings and refreshes normalized plots."""
        if enabled is not None: self.i0_calib_enabled = enabled
        if shift is not None: self.i0_energy_shift = shift
        
        # Propagate to path_pack for other modules
        sd = self.summary_dash or _GLOBAL_SUMMARY_DASH
        if sd:
            sd.ctx['path_pack']['i0_calib_enabled'] = self.i0_calib_enabled
            sd.ctx['path_pack']['i0_energy_shift'] = self.i0_energy_shift
            
            # Recalculate ext_i0_values if it's an external I0
            if sd.ctx.get('ext_i0_df') is not None:
                df = sd.ctx['ext_i0_df']
                e_col, i_col = sd.ctx['ext_i0_cols']
                x = df[e_col].values
                y = df[i_col].values
                # Apply smoothing if it was enabled (simplified for now, using stored y_preview if available)
                # Actually, better to just use the x_sorted, y_sorted we stored
                if sd.ctx.get('ext_i0_raw_xy') is not None:
                    x_s, y_s = sd.ctx['ext_i0_raw_xy']
                    x_shifted = x_s + (self.i0_energy_shift if self.i0_calib_enabled else 0.0)
                    sd.ctx['ext_i0_values'] = np.interp(sd.ctx['calibrated_energies'], x_shifted, y_s)
        
        print(f"  [I0 Energy Calib] Enabled: {self.i0_calib_enabled}, Shift: {self.i0_energy_shift}")
        if sd:
            sd.update_plots(self.current_roi, self.current_poly, self.mode)
        
    def broadcast_theme(self):
        self.use_color = not self.use_color
        for d in self.dashboards:
            d.update_theme(self.use_color)

    def broadcast(self, source, roi_or_poly, mode='rect'):
        if self.is_syncing: return
        self.is_syncing = True
        try:
            self.current_roi = roi_or_poly if mode == 'rect' else self.current_roi
            self.current_poly = roi_or_poly if mode == 'poly' else self.current_poly
            self.mode = mode
            
            # Use global registry if local ref is missing
            sd = self.summary_dash or _GLOBAL_SUMMARY_DASH
            
            for d in self.dashboards:
                d.current_roi = self.current_roi
                d.current_poly = self.current_poly
                d.sync_to_mode() # Ensure active selector matches mode

                # Update visual markers on map
                if mode == 'rect':
                    x1, x2, y1, y2 = self.current_roi
                    d.rect_patch.set_xy((x1, y1))
                    d.rect_patch.set_width(x2 - x1)
                    d.rect_patch.set_height(y2 - y1)
                    d.rect_patch.set_visible(True)
                    d.poly_line.set_visible(False)
                    # Sync the interactive selector if it's not the source
                    if d != source and hasattr(d, 'selector_rect'):
                        try: d.selector_rect.extents = (x1, x2, y1, y2)
                        except: pass
                else:
                    if len(self.current_poly) > 0:
                        d.poly_line.set_data([v[0] for v in self.current_poly], [v[1] for v in self.current_poly])
                        d.poly_line.set_visible(True)
                    d.rect_patch.set_visible(False)
                    if d != source and hasattr(d, 'selector_poly'):
                        try: d.selector_poly.verts = self.current_poly
                        except: pass
                
                d.update_spectrum()
            
            if sd: 
                print(f"  [Map Broadcast] Updating summary plots (Mode: {mode})...", flush=True)
                sd.update_plots(self.current_roi, self.current_poly, self.mode)
        except Exception as e:
            print(f"  [Broadcast Error] {e}", flush=True)
            import traceback; traceback.print_exc()
        finally:
            for d in self.dashboards: d.fig.canvas.draw_idle()
            if self.summary_dash: self.summary_dash.fig.canvas.draw_idle()
            self.is_syncing = False

    def broadcast_channel_roi(self, source, new_roi):
        if self.is_syncing: return
        self.is_syncing = True
        try:
            x1, x2 = new_roi
            self.channel_roi = new_roi
            
            # 1. Update the status widget
            if self.status_widget:
                self.status_widget.value = f"Spectral ROI: Ch{x1}-{x2}. (Energy Map updated. Click REFRESH for Average Map & Summary)"
            
            print(f"[DASHBOARD] Broadcasting new ROI: {new_roi}")
            
            # 2. Update the master context
            sd = self.summary_dash or _GLOBAL_SUMMARY_DASH
            if sd: sd.ctx['channel_roi'] = new_roi
            for d in self.dashboards: d.ctx['channel_roi'] = new_roi
            # Sync back to path_pack so exports see the updated ROI
            for d in self.dashboards:
                if 'path_pack' in d.ctx:
                    d.ctx['path_pack']['channel_roi'] = new_roi

            # 3. Synchronize ALL selectors and update map images for the current energy
            for d in self.dashboards:
                # Update the SpanSelector extents
                if d != source and hasattr(d, 'selector_span') and d.selector_span:
                    curr_ext = d.selector_span.extents
                    if abs(curr_ext[0] - x1) > 0.1 or abs(curr_ext[1] - x2) > 0.1:
                        try: d.selector_span.extents = (x1, x2)
                        except: pass
                
                # Re-draw the maps using the new integrated intensities
                try:
                    d.update_energy(self.energy_idx)
                except Exception as e:
                    print(f"    ! Error refreshing map for {d.name}: {e}")

            print(f"  [Channel Sync] Map ROI updated to Ch{x1}-{x2} for all detectors.", flush=True)
        except Exception as e:
            print(f"  [Channel Sync Error] {e}")
            import traceback; traceback.print_exc()
        finally:
            for d in self.dashboards: d.fig.canvas.draw_idle()
            self.is_syncing = False

    def broadcast_energy_roi(self, source, new_roi):
        """Synchronizes ROIs based on physical Energy (eV)."""
        if self.is_syncing: return
        self.is_syncing = True
        try:
            e1, e2 = new_roi
            self.energy_roi = new_roi
            
            if self.status_widget:
                self.status_widget.value = f"Energy ROI: {e1:.1f}-{e2:.1f} eV. (Maps updated. Click REFRESH for Summary)"
            
            print(f"[DASHBOARD] Broadcasting Energy ROI: {new_roi}")
            
            sd = self.summary_dash or _GLOBAL_SUMMARY_DASH
            if sd: sd.ctx['energy_roi'] = new_roi

            for d in self.dashboards:
                if d != source and hasattr(d, 'selector_span') and d.selector_span:
                    try: d.selector_span.extents = (e1, e2)
                    except: pass
                
                # Update map images (each detector will calculate its own channel bounds)
                try:
                    d.update_energy(self.energy_idx)
                except Exception as e:
                    print(f"    ! Error refreshing map for {d.name}: {e}")

        except Exception as e:
            print(f"  [Energy ROI Sync Error] {e}")
        finally:
            for d in self.dashboards: d.fig.canvas.draw_idle()
            self.is_syncing = False

    def broadcast_contrast(self, new_percentiles):
        """Updates contrast limits for all map plots based on percentiles."""
        self.contrast_percentiles = new_percentiles
        for d in self.dashboards:
            d.update_contrast()
        for d in self.dashboards:
            d.fig.canvas.draw_idle()

    def broadcast_log(self, use_log):
        """Toggles log scale for all map plots."""
        self.use_log = use_log
        for d in self.dashboards:
            d.update_contrast() # Contrast update also handles norm update
        for d in self.dashboards:
            d.fig.canvas.draw_idle()

    def broadcast_log_spec(self, use_log):
        """Toggles log scale for all spectral plots."""
        self.use_log_spec = use_log
        for d in self.dashboards:
            d.update_spectrum()
        for d in self.dashboards:
            d.fig.canvas.draw_idle()

    def force_global_refresh(self, btn=None):
        """Forces a full re-calculation using the current master state."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        if self.is_syncing: return
        
        x1, x2 = self.channel_roi
        if getattr(self, '_last_refresh_roi', None) == (x1, x2):
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] [GLOBAL REFRESH] Spectral ROI Ch{x1}-{x2} has not changed. Skipping redundant I/O.", flush=True)
            if self.status_widget:
                self.status_widget.value = f"[{timestamp}] REFRESH SKIPPED (ROI Unchanged)."
            return
            
        self.is_syncing = True 
        self._last_refresh_roi = (x1, x2)
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] [GLOBAL REFRESH] Re-integrating stack for Channels {x1}-{x2}...", flush=True)
            if self.status_widget:
                self.status_widget.value = f"[{timestamp}] RE-INTEGRATING STACK... (Please wait)"
            
            sd = self.summary_dash or _GLOBAL_SUMMARY_DASH
            master_ctx = sd.ctx if sd else self.dashboards[0].ctx
            
            # 1. Re-calculate the 4D stack from binary files
            path_pack = master_ctx['path_pack']
            all_energies = self.all_energies
            det_names = master_ctx['detector_names']
            num_pixels = master_ctx['x_coords'].size
            roll_shift = master_ctx.get('roll_shift', 0)
            
            for det in det_names:
                master_ctx['avg_maps'][det] = None
                
                # Determine channel bounds for this detector
                if self.use_sdd_calib:
                    e_min, e_max = self.energy_roi
                    ch1, ch2 = sdd_calib.get_calibrated_bounds(e_min, e_max, det, self.sdd_calib_data)
                else:
                    ch1, ch2 = self.channel_roi
                
                for e_idx, energy in enumerate(all_energies):
                    p = path_pack['sdd_files'][det].get(energy)
                    if p and os.path.exists(p):
                        try:
                            with open(p, 'rb') as f:
                                d1d = np.fromfile(f, dtype=np.uint32)
                            num_s = min(len(d1d) // 256, num_pixels)
                            s2d = d1d[:num_s * 256].reshape((num_s, 256))
                            if roll_shift != 0: s2d = np.roll(s2d, shift=roll_shift, axis=0)
                            inten = np.sum(s2d[:, ch1:ch2], axis=1)
                            master_ctx['stack_maps'][det][e_idx, :num_s] = inten
                            if master_ctx['avg_maps'][det] is None: master_ctx['avg_maps'][det] = inten.copy()
                            else: master_ctx['avg_maps'][det] += inten
                        except: pass

            # 2. Update all row dashboards with the new context
            for d in self.dashboards:
                d.ctx = master_ctx
                # Re-draw map and spectrum
                d.update_energy(self.energy_idx)
            
            # 3. Update the existing Summary Dashboard (NO RE-CREATION)
            if sd:
                print("  -> Updating Summary plots...", flush=True)
                sd.ctx = master_ctx
                try:
                    sd.update_plots(self.current_roi, self.current_poly, self.mode, autoscale_y=True)
                except Exception as e:
                    print(f"  [Critical Summary Plot Error] {e}", flush=True)
            else:
                print("  [Warning] No Summary Dashboard found to refresh.", flush=True)

            if self.status_widget:
                self.status_widget.value = f"[{timestamp}] REFRESH COMPLETE: Ch{x1}-{x2}."
            print(f"[{timestamp}] [GLOBAL REFRESH] Complete.", flush=True)
        except Exception as e:
            print(f"  [Refresh Error] {e}", flush=True)
            import traceback; traceback.print_exc()
        finally:
            self.is_syncing = False
            for d in self.dashboards: d.fig.canvas.draw_idle()
            if sd: sd.fig.canvas.draw_idle()

class DashboardRow:
    def __init__(self, name, sync_obj, context):
        self.name, self.sync = name, sync_obj
        self.ctx = context # dict containing all data arrays
        self.current_roi = self.sync.current_roi
        self.current_poly = self.sync.current_poly
        self.rep_e = sync_obj.all_energies[sync_obj.energy_idx]
        self.line_spec, self.rect_patch, self.poly_line, self.span_vspan = None, None, None, None
        self.sc_en, self.sc_avg = None, None
        self.s2d_rep = None 
        self.is_mcc = name.startswith('mcc')
        self.selector_rect, self.selector_poly, self.selector_span = None, None, None

    def load_rep_data(self):
        if self.is_mcc:
            # MCC is just a single channel per pixel, stored in mcc_maps [energies, pixels]
            num_pts = self.ctx['x_coords'].size
            self.s2d_rep = self.ctx['mcc_maps'][self.name][self.sync.energy_idx, :num_pts]
            return

        p = self.ctx['path_pack']['sdd_files'][self.name].get(self.rep_e)
        if not p or not os.path.exists(p):
            self.s2d_rep = np.zeros((self.ctx['x_coords'].size, 256), dtype=np.uint32)
            return
        try:
            with open(p, 'rb') as f:
                d1d = np.fromfile(f, dtype=np.uint32)
            num_s = min(len(d1d) // 256, self.ctx['x_coords'].size)
            self.s2d_rep = d1d[:num_s * 256].reshape((num_s, 256))
            if self.ctx['roll_shift'] != 0: 
                self.s2d_rep = np.roll(self.s2d_rep, shift=self.ctx['roll_shift'], axis=0)
        except Exception as e:
            print(f"Error loading data for {self.name} at {self.rep_e} eV: {e}")
            if not hasattr(self, 's2d_rep') or self.s2d_rep is None:
                self.s2d_rep = np.zeros((self.ctx['x_coords'].size, 256), dtype=np.uint32)

    def update_contrast(self):
        """Re-calculates color limits and normalization (linear/log)."""
        if not self.sc_en: return
        from matplotlib.colors import LogNorm, Normalize
        
        # Energy Map
        data = self.sc_en.get_array()
        if data is not None and len(data) > 0:
            vmin = np.nanpercentile(data, self.sync.contrast_percentiles[0])
            vmax = np.nanpercentile(data, self.sync.contrast_percentiles[1])
            if vmin == vmax: vmax = vmin + 1.0
            
            if self.sync.use_log:
                # Ensure vmin is positive for log scale
                vmin = max(vmin, np.nanmin(data[data > 0]) if np.any(data > 0) else 1.0)
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            self.sc_en.set_norm(norm)
            
        # Average Map
        if self.sc_avg:
            data_avg = self.sc_avg.get_array()
            if data_avg is not None and len(data_avg) > 0:
                vmin = np.nanpercentile(data_avg, self.sync.contrast_percentiles[0])
                vmax = np.nanpercentile(data_avg, self.sync.contrast_percentiles[1])
                if vmin == vmax: vmax = vmin + 1.0
                
                if self.sync.use_log:
                    vmin = max(vmin, np.nanmin(data_avg[data_avg > 0]) if np.any(data_avg > 0) else 1.0)
                    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
                else:
                    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
                self.sc_avg.set_norm(norm)

    def update_energy(self, new_idx):
        self.rep_e = self.sync.all_energies[new_idx]
        self.load_rep_data()
        if self.is_mcc:
            m_rep = self.s2d_rep # In MCC mode, s2d_rep is already the map for this energy
        else:
            if self.sync.use_sdd_calib:
                e1, e2 = self.sync.energy_roi
                ch1, ch2 = sdd_calib.get_calibrated_bounds(e1, e2, self.name, self.sync.sdd_calib_data)
            else:
                ch1, ch2 = self.ctx['channel_roi']
            m_rep = np.sum(self.s2d_rep[:, ch1:ch2+1], axis=1)
        tm = get_dynamic_mask(self.ctx['x_coords'][:len(m_rep)], self.ctx['y_coords'][:len(m_rep)], self.ctx['x_trim'], self.ctx['y_trim'])
        if self.sc_en:
            try:
                data = m_rep[tm]
                if len(data) == self.sc_en.get_array().size:
                    self.sc_en.set_array(data)
                    # Apply contrast scaling
                    vmin = np.nanpercentile(data, self.sync.contrast_percentiles[0])
                    vmax = np.nanpercentile(data, self.sync.contrast_percentiles[1])
                    if vmin == vmax: vmax = vmin + 1.0
                    self.sc_en.set_clim(vmin, vmax)
                
                det_id = int(self.name.replace('sdd','')) if 'sdd' in self.name else None
                title_name = (SDD_NAMES.get(det_id, self.name) if det_id else 
                             (MCC_NAMES.get(int(self.name.replace('mcc','')), self.name) if self.is_mcc else self.name))
                self.ax[0].set_title(f"{title_name} @ {self.rep_e:.2f} eV", fontsize='small')
            except Exception as e:
                print(f"    ! Error updating map for {self.name}: {e}")
        
        # Also update Average Map visually if it exists
        if self.sc_avg and not self.is_mcc:
            try:
                # Calculate average by dividing sum by number of energies
                num_en = len(self.sync.all_energies)
                avg_data = self.ctx['avg_maps'][self.name][tm] / num_en
                if len(avg_data) == self.sc_avg.get_array().size:
                    self.sc_avg.set_array(avg_data)
                    # Apply contrast scaling
                    vmin = np.nanpercentile(avg_data, self.sync.contrast_percentiles[0])
                    vmax = np.nanpercentile(avg_data, self.sync.contrast_percentiles[1])
                    if vmin == vmax: vmax = vmin + 1.0
                    self.sc_avg.set_clim(vmin, vmax)
            except: pass

        self.update_spectrum()
        # Ensure contrast/log is applied to new data
        self.update_contrast()
        self.fig.canvas.draw_idle()

    def update_theme(self, use_color):
        cmap = 'viridis' if use_color else 'gray'
        if self.sc_en: self.sc_en.set_cmap(cmap)
        if self.sc_avg: self.sc_avg.set_cmap(cmap)
        self.fig.canvas.draw_idle()

    def force_roi_sync(self, event):
        print(f"\n[MANUAL SYNC] Forcing ROI broadcast from {self.name}...", flush=True)
        self.sync.broadcast(self, self.current_roi, mode=self.sync.mode)
        if self.sync.status_widget:
            self.sync.status_widget.value = f"Manual Sync Complete: Synced from {self.name}."

    def on_rect_select(self, e1, e2):
        if e1 is None or e2 is None or e1.xdata is None or e2.xdata is None: return
        print(f"  [DEBUG] on_rect_select triggered on {self.name}!", flush=True)
        if e1.xdata == e2.xdata and e1.ydata == e2.ydata: return
        
        x1, x2 = sorted([e1.xdata, e2.xdata])
        y1, y2 = sorted([e1.ydata, e2.ydata])
        ext = (x1, x2, y1, y2)
        
        self.current_roi = list(ext)
        # Update local visual patch immediately
        self.rect_patch.set_xy((x1, y1))
        self.rect_patch.set_width(x2 - x1)
        self.rect_patch.set_height(y2 - y1)
        self.rect_patch.set_visible(True)
        
        self.update_spectrum()
        self.sync.broadcast(self, ext, mode='rect')

    def on_poly_select(self, verts):
        if not verts or len(verts) < 3: return
        self.current_poly = verts
        self.update_spectrum()
        self.sync.broadcast(self, verts, mode='poly')

    def on_span_select(self, xmin, xmax):
        if xmin is None or xmax is None: return
        if xmin > xmax: xmin, xmax = xmax, xmin
        
        if self.sync.use_sdd_calib:
            # ROI is in Energy (eV)
            new_roi = (xmin, xmax)
            self.sync.broadcast_energy_roi(self, new_roi)
        else:
            # ROI is in Channels
            x1 = int(np.clip(np.floor(xmin), 0, 255))
            x2 = int(np.clip(np.ceil(xmax), 0, 256))
            if x1 == x2:
                if x1 < 256: x2 = x1 + 1
                else: x1 = x1 - 1
            x1, x2 = sorted([x1, x2])
            new_roi = (x1, x2)
            self.sync.broadcast_channel_roi(self, new_roi)

    def sync_to_mode(self):
        if self.sync.mode == 'rect':
            self.selector_rect.set_active(True)
            self.selector_rect.set_visible(True)
            self.rect_patch.set_visible(True)
            self.selector_poly.set_active(False)
            self.selector_poly.set_visible(False)
            self.poly_line.set_visible(False)
        else:
            self.selector_rect.set_active(False)
            self.selector_rect.set_visible(False)
            self.rect_patch.set_visible(False)
            self.selector_poly.set_active(True)
            self.selector_poly.set_visible(True)
            # Only show poly_line if we have vertices
            if len(self.current_poly) > 0:
                self.poly_line.set_visible(True)

    def update_spectrum(self):
        m = get_dynamic_mask(self.ctx['x_coords'][:len(self.s2d_rep)], self.ctx['y_coords'][:len(self.s2d_rep)], self.ctx['x_trim'], self.ctx['y_trim'], 
                             roi=self.current_roi, poly=self.current_poly if self.sync.mode=='poly' else None)
        if self.is_mcc:
            if self.line_spec: self.line_spec.set_ydata(np.zeros(256))
            return

        if self.sync.use_sdd_calib and self.name in self.sync.sdd_calib_data:
            gain = self.sync.sdd_calib_data[self.name].get('gain', 1.0)
            offset = self.sync.sdd_calib_data[self.name].get('offset', 0.0)
            x_axis = sdd_calib.channel_to_energy(np.arange(256), gain, offset)
            self.ax[2].set_xlabel("Energy (eV)")
        else:
            x_axis = np.arange(256)
            self.ax[2].set_xlabel("Channel")

        if np.any(m):
            data = np.sum(self.s2d_rep[m], axis=0)
            data = np.nan_to_num(data)
            if self.line_spec: 
                self.line_spec.set_data(x_axis, data)
        else:
            if self.line_spec: 
                self.line_spec.set_data(x_axis, np.zeros(256))
        
        # Completely manual scaling to prevent X-axis expansion
        self.ax[2].relim()
        data = self.line_spec.get_ydata()
        ymax = np.nanmax(data) if len(data) > 0 else 0
        
        if self.sync.use_log_spec:
            self.ax[2].set_yscale('log')
            ymin = np.nanmin(data[data > 0]) if np.any(data > 0) else 1.0
            self.ax[2].set_ylim(ymin * 0.5, ymax * 1.5 if ymax > 0 else 10.0)
        else:
            self.ax[2].set_yscale('linear')
            self.ax[2].set_ylim(0, ymax * 1.1 if ymax > 0 else 1.0)
            
        if self.sync.use_sdd_calib and self.name in self.sync.sdd_calib_data:
            e_min, e_max = np.min(x_axis), np.max(x_axis)
            self.ax[2].set_xlim(e_min, e_max)
        else:
            self.ax[2].set_xlim(-0.5, 255.5)
        self.fig.canvas.draw_idle()

    def export_images(self):
        """Manually save the current Energy Map and the Average Map with current contrast settings."""
        from matplotlib.colors import LogNorm, Normalize
        cmap = 'viridis' if self.sync.use_color else 'gray'
        m_rep = np.sum(self.s2d_rep[:, self.ctx['channel_roi'][0]:self.ctx['channel_roi'][1]], axis=1)
        tm = get_dynamic_mask(self.ctx['x_coords'][:len(m_rep)], self.ctx['y_coords'][:len(m_rep)], self.ctx['x_trim'], self.ctx['y_trim'])
        
        # Calculate global contrast limits for export
        p_low, p_high = self.sync.contrast_percentiles

        # Export Energy Map
        path_en = get_safe_save_path(self.ctx['save_dir'], f"{self.ctx['scan_name']}_{self.name}_{self.rep_e:.2f}eV.png")
        if path_en:
            data_en = m_rep[tm]
            vmin = np.nanpercentile(data_en, p_low) if len(data_en) > 0 else 0
            vmax = np.nanpercentile(data_en, p_high) if len(data_en) > 0 else 1
            if vmin == vmax: vmax = vmin + 1.0

            if self.sync.use_log:
                vmin = max(vmin, np.nanmin(data_en[data_en > 0]) if np.any(data_en > 0) else 1.0)
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

            fig_en = Figure(figsize=(6,6))
            canvas_en = FigureCanvasAgg(fig_en)
            ax_en = fig_en.add_subplot(111)
            ax_en.tripcolor(self.ctx['x_coords'][:len(m_rep)][tm], self.ctx['y_coords'][:len(m_rep)][tm], m_rep[tm], 
                             shading='gouraud', edgecolors='none', rasterized=True, cmap=cmap, norm=norm)
            ax_en.set_aspect('equal'); ax_en.axis('off')
            fig_en.savefig(path_en, bbox_inches='tight', pad_inches=0, transparent=True)
            print(f"    -> [SAVE] Energy Map: {path_en}", flush=True)

        # Export Average Map
        path_avg = get_safe_save_path(self.ctx['save_dir'], f"{self.ctx['scan_name']}_{self.name}_StackAverage.png")
        if path_avg:
            num_en = len(self.sync.all_energies)
            m_avg = self.ctx['avg_maps'][self.name] / num_en
            data_avg = m_avg[tm]
            vmin = np.nanpercentile(data_avg, p_low) if len(data_avg) > 0 else 0
            vmax = np.nanpercentile(data_avg, p_high) if len(data_avg) > 0 else 1
            if vmin == vmax: vmax = vmin + 1.0

            if self.sync.use_log:
                vmin = max(vmin, np.nanmin(data_avg[data_avg > 0]) if np.any(data_avg > 0) else 1.0)
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

            fig_avg = Figure(figsize=(6,6))
            canvas_avg = FigureCanvasAgg(fig_avg)
            ax_avg = fig_avg.add_subplot(111)
            ax_avg.tripcolor(self.ctx['x_coords'][:len(m_avg)][tm], self.ctx['y_coords'][:len(m_avg)][tm], m_avg[tm], 
                             shading='gouraud', edgecolors='none', rasterized=True, cmap=cmap, norm=norm)
            ax_avg.set_aspect('equal'); ax_avg.axis('off')
            fig_avg.savefig(path_avg, bbox_inches='tight', pad_inches=0, transparent=True)
            print(f"    -> [SAVE] Average Map: {path_avg}", flush=True)

    def plot(self):
        # Use a unique but persistent figure ID for each detector
        fig_id = f"dash_{self.name}_{id(self.sync)}"
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15.0, 4.8), num=fig_id)
        cmap = 'viridis' if self.sync.use_color else 'gray'
        
        if self.is_mcc:
            m_rep = self.s2d_rep
        else:
            m_rep = np.sum(self.s2d_rep[:, self.ctx['channel_roi'][0]:self.ctx['channel_roi'][1]], axis=1)
            
        tm = get_dynamic_mask(self.ctx['x_coords'][:len(m_rep)], self.ctx['y_coords'][:len(m_rep)], self.ctx['x_trim'], self.ctx['y_trim'])
        tri_en = get_masked_triangulation(self.ctx['x_coords'][:len(m_rep)][tm], self.ctx['y_coords'][:len(m_rep)][tm])
        if tri_en is not None:
            self.sc_en = self.ax[0].tripcolor(tri_en, m_rep[tm], 
                                        shading='gouraud', edgecolors='none', rasterized=True, cmap=cmap)
        else:
            self.sc_en = self.ax[0].tripcolor(self.ctx['x_coords'][:len(m_rep)][tm], self.ctx['y_coords'][:len(m_rep)][tm], m_rep[tm],
                                        shading='gouraud', edgecolors='none', rasterized=True, cmap=cmap)
        plt.colorbar(self.sc_en, ax=self.ax[0]); self.ax[0].set_aspect('equal')
        
        # Explicitly set limits to the data range to avoid expansion from patches
        if np.any(tm):
            x_data = self.ctx['x_coords'][:len(m_rep)][tm]
            y_data = self.ctx['y_coords'][:len(m_rep)][tm]
            self.ax[0].set_xlim(np.min(x_data), np.max(x_data))
            self.ax[0].set_ylim(np.min(y_data), np.max(y_data))
            
        det_id = int(self.name.replace('sdd','')) if 'sdd' in self.name else None
        title_name = (SDD_NAMES.get(det_id, self.name) if det_id else 
                     (MCC_NAMES.get(int(self.name.replace('mcc','')), self.name) if self.is_mcc else self.name))
        self.ax[0].set_title(f"{title_name} @ {self.rep_e:.2f} eV", fontsize='small')
        
        m_avg = self.ctx['avg_maps'][self.name] / len(self.sync.all_energies)
        tri_avg = get_masked_triangulation(self.ctx['x_coords'][:len(m_avg)][tm], self.ctx['y_coords'][:len(m_avg)][tm])
        if tri_avg is not None:
            self.sc_avg = self.ax[1].tripcolor(tri_avg, m_avg[tm],
                                         shading='gouraud', edgecolors='none', rasterized=True, cmap=cmap)
        else:
            self.sc_avg = self.ax[1].tripcolor(self.ctx['x_coords'][:len(m_avg)][tm], self.ctx['y_coords'][:len(m_avg)][tm], m_avg[tm],
                                         shading='gouraud', edgecolors='none', rasterized=True, cmap=cmap)
        plt.colorbar(self.sc_avg, ax=self.ax[1]); self.ax[1].set_aspect('equal')
        
        if np.any(tm):
            x_data = self.ctx['x_coords'][:len(m_avg)][tm]
            y_data = self.ctx['y_coords'][:len(m_avg)][tm]
            self.ax[1].set_xlim(np.min(x_data), np.max(x_data))
            self.ax[1].set_ylim(np.min(y_data), np.max(y_data))
            
        self.ax[1].set_title(f"Stack Average", fontsize='small')
        
        p_rect = plt.Rectangle((self.current_roi[0], self.current_roi[2]), self.current_roi[1]-self.current_roi[0], 
                             self.current_roi[3]-self.current_roi[2], lw=1.5, ec='r', fc='none', ls='--', zorder=100)
        self.ax[0].add_patch(p_rect); self.rect_patch = p_rect
        p_poly, = self.ax[0].plot([], [], 'r--', lw=1.5, zorder=101, visible=False)
        self.poly_line = p_poly
        
        # Initializing selectors with transparent props to avoid double-rendering on top of our patches
        self.selector_rect = RectangleSelector(self.ax[0], self.on_rect_select, props=dict(fc='none', ec='none'), interactive=True, useblit=False)
        # props for PolygonSelector markers (the circles) to be red
        self.selector_poly = PolygonSelector(self.ax[0], self.on_poly_select, useblit=False, 
                                            props=dict(color='red', markeredgecolor='red', markerfacecolor='red', alpha=0.5))
        self.sync_to_mode() 
        
        ax_sync_roi = self.fig.add_axes([0.42, 0.88, 0.15, 0.05])
        self.btn_sync_roi = Button(ax_sync_roi, 'Sync Map ROI', color='lightgreen', hovercolor='lime')
        self.btn_sync_roi.on_clicked(self.force_roi_sync); self.btn_sync_roi.label.set_fontsize(7)

        if self.is_mcc:
            # Hide the 3rd axis for MCC maps as it is not needed
            self.ax[2].set_visible(False)
            # Use identical layout to SDD maps so MCC maps perfectly match their size/position
            self.fig.subplots_adjust(left=0.06, right=0.97, bottom=0.2, top=0.85, wspace=0.18)
            self.fig.canvas.mpl_connect('button_press_event', _on_dashboard_click)
            return

        if len(self.sync.all_energies) <= 1:
            # For single energy, we still want the XRF spectrum to see the peaks
            # but we use a 1x3 grid or similar? The figure was created as 1x3.
            # Let's just keep the 3rd axis visible and update it.
            pass
        else:
            # Multi-energy stack logic
            pass

        m_roi = get_dynamic_mask(self.ctx['x_coords'][:len(self.s2d_rep)], self.ctx['y_coords'][:len(self.s2d_rep)], self.ctx['x_trim'], self.ctx['y_trim'], 
                                roi=self.current_roi, poly=self.current_poly if self.sync.mode=='poly' else None)
        if self.sync.use_sdd_calib and self.name in self.sync.sdd_calib_data:
            gain = self.sync.sdd_calib_data[self.name].get('gain', 1.0)
            offset = self.sync.sdd_calib_data[self.name].get('offset', 0.0)
            x_axis = sdd_calib.channel_to_energy(np.arange(256), gain, offset)
            self.ax[2].set_xlabel("Energy (eV)")
        else:
            x_axis = np.arange(256)
            self.ax[2].set_xlabel("Channel")

        # Calculate initial spectrum for the ROI
        if np.any(m_roi):
            s_sum = np.sum(self.s2d_rep[m_roi], axis=0)
        else:
            s_sum = np.zeros(256)

        self.line_spec, = self.ax[2].plot(x_axis, s_sum, color='blue', lw=1)
        
        self.ax[2].set_title("Spectrum (ROI Sum)\nDrag to set ROI", fontsize='x-small')
        
        self.selector_span = SpanSelector(self.ax[2], self.on_span_select, 'horizontal', useblit=False,
                                         props=dict(alpha=0.3, facecolor='red'), interactive=True, drag_from_anywhere=True)
        
        if self.sync.use_sdd_calib:
            self.selector_span.extents = self.sync.energy_roi
            self.ax[2].set_xlim(np.min(x_axis), np.max(x_axis))
        else:
            self.selector_span.extents = (self.ctx['channel_roi'][0], self.ctx['channel_roi'][1])
            self.ax[2].set_xlim(-10, 266)
        
        self.ax[2].set_xmargin(0)
        self.ax[2].set_autoscalex_on(False)
        self.ax[2].set_ylim(bottom=0)
        
        # Tighten margins and spacing to ensure the 3rd plot is pulled inward
        self.fig.subplots_adjust(left=0.06, right=0.97, bottom=0.2, top=0.85, wspace=0.18)
        mplcursors.cursor(self.sc_en, hover=True); mplcursors.cursor(self.line_spec, hover=True)
        
        # Enable double-click to copy to clipboard
        self.fig.canvas.mpl_connect('button_press_event', _on_dashboard_click)
        # plt.show() removed to prevent double-plotting in Jupyter notebooks

class SummaryDashboard:
    def __init__(self, sync_obj, context):
        self.sync, self.ctx = sync_obj, context
        self.fig, self.ax = None, []
        self.btn_theme = None
        self.det_lines_raw = {}
        self.det_lines_norm = {}
        self.avg_line_raw = None
        self.avg_line_norm = None
        self.btn_toggle = None
        self.slider_energy = None
        self.mcc_lines_raw = {}
        self.mcc_lines_norm = {}
        self.current_roi = sync_obj.current_roi
        self.current_poly = sync_obj.current_poly
        self.mode = sync_obj.mode

    def on_slider_change(self, val):
        new_idx = int(val)
        if new_idx != self.sync.energy_idx:
            self.sync.broadcast_energy(new_idx)

    def toggle_theme(self, event):
        self.sync.broadcast_theme()
        self.btn_theme.label.set_text("Switch to Gray" if self.sync.use_color else "Switch to Color")
        self.fig.canvas.draw_idle()

    def toggle_selection_mode(self, event):
        new_mode = 'poly' if self.sync.mode == 'rect' else 'rect'
        source_dash = self.sync.dashboards[0] if self.sync.dashboards else None
        self.sync.broadcast(source_dash, self.sync.current_poly if new_mode == 'poly' else self.sync.current_roi, mode=new_mode)
        self.btn_toggle.label.set_text("Switch Polygon ROI Selection" if new_mode == 'rect' else "Switch Rectangle ROI Selection")
        self.fig.canvas.draw_idle()

    def update_plots(self, roi, poly, mode, autoscale_y=True):
        # print(f"\n[DEBUG] update_plots entered! Mode={mode}, AutoScale={autoscale_y}", flush=True)
        self.current_roi, self.current_poly, self.mode = roi, poly, mode
        mask = get_dynamic_mask(self.ctx['x_coords'], self.ctx['y_coords'], self.ctx['x_trim'], self.ctx['y_trim'], roi=roi, poly=poly if mode=='poly' else None)
        # Ensure mask is flattened to match the pixel dimension of stack_maps
        mask_flat = mask.flatten()
        num_pixels_in_mask = np.sum(mask_flat)
        # print(f"  -> SUMMARY REFRESH: {num_pixels_in_mask} pixels in mask. Mode: {mode}", flush=True)
        
        raw_summaries = {}
        # 1. Update Raw SDD Lines (Sum)
        try:
            for det, line in self.det_lines_raw.items():
                if line is None: continue
                stack_data = self.ctx['stack_maps'][det] # (energy, num_pixels)
                
                # Slice only the pixels in the mask and sum across pixels
                m_slice = mask_flat[:stack_data.shape[1]]
                if np.any(m_slice):
                    data = np.sum(stack_data[:, m_slice], axis=1)
                else:
                    data = np.zeros(stack_data.shape[0])
                
                data = np.nan_to_num(data)
                line.set_ydata(data)
                raw_summaries[det] = data
        except Exception as e:
            print(f"  [Summary] Error in Raw Lines: {e}")
        
        # Calculate new average from fresh sums
        if raw_summaries:
            raw_avg = np.nanmean(list(raw_summaries.values()), axis=0)
            raw_avg = np.nan_to_num(raw_avg)
            if self.avg_line_raw:
                self.avg_line_raw.set_ydata(raw_avg)
        
        # 2. Update MCC Lines (Mean)
        mcc_summaries_raw = {}
        mcc1_data = None
        if self.ctx['mcc_channels']:
            for ch in self.ctx['mcc_channels']:
                mcc_key = f'mcc{ch}'
                if mcc_key in self.ctx['mcc_maps']:
                    mcc_array = self.ctx['mcc_maps'][mcc_key]
                    means = []
                    for e_idx in range(mcc_array.shape[0]):
                        m_pixels = mcc_array[e_idx, mask]
                        means.append(np.mean(m_pixels) if np.any(mask) else 0.0)
                    
                    data = np.array(means)
                    mcc_summaries_raw[ch] = data
                    if ch == 1: mcc1_data = data
                    
                    line = self.mcc_lines_raw.get(ch)
                    if line: line.set_ydata(data)

        # 3. Update Normalized SDD Lines (Raw / MCC1 / Ext I0)
        active_i0 = None
        if self.ctx.get('ext_i0_values') is not None:
            active_i0 = self.ctx['ext_i0_values'].copy()
        elif mcc1_data is not None:
            active_i0 = mcc1_data.copy()

        if active_i0 is not None:
            i0_safe = np.where(active_i0 == 0, 1.0, active_i0)
            norm_summaries = []
            try:
                for det, line in self.det_lines_norm.items():
                    if line is None: continue
                    raw_data = raw_summaries.get(det, np.zeros_like(i0_safe))
                    norm_data = raw_data / i0_safe
                    if getattr(self.sync, 'ipfy_mode', False):
                        # Invert and shift to make positive with a 500 unit baseline
                        norm_data = -norm_data
                        offset = np.abs(np.min(norm_data)) + 500
                        norm_data += offset
                    norm_data = np.nan_to_num(norm_data)
                    line.set_ydata(norm_data)
                    norm_summaries.append(norm_data)
            except Exception as e:
                print(f"  [Summary] Error in Norm Lines: {e}")
            
            try:
                if self.avg_line_norm and norm_summaries:
                    avg_norm = np.nanmean(norm_summaries, axis=0)
                    avg_norm = np.nan_to_num(avg_norm)
                    self.avg_line_norm.set_ydata(avg_norm)
            except Exception as e:
                print(f"  [Summary] Error in Norm Avg: {e}")
            
            try:
                # 4. Update Normalized MCC Lines (MCC / MCC1)
                for ch, line in self.mcc_lines_norm.items():
                    if line:
                        norm_mcc = mcc_summaries_raw[ch] / i0_safe
                        if getattr(self.sync, 'ipfy_mode', False) and ch == 4: # TEY
                             norm_mcc = -norm_mcc
                             offset = np.abs(np.min(norm_mcc)) + 500
                             norm_mcc += offset
                        norm_mcc = np.nan_to_num(norm_mcc)
                        line.set_ydata(norm_mcc)
            except Exception as e:
                print(f"  [Summary] Error in MCC Lines: {e}")
            
            # Update Title to show active ROI for feedback
            roi_ch = self.ctx.get('channel_roi', (0, 255))
            i0_src = self.ctx.get('i0_source', 'mcc1')
            if self.sync.i0_calib_enabled and "Internal" not in i0_src:
                i0_src += f" (Energy Shift: {self.sync.i0_energy_shift:+.2f} eV)"
            title_str = f"Normalized Fluorescence Spectra (Ch{roi_ch[0]}-{roi_ch[1]}): {self.ctx['scan_name']} (by {i0_src})"
            try:
                # Handle both 1D and 2D axis arrays
                if self.ax.ndim == 2 and self.ax.shape == (2, 2): target_ax = self.ax[1, 0]
                elif self.ax.ndim == 2: target_ax = self.ax[2, 0]
                else: target_ax = self.ax[2]
                target_ax.set_title(title_str)
            except Exception as e:
                print(f"  [Title Update Error] {e}")

        # Completely manual scaling to ensure axes SHRINK when moving to low-signal ROIs
        for ax in self.ax.flatten():
            if not autoscale_y: continue
            
            ymax = 0.0
            ymin = 0.0
            has_data = False
            
            for line in ax.get_lines():
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    ymax = max(ymax, np.nanmax(ydata))
                    ymin = min(ymin, np.nanmin(ydata))
                    has_data = True
            
            if has_data:
                pad = abs(ymax) * 0.05
                # For Fluorescence plots, strictly prevent negative bottom scaling
                # For Fluorescence plots, strictly prevent negative bottom scaling UNLESS in IPFY mode
                if "Fluorescence" in (ax.get_title() or ""):
                    if getattr(self.sync, 'ipfy_mode', False):
                         ax.set_ylim(ymin - pad if ymin < 0 else -1.0, ymax + pad if ymax > ymin else 0.0)
                    else:
                         ax.set_ylim(0, ymax + pad if ymax > 0 else 1.0)
                else:
                    ax.set_ylim(ymin, ymax + pad if ymax > 0 else 1.0)
            else:
                ax.set_ylim(0, 1.0)
        
        self.fig.canvas.draw_idle()

    def get_metadata(self):
        if self.sync.full_metadata and self.sync.user_metadata is None:
            root = get_tk_root()
            root.attributes("-topmost", True)
            d = MetadataDialog(root, "Research Metadata Input", initial_data={"Name": self.ctx['scan_name']})
            if d.result:
                self.sync.user_metadata = d.result
        return self.sync.user_metadata or {}

    def save_summary_csv(self, event):
        roi, poly, mode = self.current_roi, self.current_poly, self.mode
        current_summary = {det: [] for det in self.ctx['detector_names']}
        current_mcc = {f'mcc{ch}': [] for ch in (self.ctx['mcc_channels'] or [])}
        mask = get_dynamic_mask(self.ctx['x_coords'], self.ctx['y_coords'], self.ctx['x_trim'], self.ctx['y_trim'], roi=roi, poly=poly if mode=='poly' else None)

        roi_ch = self.ctx.get('channel_roi', (0, 255))
        print(f"  Exporting {mode} ROI summary...")
        print(f"  -> Active Channel ROI: {roi_ch[0]} - {roi_ch[1]}")
        # 1. SDD Summaries
        for det in self.ctx['detector_names']:
            # Re-sum from the LIVE stack_maps (updated during ROI drag)
            current_summary[det] = np.sum(self.ctx['stack_maps'][det][:, mask], axis=1)
        
        # 2. MCC Summaries
        if self.ctx['mcc_channels']:
            for ch in self.ctx['mcc_channels']:
                mcc_key = f'mcc{ch}'
                if mcc_key in self.ctx['mcc_maps']:
                    mcc_array = self.ctx['mcc_maps'][mcc_key]
                    means = []
                    for e_idx in range(mcc_array.shape[0]):
                        means.append(np.mean(mcc_array[e_idx, mask]) if np.any(mask) else 0.0)
                    current_mcc[mcc_key] = means
                else:
                    current_mcc[mcc_key] = [0.0] * len(self.sync.all_energies)
        
        # 3. I0 Normalization Handling
        if self.ctx.get('ext_i0_values') is not None:
            i0_values = self.ctx['ext_i0_values'].copy()
            src = self.ctx.get('i0_source', '')
            i0_source = src if src.startswith("Internal") else f"External: {src}"
        else:
            mcc1_key = 'mcc1'
            if mcc1_key in current_mcc and np.any(current_mcc[mcc1_key]):
                i0_values = np.array(current_mcc[mcc1_key]).copy()
                i0_source = "Internal: mcc1"
            else:
                i0_values = np.ones(len(self.sync.all_energies))
                i0_source = "None (Raw Only)"
        
        # Prepare Normalized Data

        # Prepare Normalized Data
        i0_safe = np.where(i0_values == 0, 1.0, i0_values)
        normalized_summary = {det: current_summary[det] / i0_safe for det in self.ctx['detector_names']}
        norm_avg = np.nanmean([normalized_summary[det] for det in self.ctx['detector_names']], axis=0)
        
        # Prepare Normalized MCC Data
        normalized_mcc = {}
        for mcc_key, data in current_mcc.items():
            normalized_mcc[mcc_key] = np.array(data) / i0_safe

        raw_avg = np.nanmean([current_summary[det] for det in self.ctx['detector_names']], axis=0)

        roi_ch = self.ctx.get('channel_roi', (0, 255))
        roi_ch_str = f"Ch{roi_ch[0]}-{roi_ch[1]}"
        mode_str = "Poly" if mode=='poly' else "Rect"
        default_name = f"{self.ctx['scan_name']}_{mode_str}_{roi_ch_str}_summary.csv"
        save_path = get_safe_save_path(self.ctx['save_dir'], default_name)
        if save_path:
            meta = self.get_metadata()
            rows = []
            if self.sync.full_metadata:
                rows += [
                    f"# Name: {meta.get('Name', 'N/A')}",
                    f"# Formula: {meta.get('Formula', 'N/A')}",
                    f"# Authors: {meta.get('Authors', 'N/A')}",
                    f"# Affiliation: {meta.get('Affiliation', 'N/A')}",
                    f"# Facility: CLS",
                    f"# Beamline: SGM",
                    f"# Mono: Spherical Grating Monochromator",
                    f"# Website: https://sgm.lightsource.ca",
                    f"# Element: {meta.get('Element', 'N/A')}",
                    f"# Edge: {meta.get('Edge', 'N/A')}",
                    f"# Preparation Method: {meta.get('Prep', 'N/A')}",
                    f"# Calibrated To: {meta.get('Calib', 'N/A')}",
                    f"# Temperature: {meta.get('Temp', 'N/A')}",
                    f"# Scan Mode: {meta.get('Mode', 'N/A')}",
                    f"# Chamber Conditions: {meta.get('Chamber', 'N/A')}",
                    f"# Comments: {meta.get('Comments', 'N/A')}",
                    "#"
                ]
            
            roi_ch = self.ctx.get('channel_roi', (0, 255))
            roi_ch_str = f"Ch{roi_ch[0]}-{roi_ch[1]}"
            rows += [
                f"# Scan Name: {self.ctx['scan_name']}",
                f"# Project: {self.ctx['project_name']}",
                f"# Date: {self.ctx['path_pack'].get('date', 'N/A')}",
                f"# Number of Images: {len(self.sync.all_energies)}",
                f"# Energy Regions: {self.ctx['path_pack'].get('Energy Regions', 'N/A')}",
                f"# Grid: {self.ctx['path_pack'].get('nx', 'N/A')} x {self.ctx['path_pack'].get('ny', 'N/A')} (Total: {len(self.ctx['x_coords'])})",
                f"# Grating: {self.ctx['path_pack'].get('grating', 'N/A')}",
                f"# Harmonic: {self.ctx['path_pack'].get('harmonic', 'N/A')}",
                f"# Strip: {self.ctx['path_pack'].get('strip', 'N/A')}",
                f"# Polarization: {self.ctx['path_pack'].get('polarization', 'N/A')}",
                f"# Exit Slit Gap: {self.ctx['path_pack'].get('exit_slit_gap', 'N/A')}",
                f"# XPS Z: {self.ctx['path_pack'].get('xps_z', 'N/A')}",
                f"# Time Per Map: {self.ctx['path_pack'].get('time_per_map', 'N/A')}",
                f"# Number of Points: {self.ctx['path_pack'].get('number_of_points', 'N/A')}",
                f"# ROI Selection: {mode_str}",
                f"# Channels: {roi_ch_str}",
                f"# Normalization: {i0_source}",
                f"# SDD Calibration: {'ACTIVE' if self.sync.use_sdd_calib else 'DISABLED'}",
                "#"
            ]
            
            rows.append("# Column 1: Calibrated Energy (eV)")
            rows.append("# Column 2: Original Energy (eV)")
            c_idx = 3
            # List Raw Columns
            for det in self.ctx['detector_names']:
                label = SDD_NAMES.get(int(det.replace('sdd','')), det) if 'sdd' in det else det
                rows.append(f"# Column {c_idx}: RAW_{label}"); c_idx += 1
            rows.append(f"# Column {c_idx}: RAW_Average_SDD"); c_idx += 1
            
            # List Normalized Columns
            for det in self.ctx['detector_names']:
                label = SDD_NAMES.get(int(det.replace('sdd','')), det) if 'sdd' in det else det
                rows.append(f"# Column {c_idx}: NORM_{label} (by {i0_source})"); c_idx += 1
            rows.append(f"# Column {c_idx}: NORM_Average_SDD"); c_idx += 1
            
            if self.ctx['mcc_channels']:
                for ch in self.ctx['mcc_channels']:
                    label = MCC_NAMES.get(ch, f"mcc{ch}")
                    rows.append(f"# Column {c_idx}: RAW_{label}"); c_idx += 1
                for ch in self.ctx['mcc_channels']:
                    label = MCC_NAMES.get(ch, f"mcc{ch}")
                    rows.append(f"# Column {c_idx}: NORM_{label} (by {i0_source})"); c_idx += 1
            rows.append("#")
            for i, energy in enumerate(self.sync.all_energies):
                row_data = [f"{self.ctx['calibrated_energies'][i]:.2f}", f"{energy:.2f}"]
                # Raw Columns
                for det in self.ctx['detector_names']: row_data.append(f"{current_summary[det][i]:.2f}")
                row_data.append(f"{raw_avg[i]:.2f}")
                # Normalized Columns
                for det in self.ctx['detector_names']: row_data.append(f"{normalized_summary[det][i]:.6f}")
                row_data.append(f"{norm_avg[i]:.6f}")
                # MCC Columns
                if self.ctx['mcc_channels']:
                    for ch in self.ctx['mcc_channels']: row_data.append(f"{current_mcc[f'mcc{ch}'][i]:.6f}")
                    for ch in self.ctx['mcc_channels']: row_data.append(f"{normalized_mcc[f'mcc{ch}'][i]:.6f}")
                rows.append(",".join(row_data))
            save_csv_idl(save_path, rows)
            root_msg = get_tk_root()
            root_msg.attributes("-topmost", True)
            messagebox.showinfo("Save Successful", f"Exported {mode_str} XANES spectra to:\n{save_path}", parent=root_msg)

    def save_consolidated_xrd_spectrum(self, event):
        roi, poly, mode = self.sync.current_roi, self.sync.current_poly, self.sync.mode
        consolidated_specs = {det: np.zeros(256, dtype=np.float64) for det in self.ctx['detector_names']}
        
        print(f"  Integrating Consolidated XRD (Mode: {mode})...")
        for i, energy in enumerate(self.sync.all_energies):
            if (i+1) % 50 == 0 or (i+1) == len(self.sync.all_energies):
                print(f"    Energy step {i+1}/{len(self.sync.all_energies)} ({energy:.2f} eV)...")
            
            for det in self.ctx['detector_names']:
                p = self.ctx['path_pack']['sdd_files'][det].get(energy)
                if not p:
                     for k in self.ctx['path_pack']['sdd_files'][det].keys():
                         if abs(k - energy) < 0.001:
                             p = self.ctx['path_pack']['sdd_files'][det][k]; break
                if not p or not os.path.exists(p): continue
                try:
                    d1d = np.fromfile(p, dtype=np.uint32)
                    num_s = min(len(d1d) // 256, self.ctx['x_coords'].size)
                    s2d = d1d[:num_s * 256].reshape((num_s, 256))
                    if self.ctx['roll_shift'] != 0: s2d = np.roll(s2d, shift=self.ctx['roll_shift'], axis=0)
                    m = get_dynamic_mask(self.ctx['x_coords'][:num_s], self.ctx['y_coords'][:num_s], self.ctx['x_trim'], self.ctx['y_trim'], roi=roi, poly=poly if mode=='poly' else None)
                    if np.any(m): consolidated_specs[det] += np.sum(s2d[m], axis=0)
                except: continue

        roi_ch_str = f"Ch{self.ctx['channel_roi'][0]}-{self.ctx['channel_roi'][1]}"
        mode_str = "Poly" if mode=='poly' else "Rect"
        default_name = f"{self.ctx['scan_name']}_Consolidated_{mode_str}_ROI_{roi_ch_str}_XRF.csv"
        save_path = get_safe_save_path(self.ctx['save_dir'], default_name)
        
        if save_path:
            meta = self.get_metadata()
            rows = []
            if self.sync.full_metadata:
                rows += [
                    f"# Name: {meta.get('Name', 'N/A')}", f"# Formula: {meta.get('Formula', 'N/A')}",
                    f"# Authors: {meta.get('Authors', 'N/A')}", f"# Affiliation: {meta.get('Affiliation', 'N/A')}",
                    f"# Facility: CLS", f"# Beamline: SGM", f"# Element: {meta.get('Element', 'N/A')}", f"# Edge: {meta.get('Edge', 'N/A')}", "#"
                ]

            rows += [
                f"# Scan Name: {self.ctx['scan_name']}", f"# Project: {self.ctx['project_name']}",
                f"# Date: {self.ctx['path_pack'].get('date', 'N/A')}", f"# Number of Images: {len(self.sync.all_energies)}",
                f"# Energy Regions: {self.ctx['path_pack'].get('Energy Regions', 'N/A')}",
                f"# Grid: {self.ctx['path_pack'].get('nx', 'N/A')} x {self.ctx['path_pack'].get('ny', 'N/A')} (Total: {len(self.ctx['x_coords'])})",
                f"# Grating: {self.ctx['path_pack'].get('grating', 'N/A')}", f"# Harmonic: {self.ctx['path_pack'].get('harmonic', 'N/A')}",
                f"# Strip: {self.ctx['path_pack'].get('strip', 'N/A')}", f"# Polarization: {self.ctx['path_pack'].get('polarization', 'N/A')}",
                f"# Exit Slit Gap: {self.ctx['path_pack'].get('exit_slit_gap', 'N/A')}", f"# XPS Z: {self.ctx['path_pack'].get('xps_z', 'N/A')}",
                f"# Time Per Map: {self.ctx['path_pack'].get('time_per_map', 'N/A')}",
                f"# Number of Points: {self.ctx['path_pack'].get('number_of_points', 'N/A')}",
                f"# ROI Selection: {mode_str}", f"# SDD ROI Channels: {roi_ch_str}",
                f"# Image Energy: {self.sync.all_energies[self.sync.energy_idx]:.2f} eV",
                f"# Selection Coordinates: {roi if mode=='rect' else poly}", "#"
            ]
            rows.append("#")
            rows.append("# Column 1: Channel")
            c_idx = 2
            for det in self.ctx['detector_names']:
                rows.append(f"# Column {c_idx}: {det}"); c_idx += 1
            rows.append("#")
            rows.append(",".join(["Channel"] + self.ctx['detector_names']))
            for i in range(256):
                row_data = [str(i)]
                for det in self.ctx['detector_names']: row_data.append(f"{consolidated_specs[det][i]:.2f}")
                rows.append(",".join(row_data))
            save_csv_idl(save_path, rows)
            root_msg = get_tk_root()
            root_msg.attributes("-topmost", True)
            messagebox.showinfo("Save Successful", f"Exported Consolidated {mode_str} ROI XRF to:\n{save_path}", parent=root_msg)

    def save_all_images(self, event):
        total = len(self.sync.dashboards)
        save_dir_abs = os.path.abspath(self.ctx['save_dir'])
        
        # 1. Path Confirmation Popup
        root_init = get_tk_root()
        root_init.attributes("-topmost", True)
        ok = messagebox.askokcancel("Confirm Export", f"Saving {total*2} images to:\n\n{save_dir_abs}\n\nProceed?", parent=root_init)
        if not ok: return

        # 2. Update Status via Widget (Safe for Jupyter)
        if self.sync.status_widget:
            self.sync.status_widget.value = f"EXPORT START: 0/{total} processed..."
        
        plt.ioff() # Temporary disable interactive display
        print(f"\n--- [EXPORT START] Saving High-Quality PNGs for {total} Detectors ---", flush=True)
        print(f"Target Directory: {save_dir_abs}", flush=True)
        
        for i, d in enumerate(self.sync.dashboards):
            idx = i + 1
            percent = (idx / total) * 100
            msg = f"Processing {d.name}... ({idx}/{total}, {percent:.0f}%)"
            if self.sync.status_widget:
                self.sync.status_widget.value = msg
            
            print(f"  {msg}", flush=True)
            try:
                d.export_images()
            except Exception as e:
                print(f"    ! Error exporting {d.name}: {e}", flush=True)
        
        # 3. Final Summary
        if self.sync.status_widget:
            self.sync.status_widget.value = f"EXPORT COMPLETE: {total} detectors saved."
        plt.ion()

        # Safe Tkinter MessageBox
        root_msg = get_tk_root()
        root_msg.attributes("-topmost", True)
        messagebox.showinfo("Export Complete", f"Successfully exported all maps to:\n{save_dir_abs}", parent=root_msg)
        
        if self.sync.status_widget:
            self.sync.status_widget.value = f"EXPORT COMPLETE: All {total} maps saved to {save_dir_abs}"
        print(f"--- [EXPORT COMPLETE] All {total} detectors saved to {save_dir_abs} ---\n", flush=True)

    def save_pymca_stack_call(self, event):
        """Callback for saving ROI-summed PyMca stack (3D)."""
        from save_pymca_stack_h5 import save_pymca_stack_h5
        # Use current state
        roi_ch = self.ctx.get('channel_roi', (0, 255))
        roi_map = self.sync.current_roi
        
        ipfy = getattr(self.sync, 'ipfy_mode', False)
        print(f"  [EXPORT] Launching ROI-summed PyMca Stack Export (3D)... (IPFY Mode: {ipfy})")
        print(f"  -> Active Channel ROI: {roi_ch}")
        
        try:
            # Note: save_pymca_stack_h5 will handle the file dialog
            self.ctx['path_pack']['ipfy_mode'] = ipfy
            save_path = save_pymca_stack_h5(self.ctx['path_pack'], channel_roi=roi_ch, map_roi=roi_map)
            
            # Automatically update the Jupyter global namespace
            if save_path:
                try:
                    from IPython import get_ipython
                    ip = get_ipython()
                    if ip:
                        ip.user_ns['saved_pymca_stack'] = save_path
                        print(f"  [AUTO] Variable 'saved_pymca_stack' updated in notebook.")
                except: pass
        except Exception as e:
            print(f"Error during 3D Stack Export: {e}")
            traceback.print_exc()

    def save_pymca_4d_stack_call(self, event):
        """Callback for saving full 4D PyMca stack."""
        from save_pymca_4d_stack_h5 import save_pymca_4d_stack_h5
        # Use current state
        roi_ch = self.ctx.get('channel_roi', (0, 255))
        
        ipfy = getattr(self.sync, 'ipfy_mode', False)
        print(f"  [EXPORT] Launching Full 4D PyMca Stack Export (4D Cube)... (IPFY Mode: {ipfy})")
        print(f"  -> Active Channel ROI (for 3D preview): {roi_ch}")
        
        try:
            # Note: save_pymca_4d_stack_h5 will handle the file dialog
            self.ctx['path_pack']['ipfy_mode'] = ipfy
            save_path = save_pymca_4d_stack_h5(self.ctx['path_pack'], normalize=True, channel_roi=roi_ch)
            
            # Automatically update the Jupyter global namespace
            if save_path:
                try:
                    from IPython import get_ipython
                    ip = get_ipython()
                    if ip:
                        ip.user_ns['saved_pymca_4d_stack'] = save_path
                        ip.user_ns['saved_pymca_stack'] = save_path # Often used interchangeably for the next step
                        print(f"  [AUTO] Variables 'saved_pymca_4d_stack' and 'saved_pymca_stack' updated in notebook.")
                except: pass
        except Exception as e:
            print(f"Error during 4D Stack Export: {e}")
            traceback.print_exc()

    def plot(self):
        has_mcc_data = (self.ctx['mcc_channels'] and len(self.ctx['mcc_channels']) > 0)
        is_single_energy = (len(self.sync.all_energies) <= 1)
        
        # Determine Grid Layout
        if is_single_energy:
            # Only 1 row: Raw Map(s) if multiple, or just 1 map?
            # Actually, the SummaryDashboard plots spectra. If single energy, spectra are points.
            # User says "we do not need to show the xanes spectra plotting".
            # So we only show maps? But this IS the summary dashboard.
            # If we don't show spectra, this whole figure might be empty or just show 1 point.
            # Let's hide the axes if single energy.
            rows, cols = 1, 1
            # But the SummaryDashboard is designed to show 2x2.
            # Let's just make the axes invisible or return early if single energy?
            # Actually, the user likely wants the WHOLE summary dashboard to be hidden if single energy,
            # but usually they want to see the MAPS. The maps are in DashboardRow.
            # So if single energy, we hide the spectra in SummaryDashboard.
            rows, cols = 1, 1 
        else:
            rows, cols = 2, 2
        fig_id = f"dash_summary_{id(self.sync)}"
        if is_single_energy:
            # We want the UI buttons but no plots. 
            # We can't return early or we lose all button setup.
            # Create a thin empty axis for the plots area or just skip plot creation.
            rows, cols = 1, 1
            # Increased height from 2.0 to 4.0 to give buttons more room
            self.fig, self.ax = plt.subplots(rows, cols, figsize=(12.75, 4.0), squeeze=False, num=fig_id)
            self.ax[0,0].axis('off')
            btn_y = 0.20 # Higher starting Y for buttons in a smaller figure
            btn_h = 0.15 # Taller buttons for the small figure
            chk_y = 0.45
        else:
            # Each plot is ~6.375 wide by ~9.6 tall to satisfy "50% smaller width, height 1.5x width"
            self.fig, self.ax = plt.subplots(rows, cols, figsize=(12.75, 19.2), squeeze=False, num=fig_id)
            self.fig.subplots_adjust(hspace=0.3, wspace=0.2, bottom=0.15, top=0.92, left=0.08, right=0.95)
            btn_y = 0.015
            btn_h = 0.04
            chk_y = 0.06
            
            ax_raw_sdd = self.ax[0, 0]
            ax_raw_mcc = self.ax[0, 1]
            ax_norm_sdd = self.ax[1, 0]
            ax_norm_mcc = self.ax[1, 1]
            
            fmt = 'o-' if self.ctx.get('show_markers', True) else '-'
            
            # 1. Plot Raw Fluorescence (Top)
            for det in self.ctx['detector_names']:
                det_id = int(det.replace('sdd','')) if 'sdd' in det else None
                label = SDD_NAMES.get(det_id, det) if det_id else det
                l, = ax_raw_sdd.plot(self.ctx['calibrated_energies'], self.ctx['summary_data'][det], fmt, label=f"Raw {label}", alpha=0.7)
                self.det_lines_raw[det] = l
            self.avg_line_raw, = ax_raw_sdd.plot(self.ctx['calibrated_energies'], self.ctx['avg_dependence'], 'k--', lw=2, label='Raw Average')
            ax_raw_sdd.set_title(f"Raw Fluorescence Spectra: {self.ctx['scan_name']}"); ax_raw_sdd.legend(fontsize='xx-small')
            ax_raw_sdd.set_xlabel("Energy (eV)"); ax_raw_sdd.set_ylabel("Raw Counts")

            # 2. Plot Raw MCC
            for ch in (self.ctx['mcc_channels'] or []):
                data = self.ctx['mcc_data'][f'mcc{ch}']
                if not np.all(np.array(data) == 0):
                    m_fmt = 's-' if self.ctx.get('show_markers', True) else '-'
                    label = MCC_NAMES.get(ch, f'mcc{ch}')
                    l, = ax_raw_mcc.plot(self.ctx['calibrated_energies'], data, m_fmt, label=f"Raw {label}")
                    self.mcc_lines_raw[ch] = l
            ax_raw_mcc.set_title("Raw I0 and Total Electron Yield (TEY) Spectra"); ax_raw_mcc.legend(fontsize='xx-small')
            ax_raw_mcc.set_xlabel("Energy (eV)"); ax_raw_mcc.set_ylabel("Counts/Intensity")

            # 3. Plot Normalized Fluorescence
            if self.ctx.get('ext_i0_values') is not None:
                i0_init = self.ctx['ext_i0_values']
            else:
                i0_init = np.array(self.ctx['mcc_data'].get('mcc1', np.ones(len(self.ctx['calibrated_energies']))))
            
            i0_init = np.where(i0_init == 0, 1.0, i0_init)
            
            norm_avg_init = []
            for det in self.ctx['detector_names']:
                det_id = int(det.replace('sdd','')) if 'sdd' in det else None
                label = SDD_NAMES.get(det_id, det) if det_id else det
                norm_data = self.ctx['summary_data'][det] / i0_init
                l, = ax_norm_sdd.plot(self.ctx['calibrated_energies'], norm_data, fmt, label=f"Normalized {label}", alpha=0.7)
                self.det_lines_norm[det] = l
                norm_avg_init.append(norm_data)
            
            self.avg_line_norm, = ax_norm_sdd.plot(self.ctx['calibrated_energies'], np.nanmean(norm_avg_init, axis=0), 'k--', lw=2, label='Normalized Average')
            i0_src = self.ctx.get('i0_source', 'mcc1')
            ax_norm_sdd.set_title(f"Normalized Fluorescence Spectra (Ch{self.sync.channel_roi[0]}-{self.sync.channel_roi[1]}): {self.ctx['scan_name']} (by {i0_src})")
            ax_norm_sdd.legend(fontsize='xx-small')
            ax_norm_sdd.set_xlabel("Energy (eV)"); ax_norm_sdd.set_ylabel("Normalized Intensity")

            # 4. Plot Normalized MCC
            for ch in (self.ctx['mcc_channels'] or []):
                data = np.array(self.ctx['mcc_data'][f'mcc{ch}'])
                if not np.all(data == 0):
                    i0_safe = np.where(i0_init <= 0, 1.0, i0_init)
                    norm_mcc = data / i0_safe
                    m_fmt = 's-' if self.ctx.get('show_markers', True) else '-'
                    label = MCC_NAMES.get(ch, f'mcc{ch}')
                    l, = ax_norm_mcc.plot(self.ctx['calibrated_energies'], norm_mcc, m_fmt, label=f"Normalized {label}")
                    self.mcc_lines_norm[ch] = l
            i0_src = self.ctx.get('i0_source', 'mcc1')
            ax_norm_mcc.set_title(f"Normalized I0 and TEY Spectra (by {i0_src})"); ax_norm_mcc.legend(fontsize='xx-small')
            ax_norm_mcc.set_xlabel("Energy (eV)"); ax_norm_mcc.set_ylabel("Normalized Intensity")
        
        # Bottom row buttons - Adjusted positions to avoid overlapping x-axis labels
        ax_chk = self.fig.add_axes([0.02, chk_y, 0.40, 0.03])
        self.chk_meta = CheckButtons(ax_chk, ["Add Sample Specific Information to File Header"], [self.sync.full_metadata])
        for text in self.chk_meta.labels:
            text.set_fontsize(7)
        def toggle_meta(label):
            self.sync.full_metadata = not self.sync.full_metadata
        self.chk_meta.on_clicked(toggle_meta)

        ax_save_xanes = self.fig.add_axes([0.02, btn_y, 0.06, btn_h])
        self.btn_save_xanes = Button(ax_save_xanes, 'Save XANES\nSpectra', color='yellow', hovercolor='khaki')
        self.btn_save_xanes.on_clicked(self.save_summary_csv); self.btn_save_xanes.label.set_fontsize(7)

        ax_save_xrd = self.fig.add_axes([0.10, btn_y, 0.07, btn_h])
        self.btn_save_xrd = Button(ax_save_xrd, 'Save XRD\nSpectra', color='yellow', hovercolor='khaki')
        self.btn_save_xrd.on_clicked(self.save_consolidated_xrd_spectrum); self.btn_save_xrd.label.set_fontsize(7)
        
        ax_toggle = self.fig.add_axes([0.18, btn_y, 0.09, btn_h])
        label_toggle = "Switch to\nPolygon" if self.sync.mode == 'rect' else "Switch to\nRectangle"
        self.btn_toggle = Button(ax_toggle, label_toggle, color='yellow', hovercolor='khaki')
        self.btn_toggle.on_clicked(self.toggle_selection_mode); self.btn_toggle.label.set_fontsize(7)

        ax_theme = self.fig.add_axes([0.28, btn_y, 0.08, btn_h])
        label_theme = "Switch to\nGray" if self.sync.use_color else "Switch to\nColor"
        self.btn_theme = Button(ax_theme, label_theme, color='yellow', hovercolor='khaki')
        self.btn_theme.on_clicked(self.toggle_theme); self.btn_theme.label.set_fontsize(7)

        ax_save_imgs = self.fig.add_axes([0.37, btn_y, 0.08, btn_h])
        self.btn_save_imgs = Button(ax_save_imgs, 'Save All\nImages', color='yellow', hovercolor='khaki')
        self.btn_save_imgs.on_clicked(self.save_all_images); self.btn_save_imgs.label.set_fontsize(7)

        ax_save_pm3d = self.fig.add_axes([0.46, btn_y, 0.14, btn_h])
        self.btn_save_pm3d = Button(ax_save_pm3d, 'Save XANES Spectra\nfor PCA/CA', color='yellow', hovercolor='khaki')
        self.btn_save_pm3d.on_clicked(self.save_pymca_stack_call); self.btn_save_pm3d.label.set_fontsize(7)

        ax_save_pm4d = self.fig.add_axes([0.61, btn_y, 0.24, btn_h])
        self.btn_save_pm4d = Button(ax_save_pm4d, 'Save XRF Spectra for Elemental\nAnalysis using PyMca', color='yellow', hovercolor='khaki')
        self.btn_save_pm4d.on_clicked(self.save_pymca_4d_stack_call); self.btn_save_pm4d.label.set_fontsize(7)
        
        # top row elements removed as energy selector is now a decoupled widget
        
        # Enable double-click to copy to clipboard
        self.fig.canvas.mpl_connect('button_press_event', _on_dashboard_click)
        # plt.show() removed to prevent double-plotting in Jupyter notebooks

# --- Main Entry Point ---

def plot_sgm_bsky_data(path_pack, representative_energy=None, channel_roi=(0, 256), roll_shift=0, as_scatter_plot: bool = False, map_roi=None, contrast=None, show_markers: bool = True, output_csv: bool = False, energy_shift=0.0, mcc_channels=None, mcc_to_map=None, x_trim=0.0, y_trim=0.0, save_images: bool = False, fixed_roi: bool = False, use_color: bool = True, use_full_metadata: bool = False):
    """
    Beautiful Stack Dashboard (Refactored for Fresh Starts).
    """
    # REFINED RESET: Close only existing dashboard figures to prevent ghosting 
    # without destroying unrelated user plots.
    for i in plt.get_fignums():
        label = plt.figure(i).get_label()
        if any(prefix in label for prefix in ['dash_', 'align_fig_', 'merger_fig_']):
            plt.close(i)

    global _GLOBAL_SUMMARY_DASH, _GLOBAL_SYNC_OBJ
    _GLOBAL_SUMMARY_DASH = None; _GLOBAL_SYNC_OBJ = None

    try:
        if not path_pack: return
        
        # Disable auto-display temporarily to allow wrapping plots in VBox
        plt.ioff()
        
        # 1. Path Setup
        h5_path = path_pack.get('h5_file_path') or path_pack.get('h5_dir')
        save_dir = os.path.dirname(os.path.abspath(h5_path)) if h5_path and os.path.isfile(h5_path) else (os.path.abspath(h5_path) if h5_path and os.path.isdir(h5_path) else os.getcwd())
        scan_name = path_pack.get('scan_name', 'sdd_stack')
        
        # Override alignment params if present in path_pack (e.g. from interactive_roll_align)
        if roll_shift == 0 and 'roll_shift' in path_pack:
            roll_shift = path_pack['roll_shift']
        if x_trim == 0.0 and 'x_trim' in path_pack:
            x_trim = path_pack['x_trim']
        if y_trim == 0.0 and 'y_trim' in path_pack:
            y_trim = path_pack['y_trim']
        
        # 2. Data Preparation
        all_energies = np.array(sorted(path_pack['energies']))
        if representative_energy is None:
            if len(all_energies) > 0:
                representative_energy = all_energies[len(all_energies) // 2]
                print(f"  [Auto Select] Defaulting to middle energy: {representative_energy:.2f} eV")
            else:
                representative_energy = 0.0
                print("  [Warning] No energies found in path_pack.")

        x_coords_raw, y_coords_raw = path_pack.get('x', np.array([])), path_pack.get('y', np.array([]))
        detector_names = sorted(path_pack['sdd_files'].keys())
        calibrated_energies = all_energies + energy_shift
        
        # 2.5 Determine actual data point count and ensure coordinate lengths match
        # Check first available detector/energy to find the real number of scanned points
        actual_num_s = min(x_coords_raw.size, y_coords_raw.size)
        try:
            first_det = detector_names[0]
            first_en = all_energies[0]
            first_path = path_pack['sdd_files'][first_det].get(first_en)
            if first_path and os.path.exists(first_path):
                # 256 channels * 4 bytes per channel (uint32)
                actual_num_s = min(os.path.getsize(first_path) // 1024, actual_num_s)
        except Exception:
            pass

        x_coords = x_coords_raw[:actual_num_s]
        y_coords = y_coords_raw[:actual_num_s]
        
        eb_x1, eb_x2 = np.min(x_coords) + x_trim, np.max(x_coords) - x_trim
        eb_y1, eb_y2 = np.min(y_coords) + y_trim, np.max(y_coords) - y_trim

        if map_roi is None:
            # Default to 1x1 mm square at the center of the trimmed area
            center_x = (eb_x1 + eb_x2) / 2.0
            center_y = (eb_y1 + eb_y2) / 2.0
            map_roi = [center_x - 0.5, center_x + 0.5, center_y - 0.5, center_y + 0.5]
        
        # Ensure map_roi is within bounds and properly formatted [x1, x2, y1, y2]
        map_roi = [max(min(map_roi[0], map_roi[1]), eb_x1), min(max(map_roi[0], map_roi[1]), eb_x2),
                   max(min(map_roi[2], map_roi[3]), eb_y1), min(max(map_roi[2], map_roi[3]), eb_y2)]

        print(f"--- SDD Stack Dashboard: {scan_name} ---")
        print(f"  [Metadata] Full Headers: {'Enabled' if use_full_metadata else 'Disabled'}")
        print(f"  Location: {save_dir}")
        print(f"  [Alignment] Roll Shift: {roll_shift} | Trim: X={x_trim:.3f} mm, Y={y_trim:.3f} mm")

        summary_data = {det: [] for det in detector_names}
        avg_maps = {det: None for det in detector_names} 
        stack_maps = {det: np.full((len(all_energies), x_coords.size), 0.0, dtype=np.float32) for det in detector_names}
        
        # mcc_channels are for the spectrum; mcc_to_map are for spatial images
        all_mccs = sorted(list(set((mcc_channels or []) + (mcc_to_map or []))))
        mcc_maps = {f'mcc{ch}': np.full((len(all_energies), x_coords.size), 0.0, dtype=np.float32) for ch in all_mccs}
        mcc_data = {f'mcc{ch}': [] for ch in (mcc_channels or [])}
        
        # We also need average maps for MCCs that are mapped
        for ch in (mcc_to_map or []):
            avg_maps[f'mcc{ch}'] = None

        print("  Pass 1: Pre-loading energy dependence...")
        for e_idx, energy in enumerate(all_energies):
            for det_name in detector_names:
                p = path_pack['sdd_files'][det_name].get(energy)
                if not p or not os.path.exists(p):
                    summary_data[det_name].append(np.nan); continue
                try:
                    d1d = np.fromfile(p, dtype=np.uint32)
                    num_s = min(len(d1d) // 256, x_coords.size)
                    s2d = d1d[:num_s * 256].reshape((num_s, 256))
                    if roll_shift != 0: s2d = np.roll(s2d, shift=roll_shift, axis=0)
                    inten = np.sum(s2d[:, channel_roi[0]:channel_roi[1]+1], axis=1)
                    if avg_maps[det_name] is None: avg_maps[det_name] = inten.copy()
                    else: avg_maps[det_name] += inten
                    m = get_dynamic_mask(x_coords[:num_s], y_coords[:num_s], x_trim, y_trim, roi=map_roi)
                    summary_data[det_name].append(np.sum(s2d[m, channel_roi[0]:channel_roi[1]+1]))
                    stack_maps[det_name][e_idx, :num_s] = inten
                except: summary_data[det_name].append(np.nan)
            
            if mcc_channels:
                mf = path_pack['mcc_files'].get(energy)
                current_mcc_vals = {ch: 0.0 for ch in mcc_channels}
                if mf and os.path.exists(mf):
                    try:
                        # Robustly read CSV/TSV without skipping the header if it contains column names
                        df = pd.read_csv(mf) if mf.endswith('.csv') else pd.read_table(mf)
                        
                        # Clean column names (handle headers starting with # and remove whitespace)
                        df.columns = [c.replace('#', '').strip() for c in df.columns]
                        
                        num_pts = min(len(df), x_coords.size, y_coords.size)
                        m_mcc = get_dynamic_mask(x_coords[:num_pts], y_coords[:num_pts], x_trim, y_trim, roi=map_roi)
                        
                        for ch in mcc_channels:
                            # Try multiple variants for MCC column names
                            col = None
                            for variant in [f'ch{ch}', f'mcc{ch}', str(ch)]:
                                if variant in df.columns:
                                    col = variant
                                    break
                            
                            if col is not None:
                                col_data = df[col].values
                                pts = min(len(col_data), x_coords.size)
                                mcc_maps[f'mcc{ch}'][e_idx, :pts] = col_data[:pts]
                                
                                # Update average map for mapping
                                if f'mcc{ch}' in avg_maps:
                                    if avg_maps[f'mcc{ch}'] is None: avg_maps[f'mcc{ch}'] = col_data[:pts].copy()
                                    else: avg_maps[f'mcc{ch}'] += col_data[:pts]

                                val = df[col].values[:num_pts][m_mcc].mean() if np.any(m_mcc) else 0.0
                                current_mcc_vals[ch] = 0.0 if np.isnan(val) else val
                            else:
                                if energy == all_energies[0]: # Only warn once
                                    print(f"  [Warning] MCC channel {ch} not found in {os.path.basename(mf)}. Available: {list(df.columns)}")
                    except Exception as e:
                        print(f"  [Error] Failed to load MCC file {mf}: {e}")
                
                # Commit values once per energy step to prevent double-appends or mismatches
                for ch in mcc_channels:
                    mcc_data[f'mcc{ch}'].append(current_mcc_vals[ch])

            
        if detector_names and any(summary_data[det] for det in detector_names):
            avg_dependence = np.nanmean([summary_data[det] for det in detector_names], axis=0)
        else:
            avg_dependence = np.zeros(len(all_energies))

        # ---- EXTERNAL I0 PROMPT ----
        root_i0 = get_tk_root(); root_i0.attributes("-topmost", True)
        
        # If single energy, skip the prompt and default to internal I0 (mcc1)
        if len(all_energies) <= 1:
            use_internal = True
        else:
            use_internal = messagebox.askyesno("I0 Selection", "Use INTERNAL mcc1 for I0 normalization?\n\n(Select 'No' to load an EXTERNAL I0 CSV)")
        
        use_ext = not use_internal
        
        ext_i0_values = None
        i0_source = "mcc1" if (mcc_channels and 1 in mcc_channels) else "None (Raw Only)"
        
        if use_ext:
            ext_path = filedialog.askopenfilename(title="Select External I0 CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if ext_path:
                try:
                    ext_df = pd.read_csv(ext_path, comment='#')
                    # Trim whitespace from columns to prevent lookup errors
                    ext_df.columns = [c.strip() for c in ext_df.columns]
                    
                    e_col = next((c for c in ext_df.columns if 'energy' in c.lower()), ext_df.columns[0])
                    i_col = next((c for c in ext_df.columns if any(k in c.lower() for k in ['i0', 'intensity', 'norm', 'tey'])), ext_df.columns[1])
                    
                    dialog = ExternalI0PreviewDialog(root_i0, ext_df, e_col, i_col)
                    dialog.mainloop()
                    
                    if dialog.result:
                        selected_e_col, selected_i_col, x_sorted, y_sorted, extra_str, cal_en, cal_val = dialog.result
                        
                        # Store raw sorted data for on-the-fly shift adjustments
                        context_ext_i0_df = ext_df
                        context_ext_i0_cols = (selected_e_col, selected_i_col)
                        context_ext_i0_raw_xy = (x_sorted, y_sorted)

                        # Use np.interp (requires strictly increasing x, which happens after sorting)
                        shift = cal_val if cal_en else 0.0
                        ext_i0_values = np.interp(calibrated_energies, x_sorted + shift, y_sorted)
                        i0_source = f"{os.path.basename(ext_path)} [{selected_i_col}]{extra_str}"
                        # Set initial calibration state from dialog
                        path_pack['i0_calib_enabled'] = cal_en
                        path_pack['i0_energy_shift'] = cal_val
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load external I0: {e}")
        elif use_internal and mcc_channels and 1 in mcc_channels and np.any(mcc_data['mcc1']):
            try:
                # Internal I0 smoothing capability
                # Ensure matching lengths for DataFrame creation
                mcc1_vals = mcc_data['mcc1']
                min_len = min(len(calibrated_energies), len(mcc1_vals))
                int_df = pd.DataFrame({'Energy': calibrated_energies[:min_len], 'mcc1': mcc1_vals[:min_len]})
                dialog = ExternalI0PreviewDialog(root_i0, int_df, 'Energy', 'mcc1')
                dialog.title("Internal I0 Preview")
                dialog.mainloop()
                
                if dialog.result:
                    selected_e_col, selected_i_col, x_sorted, y_sorted, extra_str, cal_en, cal_val = dialog.result
                    ext_i0_values = np.interp(calibrated_energies, x_sorted, y_sorted)
                    i0_source = f"Internal: mcc1{extra_str}"
                    # Note: Energy shift NOT applied to internal I0 as per user request
                    path_pack['i0_calib_enabled'] = False
                    path_pack['i0_energy_shift'] = 0.0
            except Exception as e:
                print(f"Error previewing internal I0: {e}")

        # Save selection to path_pack for downstream modules (e.g. save_pymca_stack_h5, PCA, clustering)
        path_pack['ext_i0_values'] = ext_i0_values
        path_pack['i0_source'] = i0_source

        # 3. Class Instantiation (Fresh State)
        sync = Synchronizer(all_energies, representative_energy, map_roi, use_color=use_color, use_full_metadata=use_full_metadata, calibrated_energies=calibrated_energies)
        sync.channel_roi = channel_roi # Set initial range from command line
        
        # Load initial calibration state from path_pack if set by dialog
        sync.i0_calib_enabled = path_pack.get('i0_calib_enabled', False)
        sync.i0_energy_shift = path_pack.get('i0_energy_shift', 0.0)
        
        context = {
            'path_pack': path_pack, 'x_coords': x_coords, 'y_coords': y_coords,
            'x_trim': x_trim, 'y_trim': y_trim, 'save_dir': save_dir,
            'scan_name': scan_name, 'project_name': path_pack.get('project', 'N/A'),
            'channel_roi': channel_roi, 'detector_names': detector_names,
            'avg_maps': avg_maps, 'stack_maps': stack_maps, 'roll_shift': roll_shift,
            'calibrated_energies': calibrated_energies, 'summary_data': summary_data,
            'avg_dependence': avg_dependence, 'mcc_channels': mcc_channels, 
            'mcc_data': mcc_data, 'mcc_maps': mcc_maps, 'show_markers': show_markers,
            'ext_i0_values': ext_i0_values, 'i0_source': i0_source,
            'ext_i0_df': context_ext_i0_df if use_ext else None,
            'ext_i0_cols': context_ext_i0_cols if use_ext else None,
            'ext_i0_raw_xy': context_ext_i0_raw_xy if use_ext else None
        }

        for det in detector_names:
            row = DashboardRow(det, sync, context)
            row.load_rep_data(); row.plot()
            sync.dashboards.append(row)
        
        # Add MCC mapping rows
        for ch in (mcc_to_map or []):
            name = f'mcc{ch}'
            row = DashboardRow(name, sync, context)
            row.load_rep_data(); row.plot()
            sync.dashboards.append(row)
        
        summary = SummaryDashboard(sync, context); summary.plot()
        sync.summary_dash = summary
        
        # Register in global scope
        _GLOBAL_SUMMARY_DASH = summary
        _GLOBAL_SYNC_OBJ = sync

        # 8. Decoupled Progress Widget and Control Buttons
        sync.status_widget = widgets.Label(value="[ROI ENGINE v2.6 Ready] Dashboard Active.")
        
        btn_refresh = widgets.Button(description="REFRESH PLOTS", button_style='success', tooltip='Force a full re-calculation of all spectra')
        btn_refresh.on_click(sync.force_global_refresh)
        
        ipfy_chk = widgets.Checkbox(value=False, description='IPFY Mode (Invert for PCA)', indent=False)
        def on_ipfy_change(change): 
            sync.ipfy_mode = change['new']
            if sync.summary_dash:
                sync.summary_dash.update_plots(sync.current_roi, sync.current_poly, sync.mode)
        ipfy_chk.observe(on_ipfy_change, names='value')

        controls = widgets.HBox([sync.status_widget, btn_refresh, ipfy_chk])

        # 9. Decoupled Energy Slider (moved to just before images)
        energy_slider = widgets.IntSlider(
            value=sync.energy_idx,
            min=0,
            max=len(all_energies) - 1,
            step=1,
            description='Select Image Energy for Display:',
            layout=widgets.Layout(width='800px'),
            style={'description_width': 'initial', 'handle_color': 'yellow'}
        )
        
        # Use a Button-styled label to add some color feedback
        energy_label = widgets.Button(description=f"{sync.calibrated_energies[sync.energy_idx]:.2f} eV", 
                                    button_style='warning', layout=widgets.Layout(width='120px'))
        
        def on_energy_change(change):
            new_idx = change['new']
            energy_label.description = f"{sync.calibrated_energies[new_idx]:.2f} eV"
            sync.broadcast_energy(new_idx)
            
        energy_slider.observe(on_energy_change, names='value')
        
        # 10. Global Contrast Slider
        contrast_slider = widgets.FloatRangeSlider(
            value=[0.0, 100.0], min=0.0, max=100.0, step=0.1,
            description='Image Contrast %:',
            layout=widgets.Layout(width='800px'),
            style={'description_width': 'initial', 'handle_color': 'cyan'}
        )
        
        def on_contrast_change(change):
            sync.broadcast_contrast(change['new'])
        
        contrast_slider.observe(on_contrast_change, names='value')

        # 11. Log Scale Checkboxes
        log_toggle = widgets.Checkbox(value=False, description='Use Log Scale (Images)', indent=False)
        log_toggle.observe(lambda c: sync.broadcast_log(c['new']), names='value')

        log_spec_toggle = widgets.Checkbox(value=False, description='Use Log Scale (Spectra)', indent=False)
        log_spec_toggle.observe(lambda c: sync.broadcast_log_spec(c['new']), names='value')

        energy_controls = widgets.VBox([
            widgets.HBox([energy_slider, energy_label]),
            widgets.HBox([contrast_slider, log_toggle, log_spec_toggle]),
            widgets.HBox([
                widgets.Label("Calibration Toggles:", layout=widgets.Layout(width='200px')),
                widgets.Checkbox(value=sync.i0_calib_enabled, description='Enable I0 Shift', indent=False),
                widgets.FloatText(value=sync.i0_energy_shift, description='I0 Shift (eV):', layout=widgets.Layout(width='180px')),
                widgets.Label("|", layout=widgets.Layout(width='20px')),
                widgets.Checkbox(value=sync.use_sdd_calib, description='Enable SDD Energy Calib', indent=False, tooltip='Sync ROIs by Energy instead of Channels'),
            ]),
        ])
        
        # Set up I0 & SDD Calibration Observers
        i0_shift_chk = energy_controls.children[2].children[1]
        i0_shift_val = energy_controls.children[2].children[2]
        sdd_cal_chk = energy_controls.children[2].children[4]
        
        i0_shift_chk.observe(lambda c: sync.broadcast_i0_calib(enabled=c['new']), names='value')
        i0_shift_val.observe(lambda c: sync.broadcast_i0_calib(shift=c['new']), names='value')
        
        def on_sdd_cal_change(change):
            sync.use_sdd_calib = change['new']
            print(f"  [SDD Calibration] {'ENABLED' if sync.use_sdd_calib else 'DISABLED'}")
            # Refresh all plots to show Energy vs Channel axis
            for d in sync.dashboards:
                d.update_spectrum()
            if sync.status_widget:
                sync.status_widget.value = f"SDD Calibration {'Enabled (ROI in eV)' if sync.use_sdd_calib else 'Disabled (ROI in Channels)'}."
        
        sdd_cal_chk.observe(on_sdd_cal_change, names='value')

        try:
            get_ipython()
            in_jupyter = True
        except NameError:
            in_jupyter = False

        if in_jupyter:
            summary_canvases = []
            map_canvases = []
            
            # Place summary dashboard first so it appears above the maps
            if sync.summary_dash:
                try: sync.summary_dash.fig.canvas.layout.min_width = '1450px'
                except: pass
                summary_canvases.append(sync.summary_dash.fig.canvas)
                
            for d in sync.dashboards:
                try: d.fig.canvas.layout.min_width = '1450px'
                except: pass
                map_canvases.append(d.fig.canvas)
            
            # Separate scrollable boxes for independent horizontal scrolling
            summary_scroll_box = widgets.VBox(summary_canvases, layout=widgets.Layout(width='100%', overflow_x='auto'))
            map_scroll_box = widgets.VBox(map_canvases, layout=widgets.Layout(width='100%', overflow_x='auto'))
            
            # RE-ORDERED DISPLAY as requested
            display(controls)             # 1. REFRESH button / ROI engine
            display(summary_scroll_box)   # 2. XANES spectra
            display(energy_controls)      # 3. Energy selector
            display(map_scroll_box)       # 4. Images
        else:
            plt.show()  # Trigger display in CLI

        plt.ion() # Restore interactive mode
        
        # Keep references alive to prevent garbage collection in Jupyter
        if sync.dashboards:
            sync.dashboards[0].fig._sync_ref = sync 
        return sync
        
    except Exception as e:
        plt.ion() # Safety restore
        print(f"Error in plot_sgm_bsky_data: {e}"); traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", help="HDF5 file path")
    parser.add_argument("--energy", type=float, help="Rep energy (defaults to middle of stack)")
    parser.add_argument("--roi_s", type=int, default=0)
    parser.add_argument("--roi_e", type=int, default=255)
    parser.add_argument("--roll_shift", type=int, default=0)
    parser.add_argument("--x_trim", type=float, default=0.0)
    parser.add_argument("--map_roi", type=float, nargs=4, required=True)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--minimum_metadata", action="store_false", dest="full_metadata")
    parser.set_defaults(full_metadata=False)
    args = parser.parse_args()
    pp = analyze_sgm_bsky_data(args.h5)
    if pp:
        pp['h5_dir'] = os.path.dirname(args.h5)
        plot_sgm_bsky_data(pp, args.energy, (args.roi_s, args.roi_e), roll_shift=args.roll_shift, x_trim=args.x_trim, map_roi=args.map_roi, save_images=args.save_images, use_full_metadata=args.full_metadata)
