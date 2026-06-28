import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import find_peaks
import ipywidgets as widgets
from IPython.display import display, clear_output

# Import project utilities
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
import sdd_calibration_utils as calib_utils

class SDDCalibrationGUI:
    def __init__(self):
        self.data_pack = None
        self.current_spectra = {} # {sdd_id: array_of_256}
        self.calibrations = calib_utils.load_calibration()
        self.detected_peaks_all = {} # {sdd_id: array_of_peak_channels}
        self.detector_active = {} # {sdd_id: Checkbox}
        
        # Known peaks for reference (User can override)
        self.standard_peaks = {
            # K-Edges
            "B-K": 183.3, "C-K": 277.0, "N-K": 392.4, "O-K": 524.9, "F-K": 677.0, 
            "Na-K": 1041.0, "Mg-K": 1253.6, "Al-K": 1486.7, "Si-K": 1739.0,
            # L-Edges (L3 typically)
            "K-L": 294.6, "Ca-L": 346.4, "Sc-L": 398.8, "Ti-L": 453.8, "V-L": 512.1, 
            "Cr-L": 574.1, "Mn-L": 638.7, "Fe-L": 706.8, "Co-L": 778.1, "Ni-L": 852.7, 
            "Cu-L": 932.7, "Zn-L": 1021.8, "Ga-L": 1116.4, "Ge-L": 1217.0, "As-L": 1323.2, 
            "Se-L": 1433.9, "Br-L": 1550.0,
            # Rare Earth M-Edges (M-alpha emission)
            "La-M": 833.3, "Ce-M": 883.3, "Pr-M": 928.8, "Nd-M": 977.7, "Sm-M": 1080.9, 
            "Eu-M": 1130.9, "Gd-M": 1185.2, "Tb-M": 1241.1, "Dy-M": 1292.8, "Ho-M": 1351.1, 
            "Er-M": 1405.7, "Tm-M": 1462.3, "Yb-M": 1521.3, "Lu-M": 1581.2
        }
        
        self.detector_widgets = {} # {sdd_id: [widgets for each point]}
        self.setup_ui()

    def setup_ui(self):
        self.out = widgets.Output()
        self.out_plot = widgets.Output()
        
        # 1. File Loading Row
        self.btn_load = widgets.Button(description="Load Standard Scan", button_style='info')
        self.btn_load.on_click(self.on_load_clicked)
        self.txt_file = widgets.Label(value="No file loaded.")
        
        # 2. Global Detection Settings
        # Increased sensitivity range (min 0.001)
        self.threshold = widgets.FloatSlider(value=0.05, min=0.001, max=0.9, step=0.001, 
                                             description="Threshold:", continuous_update=False, readout_format='.3f')
        self.distance = widgets.IntSlider(value=10, min=2, max=100, step=1, description="Distance:", continuous_update=False)
        self.log_scale = widgets.Checkbox(value=False, description="Log Scale")
        self.num_points = widgets.Dropdown(options=[2, 3, 4], value=2, description="Points:", layout=widgets.Layout(width='150px'))
        
        self.threshold.observe(self.update_ui_on_params, names='value')
        self.distance.observe(self.update_ui_on_params, names='value')
        self.log_scale.observe(self.update_ui_on_params, names='value')
        self.num_points.observe(self.update_point_count, names='value')

        settings_box = widgets.HBox([self.threshold, self.distance, self.log_scale, self.num_points])
        
        # 3. Target Energy Points
        self.point_configs = []
        for i in range(4):
            p_name = widgets.Dropdown(options=list(self.standard_peaks.keys()) + ["Custom"], 
                                      value=list(self.standard_peaks.keys())[i] if i < len(self.standard_peaks) else "Custom",
                                      description=f"Point {i+1}:", layout=widgets.Layout(width='180px'))
            p_val = widgets.FloatText(value=self.standard_peaks.get(p_name.value, 0.0), description="eV:", layout=widgets.Layout(width='150px'))
            
            def make_updater(name_w, val_w):
                def update_val(change):
                    if change['new'] != "Custom":
                        val_w.value = self.standard_peaks.get(change['new'], 0.0)
                return update_val
            
            p_name.observe(make_updater(p_name, p_val), names='value')
            
            row = widgets.HBox([p_name, p_val])
            if i >= 2: row.layout.display = 'none' # Start with 2 points
            self.point_configs.append({'name': p_name, 'val': p_val, 'row': row})
            
        points_box = widgets.VBox([p['row'] for p in self.point_configs])
        
        # 4. Detector Assignments Area
        self.detector_box = widgets.VBox([])
        
        # 5. Actions
        self.btn_calc = widgets.Button(description="Calculate & Save Calibration", button_style='success', disabled=True, layout=widgets.Layout(width='300px'))
        self.btn_calc.on_click(self.on_save_clicked)
        
        self.status = widgets.HTML(value="<b>Status:</b> Please load a standard scan (e.g. AlPO4, Cryolite, MgO)")
        
        # Main Layout
        self.ui = widgets.VBox([
            widgets.HTML("<h2>XRF SDD Energy Calibration Tool</h2>"),
            widgets.HBox([self.btn_load, self.txt_file]),
            widgets.HTML("<hr><b>1. Detection Settings</b>"),
            settings_box,
            widgets.HTML("<hr><b>2. Target Energies (eV)</b>"),
            points_box,
            widgets.HTML("<hr><b>3. Per-Detector Peak Selection</b>"),
            self.detector_box,
            widgets.HTML("<hr>"),
            self.btn_calc,
            self.status,
            self.out_plot,
            self.out
        ])

    def update_point_count(self, change):
        n = change['new']
        for i, config in enumerate(self.point_configs):
            config['row'].layout.display = 'flex' if i < n else 'none'
        self.rebuild_detector_widgets()

    def update_ui_on_params(self, change):
        if self.data_pack:
            self.plot_spectra()
            self.rebuild_detector_widgets()

    def on_load_clicked(self, b):
        with self.out:
            clear_output()
            print("Opening file browser...")
            self.data_pack = analyze_sgm_bsky_data()
            if not self.data_pack:
                self.status.value = "<b>Status:</b> No file selected."
                return
            
            self.txt_file.value = os.path.basename(self.data_pack['h5_file_path'])
            self.process_data()
            self.plot_spectra()
            self.rebuild_detector_widgets()
            self.btn_calc.disabled = False
            self.status.value = "<b>Status:</b> Data loaded. Assign peaks for each detector."

    def process_data(self):
        """Sums spectra for each detector across a subset of energies for speed."""
        self.current_spectra = {}
        all_energies = self.data_pack['energies']
        
        # Sample up to 20 energies to get a good representative spectrum
        step = max(1, len(all_energies) // 20)
        energies_to_sum = all_energies[::step]
        
        for sdd_id in sorted(self.data_pack['sdd_files'].keys()):
            sum_spec = np.zeros(256)
            for energy in energies_to_sum:
                p = self.data_pack['sdd_files'][sdd_id].get(energy)
                if p and os.path.exists(p):
                    try:
                        d1d = np.fromfile(p, dtype=np.uint32)
                        num_pixels = len(d1d) // 256
                        if num_pixels > 0:
                            spec_2d = d1d[:num_pixels*256].reshape((num_pixels, 256))
                            sum_spec += np.sum(spec_2d, axis=0)
                    except: pass
            self.current_spectra[sdd_id] = sum_spec

    def plot_spectra(self):
        with self.out_plot:
            clear_output(wait=True)
            num_dets = len(self.current_spectra)
            if num_dets == 0: return
            
            fig, axes = plt.subplots(num_dets, 1, figsize=(10, 3 * num_dets), sharex=True)
            if num_dets == 1: axes = [axes]
            
            self.detected_peaks_all = {}
            
            for i, (sdd_id, spec) in enumerate(self.current_spectra.items()):
                ax = axes[i]
                channels = np.arange(256)
                ax.plot(channels, spec, label=f"{sdd_id} Raw", color='blue', alpha=0.6)
                if self.log_scale.value: ax.set_yscale('log')
                
                # Peak detection
                h = np.max(spec) * self.threshold.value
                d = self.distance.value
                peaks, _ = find_peaks(spec, height=h, distance=d)
                self.detected_peaks_all[sdd_id] = peaks
                
                for idx, p in enumerate(peaks):
                    ax.axvline(p, color='gray', linestyle=':', alpha=0.3)
                    ax.text(p, np.max(spec)*0.8 if not self.log_scale.value else np.max(spec), 
                            f"ID:{idx}\nCh:{p}", color='black', fontsize=8, ha='center')
                
                ax.set_ylabel("Counts")
                ax.set_title(f"Detector: {sdd_id}")
                ax.legend()
            
            axes[-1].set_xlabel("Channel")
            plt.tight_layout()
            plt.show()

    def rebuild_detector_widgets(self):
        rows = []
        n_pts = self.num_points.value
        
        for sdd_id in sorted(self.current_spectra.keys()):
            # Detect if flat line (all zeros or constant value)
            spec = self.current_spectra.get(sdd_id, np.zeros(256))
            is_flat = np.all(spec == 0) or np.std(spec) == 0
            
            # Retrieve or create active checkbox
            if sdd_id not in self.detector_active:
                self.detector_active[sdd_id] = widgets.Checkbox(
                    value=not is_flat, 
                    description="Active", 
                    layout=widgets.Layout(width='80px')
                )
            
            active_checkbox = self.detector_active[sdd_id]
            
            peaks = self.detected_peaks_all.get(sdd_id, [])
            peak_options = [("Manual", -1)] + [(f"Peak {i} (Ch {p})", i) for i, p in enumerate(peaks)]
            
            det_label = widgets.HTML(value=f"<b>{sdd_id} Assignments:</b>", layout=widgets.Layout(width='130px'))
            
            point_selectors = []
            for i in range(n_pts):
                sel = widgets.Dropdown(options=peak_options, value=i if i < len(peaks) else -1, 
                                       description=f"P{i+1}:", layout=widgets.Layout(width='140px'))
                man = widgets.IntText(value=0, description="Manual Ch:", layout=widgets.Layout(width='120px'))
                
                # Default manual value if no peak found
                if i < len(peaks): man.value = peaks[i]
                
                # Hide manual if peak ID selected
                def make_man_toggle(s, m):
                    def toggle(change): m.layout.display = 'flex' if change['new'] == -1 else 'none'
                    return toggle
                sel.observe(make_man_toggle(sel, man), names='value')
                man.layout.display = 'none' if sel.value != -1 else 'flex'
                
                point_selectors.append({'sel': sel, 'man': man, 'box': widgets.HBox([sel, man])})
            
            # Setup active state change observer to disable/enable dropdowns
            def make_active_observer(selectors_list):
                def observer(change):
                    disabled = not change['new']
                    for ps in selectors_list:
                        ps['sel'].disabled = disabled
                        ps['man'].disabled = disabled
                return observer
            active_checkbox.observe(make_active_observer(point_selectors), names='value')
            
            # Initial state setup
            disabled = not active_checkbox.value
            for ps in point_selectors:
                ps['sel'].disabled = disabled
                ps['man'].disabled = disabled
            
            self.detector_widgets[sdd_id] = point_selectors
            rows.append(widgets.HBox([det_label, active_checkbox] + [ps['box'] for ps in point_selectors]))
        
        self.detector_box.children = rows

    def on_save_clicked(self, b):
        new_calib = {}
        n_pts = self.num_points.value
        target_energies = [p['val'].value for p in self.point_configs[:n_pts]]
        
        summary_text = "<b>Calibration Summary:</b><br>"
        
        for sdd_id, selectors in self.detector_widgets.items():
            # Check if active checkbox is checked
            is_active = self.detector_active[sdd_id].value if sdd_id in self.detector_active else True
            
            if not is_active:
                gain, offset = 1.0, 0.0
                new_calib[sdd_id] = {"gain": round(gain, 4), "offset": round(offset, 4)}
                summary_text += f" - {sdd_id}: Inactive (Default Gain=1.0, Offset=0.0)<br>"
                continue
                
            channels = []
            for i in range(n_pts):
                sel_idx = selectors[i]['sel'].value
                if sel_idx == -1:
                    channels.append(selectors[i]['man'].value)
                else:
                    channels.append(self.detected_peaks_all[sdd_id][sel_idx])
            
            # Check for duplicate channel selection
            if len(set(channels)) < len(channels):
                self.status.value = f"<b>Status:</b> <font color='red'>Error: Duplicate channels {channels} assigned for detector '{sdd_id}'. Please select unique peaks or manual values, or uncheck 'Active'.</font>"
                return
            
            # Linear fit: Energy = Gain * Channel + Offset
            # We sort both to ensure correct mapping (lower channel -> lower energy)
            sorted_channels = sorted(channels)
            sorted_energies = sorted(target_energies)
            
            gain, offset = calib_utils.calculate_calibration_params(sorted_channels, sorted_energies)
            new_calib[sdd_id] = {"gain": round(gain, 4), "offset": round(offset, 4)}
            summary_text += f" - {sdd_id}: Gain={gain:.3f}, Offset={offset:.1f}<br>"
        
        self.calibrations.update(new_calib)
        if calib_utils.save_calibration(self.calibrations):
            self.status.value = summary_text + "<br><font color='green'><b>Successfully saved to sdd_calibration.json</b></font>"
        else:
            self.status.value = "<b>Status:</b> <font color='red'>Error saving to file.</font>"

def run_calibration():
    gui = SDDCalibrationGUI()
    display(gui.ui)

if __name__ == "__main__":
    run_calibration()
