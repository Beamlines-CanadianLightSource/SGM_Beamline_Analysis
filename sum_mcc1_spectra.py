import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import existing analysis logic
from analyze_sgm_bsky_data import analyze_sgm_bsky_data

class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

def extract_mcc1_spectrum(file_path):
    print(f"Loading {os.path.basename(file_path)}...")
    path_pack = analyze_sgm_bsky_data(file_path, verbose=False)
    if not path_pack:
        return None, None
        
    mcc_channels = path_pack.get('mcc_channel_names', [])
    mcc1_idx = -1
    for idx, name in enumerate(mcc_channels):
        if any(k == name.lower().strip() for k in ['ch1', 'mcc1', '1']):
            mcc1_idx = idx
            break
            
    if mcc1_idx == -1:
        print(f"  Warning: No mcc1 channel found in {os.path.basename(file_path)}.")
        return None, None
        
    all_energies = path_pack.get('energies', np.array([]))
    if len(all_energies) == 0:
        return None, None
        
    mcc_data_dict = path_pack.get('mcc_data', {})
    mcc1_means = []
    
    for energy in all_energies:
        energy_mcc = mcc_data_dict.get(energy)
        if energy_mcc is not None and energy_mcc.shape[0] > 0:
            if len(energy_mcc.shape) > 1:
                mcc1_col = energy_mcc[:, mcc1_idx]
            else:
                mcc1_col = np.array([energy_mcc[mcc1_idx]])
            mean_val = np.nanmean(mcc1_col)
            mcc1_means.append(0.0 if np.isnan(mean_val) else mean_val)
        else:
            mcc1_means.append(0.0)
            
    return all_energies, np.array(mcc1_means)

class MCC1SummatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MCC1 Spectra Summator")
        self.geometry("900x700")
        
        self.spectra_data = [] # List of dicts: {'file': path, 'energies': e, 'mcc1': m, 'active': bool}
        self.common_energies = None
        
        self._build_ui()
        
    def _build_ui(self):
        # Left Panel: File list and controls
        left_frame = tk.Frame(self, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        btn_load = tk.Button(left_frame, text="Load Stack HDF5 Files", command=self.load_files, bg='lightblue')
        btn_load.pack(fill=tk.X, pady=5)
        
        tk.Label(left_frame, text="Select files to include in average:").pack(anchor=tk.W, pady=(10, 0))
        
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(btn_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2, 0))
        
        # Scrollable frame for checkboxes
        self.chk_container = ScrollableFrame(left_frame)
        self.chk_container.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chk_frame = self.chk_container.scrollable_frame
        
        self.chk_widgets = []
        
        btn_save = tk.Button(left_frame, text="Save Average to CSV", command=self.save_csv, bg='lightgreen')
        btn_save.pack(fill=tk.X, pady=10)
        
        btn_close = tk.Button(left_frame, text="Close Dashboard", command=self.close_app, bg='lightcoral')
        btn_close.pack(fill=tk.X, pady=(0, 10))
        
        # Ensure 'X' in the corner also properly shuts down
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        
        # Right Panel: Plot
        right_frame = tk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(8, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def close_app(self):
        self.quit()
        self.destroy()
        
    def select_all(self):
        for sd in self.spectra_data:
            sd['active'].set(True)
        self.update_plot()
        
    def deselect_all(self):
        for sd in self.spectra_data:
            sd['active'].set(False)
        self.update_plot()
        
    def load_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select HDF5 Stack Files",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if not file_paths:
            return
            
        for path in file_paths:
            # Check if already loaded
            if any(sd['file'] == path for sd in self.spectra_data):
                continue
                
            e, m = extract_mcc1_spectrum(path)
            if e is not None and m is not None:
                # Setup interpolation basis if this is the first one
                if self.common_energies is None:
                    self.common_energies = e
                    
                # Interpolate to common energies to be safe
                m_interp = np.interp(self.common_energies, e, m)
                
                self.spectra_data.append({
                    'file': path,
                    'mcc1': m_interp,
                    'active': tk.BooleanVar(self, value=True)
                })
            else:
                print(f"Skipping {os.path.basename(path)} (could not extract mcc1).")
        
        self._refresh_checkboxes()
        self.update_plot()
        
    def _refresh_checkboxes(self):
        for w in self.chk_widgets:
            w.destroy()
        self.chk_widgets.clear()
        
        for sd in self.spectra_data:
            fname = os.path.basename(sd['file'])
            chk = tk.Checkbutton(self.chk_frame, text=fname, variable=sd['active'], onvalue=True, offvalue=False, command=self.update_plot)
            chk.pack(anchor=tk.W)
            self.chk_widgets.append(chk)
            
    def update_plot(self):
        self.ax.clear()
        if not self.spectra_data:
            self.ax.set_title("No Data Loaded")
            self.canvas.draw()
            return
            
        all_spectra = []
        active_spectra = []
        for sd in self.spectra_data:
            all_spectra.append(sd['mcc1'])
            if sd['active'].get():
                self.ax.plot(self.common_energies, sd['mcc1'], alpha=0.5, label=os.path.basename(sd['file']))
                active_spectra.append(sd['mcc1'])
                
        if all_spectra:
            global_avg = np.mean(all_spectra, axis=0)
            self.ax.plot(self.common_energies, global_avg, 'k--', linewidth=2, label="Global Average (All Files)")
            
        if active_spectra:
            avg_mcc1 = np.mean(active_spectra, axis=0)
            
            # Plot average
            self.ax.plot(self.common_energies, avg_mcc1, 'r-', linewidth=2, label="Selected Average")
            
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("MCC1 Mean Intensity")
        self.ax.set_title("MCC1 Spectra Comparison")
        
        # Display legend nicely outside the plot
        if len(self.spectra_data) < 15:
            self.ax.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))
        else:
            # If too many files, just show the averages in legend
            handles, labels = self.ax.get_legend_handles_labels()
            # The last handles are Global and Selected averages
            num_handles = min(2, len(handles))
            if num_handles > 0:
                self.ax.legend(handles[-num_handles:], labels[-num_handles:], fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))
                
        self.fig.tight_layout()
        self.canvas.draw()
        
    def save_csv(self):
        if not self.spectra_data or self.common_energies is None:
            messagebox.showwarning("Warning", "No data to save.")
            return
            
        active_spectra = [sd['mcc1'] for sd in self.spectra_data if sd['active'].get()]
        if not active_spectra:
            messagebox.showwarning("Warning", "No active spectra selected.")
            return
            
        avg_mcc1 = np.mean(active_spectra, axis=0)
        
        # Propose save path based on first active file
        first_active = next((sd for sd in self.spectra_data if sd['active'].get()), None)
        initial_dir = os.path.dirname(first_active['file']) if first_active else os.getcwd()
        
        save_path = filedialog.asksaveasfilename(
            title="Save Averaged MCC1 to CSV",
            initialdir=initial_dir,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not save_path:
            return
            
        try:
            df = pd.DataFrame({
                'Energy': self.common_energies,
                'Avg_MCC1': avg_mcc1
            })
            df.to_csv(save_path, index=False)
            
            # This print ensures the pathway is available in Jupyter Notebook output
            print(f"\n[SAVE SUCCESS] MCC1 Average saved to:")
            print(f"--> {os.path.abspath(save_path)}")
            
            messagebox.showinfo("Success", f"Saved successfully to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

def sum_mcc1_spectra():
    app = MCC1SummatorApp()
    app.mainloop()

if __name__ == '__main__':
    sum_mcc1_spectra()
