import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import ipywidgets as widgets
from IPython.display import display
import tkinter as tk
from tkinter import messagebox, filedialog

def interactive_cluster_merger(h5_path, dataset_name='average'):
    """
    Provides an interactive widget to select specific clusters and view/extract 
    their combined averaged XANES spectrum. Supports switching between different detectors.
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return

    # --- Setup Main UI Containers ---
    output = widgets.Output()
    main_vbox = widgets.VBox([])
    display(main_vbox)

    # State variables
    state = {
        'current_dataset': dataset_name,
        'h5_path': h5_path,
        'cluster_map': None,
        'stack_flat': None,
        'energy': None,
        'x_axis': None,
        'y_axis': None,
        'checkboxes': [],
        'meta_header': []
    }

    fig_id = f"merger_fig_{id(h5_path)}"
    plt.close(fig_id) # Ensure fresh start for this instance
    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(10, 4.5), num=fig_id)
    line, = ax_spec.plot([], [], color='black', linewidth=2)
    im = ax_map.imshow(np.zeros((1,1)), cmap='tab10', interpolation='nearest')
    ax_spec.grid(True, linestyle='--', alpha=0.7)

    def load_dataset_data(ds_name):
        try:
            with h5py.File(h5_path, 'r') as f:
                pca_path = f"entry/pca_results/{ds_name}/clustering"
                if pca_path not in f:
                    with output:
                        print(f"Error: Clustering results for '{ds_name}' not found.")
                    return False
                
                state['current_dataset'] = ds_name
                state['cluster_map'] = f[f"{pca_path}/cluster_map"][()]
                state['stack'] = f[f"entry/measurement/{ds_name}"][()]
                state['energy'] = f["entry/measurement/energy"][()]
                state['x_axis'] = f["entry/measurement/x"][()]
                state['y_axis'] = f["entry/measurement/y"][()]
                
                ny, nx, n_en = state['stack'].shape
                state['stack_flat'] = state['stack'].reshape(ny * nx, -1)
                
                # Metadata
                state['meta_header'] = [f"# Facility: CLS", f"# Beamline: SGM"]
                meta_map = {'beamline': 'Beamline', 'date': 'Date', 'project': 'Project', 'exit_slit_gap': 'Exit Slit Gap'}
                for source in [f['entry'], f['entry/measurement']]:
                    for k, label in meta_map.items():
                        if k in source.attrs:
                            val = source.attrs[k]
                            if isinstance(val, (bytes, np.bytes_)): val = val.decode('utf-8')
                            state['meta_header'].append(f"# {label}: {val}")
                return True
        except Exception as e:
            with output:
                print(f"Error loading {ds_name}: {e}")
            return False

    def update_plots(change=None):
        selected_c = [c for c, cb in state['checkboxes'] if cb.value]
        
        # Update Map
        if len(selected_c) > 0:
            mask = np.isin(state['cluster_map'], selected_c)
            masked_map = np.ma.masked_where(~mask, state['cluster_map'])
            im.set_data(masked_map)
            im.set_extent([state['x_axis'][0], state['x_axis'][-1], state['y_axis'][-1], state['y_axis'][0]])
            im.set_clim(0, np.max(state['cluster_map']))
        else:
            im.set_data(np.ma.masked_all(state['cluster_map'].shape))
            
        # Update Spectrum
        if len(selected_c) > 0:
            mask_flat = np.isin(state['cluster_map'].flatten(), selected_c)
            selected_pixels = state['stack_flat'][mask_flat]
            merged_spec = np.mean(selected_pixels, axis=0) if len(selected_pixels) > 0 else np.zeros_like(state['energy'])
        else:
            merged_spec = np.zeros_like(state['energy'])
            
        line.set_data(state['energy'], merged_spec)
        ax_spec.set_xlim(state['energy'][0], state['energy'][-1])
        if np.max(merged_spec) > 0:
            ax_spec.set_ylim(min(0, np.min(merged_spec)), np.max(merged_spec) * 1.05)
        else:
            ax_spec.set_ylim(-0.1, 1.0)
            
        fig.canvas.draw_idle()

    def build_ui(ds_name):
        if not load_dataset_data(ds_name):
            return

        unique_clusters = sorted([c for c in np.unique(state['cluster_map']) if c >= 0])
        state['checkboxes'] = []
        for c in unique_clusters:
            cb = widgets.Checkbox(value=True, description=f'Cluster {c+1}', indent=False, layout=widgets.Layout(width='auto'))
            cb.observe(update_plots, names='value')
            state['checkboxes'].append((c, cb))
        
        cb_box = widgets.HBox([cb for _, cb in state['checkboxes']], layout=widgets.Layout(flex_flow='row wrap'))
        
        # Get all available datasets with clustering
        available_ds = []
        with h5py.File(h5_path, 'r') as f:
            if 'entry/pca_results' in f:
                for k in f['entry/pca_results'].keys():
                    if 'clustering' in f[f'entry/pca_results/{k}']:
                        available_ds.append(k)
        
        ds_dropdown = widgets.Dropdown(options=available_ds, value=ds_name, description='Detector:')
        
        def on_ds_change(change):
            if change['new'] != change['old']:
                build_ui(change['new'])
        
        ds_dropdown.observe(on_ds_change, names='value')
        
        save_btn = widgets.Button(description="Save Merged Spectrum", button_style='success', icon='save')
        save_btn.on_click(run_save)
        
        main_vbox.children = [
            widgets.HBox([ds_dropdown, save_btn]),
            widgets.HTML("<b>Select Clusters to Merge:</b>"),
            cb_box,
            output
        ]
        
        ax_map.set_title(f"Cluster Map: {ds_name}")
        ax_spec.set_title(f"Merged Spectrum: {ds_name}")
        update_plots()

    def run_save(b):
        selected_c = [c for c, cb in state['checkboxes'] if cb.value]
        if not selected_c:
            with output: print("Error: No clusters selected.")
            return

        mask_flat = np.isin(state['cluster_map'].flatten(), selected_c)
        merged_spec = np.mean(state['stack_flat'][mask_flat], axis=0)
        
        root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
        scan_name = os.path.splitext(os.path.basename(h5_path))[0]
        cluster_str = "_".join([str(c+1) for c in selected_c])
        default_name = f"{scan_name}_{state['current_dataset']}_merged_clusters_{cluster_str}.csv"
        save_path = filedialog.asksaveasfilename(title="Save Merged Spectrum", initialdir=os.path.dirname(h5_path), initialfile=default_name, defaultextension=".csv", parent=root)
        root.destroy()
        
        if save_path:
            try:
                df = pd.DataFrame({'Energy_eV': state['energy'], f'Merged_Clusters_{cluster_str}': merged_spec})
                header = list(state['meta_header'])
                header.insert(0, f"# Scan Name: {scan_name}")
                header.insert(1, f"# Dataset Source: {state['current_dataset']}")
                header.insert(2, f"# Merged Clusters: {[c+1 for c in selected_c]}")
                with open(save_path, 'w', newline='') as fh:
                    fh.write("\n".join(header) + "\n")
                    df.to_csv(fh, index=False, lineterminator='\n')
                with output: print(f"Saved to: {save_path}")
            except Exception as e:
                with output: print(f"Save error: {e}")

    # Initial UI build
    build_ui(dataset_name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python interactive_cluster_merger.py [h5_file_path] [dataset_name]")
    else:
        interactive_cluster_merger(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'average')

