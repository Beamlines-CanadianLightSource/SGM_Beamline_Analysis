import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import sys
import tkinter as tk
from tkinter import simpledialog

# Global cache to prevent multiple prompts during a multi-detector run
_USER_METADATA_CACHE = None

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

def get_h5_metadata(h5_path):
    """Extracts metadata from various HDF5 groups (robust search)."""
    meta = {}
    try:
        with h5py.File(h5_path, 'r') as f:
            # List of groups to search for metadata attributes
            search_groups = ['entry', 'entry/measurement', 'stack_metadata', 'scan_metadata', 'entry/xanes_measurement']
            
            for group_path in search_groups:
                if group_path in f:
                    group = f[group_path]
                    for attr in group.attrs:
                        val = group.attrs[attr]
                        if isinstance(val, bytes): val = val.decode('utf-8')
                        # Don't overwrite if we already have a better value (non-N/A)
                        if attr not in meta or meta[attr] == 'N/A':
                            meta[attr] = val
            
            # Special handling for energy regions if present as dataset
            if 'entry/measurement/Energy Regions' in f:
                 meta['Energy Regions'] = f['entry/measurement/Energy Regions'][()].decode('utf-8')

            # Search for nx/ny if missing from attributes
            if 'nx' not in meta or meta['nx'] == 'N/A':
                if 'entry/measurement/x' in f:
                    meta['nx'] = f['entry/measurement/x'].shape[0]
            if 'ny' not in meta or meta['ny'] == 'N/A':
                if 'entry/measurement/y' in f:
                    meta['ny'] = f['entry/measurement/y'].shape[0]

            if 'scan_name' not in meta or meta['scan_name'] == 'N/A':
                meta['scan_name'] = os.path.splitext(os.path.basename(h5_path))[0]
    except Exception as e:
        print(f"  [Metadata] Warning: Error during robust attribute search: {e}")
    return meta

def save_csv_with_header(csv_path, df, scan_info, full_meta=None):
    """Saves a DataFrame to CSV with a commented metadata header (matching plot_sgm_bsky_data)."""
    rows = []
    if full_meta:
        rows += [
            f"# Name: {full_meta.get('Name', 'N/A')}",
            f"# Formula: {full_meta.get('Formula', 'N/A')}",
            f"# Authors: {full_meta.get('Authors', 'N/A')}",
            f"# Affiliation: {full_meta.get('Affiliation', 'N/A')}",
            f"# Facility: CLS",
            f"# Beamline: SGM",
            f"# Mono: Spherical Grating Monochromator",
            f"# Website: https://sgm.lightsource.ca",
            f"# Element: {full_meta.get('Element', 'N/A')}",
            f"# Edge: {full_meta.get('Edge', 'N/A')}",
            f"# Preparation Method: {full_meta.get('Prep', 'N/A')}",
            f"# Calibrated To: {full_meta.get('Calib', 'N/A')}",
            f"# Temperature: {full_meta.get('Temp', 'N/A')}",
            f"# Scan Mode: {full_meta.get('Mode', 'N/A')}",
            f"# Chamber Conditions: {full_meta.get('Chamber', 'N/A')}",
            f"# Comments: {full_meta.get('Comments', 'N/A')}",
            "#"
        ]
    
    # Try to find ny/nx if available (robust)
    nx = scan_info.get('nx', 'N/A')
    ny = scan_info.get('ny', 'N/A')
    grid_str = f"{nx} x {ny}" if nx != 'N/A' else 'N/A'

    rows += [
        f"# Scan Name: {scan_info.get('scan_name', 'N/A')}",
        f"# Date: {scan_info.get('date', 'N/A')}",
        f"# Project: {scan_info.get('project', 'N/A')}",
        f"# Energy Regions: {scan_info.get('Energy Regions', 'N/A')}",
        f"# Grid: {grid_str}",
        f"# Grating: {scan_info.get('grating', 'N/A')}",
        f"# Harmonic: {scan_info.get('harmonic', 'N/A')}",
        f"# Strip: {scan_info.get('strip', 'N/A')}",
        f"# Polarization: {scan_info.get('polarization', 'N/A')}",
        f"# Exit Slit Gap: {scan_info.get('exit_slit_gap', 'N/A')}",
        "#"
    ]

    # Add Processing Metadata
    rows += [
        f"# --- Processing ---",
        f"# Normalized: {scan_info.get('normalized', 'N/A')} (I0 Source: {scan_info.get('i0_source', 'N/A')})",
        f"# Trimmed: {'Yes' if scan_info.get('x_trim') or scan_info.get('y_trim') else 'No'} (X: {scan_info.get('x_trim', 0)}, Y: {scan_info.get('y_trim', 0)})",
        f"# Roll Shift: {scan_info.get('roll_shift', 0)}",
        "#"
    ]

    # Add Column Descriptions
    rows.append("# Column 1: Energy (eV)")
    col_names = df.columns.tolist()
    for i, col in enumerate(col_names[1:], start=2):
        rows.append(f"# Column {i}: {col}")
    rows.append("#")
    
    try:
        with open(csv_path, 'w') as f:
            for row in rows:
                f.write(row + "\n")
            df.to_csv(f, index=False, header=False)
    except Exception as e:
        print(f"  [CSV Export] Error saving to {csv_path}: {e}")

def cluster_xanes_analysis(h5_path, dataset_name='average', n_clusters=4, show_plot=True, return_dict=False, use_full_metadata=False, metadata=None):
    """
    Performs K-Means clustering on the PCA scores and extracts averaged XANES spectra for each cluster.
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return None

    print(f"Loading PCA results for '{dataset_name}' from {h5_path}...")
    
    # Use provided metadata dictionary if available, otherwise read from H5
    if metadata and isinstance(metadata, dict):
        scan_info = metadata.copy()
        if 'scan_name' not in scan_info or scan_info['scan_name'] == 'N/A':
            scan_info['scan_name'] = os.path.splitext(os.path.basename(h5_path))[0]
    else:
        scan_info = get_h5_metadata(h5_path)

    global _USER_METADATA_CACHE
    full_meta = None
    if use_full_metadata:
        if _USER_METADATA_CACHE is None:
            root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
            d = MetadataDialog(root, "Clustering Metadata Input", initial_data={"Name": scan_info.get('scan_name', 'N/A')})
            if d.result: _USER_METADATA_CACHE = d.result
            root.destroy()
        full_meta = _USER_METADATA_CACHE

    try:
        with h5py.File(h5_path, 'r+') as f:
            pca_path = f"entry/pca_results/{dataset_name}"
            if pca_path not in f:
                print(f"Error: PCA results for '{dataset_name}' not found. Run pca_xanes_analysis.py first.")
                return None
            
            # Load PCA scores (eigenimages) and original stack
            eigenimages = f[f"{pca_path}/eigenimages"][()] # (ny, nx, n_components)
            stack = f[f"entry/measurement/{dataset_name}"][()] # (ny, nx, n_energies)
            energy = f["entry/measurement/energy"][()]
            x_axis = f["entry/measurement/x"][()]
            y_axis = f["entry/measurement/y"][()]
            
            ny, nx, n_comp = eigenimages.shape
            
            # 1. Prepare data for Clustering
            scores_flat = eigenimages.reshape(-1, n_comp)
            valid_mask = np.sum(np.abs(scores_flat), axis=1) > 0
            scores_valid = scores_flat[valid_mask]
            
            if scores_valid.shape[0] < n_clusters:
                print(f"Warning: Not enough valid pixels for {n_clusters} clusters in {dataset_name}. Adjusting.")
                n_clusters = max(1, scores_valid.shape[0])

            # 2. Run K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels_valid = kmeans.fit_predict(scores_valid)
            
            cluster_map_flat = np.full(ny * nx, -1, dtype=np.int32)
            cluster_map_flat[valid_mask] = labels_valid
            cluster_map = cluster_map_flat.reshape(ny, nx)

            # 3. Extract Averaged and Summed XANES
            stack_flat = stack.reshape(-1, stack.shape[-1])
            cluster_spectra = []
            cluster_sums = []
            for i in range(n_clusters):
                cluster_pixels = stack_flat[cluster_map_flat == i]
                if len(cluster_pixels) > 0:
                    cluster_spectra.append(np.mean(cluster_pixels, axis=0))
                    cluster_sums.append(np.sum(cluster_pixels, axis=0))
                else:
                    cluster_spectra.append(np.zeros(stack.shape[-1], dtype=np.float32))
                    cluster_sums.append(np.zeros(stack.shape[-1], dtype=np.float32))
            cluster_spectra = np.array(cluster_spectra)
            cluster_sums = np.array(cluster_sums)

            # 4. Save CSV
            scan_name = os.path.splitext(os.path.basename(h5_path))[0]
            output_dir = os.path.dirname(h5_path)
            csv_path = os.path.join(output_dir, f"{scan_name}_{dataset_name}_cluster_spectra_summary.csv")
            
            cols = {'Energy_eV': energy}
            for i in range(n_clusters):
                cols[f'Cluster_{i+1}_Mean'] = cluster_spectra[i, :]
                cols[f'Cluster_{i+1}_Sum'] = cluster_sums[i, :]

            df = pd.DataFrame(cols)
            save_csv_with_header(csv_path, df, scan_info, full_meta)

            # 5. Save back to HDF5
            cluster_group_path = f"entry/pca_results/{dataset_name}/clustering"
            if cluster_group_path in f:
                del f[cluster_group_path]
            
            cluster_group = f.create_group(cluster_group_path)
            cluster_group.attrs['n_clusters'] = n_clusters
            cluster_group.create_dataset('cluster_map', data=cluster_map, compression="gzip")
            cluster_group.create_dataset('cluster_spectra', data=cluster_spectra)
            cluster_group.create_dataset('cluster_sums', data=cluster_sums)
            
            print(f"    -> {dataset_name} cluster results saved.")

        if show_plot:
            plot_results(x_axis, y_axis, energy, cluster_map, cluster_spectra, dataset_name, output_dir, scan_name)
        
        results = {
            'dataset': dataset_name,
            'cluster_map': cluster_map,
            'cluster_spectra': cluster_spectra,
            'cluster_sums': cluster_sums,
            'energy': energy,
            'x': x_axis,
            'y': y_axis
        }
        
        return results if return_dict else h5_path

    except Exception as e:
        print(f"An error occurred during clustering on {dataset_name}: {e}")
        return None

def run_clustering_all_detectors(h5_path, n_clusters=4, use_full_metadata=False, metadata=None):
    """
    Performs K-Means clustering on sdd1-4 and average, then plots comparison.
    """
    print(f"\n{'='*60}\nRunning Multi-Detector Clustering Analysis\n{'='*60}")
    
    datasets = ['sdd1', 'sdd2', 'sdd3', 'sdd4', 'average']
    all_results = []
    
    for ds in datasets:
        res = cluster_xanes_analysis(h5_path, dataset_name=ds, n_clusters=n_clusters, show_plot=False, return_dict=True, use_full_metadata=use_full_metadata, metadata=metadata)
        if res:
            all_results.append(res)
            
    if not all_results:
        print("Error: No datasets were successfully clustered.")
        return

    plot_multi_cluster_results(all_results, h5_path)
    
    # NEW: Save combined cluster sums for each detector
    save_combined_cluster_sums(all_results, h5_path, use_full_metadata=use_full_metadata, metadata=metadata)
    
    return h5_path

def _display_scrollable_figure(fig):
    """
    Attempts to display a matplotlib figure with a horizontal scrollbar in Jupyter Notebooks.
    If not in Jupyter or if an error occurs, it leaves the figure open for standard plt.show().
    """
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display, HTML
            import io
            import base64
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            
            html = f'<div style="width: 100%; overflow-x: auto; white-space: nowrap;"><img src="data:image/png;base64,{img_b64}" style="max-width: none; margin: 10px 0; border: 1px solid #ccc;"/></div>'
            display(HTML(html))
            plt.close(fig)
            return True
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not display scrollable figure: {e}")
    return False

def plot_multi_cluster_results(all_results, h5_path):
    """
    Plots cluster maps and spectra for all detectors side-by-side.
    """
    n_det = len(all_results)
    n_clusters = all_results[0]['cluster_spectra'].shape[0]
    
    x_axis = all_results[0]['x']
    y_axis = all_results[0]['y']
    energy = all_results[0]['energy']
    
    scan_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_dir = os.path.dirname(h5_path)
    
    # --- Figure 1: Cluster Maps ---
    fig_map, axes_map = plt.subplots(1, n_det, figsize=(2.8*n_det, 3.5), squeeze=False)
    fig_map.suptitle(f"Multi-Detector Cluster Maps: {scan_name}", fontsize=16)
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    for i, res in enumerate(all_results):
        ax = axes_map[0, i]
        masked_map = np.ma.masked_where(res['cluster_map'] == -1, res['cluster_map'])
        ax.imshow(masked_map, extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]], cmap=cmap, interpolation='nearest')
        ax.set_title(f"{res['dataset']}")
        ax.set_xlabel("X (mm)")
        if i == 0: ax.set_ylabel("Y (mm)")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    map_plot_path = os.path.join(output_dir, f"{scan_name}_cluster_comparison_maps.png")
    plt.savefig(map_plot_path, dpi=150)

    # --- Figure 2: Cluster Spectra ---
    fig_spec, axes_spec = plt.subplots(1, n_det, figsize=(2.8*n_det, 3.5), squeeze=False)
    fig_spec.suptitle(f"Multi-Detector Cluster Spectra: {scan_name}", fontsize=16)
    
    for i, res in enumerate(all_results):
        ax = axes_spec[0, i]
        spectra = res['cluster_spectra']
        for k in range(n_clusters):
            ax.plot(energy, spectra[k, :], color=cmap(k), linewidth=1.5, label=f"C{k+1}")
        ax.set_title(f"{res['dataset']}")
        ax.set_xlabel("Energy (eV)")
        if i == 0: ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        if i == n_det - 1:
            ax.legend(loc='upper right', fontsize='x-small')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    spec_plot_path = os.path.join(output_dir, f"{scan_name}_cluster_comparison_spectra.png")
    plt.savefig(spec_plot_path, dpi=150)
    
    print(f"\nMulti-detector clustering plots saved to:\n  -> {map_plot_path}\n  -> {spec_plot_path}")
    
    scrolled_map = _display_scrollable_figure(fig_map)
    scrolled_spec = _display_scrollable_figure(fig_spec)
    
    if not (scrolled_map or scrolled_spec):
        plt.show()

def save_combined_cluster_sums(all_results, h5_path, use_full_metadata=False, metadata=None):
    """
    Saves a master CSV containing summed spectra for all detectors and clusters.
    """
    scan_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_dir = os.path.dirname(h5_path)
    csv_path = os.path.join(output_dir, f"{scan_name}_all_detectors_cluster_sums.csv")
    
    energy = all_results[0]['energy']
    cols = {'Energy_eV': energy}
    
    for res in all_results:
        ds_name = res['dataset']
        sums = res['cluster_sums']
        n_clusters = sums.shape[0]
        for k in range(n_clusters):
            cols[f'{ds_name}_cluster_{k+1}_sum'] = sums[k, :]
            
    df = pd.DataFrame(cols)
    
    # Use provided metadata dictionary if available, otherwise read from H5
    if metadata and isinstance(metadata, dict):
        scan_info = metadata.copy()
        if 'scan_name' not in scan_info or scan_info['scan_name'] == 'N/A':
            scan_info['scan_name'] = os.path.splitext(os.path.basename(h5_path))[0]
    else:
        scan_info = get_h5_metadata(h5_path)

    full_meta = _USER_METADATA_CACHE if use_full_metadata else None
    save_csv_with_header(csv_path, df, scan_info, full_meta)
    
    print(f"    -> Combined cluster sums saved to: {csv_path}")

def plot_results(x_coords, y_coords, energy, cluster_map, spectra, dataset_name, output_dir, scan_name):
    """
    Plots the cluster map and the averaged XANES spectra for a single dataset.
    """
    n_clusters = spectra.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    fig.suptitle(f"K-Means Cluster Analysis: {dataset_name}", fontsize=18)

    cmap = plt.cm.get_cmap('tab10', n_clusters)
    masked_map = np.ma.masked_where(cluster_map == -1, cluster_map)
    
    im = ax1.imshow(masked_map, extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]], 
                    cmap=cmap, interpolation='nearest')
    
    cbar = fig.colorbar(im, ax=ax1, ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"{i+1}" for i in range(n_clusters)])
    cbar.set_label('Cluster ID')
    
    ax1.set_title(f"Cluster Map (k={n_clusters})")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")

    for i in range(n_clusters):
        ax2.plot(energy, spectra[i, :], color=cmap(i), linewidth=2, label=f"Cluster {i+1}")

    ax2.set_title("Cluster-Averaged XANES Spectra")
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("Normalized Intensity")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    plot_path = os.path.join(output_dir, f"{scan_name}_{dataset_name}_cluster_preview.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    -> Preview plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cluster_xanes_analysis.py [h5_file_path] [dataset_name] [n_clusters]")
        print("Example: python cluster_xanes_analysis.py my_stack.h5 average 4")
        print("To run for all detectors: python cluster_xanes_analysis.py my_stack.h5 all 4")
    else:
        h5_file = sys.argv[1]
        ds_name = sys.argv[2] if len(sys.argv) > 2 else 'average'
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        
        if ds_name.lower() == 'all':
            run_clustering_all_detectors(h5_file, n_clusters=k)
        else:
            cluster_xanes_analysis(h5_file, ds_name, k)
