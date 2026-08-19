import os
import sys

# Prevent OpenMP duplicate runtime warnings from threadpoolctl / scikit-learn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*OpenMP.*")

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import simpledialog
from alignment_utils import format_num_val, safe_metadata_dialog_call

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
            ("Calibration Reference", "CalibRef"),
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

            # Ensure normalized and i0_source are properly populated
            if 'i0_source' not in meta or not meta['i0_source']:
                meta['i0_source'] = 'Internal (mcc1 / Au Mesh)'
            
            s = str(meta['i0_source'])
            if "mcc1" in s.lower() and "au mesh" not in s.lower():
                meta['i0_source'] = s.replace("mcc1", "mcc1 (Au Mesh)")
            elif s.lower() in ["internal", "mcc1"] and "au mesh" not in s.lower():
                meta['i0_source'] = "Internal (mcc1 / Au Mesh)"

            if meta['i0_source'] in ['None', 'Unknown']:
                meta['normalized'] = 'No'
            elif 'normalized' not in meta:
                meta['normalized'] = 'Yes'
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
            f"# Facility: Canadian Light Source (CLS)",
            f"# Beamline: Spherical Grating Monochromator (SGM) (11ID-1)",
            f"# Mono: Spherical Grating Monochromator",
            f"# Website: https://sgm.lightsource.ca",
            f"# Element: {full_meta.get('Element', 'N/A')}",
            f"# Edge: {full_meta.get('Edge', 'N/A')}",
            f"# Preparation Method: {full_meta.get('Prep', 'N/A')}",
            f"# Calibrated To: {full_meta.get('Calib', 'N/A')}",
            f"# Calibration Reference: {full_meta.get('CalibRef', 'N/A')}",
            f"# Temperature: {full_meta.get('Temp', 'N/A')}",
            f"# Scan Mode: {full_meta.get('Mode', 'N/A')}",
            f"# Chamber Conditions: {full_meta.get('Chamber', 'N/A')}",
            f"# Comments: {full_meta.get('Comments', 'N/A')}",
            "#"
        ]
    
    nx = scan_info.get('nx', 'N/A')
    ny = scan_info.get('ny', 'N/A')
    pts_str = f" ({nx * ny} points)" if isinstance(nx, (int, float, np.number)) and isinstance(ny, (int, float, np.number)) else ""
    grid_str = f"{nx} x {ny}{pts_str}" if nx != 'N/A' else 'N/A'

    rows += [
        f"# Scan Name: {scan_info.get('scan_name', 'N/A')}",
        f"# Scan Type: {scan_info.get('scan_type', 'N/A')}",
        f"# Date: {scan_info.get('date', 'N/A')}",
        f"# Project: {scan_info.get('project', 'N/A')}",
        f"# Energy Regions: {scan_info.get('Energy Regions', 'N/A')}",
        f"# Grid Dimensions: {grid_str}",
        f"# Grating: {scan_info.get('grating', 'N/A')}",
        f"# Harmonic: {scan_info.get('harmonic', 'N/A')}",
        f"# Strip: {scan_info.get('strip', 'N/A')}",
        f"# Polarization: {scan_info.get('polarization', 'N/A')}",
        f"# Exit Slit Gap: {format_num_val(scan_info.get('exit_slit_gap'))}",
        f"# XPS Z: {format_num_val(scan_info.get('xps_z'))}",
    ]
    t_per_img = scan_info.get('time_per_map') or scan_info.get('time_per_image')
    if t_per_img and str(t_per_img).strip() not in ('N/A', 'None', ''):
        rows.append(f"# Time Per Image: {t_per_img}")
    rows += [
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

from pca_xanes_analysis import resolve_pca_h5_path

def cluster_xanes_analysis(h5_path, dataset_name='average', n_clusters=4, show_plot=True, return_dict=False, use_full_metadata=False, metadata=None,
                           apply_waterfall=False, waterfall_offset=None, energy_range=None, figsize=(11, 5)):
    """
    Performs K-Means clustering on the PCA scores and extracts averaged XANES spectra for each cluster.
    """
    h5_path = resolve_pca_h5_path(h5_path)
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return None

    # Use provided metadata dictionary if available, otherwise read from H5
    if metadata and isinstance(metadata, dict):
        scan_info = metadata.copy()
        if 'scan_name' not in scan_info or scan_info['scan_name'] == 'N/A':
            scan_info['scan_name'] = os.path.splitext(os.path.basename(h5_path))[0]
    else:
        scan_info = get_h5_metadata(h5_path)

    i0_source = scan_info.get('i0_source', 'Internal (mcc1)')
    norm_status = scan_info.get('normalized', 'Yes')
    i0_info = f"Yes (I0 Source: {i0_source})" if norm_status == 'Yes' else "No (Raw Intensity)"

    print(f"Loading PCA results for '{dataset_name}' from {h5_path}...")
    print(f"    [NORMALIZATION] Data is Normalized: {i0_info}")
    
    global _USER_METADATA_CACHE
    full_meta = None
    if use_full_metadata:
        if _USER_METADATA_CACHE is None:
            res = safe_metadata_dialog_call(initial_data={"Name": scan_info.get('scan_name', 'N/A')})
            if res: _USER_METADATA_CACHE = res
        full_meta = _USER_METADATA_CACHE

    try:
        with h5py.File(h5_path, 'r+') as f:
            pca_path = f"entry/pca_results/{dataset_name}"
            if pca_path not in f:
                print(f"Error: PCA results for '{dataset_name}' not found. Run pca_xanes_analysis.py first.")
                return None
            
            # Load PCA scores (eigenimages) and original stack
            eigenimages = f[f"{pca_path}/eigenimages"][()] # (ny, nx, n_components)
            meas = f['entry/measurement'] if 'entry/measurement' in f else f['measurement']
            stack = meas[dataset_name][()] # (ny, nx, n_energies)
            energy = meas["energy"][()]
            x_axis = meas["x"][()]
            y_axis = meas["y"][()]
            
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
            ipfy_mode = bool(meas.attrs.get('ipfy_mode', False))
            if ipfy_mode:
                print("    [IPFY] IPFY Mode detected in HDF5 metadata. Saving both Original and PFY (inverted) spectra.")
            else:
                print("    [IPFY] Standard PFY Mode (no inversion).")

            stack_flat = stack.reshape(-1, stack.shape[-1])
            cluster_spectra_orig = []
            cluster_spectra_pfy = []
            cluster_sums = []
            for i in range(n_clusters):
                cluster_pixels = stack_flat[cluster_map_flat == i]
                if len(cluster_pixels) > 0:
                    mean_orig = np.mean(cluster_pixels, axis=0)
                    cluster_spectra_orig.append(mean_orig)
                    pfy_spec = -mean_orig if ipfy_mode else mean_orig
                    if ipfy_mode:
                        pfy_spec = pfy_spec + (np.abs(np.min(pfy_spec)) + 500)
                    cluster_spectra_pfy.append(pfy_spec)
                    cluster_sums.append(np.sum(cluster_pixels, axis=0))
                else:
                    cluster_spectra_orig.append(np.zeros(stack.shape[-1], dtype=np.float32))
                    cluster_spectra_pfy.append(np.zeros(stack.shape[-1], dtype=np.float32))
                    cluster_sums.append(np.zeros(stack.shape[-1], dtype=np.float32))
            # Prepare for saving
            cluster_spectra_orig = np.array(cluster_spectra_orig)
            cluster_spectra_pfy = np.array(cluster_spectra_pfy)
            cluster_sums = np.array(cluster_sums)

            # 4. Save CSV
            scan_name = os.path.splitext(os.path.basename(h5_path))[0]
            output_dir = os.path.dirname(h5_path)
            csv_path = os.path.join(output_dir, f"{scan_name}_{dataset_name}_cluster_spectra_summary.csv")
            
            cols = {'Energy_eV': energy}
            for i in range(n_clusters):
                if ipfy_mode:
                    cols[f'Cluster_{i+1}_Mean_Original'] = cluster_spectra_orig[i]
                    cols[f'Cluster_{i+1}_Mean_PFY'] = cluster_spectra_pfy[i]
                else:
                    cols[f'Cluster_{i+1}_Mean'] = cluster_spectra_orig[i, :]
                cols[f'Cluster_{i+1}_Sum'] = cluster_sums[i, :]

            df = pd.DataFrame(cols)
            save_csv_with_header(csv_path, df, scan_info, full_meta)

            # 5. Save back to HDF5
            cluster_group_path = f"entry/pca_results/{dataset_name}/clustering"
            if cluster_group_path in f:
                del f[cluster_group_path]
            
            cluster_group = f.create_group(cluster_group_path)
            cluster_group.attrs['n_clusters'] = n_clusters
            cluster_group.attrs['i0_source'] = i0_source
            cluster_group.attrs['normalized_info'] = i0_info
            cluster_group.create_dataset('cluster_map', data=cluster_map, compression="gzip")
            cluster_group.create_dataset('cluster_spectra', data=cluster_spectra_orig)
            if ipfy_mode:
                cluster_group.create_dataset('cluster_spectra_pfy', data=cluster_spectra_pfy)
            cluster_group.create_dataset('cluster_sums', data=cluster_sums)
            
            print(f"    -> {dataset_name} cluster results saved.")

        if show_plot:
            # For plotting and interactive use, we use the PFY (peaks up) version
            plot_results(x_axis, y_axis, energy, cluster_map, cluster_spectra_pfy, dataset_name, output_dir, scan_name, i0_info=i0_info,
                         apply_waterfall=apply_waterfall, waterfall_offset=waterfall_offset, energy_range=energy_range, figsize=figsize)
        
        results = {
            'dataset': dataset_name,
            'cluster_map': cluster_map,
            'cluster_spectra': cluster_spectra_pfy,
            'cluster_sums': cluster_sums,
            'energy': energy,
            'x': x_axis,
            'y': y_axis,
            'i0_info': i0_info
        }
        
        return results if return_dict else h5_path

    except Exception as e:
        print(f"An error occurred during clustering on {dataset_name}: {e}")
        return None

def run_clustering_all_detectors(h5_path, n_clusters=4, use_full_metadata=False, metadata=None,
                                 show_individual_plots=True, apply_waterfall=False, waterfall_offset=None,
                                 energy_range=None, figsize=(11, 5)):
    """
    Performs K-Means clustering on sdd1-4 and average, then plots individual and comparison results.
    """
    h5_path = resolve_pca_h5_path(h5_path)
    scan_info = get_h5_metadata(h5_path)
    i0_source = scan_info.get('i0_source', 'Internal (mcc1)')
    norm_status = scan_info.get('normalized', 'Yes')
    i0_info = f"Yes (I0 Source: {i0_source})" if norm_status == 'Yes' else "No (Raw Intensity)"

    print(f"\n{'='*60}\nRunning Multi-Detector Clustering Analysis\n{'='*60}")
    print(f"Dataset File: {os.path.basename(h5_path)}")
    print(f"Data Normalization: {i0_info}")
    
    datasets = ['sdd1', 'sdd2', 'sdd3', 'sdd4', 'average']
    try:
        with h5py.File(h5_path, 'r') as f:
            for grp in ['entry/measurement', 'measurement', 'entry']:
                if grp in f and 'selected_average' in f[grp]:
                    datasets.append('selected_average')
                    print(f"  [Notice] Found 'selected_average' dataset in HDF5 stack. Including in Clustering analysis.")
                    break
    except Exception as _e_sel:
        pass

    all_results = []
    
    for ds in datasets:
        res = cluster_xanes_analysis(
            h5_path, dataset_name=ds, n_clusters=n_clusters,
            show_plot=show_individual_plots, return_dict=True,
            use_full_metadata=use_full_metadata, metadata=metadata,
            apply_waterfall=apply_waterfall, waterfall_offset=waterfall_offset,
            energy_range=energy_range, figsize=figsize
        )
        if res:
            all_results.append(res)
            
    if not all_results:
        print("Error: No datasets were successfully clustered.")
        return

    plot_multi_cluster_results(all_results, h5_path, i0_info=i0_info,
                               apply_waterfall=apply_waterfall, waterfall_offset=waterfall_offset,
                               energy_range=energy_range)
    
    # Save combined cluster sums for each detector
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

def plot_multi_cluster_results(all_results, h5_path, i0_info=None, apply_waterfall=False, waterfall_offset=None, energy_range=None, figsize=None):
    """
    Plots cluster maps and spectra for all detectors side-by-side.
    """
    if i0_info is None:
        scan_info = get_h5_metadata(h5_path)
        i0_source = scan_info.get('i0_source', 'Internal (mcc1)')
        norm_status = scan_info.get('normalized', 'Yes')
        i0_info = f"Yes (I0 Source: {i0_source})" if norm_status == 'Yes' else "No (Raw Intensity)"

    n_det = len(all_results)
    n_clusters = all_results[0]['cluster_spectra'].shape[0]
    
    x_axis = all_results[0]['x']
    y_axis = all_results[0]['y']
    energy = all_results[0]['energy']
    
    scan_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_dir = os.path.dirname(h5_path)
    
    # --- Figure 1: Cluster Maps ---
    map_figsize = (3.2*n_det, 4.0) if figsize is None else (figsize[0], figsize[1]*0.8)
    fig_map, axes_map = plt.subplots(1, n_det, figsize=map_figsize, squeeze=False)
    fig_map.suptitle(f"Multi-Detector Cluster Maps: {scan_name}\n[Normalized: {i0_info}]", fontsize=14, fontweight='semibold')
    try:
        cmap = plt.colormaps['tab10'].resampled(n_clusters)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    for i, res in enumerate(all_results):
        ax = axes_map[0, i]
        masked_map = np.ma.masked_where(res['cluster_map'] == -1, res['cluster_map'])
        ax.imshow(masked_map, extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]], cmap=cmap, interpolation='nearest')
        ax.set_title(f"{res['dataset']}", fontweight='semibold')
        ax.set_xlabel("X (mm)")
        if i == 0: ax.set_ylabel("Y (mm)")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    map_plot_path = os.path.join(output_dir, f"{scan_name}_cluster_comparison_maps.png")
    plt.savefig(map_plot_path, dpi=150, bbox_inches='tight')

    # --- Figure 2: Cluster Spectra ---
    spec_figsize = (3.8*n_det, 5.5) if figsize is None else figsize
    fig_spec, axes_spec = plt.subplots(1, n_det, figsize=spec_figsize, squeeze=False)
    fig_spec.suptitle(f"Multi-Detector Cluster Spectra: {scan_name}\n[Normalized: {i0_info}]", fontsize=14, fontweight='semibold')
    
    for i, res in enumerate(all_results):
        ax = axes_spec[0, i]
        spectra = res['cluster_spectra']

        if apply_waterfall:
            if waterfall_offset is None or waterfall_offset <= 0:
                spec_range = np.nanmax(spectra) - np.nanmin(spectra)
                offset_step = spec_range * 0.4 if spec_range > 0 else 1.0
            else:
                offset_step = waterfall_offset
        else:
            offset_step = 0.0

        for k in range(n_clusters):
            y_val = spectra[k, :] + (k * offset_step)
            lbl = f"C{k+1}" + (f" (+{k*offset_step:.1f})" if apply_waterfall and k > 0 else "")
            ax.plot(energy, y_val, color=cmap(k), linewidth=1.5, label=lbl)

        ax.set_title(f"{res['dataset']}", fontweight='semibold')
        ax.set_xlabel("Energy (eV)")
        if i == 0: ax.set_ylabel("Intensity" + (" (Waterfall Offset)" if apply_waterfall else ""))
        ax.grid(True, alpha=0.4, linestyle='--')
        if energy_range is not None and len(energy_range) == 2:
            ax.set_xlim(energy_range[0], energy_range[1])
        if i == n_det - 1:
            ax.legend(loc='upper right', fontsize='x-small', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    spec_plot_path = os.path.join(output_dir, f"{scan_name}_cluster_comparison_spectra.png")
    plt.savefig(spec_plot_path, dpi=150, bbox_inches='tight')
    
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

def plot_results(x_coords, y_coords, energy, cluster_map, spectra, dataset_name, output_dir, scan_name, i0_info=None,
                 apply_waterfall=False, waterfall_offset=None, energy_range=None, figsize=(11, 5)):
    """
    Plots the cluster map and the averaged XANES spectra for a single dataset.
    """
    if i0_info is None:
        i0_info = "Yes"

    n_clusters = spectra.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    fig.suptitle(f"K-Means Cluster Analysis: {dataset_name}\n[Normalized: {i0_info}]", fontsize=14, fontweight='semibold')

    try:
        cmap = plt.colormaps['tab10'].resampled(n_clusters)
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab10', n_clusters)
    masked_map = np.ma.masked_where(cluster_map == -1, cluster_map)
    
    im = ax1.imshow(masked_map, extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]], 
                    cmap=cmap, interpolation='nearest')
    
    cbar = fig.colorbar(im, ax=ax1, ticks=range(n_clusters))
    cbar.ax.set_yticklabels([f"{i+1}" for i in range(n_clusters)])
    cbar.set_label('Cluster ID')
    
    ax1.set_title(f"Cluster Map ({dataset_name}, k={n_clusters})", fontsize=11, fontweight='semibold')
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")

    # Calculate Waterfall offset if requested
    if apply_waterfall:
        if waterfall_offset is None or waterfall_offset <= 0:
            spec_range = np.nanmax(spectra) - np.nanmin(spectra)
            offset_step = spec_range * 0.4 if spec_range > 0 else 1.0
        else:
            offset_step = waterfall_offset
    else:
        offset_step = 0.0

    for i in range(n_clusters):
        y_val = spectra[i, :] + (i * offset_step)
        lbl = f"Cluster {i+1}" + (f" (+{i*offset_step:.2f})" if apply_waterfall and i > 0 else "")
        ax2.plot(energy, y_val, color=cmap(i), linewidth=1.8, label=lbl)

    ax2.set_title(f"Cluster-Averaged XANES Spectra ({dataset_name})", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Energy (eV)")
    ax2.set_ylabel("Intensity" + (" (Waterfall Offset)" if apply_waterfall else ""))
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    if energy_range is not None and len(energy_range) == 2:
        ax2.set_xlim(energy_range[0], energy_range[1])

    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', frameon=True)

    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    
    plot_path = os.path.join(output_dir, f"{scan_name}_{dataset_name}_cluster_preview.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    -> Preview plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Perform K-Means clustering on PCA XANES stack scan data.")
    parser.add_argument("h5_file_path", help="Path to the HDF5 file.")
    parser.add_argument("dataset_name", nargs="?", default="all", help="Dataset name ('average', 'sdd1'..'sdd4', or 'all'). Default is 'all'.")
    parser.add_argument("n_clusters", nargs="?", type=int, default=4, help="Number of clusters for K-Means.")
    parser.add_argument("--waterfall", action="store_true", help="Apply vertical waterfall offset between cluster spectra.")
    parser.add_argument("--offset", type=float, default=None, help="Custom waterfall offset value.")
    parser.add_argument("--energy-min", type=float, default=None, help="Minimum energy bound for zooming plot.")
    parser.add_argument("--energy-max", type=float, default=None, help="Maximum energy bound for zooming plot.")
    parser.add_argument("--no-individual", action="store_true", help="Suppress individual per-detector plots in multi-detector run.")
    
    args = parser.parse_args()
    
    energy_range = None
    if args.energy_min is not None and args.energy_max is not None:
        energy_range = (args.energy_min, args.energy_max)

    if args.dataset_name.lower() == 'all':
        run_clustering_all_detectors(
            args.h5_file_path, n_clusters=args.n_clusters,
            show_individual_plots=not args.no_individual,
            apply_waterfall=args.waterfall, waterfall_offset=args.offset,
            energy_range=energy_range
        )
    else:
        cluster_xanes_analysis(
            args.h5_file_path, dataset_name=args.dataset_name, n_clusters=args.n_clusters,
            apply_waterfall=args.waterfall, waterfall_offset=args.offset,
            energy_range=energy_range
        )
