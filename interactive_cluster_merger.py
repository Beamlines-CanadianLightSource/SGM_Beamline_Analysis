import os
import sys

# Prevent OpenMP duplicate runtime warnings from threadpoolctl / scikit-learn
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*OpenMP.*")
warnings.filterwarnings("ignore", module="threadpoolctl")

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

try:
    import tkinter as tk
    from tkinter import messagebox, filedialog
except (ImportError, ModuleNotFoundError):
    tk = None
    messagebox = None
    filedialog = None

from alignment_utils import safe_filedialog_call, format_num_val, get_safe_save_path, HAS_GUI
from pca_xanes_analysis import resolve_pca_h5_path
from analyze_sgm_bsky_data import get_sdd_label
from cluster_xanes_analysis import get_h5_metadata

def interactive_cluster_merger(h5_path, dataset_name='average'):
    """
    Provides an interactive Jupyter widget to select specific clusters and view/extract 
    their combined averaged XANES spectrum. Supports switching between different detectors,
    select all/clear all, non-blocking rendering, and overwrite-safe CSV saving.
    """
    if isinstance(h5_path, dict):
        if 'h5_path' in h5_path:
            h5_path = h5_path['h5_path']
        elif 'dataset' in h5_path and dataset_name == 'average':
            dataset_name = h5_path.get('dataset', 'average')

    h5_path = resolve_pca_h5_path(h5_path)
    if not h5_path or not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return None

    # Retrieve all available datasets that have clustering results
    available_ds = []
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'entry/pca_results' in f:
                for k in f['entry/pca_results'].keys():
                    if 'clustering' in f[f'entry/pca_results/{k}']:
                        available_ds.append(k)
    except Exception as e:
        print(f"Error inspecting {h5_path}: {e}")
        return None

    if not available_ds:
        print(f"Error: No clustering results found in {h5_path}. Run run_clustering_all_detectors first.")
        return None

    if dataset_name not in available_ds:
        dataset_name = available_ds[0]

    # --- UI Containers ---
    plot_out = widgets.Output()
    status_out = widgets.Output()
    cb_box = widgets.HBox([], layout=widgets.Layout(flex_flow='row wrap', margin='6px 0'))
    title_html = widgets.HTML()

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
        'meta_header': [],
        'ipfy_mode': False,
        'i0_info': 'Yes'
    }

    try:
        cmap = plt.colormaps['tab10']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab10')

    def load_dataset_data(ds_name):
        try:
            with h5py.File(h5_path, 'r') as f:
                pca_path = f"entry/pca_results/{ds_name}/clustering"
                if pca_path not in f:
                    with status_out:
                        status_out.clear_output()
                        print(f"Error: Clustering results for '{ds_name}' not found.")
                    return False
                
                state['current_dataset'] = ds_name
                state['cluster_map'] = f[f"{pca_path}/cluster_map"][()]
                
                meas = f['entry/measurement'] if 'entry/measurement' in f else f.get('measurement', None)
                if meas is None or ds_name not in meas:
                    with status_out:
                        status_out.clear_output()
                        print(f"Error: Measurement data for '{ds_name}' not found.")
                    return False

                stack = meas[ds_name][()]
                state['energy'] = meas["energy"][()]
                state['x_axis'] = meas["x"][()]
                state['y_axis'] = meas["y"][()]

                # IPFY Mode detection
                ipfy_val = meas.attrs.get('ipfy_mode', False)
                if isinstance(ipfy_val, (np.bool_, bool)):
                    state['ipfy_mode'] = bool(ipfy_val)
                elif isinstance(ipfy_val, (int, np.integer)):
                    state['ipfy_mode'] = bool(ipfy_val != 0)
                elif isinstance(ipfy_val, str):
                    state['ipfy_mode'] = (ipfy_val.strip().lower() == 'true')
                else:
                    state['ipfy_mode'] = False

                ny, nx, n_en = stack.shape
                state['stack_flat'] = stack.reshape(ny * nx, -1)

                # Metadata & Normalization
                meta = get_h5_metadata(h5_path)
                state['i0_info'] = f"Yes (I0 Source: {meta.get('i0_source', 'Unknown')})" if meta.get('normalized') == 'Yes' else "No (Raw Intensity)"
                
                state['meta_header'] = [
                    f"# Facility: Canadian Light Source (CLS)",
                    f"# Beamline: Spherical Grating Monochromator (SGM) (11ID-1)"
                ]
                meta_labels = [
                    ('scan_name', 'Scan Name'), ('date', 'Date'), ('project', 'Project'),
                    ('exit_slit_gap', 'Exit Slit Gap'), ('xps_z', 'XPS Z'),
                    ('time_per_image', 'Time Per Image'), ('i0_source', 'I0 Source')
                ]
                for key, label in meta_labels:
                    if key in meta and meta[key] and str(meta[key]).strip() not in ('N/A', 'None', ''):
                        val = meta[key]
                        if key in ('exit_slit_gap', 'xps_z'):
                            val = format_num_val(val)
                        state['meta_header'].append(f"# {label}: {val}")

                return True
        except Exception as e:
            with status_out:
                status_out.clear_output()
                print(f"Error loading {ds_name}: {e}")
            return False

    def update_plots(change=None):
        selected_c = [c for c, cb in state['checkboxes'] if cb.value]
        
        with plot_out:
            plot_out.clear_output(wait=True)
            # Reduced figure size by 30% for compact notebook presentation
            fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=(7.7, 3.1))
            
            ds_label = get_sdd_label(state['current_dataset'], h5_path)
            ipfy = state.get('ipfy_mode', False)

            # --- Left: Cluster Map ---
            ax_map.set_facecolor('#e4e4e4')
            # Render unselected/all clusters as soft gray backdrop
            valid_mask = (state['cluster_map'] >= 0)
            if np.any(valid_mask):
                gray_map = np.ma.masked_where(~valid_mask, state['cluster_map'])
                ax_map.imshow(gray_map, cmap='Greys', alpha=0.18, interpolation='nearest',
                              extent=[state['x_axis'][0], state['x_axis'][-1], state['y_axis'][-1], state['y_axis'][0]])

            if selected_c:
                display_map = np.full_like(state['cluster_map'], fill_value=np.nan, dtype=float)
                for c in selected_c:
                    display_map[state['cluster_map'] == c] = c
                masked_map = np.ma.masked_invalid(display_map)
                ax_map.imshow(masked_map, cmap=cmap, vmin=0, vmax=max(9, np.max(state['cluster_map'])),
                              interpolation='nearest',
                              extent=[state['x_axis'][0], state['x_axis'][-1], state['y_axis'][-1], state['y_axis'][0]])

            sel_names = ", ".join([f"C{c+1}" for c in selected_c]) if selected_c else "None"
            ax_map.set_title(f"Cluster Map: {ds_label}\nSelected: [{sel_names}]", fontsize=9.5, fontweight='bold')
            ax_map.set_xlabel("X (mm)")
            ax_map.set_ylabel("Y (mm)")

            # --- Right: Merged Spectrum ---
            offset = 0.0
            if selected_c:
                # Plot individual selected cluster spectra as thin dashed curves
                for c in selected_c:
                    c_mask = (state['cluster_map'] == c).flatten()
                    c_pixels = state['stack_flat'][c_mask]
                    if len(c_pixels) > 0:
                        c_spec = np.mean(c_pixels, axis=0)
                        if ipfy:
                            c_spec_pfy = -c_spec
                            if offset == 0.0:
                                offset = np.abs(np.min(c_spec_pfy)) + 500
                            c_spec = c_spec_pfy + offset
                        ax_spec.plot(state['energy'], c_spec, linestyle='--', alpha=0.45, linewidth=1.2,
                                     color=cmap(c % 10), label=f"Cluster {c+1}")

                # Plot merged average in thick black
                mask_flat = np.isin(state['cluster_map'].flatten(), selected_c)
                selected_pixels = state['stack_flat'][mask_flat]
                if len(selected_pixels) > 0:
                    merged_spec = np.mean(selected_pixels, axis=0)
                    if ipfy:
                        merged_spec = (-merged_spec) + offset
                    ax_spec.plot(state['energy'], merged_spec, color='black', linewidth=2.2,
                                 label=f"Merged Average ({len(selected_c)} clusters)")
                    
                    y_min, y_max = np.min(merged_spec), np.max(merged_spec)
                    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.5
                    ax_spec.set_ylim(y_min - y_pad, y_max + y_pad)
                    ax_spec.legend(loc='best', fontsize='x-small', frameon=True)
            else:
                ax_spec.text(0.5, 0.5, "No clusters selected\nCheck one or more cluster boxes above",
                             transform=ax_spec.transAxes, ha='center', va='center', color='#777', fontsize=10)
                ax_spec.set_ylim(-0.1, 1.0)

            ax_spec.set_xlim(state['energy'][0], state['energy'][-1])
            y_label = "Intensity (PFY - Peaks Up)" if ipfy else "Intensity"
            ax_spec.set_ylabel(y_label)
            ax_spec.set_xlabel("Energy (eV)")
            ax_spec.set_title(f"Merged Spectrum: {ds_label}\n[Normalized: {state.get('i0_info', 'Yes')}]", fontsize=9.5, fontweight='bold')
            ax_spec.grid(True, linestyle='--', alpha=0.5)
            ax_spec.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def on_select_all(b):
        for _, cb in state['checkboxes']:
            cb.unobserve(update_plots, names='value')
            cb.value = True
            cb.observe(update_plots, names='value')
        update_plots()

    def on_deselect_all(b):
        for _, cb in state['checkboxes']:
            cb.unobserve(update_plots, names='value')
            cb.value = False
            cb.observe(update_plots, names='value')
        update_plots()

    def run_save(b):
        selected_c = [c for c, cb in state['checkboxes'] if cb.value]
        if not selected_c:
            with status_out:
                status_out.clear_output()
                print("Error: No clusters selected to save.")
            return

        mask_flat = np.isin(state['cluster_map'].flatten(), selected_c)
        merged_spec_orig = np.mean(state['stack_flat'][mask_flat], axis=0)
        
        cluster_str = "_".join([str(c+1) for c in selected_c])
        scan_name = os.path.splitext(os.path.basename(h5_path))[0]
        
        ipfy = state.get('ipfy_mode', False)
        data_to_save = {'Energy_eV': state['energy']}
        
        if ipfy:
            data_to_save[f'Merged_Clusters_{cluster_str}_Original'] = merged_spec_orig
            pfy_spec = -merged_spec_orig
            offset = np.abs(np.min(pfy_spec)) + 500
            data_to_save[f'Merged_Clusters_{cluster_str}_PFY'] = pfy_spec + offset
        else:
            data_to_save[f'Merged_Clusters_{cluster_str}'] = merged_spec_orig
        
        default_name = f"{scan_name}_{state['current_dataset']}_merged_clusters_{cluster_str}.csv"
        save_dir = os.path.dirname(h5_path)
        
        save_path = None
        if HAS_GUI and filedialog is not None:
            func = getattr(filedialog, 'asksaveasfilename', None)
            if func:
                try:
                    save_path = safe_filedialog_call(
                        func,
                        title="Save Merged Spectrum",
                        initialdir=save_dir,
                        initialfile=default_name,
                        defaultextension=".csv"
                    )
                except Exception:
                    save_path = None
        if not save_path:
            save_path = get_safe_save_path(save_dir, default_name)
        
        try:
            df = pd.DataFrame(data_to_save)
            header = list(state.get('meta_header', []))
            header.insert(0, f"# Scan Name: {scan_name}")
            header.insert(1, f"# Dataset Source: {state['current_dataset']}")
            header.insert(2, f"# Merged Clusters: {[c+1 for c in selected_c]}")
            header.insert(3, f"# Normalized: {state.get('i0_info', 'Yes')}")
            with open(save_path, 'w', newline='') as fh:
                fh.write("\n".join(header) + "\n")
                df.to_csv(fh, index=False, lineterminator='\n')
            with status_out:
                status_out.clear_output()
                print(f"Successfully saved merged spectrum to:\n  -> {save_path}")
        except Exception as e:
            with status_out:
                status_out.clear_output()
                print(f"Save error: {e}")

    def build_detector_ui(ds_name):
        if not load_dataset_data(ds_name):
            return

        unique_clusters = sorted([c for c in np.unique(state['cluster_map']) if c >= 0])
        state['checkboxes'] = []
        cb_widgets = []
        for c in unique_clusters:
            # Color indicator swatch
            hex_color = '#%02x%02x%02x' % tuple(int(x * 255) for x in cmap(c % 10)[:3])
            desc_html = f"<span style='display:inline-block; width:12px; height:12px; background-color:{hex_color}; margin-right:4px; vertical-align:middle; border-radius:2px;'></span>Cluster {c+1}"
            cb = widgets.Checkbox(value=True, description=f'Cluster {c+1}', indent=False, layout=widgets.Layout(width='auto', margin='2px 8px'))
            cb.observe(update_plots, names='value')
            state['checkboxes'].append((c, cb))
            cb_widgets.append(cb)
        
        cb_box.children = cb_widgets
        ds_label = get_sdd_label(ds_name, h5_path)
        title_html.value = (
            f"<div style='margin-bottom: 4px;'>"
            f"<b>Select Clusters to Merge:</b> <span style='color: #2b579a; font-weight:600;'>[{ds_label}]</span> "
            f"<span style='color: #555;'>({state.get('i0_info', 'Yes')})</span>"
            f"</div>"
        )
        update_plots()

    # --- Header Controls ---
    ds_dropdown = widgets.Dropdown(
        options=[(get_sdd_label(d, h5_path), d) for d in available_ds],
        value=dataset_name,
        description='Detector:',
        layout=widgets.Layout(width='240px')
    )

    def on_ds_change(change):
        if change['new'] != change['old']:
            build_detector_ui(change['new'])

    ds_dropdown.observe(on_ds_change, names='value')

    btn_all = widgets.Button(description="Select All", button_style='info', icon='check-square', layout=widgets.Layout(width='120px'))
    btn_all.on_click(on_select_all)

    btn_none = widgets.Button(description="Clear All", button_style='', icon='square', layout=widgets.Layout(width='120px'))
    btn_none.on_click(on_deselect_all)

    save_btn = widgets.Button(description="Save Merged Spectrum", button_style='success', icon='save', layout=widgets.Layout(width='200px'))
    save_btn.on_click(run_save)

    top_bar = widgets.HBox([ds_dropdown, btn_all, btn_none, save_btn], layout=widgets.Layout(margin='0 0 6px 0'))
    
    main_ui = widgets.VBox([
        top_bar,
        title_html,
        cb_box,
        plot_out,
        status_out
    ], layout=widgets.Layout(border='1px solid #ddd', padding='10px', border_radius='6px', margin='10px 0'))

    # Build initial detector UI and render exactly once
    build_detector_ui(dataset_name)
    display(main_ui)
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python interactive_cluster_merger.py [h5_file_path] [dataset_name]")
    else:
        interactive_cluster_merger(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'average')
