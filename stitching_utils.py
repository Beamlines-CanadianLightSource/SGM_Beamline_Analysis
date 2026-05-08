import numpy as np
import h5py
import os
import shutil
import re
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from analyze_sgm_bsky_data import analyze_sgm_bsky_data

def apply_asymmetric_trim(x, y, data_indices, left=0, right=0, top=0, bottom=0):
    """
    Filters coordinates and pixel indices based on asymmetric mm distances from boundaries.
    data_indices is the array [0, 1, 2, ...] corresponding to the raw pixel sequence.
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    mask = (x >= x_min + left) & (x <= x_max - right) & \
           (y >= y_min + bottom) & (y <= y_max - top)
           
    return x[mask], y[mask], data_indices[mask]

def browse_for_quadrant_files(num_files=4):
    """
    Opens file dialogs to select HDF5 files one by one.
    This allows files to be in different folders.
    """
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    files = []
    labels = ["SW (South-West)", "SE (South-East)", "NE (North-East)", "NW (North-West)"]
    
    print(f"Please select {num_files} HDF5 quadrant files...")
    
    for i in range(num_files):
        label = labels[i] if i < len(labels) else f"Map {i+1}"
        f = filedialog.askopenfilename(
            title=f"Select {label} HDF5 File",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if not f:
            break
        files.append(f)
        print(f"  -> Selected {label}: {os.path.basename(f)}")
    
    root.destroy()
    return files

def interactive_stitching_trim(h5_files=None, channel_roi=(30, 50)):
    """
    Interactive widget to adjust trims for 4 quadrant maps before stitching.
    Includes contrast control for global visualization.
    """
    if h5_files is None:
        h5_files = browse_for_quadrant_files()
    
    if not h5_files:
        print("No files selected.")
        return

    data_packs = [analyze_sgm_bsky_data(f, verbose=False) for f in h5_files]
    num_maps = len(data_packs)
    
    # Pre-load intensity for first detector and representative energy
    intensities = []
    all_vals = []
    for dp in data_packs:
        det = list(dp['sdd_files'].keys())[0]
        en = dp['representative_energy']
        f_path = dp['sdd_files'][det].get(en)
        if f_path:
            raw = np.fromfile(f_path, dtype=np.uint32).reshape(-1, 256)
            inten = np.sum(raw[:, channel_roi[0]:channel_roi[1]], axis=1)
            intensities.append(inten)
            all_vals.extend(inten)
        else:
            intensities.append(np.zeros_like(dp['x']))

    # Determine global data range for contrast
    if all_vals:
        global_min = np.min(all_vals)
        global_max = np.max(all_vals)
    else:
        global_min, global_max = 0, 1

    # --- UI Components ---
    sliders = []
    labels = ["SW", "SE", "NE", "NW"]
    
    for i in range(num_maps):
        label = labels[i] if i < len(labels) else f"Map {i+1}"
        x_range = np.max(data_packs[i]['x']) - np.min(data_packs[i]['x'])
        y_range = np.max(data_packs[i]['y']) - np.min(data_packs[i]['y'])
        
        l_s = widgets.FloatSlider(value=0.0, min=0.0, max=x_range*0.5, step=0.01, description=f'{label} Left:')
        r_s = widgets.FloatSlider(value=0.0, min=0.0, max=x_range*0.5, step=0.01, description=f'{label} Right:')
        t_s = widgets.FloatSlider(value=0.0, min=0.0, max=y_range*0.5, step=0.01, description=f'{label} Top:')
        b_s = widgets.FloatSlider(value=0.0, min=0.0, max=y_range*0.5, step=0.01, description=f'{label} Bottom:')
        sliders.append({'L': l_s, 'R': r_s, 'T': t_s, 'B': b_s})

    # Contrast Sliders
    contrast_slider = widgets.FloatRangeSlider(
        value=[0, 100], min=0, max=100, step=0.1,
        description='Contrast %:', layout=widgets.Layout(width='80%')
    )

    stitch_btn = widgets.Button(description="Bake Stitched Map", button_style='success', layout=widgets.Layout(width='200px'))
    output = widgets.Output()
    
    fig_id = f"stitch_trim_{id(h5_files)}"

    def update_plot(change=None):
        with output:
            if not plt.fignum_exists(fig_id):
                output.clear_output(wait=True)
                fig, ax = plt.subplots(1, 1, figsize=(10, 8), num=fig_id)
            else:
                fig = plt.figure(fig_id)
                ax = fig.gca()
                ax.clear()

            # Calculate contrast limits based on percentiles
            p_low, p_high = contrast_slider.value
            flat_intensities = np.concatenate(intensities)
            vmin = np.percentile(flat_intensities, p_low)
            vmax = np.percentile(flat_intensities, p_high)
            if vmin == vmax: vmax = vmin + 1

            colors = ['r', 'g', 'b', 'c']
            for i in range(num_maps):
                dp = data_packs[i]
                s = sliders[i]
                tx, ty, ti_idx = apply_asymmetric_trim(
                    dp['x'], dp['y'], np.arange(len(dp['x'])),
                    left=s['L'].value, right=s['R'].value, top=s['T'].value, bottom=s['B'].value
                )
                
                if len(tx) > 0:
                    ax.tripcolor(tx, ty, intensities[i][ti_idx], shading='gouraud', 
                                 edgecolors='none', vmin=vmin, vmax=vmax, cmap='viridis', alpha=0.8)
                    # Draw border
                    ax.plot([np.min(tx), np.max(tx), np.max(tx), np.min(tx), np.min(tx)], 
                            [np.min(ty), np.min(ty), np.max(ty), np.max(ty), np.min(ty)], 
                            color=colors[i % 4], lw=1, ls='--')

            ax.set_aspect('equal')
            ax.set_title(f"Stitching Preview (Contrast: {p_low}% - {p_high}%)")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            plt.tight_layout()
            fig.canvas.draw_idle()

    # Link all sliders to update_plot
    for s_set in sliders:
        for s in s_set.values():
            s.observe(update_plot, names='value')
    contrast_slider.observe(update_plot, names='value')

    def on_stitch_clicked(b):
        final_trims = []
        for s in sliders:
            final_trims.append({
                'left': s['L'].value, 'right': s['R'].value,
                'top': s['T'].value, 'bottom': s['B'].value
            })
        
        with output:
            print("\nStarting stitching with selected trims...")
            stitch_quadrant_maps(h5_files, trims=final_trims)

    stitch_btn.on_click(on_stitch_clicked)

    # Layout
    controls = []
    for i in range(0, num_maps, 2):
        row = []
        for j in range(2):
            if i + j < num_maps:
                col = widgets.VBox([sliders[i+j]['L'], sliders[i+j]['R'], sliders[i+j]['T'], sliders[i+j]['B']])
                row.append(col)
        controls.append(widgets.HBox(row))
        
    display(widgets.VBox(controls + [widgets.Label("Global Contrast Control:"), contrast_slider, stitch_btn, output]))
    update_plot()

def stitch_quadrant_maps(h5_files=None, output_dir=None, trims=None, verbose=True):
    """
    Stitches multiple map datasets into a single unified dataset.
    """
    if h5_files is None:
        h5_files = browse_for_quadrant_files()
        
    if not h5_files:
        print("No files selected. Aborting.")
        return None
        
    if output_dir is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        output_dir = filedialog.askdirectory(title="Select Output Directory for Stitched Map")
        root.destroy()
        
    if not output_dir:
        print("No output directory selected. Aborting.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data_packs = []
    for f in h5_files:
        data_packs.append(analyze_sgm_bsky_data(f, verbose=False))
        
    if not data_packs:
        return None

    # Determine common energies and detectors
    common_energies = sorted(list(set.intersection(*[set(dp['energies']) for d in data_packs for dp in [d]])))
    common_detectors = sorted(list(set.intersection(*[set(dp['sdd_files'].keys()) for d in data_packs for dp in [d]])))
    
    if verbose:
        print(f"\nStitching {len(data_packs)} maps...")
        print(f"Common Energies: {common_energies}")
        print(f"Common Detectors: {common_detectors}")

    # Calculate master coordinates
    master_x = []
    master_y = []
    map_pixel_masks = [] # List of (indices_to_keep) for each map

    for i, dp in enumerate(data_packs):
        t = trims[i] if (trims and i < len(trims)) else {}
        x, y = dp['x'], dp['y']
        indices = np.arange(len(x))
        
        tx, ty, t_indices = apply_asymmetric_trim(
            x, y, indices, 
            left=t.get('left', 0), 
            right=t.get('right', 0), 
            top=t.get('top', 0), 
            bottom=t.get('bottom', 0)
        )
        
        master_x.append(tx)
        master_y.append(ty)
        map_pixel_masks.append(t_indices)
        
    master_x = np.concatenate(master_x)
    master_y = np.concatenate(master_y)
    
    # Create the new HDF5 file
    ref_dp = data_packs[0]
    new_h5_name = f"Stitched_{ref_dp['scan_name']}.h5"
    new_h5_path = os.path.join(output_dir, new_h5_name)
    
    with h5py.File(new_h5_path, 'w') as f_out:
        with h5py.File(h5_files[0], 'r') as f_in:
            if 'scan_metadata' in f_in:
                f_in.copy('scan_metadata', f_out)
            if 'initial_motor_positions' in f_in:
                f_in.copy('initial_motor_positions', f_out)
        
        hw = f_out.create_group('hexapod_waves')
        hw.create_dataset('x', data=master_x)
        hw.create_dataset('y', data=master_y)
        
        md = f_out.create_group('map_data')
        md.create_dataset('energy', data=np.array(common_energies))

    # Process raw data files
    for energy in common_energies:
        energy_str = f"{energy:.2f}".replace('.', '_')
        energy_subdir = f"Stitched_{ref_dp['scan_name']}_{energy_str}eV"
        energy_path = os.path.join(output_dir, energy_subdir)
        if not os.path.exists(energy_path):
            os.makedirs(energy_path)
            
        for det in common_detectors:
            stitched_data = []
            for i, dp in enumerate(data_packs):
                sdd_path = dp['sdd_files'][det].get(energy)
                if sdd_path and os.path.exists(sdd_path):
                    raw_data = np.fromfile(sdd_path, dtype=np.uint32).reshape(-1, 256)
                    trimmed_raw = raw_data[map_pixel_masks[i]]
                    stitched_data.append(trimmed_raw)
            
            if stitched_data:
                final_sdd = np.concatenate(stitched_data, axis=0)
                out_sdd_path = os.path.join(energy_path, f"{det}_0.bin")
                final_sdd.tofile(out_sdd_path)
                
        for i_mcc in range(1, 5):
            mcc_name = f"mcc{i_mcc}"
            stitched_mcc = []
            for i, dp in enumerate(data_packs):
                mcc_data = dp.get('mcc_data', {}).get(energy)
                if mcc_data is not None:
                    trimmed_mcc = mcc_data[map_pixel_masks[i]]
                    stitched_mcc.append(trimmed_mcc)
            
            if stitched_mcc:
                final_mcc = np.concatenate(stitched_mcc, axis=0)
                if i_mcc == 1:
                    header = ",".join(ref_dp.get('mcc_channel_names', ['mcc1', 'mcc2', 'mcc3', 'mcc4']))
                    np.savetxt(os.path.join(energy_path, "mcc1.csv"), final_mcc, delimiter=",", header=header, comments='')

    if verbose:
        print(f"\n[SUCCESS] Stitched dataset created at: {output_dir}")
        print(f"New Master H5: {new_h5_path}")
        
    return new_h5_path
