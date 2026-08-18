import numpy as np
import h5py
import os
import shutil
import re
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
from save_pymca_stack_h5 import get_user_file_action

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

def browse_for_quadrant_files(num_files=None):
    """
    Opens file dialogs to select HDF5 files one by one.
    This allows files to be in different folders.
    """
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import simpledialog
    
    root = tk.Tk()
    # Keep the window in the taskbar but make it invisible so the dialog can't get permanently lost
    root.attributes("-alpha", 0.0)
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()
    
    if num_files is None:
        num_files = simpledialog.askinteger(
            "Number of Images",
            "How many images do you want to stitch together?",
            parent=root,
            minvalue=1,
            maxvalue=20,
            initialvalue=4
        )
        if num_files is None:
            root.destroy()
            return []
            
    root.withdraw() # Hide completely for the file dialogs
    
    files = []
    labels = []
    
    print(f"Please select {num_files} HDF5 files...")
    
    for i in range(num_files):
        label = labels[i] if i < len(labels) else f"Image {i+1}"
        f = filedialog.askopenfilename(
            title=f"Select {label} HDF5 File (Cancel to skip this file)",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if not f:
            print(f"  -> Skipped {label}")
            continue
        files.append(f)
        print(f"  -> Selected {label}: {os.path.basename(f)}")
    
    root.destroy()
    return files


def calculate_initial_zero_overlap_trims(data_packs):
    """
    Analyzes physical spatial bounds of loaded images to calculate initial
    trim distances (in mm) that remove spatial overlaps between adjacent grid images.
    
    Returns a list of dicts: [{'left': l, 'right': r, 'top': t, 'bottom': b}, ...]
    """
    num_maps = len(data_packs)
    auto_trims = [{'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0} for _ in range(num_maps)]
    
    if num_maps <= 1:
        return auto_trims
        
    maps_info = []
    for i, dp in enumerate(data_packs):
        x_min, x_max = float(np.min(dp['x'])), float(np.max(dp['x']))
        y_min, y_max = float(np.min(dp['y'])), float(np.max(dp['y']))
        xc = (x_min + x_max) / 2.0
        yc = (y_min + y_max) / 2.0
        maps_info.append({
            'idx': i,
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'xc': xc, 'yc': yc
        })

    # Sort maps into spatial rows (group by similar Y center)
    maps_info.sort(key=lambda m: m['yc'], reverse=True) # Top to bottom
    
    rows = []
    for m in maps_info:
        placed = False
        for row in rows:
            if abs(row[0]['yc'] - m['yc']) < 2.5:
                row.append(m)
                placed = True
                break
        if not placed:
            rows.append([m])
            
    # Sort each row from left to right (by X center)
    for row in rows:
        row.sort(key=lambda m: m['xc'])

    # 1. Calculate Horizontal Overlaps (Left/Right trims between adjacent columns)
    for row in rows:
        for c in range(len(row) - 1):
            left_map = row[c]
            right_map = row[c + 1]
            overlap_x = left_map['x_max'] - right_map['x_min']
            if overlap_x > 0:
                half_overlap = round(float(overlap_x / 2.0) + 0.005, 3) # add slight 5-micron safety margin
                auto_trims[left_map['idx']]['right'] = max(auto_trims[left_map['idx']]['right'], half_overlap)
                auto_trims[right_map['idx']]['left'] = max(auto_trims[right_map['idx']]['left'], half_overlap)

    # 2. Calculate Vertical Overlaps (Top/Bottom trims between adjacent rows)
    for r in range(len(rows) - 1):
        top_row = rows[r]
        bot_row = rows[r + 1]
        
        for t_map in top_row:
            for b_map in bot_row:
                if abs(t_map['xc'] - b_map['xc']) < 3.0: # Same column
                    overlap_y = b_map['y_max'] - t_map['y_min']
                    if overlap_y > 0:
                        half_overlap = round(float(overlap_y / 2.0) + 0.005, 3)
                        auto_trims[t_map['idx']]['bottom'] = max(auto_trims[t_map['idx']]['bottom'], half_overlap)
                        auto_trims[b_map['idx']]['top'] = max(auto_trims[b_map['idx']]['top'], half_overlap)

    return auto_trims


def interactive_stitching_trim(h5_files=None, channel_roi=(30, 50)):
    """
    Interactive widget to adjust trims for multiple maps before stitching.
    Includes contrast control for global visualization.
    """
    if h5_files is None:
        h5_files = browse_for_quadrant_files()
    
    if not h5_files:
        print("No files selected.")
        return

    # Check for duplicates
    if len(h5_files) != len(set(h5_files)):
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        messagebox.showwarning("Duplicate Files", 
                               "Warning: You have selected the same file for multiple quadrants.\n\n"
                               "This may cause data overlap and visualization artifacts (like 'filling in' gaps).\n"
                               "If you intended to skip a quadrant, please click 'Cancel' in one of the file dialogs next time.")
        root.destroy()


    data_packs = [analyze_sgm_bsky_data(f, verbose=False) for f in h5_files]
    num_maps = len(data_packs)
    
    # Calculate initial zero-overlap trims automatically
    auto_trims = calculate_initial_zero_overlap_trims(data_packs)
    
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
    labels = []
    
    for i in range(num_maps):
        label = labels[i] if i < len(labels) else f"Image {i+1}"
        x_range = np.max(data_packs[i]['x']) - np.min(data_packs[i]['x'])
        y_range = np.max(data_packs[i]['y']) - np.min(data_packs[i]['y'])
        
        init_l = min(auto_trims[i]['left'], x_range * 0.45)
        init_r = min(auto_trims[i]['right'], x_range * 0.45)
        init_t = min(auto_trims[i]['top'], y_range * 0.45)
        init_b = min(auto_trims[i]['bottom'], y_range * 0.45)
        
        l_s = widgets.FloatSlider(value=init_l, min=0.0, max=x_range*0.5, step=0.01, description=f'{label} Left:')
        r_s = widgets.FloatSlider(value=init_r, min=0.0, max=x_range*0.5, step=0.01, description=f'{label} Right:')
        t_s = widgets.FloatSlider(value=init_t, min=0.0, max=y_range*0.5, step=0.01, description=f'{label} Top:')
        b_s = widgets.FloatSlider(value=init_b, min=0.0, max=y_range*0.5, step=0.01, description=f'{label} Bottom:')
        sliders.append({'L': l_s, 'R': r_s, 'T': t_s, 'B': b_s})

    # Contrast Sliders
    contrast_slider = widgets.FloatRangeSlider(
        value=[0, 100], min=0, max=100, step=0.1,
        description='Contrast %:', layout=widgets.Layout(width='80%')
    )

    stitch_btn = widgets.Button(description="Bake Stitched Image", button_style='success', layout=widgets.Layout(width='200px'))
    output = widgets.Output()
    
    fig_id = f"stitch_trim_{id(h5_files)}"

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)
            if not plt.fignum_exists(fig_id):
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

            colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
            
            print("--- LIVE TRIMMING & STITCHING STATUS ---")
            for i in range(num_maps):
                lbl = labels[i] if i < len(labels) else f"Image {i+1}"
                dp = data_packs[i]
                s = sliders[i]
                L, R, T, B = s['L'].value, s['R'].value, s['T'].value, s['B'].value
                
                orig_x, orig_y = dp['x'], dp['y']
                all_idx = np.arange(len(orig_x))
                
                tx, ty, ti_idx = apply_asymmetric_trim(
                    orig_x, orig_y, all_idx,
                    left=L, right=R, top=T, bottom=B
                )
                
                num_kept = len(tx)
                num_total = len(orig_x)
                num_trimmed = num_total - num_kept
                
                print(f"  [{lbl}] Trims: Left={L:.2f}mm, Right={R:.2f}mm, Top={T:.2f}mm, Bottom={B:.2f}mm | Kept {num_kept}/{num_total} pts ({num_trimmed} trimmed)")

                col = colors[i % len(colors)]
                
                # 1. Draw Original Boundary (faint grey dotted box)
                ox_min, ox_max = np.min(orig_x), np.max(orig_x)
                oy_min, oy_max = np.min(orig_y), np.max(orig_y)
                ax.plot([ox_min, ox_max, ox_max, ox_min, ox_min],
                        [oy_min, oy_min, oy_max, oy_max, oy_min],
                        color='gray', lw=1.2, ls=':', alpha=0.6)

                # 2. Draw Trimmed Out Points (Subtle red x markers)
                if num_trimmed > 0:
                    kept_set = set(ti_idx)
                    trimmed_mask = np.array([idx not in kept_set for idx in all_idx])
                    ax.scatter(orig_x[trimmed_mask], orig_y[trimmed_mask], 
                               color='red', marker='x', s=14, alpha=0.6, label='Trimmed Points' if i==0 else "")

                # 3. Draw Kept Data Area and Active Trimmed Boundary Box
                if num_kept > 0:
                    ax.tripcolor(tx, ty, intensities[i][ti_idx], shading='gouraud', 
                                 edgecolors='none', vmin=vmin, vmax=vmax, cmap='viridis', alpha=0.85)
                    
                    # Draw prominent active boundary box
                    kx_min, kx_max = np.min(tx), np.max(tx)
                    ky_min, ky_max = np.min(ty), np.max(ty)
                    ax.plot([kx_min, kx_max, kx_max, kx_min, kx_min], 
                            [ky_min, ky_min, ky_max, ky_max, ky_min], 
                            color=col, lw=2.0, ls='-', label=f"{lbl} Kept Bounds")
                    
                    # Text label for image
                    ax.text((kx_min + kx_max) / 2.0, (ky_min + ky_max) / 2.0, lbl, 
                            color=col, fontsize=9, fontweight='bold', ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor=col))

            ax.set_aspect('equal')
            ax.set_title(f"Stitching Preview (Contrast: {p_low}% - {p_high}%)")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            fig.canvas.draw_idle()

    # Link all sliders to update_plot
    for s_set in sliders:
        for s in s_set.values():
            s.observe(update_plot, names='value')
    contrast_slider.observe(update_plot, names='value')

    def on_stitch_clicked(b):
        stitch_btn.description = "Baking Stitched Image..."
        stitch_btn.button_style = 'warning'
        
        final_trims = []
        for s in sliders:
            final_trims.append({
                'left': s['L'].value, 'right': s['R'].value,
                'top': s['T'].value, 'bottom': s['B'].value
            })
        
        with output:
            print("\n[PROCESS STARTED] Baking Stitched Image with selected trims... Please wait.")
            try:
                stitch_quadrant_maps(h5_files, trims=final_trims)
            finally:
                stitch_btn.description = "Bake Stitched Image"
                stitch_btn.button_style = 'success'

    stitch_btn.on_click(on_stitch_clicked)

    # Layout using Tab for clear image selection
    tab_widget = widgets.Tab()
    tab_children = []
    
    for i in range(num_maps):
        label = labels[i] if i < len(labels) else f"Image {i+1}"
        col = widgets.VBox([sliders[i]['L'], sliders[i]['R'], sliders[i]['T'], sliders[i]['B']])
        tab_children.append(col)
        
    tab_widget.children = tab_children
    for i in range(num_maps):
        label = labels[i] if i < len(labels) else f"Image {i+1}"
        tab_widget.set_title(i, f"{label} Trims")
        
    notice_label = widgets.HTML(
        "<div style='background-color: #e8f4f8; padding: 8px 12px; border-left: 4px solid #0056b3; margin-bottom: 8px;'>"
        "<b style='color:#004085;'>[AUTO-ALIGNMENT ACTIVE]</b> Trim sliders have been pre-populated with calculated zero-overlap values.<br>"
        "You can inspect and fine-tune trim values in the tabs below, then click <b>'Bake Stitched Image'</b> when ready."
        "</div>"
    )
    display(widgets.VBox([notice_label, tab_widget, widgets.Label("Global Contrast Control:"), contrast_slider, stitch_btn, output]))
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
        
    ref_dp = data_packs[0]

    # Determine common energies and detectors
    all_energies_sets = [set(np.round(dp['energies'], 2)) for dp in data_packs if len(dp.get('energies', [])) > 0]
    if all_energies_sets:
        common_energies = sorted(list(set.intersection(*all_energies_sets)))
    else:
        common_energies = []
        
    if not common_energies:
        # Fallback: Check if candidate energies across maps are near each other (e.g. single energy maps)
        candidate_energies = []
        for dp in data_packs:
            ens = dp.get('energies', np.array([]))
            if len(ens) > 0 and ens[0] > 0:
                candidate_energies.append(float(ens[0]))
            elif 'representative_energy' in dp and dp['representative_energy'] > 0:
                candidate_energies.append(float(dp['representative_energy']))
                
        if candidate_energies and len(candidate_energies) == len(data_packs):
            mean_en = float(np.round(np.mean(candidate_energies), 2))
            if all(abs(e - mean_en) < 1.0 for e in candidate_energies):
                common_energies = [mean_en]
                for dp in data_packs:
                    if len(dp.get('energies', [])) == 0 or (len(dp['energies']) == 1 and dp['energies'][0] == -1.0):
                        dp['energies'] = np.array([mean_en])

    if not common_energies:
        print("[ERROR] Stitching aborted: No common energy data was found across the selected maps.", file=sys.stderr)
        print("        Please ensure input HDF5 files or their parent directories contain valid energy information.", file=sys.stderr)
        return None

    # Determine common detectors present in all data packs
    common_detectors = sorted(list(set.intersection(*[set(dp['sdd_files'].keys()) for dp in data_packs if 'sdd_files' in dp])))
    if not common_detectors:
        # Fallback to union if intersection is empty
        all_dets = set()
        for dp in data_packs:
            all_dets.update(dp.get('sdd_files', {}).keys())
        common_detectors = sorted(list(all_dets))

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
    
    # Determine spatial corner scans (Top-Left to Bottom-Right in physical coordinates)
    spatial_scores = []
    for dp in data_packs:
        xc = (np.min(dp['x']) + np.max(dp['x'])) / 2.0
        yc = (np.min(dp['y']) + np.max(dp['y'])) / 2.0
        score = yc - xc
        scan_name = str(dp.get('scan_name', 'scan'))
        spatial_scores.append((score, scan_name))
        
    spatial_scores.sort(key=lambda item: item[0], reverse=True)
    top_left_scan = spatial_scores[0][1]
    bottom_right_scan = spatial_scores[-1][1]
    
    if top_left_scan != bottom_right_scan:
        stitched_tag = f"{top_left_scan}_to_{bottom_right_scan}"
    else:
        stitched_tag = top_left_scan

    # Determine master folder name for the stitched dataset
    if len(common_energies) == 1:
        energy_str = f"{common_energies[0]:.2f}".replace('.', '_')
        master_folder_name = f"Stitched_{stitched_tag}_{energy_str}eV"
    else:
        master_folder_name = f"Stitched_{stitched_tag}"

    master_folder_path = os.path.join(output_dir, master_folder_name)
    new_h5_name = f"Stitched_{stitched_tag}.h5"
    new_h5_path = os.path.join(master_folder_path, new_h5_name)
    
    # --- Overwrite & Rename Check ---
    if os.path.exists(master_folder_path) or os.path.exists(new_h5_path):
        import tkinter as tk
        from tkinter import messagebox, simpledialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        msg = (f"The dataset folder or file '{master_folder_name}' already exists.\n\n"
               f"Do you want to overwrite it?\n\n"
               f"• Click 'Yes' to overwrite existing data.\n"
               f"• Click 'No' to save under a new name in a NEW folder.\n"
               f"• Click 'Cancel' to abort.")
        response = messagebox.askyesnocancel("Stitched Dataset Exists", msg, parent=root)
        
        if response is True:  # Overwrite
            print(f"  -> Overwriting existing dataset folder: {master_folder_name}")
        elif response is False:  # Rename / Save as New Folder
            new_tag = simpledialog.askstring(
                "Save as New Folder", 
                "Enter new name/tag for stitched dataset:", 
                initialvalue=f"{stitched_tag}_v2", 
                parent=root
            )
            if new_tag:
                new_tag = new_tag.strip()
                if new_tag.startswith("Stitched_"):
                    new_tag = new_tag[9:]
                if new_tag.lower().endswith(".h5"):
                    new_tag = new_tag[:-3]
                    
                stitched_tag = new_tag
                if len(common_energies) == 1:
                    energy_str = f"{common_energies[0]:.2f}".replace('.', '_')
                    master_folder_name = f"Stitched_{stitched_tag}_{energy_str}eV"
                else:
                    master_folder_name = f"Stitched_{stitched_tag}"
                    
                master_folder_path = os.path.join(output_dir, master_folder_name)
                new_h5_name = f"Stitched_{stitched_tag}.h5"
                new_h5_path = os.path.join(master_folder_path, new_h5_name)
                print(f"  -> Saving dataset under new folder name: {master_folder_name}")
            else:
                print("  -> Stitching operation cancelled by user.")
                root.destroy()
                return None
        else:  # Cancel or dialog closed
            print("  -> Stitching operation cancelled by user.")
            root.destroy()
            return None
            
        root.destroy()

    if not os.path.exists(master_folder_path):
        os.makedirs(master_folder_path)
    
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

        # --- Add Stitching & Trim Metadata ---
        sm_group = f_out.create_group('stitching_metadata')
        sm_group.attrs['num_source_maps'] = len(data_packs)
        sm_group.attrs['source_files'] = [os.path.abspath(f) for f in h5_files]
        sm_group.attrs['top_left_scan'] = top_left_scan
        sm_group.attrs['bottom_right_scan'] = bottom_right_scan
        sm_group.attrs['stitched_tag'] = stitched_tag
        
        for i, dp in enumerate(data_packs):
            t = trims[i] if (trims and i < len(trims)) else {}
            left = float(t.get('left', 0.0))
            right = float(t.get('right', 0.0))
            top = float(t.get('top', 0.0))
            bottom = float(t.get('bottom', 0.0))
            
            x_orig, y_orig = dp['x'], dp['y']
            t_indices = map_pixel_masks[i]
            
            all_indices = np.arange(len(x_orig))
            kept_set = set(t_indices)
            trimmed_indices = np.array([idx for idx in all_indices if idx not in kept_set], dtype=int)
            
            map_grp = sm_group.create_group(f"map_{i+1}")
            map_grp.attrs['source_file'] = os.path.abspath(h5_files[i])
            map_grp.attrs['scan_name'] = str(dp.get('scan_name', f'Map_{i+1}'))
            
            # Store trim distance parameters (in mm)
            map_grp.create_dataset('trim_left_mm', data=left)
            map_grp.create_dataset('trim_right_mm', data=right)
            map_grp.create_dataset('trim_top_mm', data=top)
            map_grp.create_dataset('trim_bottom_mm', data=bottom)
            
            # Store original & kept coordinate bounds
            map_grp.attrs['orig_x_min'] = float(np.min(x_orig)) if len(x_orig) > 0 else 0.0
            map_grp.attrs['orig_x_max'] = float(np.max(x_orig)) if len(x_orig) > 0 else 0.0
            map_grp.attrs['orig_y_min'] = float(np.min(y_orig)) if len(y_orig) > 0 else 0.0
            map_grp.attrs['orig_y_max'] = float(np.max(y_orig)) if len(y_orig) > 0 else 0.0
            
            map_grp.attrs['kept_x_min'] = float(np.min(x_orig[t_indices])) if len(t_indices) > 0 else 0.0
            map_grp.attrs['kept_x_max'] = float(np.max(x_orig[t_indices])) if len(t_indices) > 0 else 0.0
            map_grp.attrs['kept_y_min'] = float(np.min(y_orig[t_indices])) if len(t_indices) > 0 else 0.0
            map_grp.attrs['kept_y_max'] = float(np.max(y_orig[t_indices])) if len(t_indices) > 0 else 0.0
            
            map_grp.attrs['total_points'] = len(x_orig)
            map_grp.attrs['kept_points'] = len(t_indices)
            map_grp.attrs['trimmed_points'] = len(trimmed_indices)
            
            # Store coordinate arrays for trimmed (excluded) points
            map_grp.create_dataset('trimmed_x', data=x_orig[trimmed_indices])
            map_grp.create_dataset('trimmed_y', data=y_orig[trimmed_indices])
            map_grp.create_dataset('trimmed_indices', data=trimmed_indices)
            
            # Store coordinate arrays for kept points
            map_grp.create_dataset('kept_x', data=x_orig[t_indices])
            map_grp.create_dataset('kept_y', data=y_orig[t_indices])
            map_grp.create_dataset('kept_indices', data=t_indices)

    # Process raw data files inside master folder
    for energy in common_energies:
        if len(common_energies) == 1:
            # Single energy map: sdd and mcc data sit directly inside master_folder_path with the .h5 file
            energy_path = master_folder_path
        else:
            # Multi-energy stack: create energy subdirectories inside master_folder_path
            energy_str = f"{energy:.2f}".replace('.', '_')
            energy_subdir = f"Stitched_{stitched_tag}_{energy_str}eV"
            energy_path = os.path.join(master_folder_path, energy_subdir)
            if not os.path.exists(energy_path):
                os.makedirs(energy_path)
            
        for det in common_detectors:
            stitched_data = []
            for i, dp in enumerate(data_packs):
                sdd_dict = dp.get('sdd_files', {}).get(det, {})
                sdd_path = sdd_dict.get(energy)
                if not sdd_path and len(sdd_dict) > 0:
                    sdd_path = list(sdd_dict.values())[0]
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
                mcc_dict = dp.get('mcc_data', {})
                mcc_data = mcc_dict.get(energy)
                if mcc_data is None and len(mcc_dict) > 0:
                    mcc_data = list(mcc_dict.values())[0]
                if mcc_data is not None:
                    if mcc_data.ndim == 1:
                        trimmed_mcc = mcc_data[map_pixel_masks[i]]
                    else:
                        trimmed_mcc = mcc_data[map_pixel_masks[i], :]
                    stitched_mcc.append(trimmed_mcc)
            
            if stitched_mcc:
                final_mcc = np.concatenate(stitched_mcc, axis=0)
                mcc_filename = f"{mcc_name}.csv"
                header = ",".join(ref_dp.get('mcc_channel_names', [f'ch{j}' for j in range(final_mcc.shape[1] if final_mcc.ndim > 1 else 1)]))
                np.savetxt(os.path.join(energy_path, mcc_filename), final_mcc, delimiter=",", header=header, comments='')


    if verbose:
        print(f"\n[SUCCESS] Stitched dataset created at: {output_dir}")
        print(f"New Master H5: {new_h5_path}")
        print("  -> Trimming metadata successfully recorded in 'stitching_metadata' group inside HDF5.")
        
    return new_h5_path


def read_stitching_trim_info(h5_file_path, verbose=True):
    """
    Reads and displays the trim and stitching metadata stored inside a stitched HDF5 file.
    
    Returns a dictionary containing map-by-map trim parameters, coordinate bounds,
    and trimmed/kept point counts and coordinate arrays.
    """
    if not os.path.exists(h5_file_path):
        print(f"[Error] File not found: {h5_file_path}")
        return None
        
    trim_info = {}
    with h5py.File(h5_file_path, 'r') as f:
        if 'stitching_metadata' not in f:
            if verbose:
                print(f"[Info] No 'stitching_metadata' group found in '{os.path.basename(h5_file_path)}'.")
                print("       (Note: This file was created before trim metadata recording was added. Re-bake using 'interactive_stitching_trim()' to save metadata.)")
            return None
            
        sm = f['stitching_metadata']
        num_maps = sm.attrs.get('num_source_maps', 0)
        source_files = sm.attrs.get('source_files', [])
        
        trim_info['num_source_maps'] = num_maps
        trim_info['source_files'] = list(source_files)
        trim_info['maps'] = {}
        
        if verbose:
            print(f"\n--- Stitching Trim Information for {os.path.basename(h5_file_path)} ---")
            print(f"Source Maps Count: {num_maps}")
            
        for key in sorted(sm.keys()):
            if key.startswith('map_'):
                map_grp = sm[key]
                map_data = {
                    'source_file': map_grp.attrs.get('source_file', 'N/A'),
                    'scan_name': map_grp.attrs.get('scan_name', key),
                    'trim_left_mm': float(map_grp['trim_left_mm'][()]),
                    'trim_right_mm': float(map_grp['trim_right_mm'][()]),
                    'trim_top_mm': float(map_grp['trim_top_mm'][()]),
                    'trim_bottom_mm': float(map_grp['trim_bottom_mm'][()]),
                    'orig_x_range': (float(map_grp.attrs.get('orig_x_min', 0)), float(map_grp.attrs.get('orig_x_max', 0))),
                    'orig_y_range': (float(map_grp.attrs.get('orig_y_min', 0)), float(map_grp.attrs.get('orig_y_max', 0))),
                    'kept_x_range': (float(map_grp.attrs.get('kept_x_min', 0)), float(map_grp.attrs.get('kept_x_max', 0))),
                    'kept_y_range': (float(map_grp.attrs.get('kept_y_min', 0)), float(map_grp.attrs.get('kept_y_max', 0))),
                    'total_points': int(map_grp.attrs.get('total_points', 0)),
                    'kept_points': int(map_grp.attrs.get('kept_points', 0)),
                    'trimmed_points': int(map_grp.attrs.get('trimmed_points', 0)),
                    'trimmed_x': map_grp['trimmed_x'][:],
                    'trimmed_y': map_grp['trimmed_y'][:],
                    'kept_x': map_grp['kept_x'][:],
                    'kept_y': map_grp['kept_y'][:],
                }
                trim_info['maps'][key] = map_data
                
                if verbose:
                    print(f"\n  [{key.upper()}] Scan: {map_data['scan_name']}")
                    print(f"    Source File: {os.path.basename(map_data['source_file'])}")
                    print(f"    Trims Applied (mm): Left={map_data['trim_left_mm']:.2f}, Right={map_data['trim_right_mm']:.2f}, Top={map_data['trim_top_mm']:.2f}, Bottom={map_data['trim_bottom_mm']:.2f}")
                    print(f"    Original Bounds: X=[{map_data['orig_x_range'][0]:.2f}, {map_data['orig_x_range'][1]:.2f}], Y=[{map_data['orig_y_range'][0]:.2f}, {map_data['orig_y_range'][1]:.2f}]")
                    print(f"    Kept Bounds:     X=[{map_data['kept_x_range'][0]:.2f}, {map_data['kept_x_range'][1]:.2f}], Y=[{map_data['kept_y_range'][0]:.2f}, {map_data['kept_y_range'][1]:.2f}]")
                    print(f"    Points: Total={map_data['total_points']}, Kept={map_data['kept_points']}, Trimmed={map_data['trimmed_points']}")

    return trim_info


def check_stitching_gaps(h5_file_path, verbose=True):
    """
    Analyzes spatial boundaries between adjacent images in a stitched HDF5 dataset
    to detect physical gaps or overlaps in X and Y directions.
    
    Provides exact trim adjustment recommendations (in mm).
    """
    trim_info = read_stitching_trim_info(h5_file_path, verbose=False)
    if not trim_info or 'maps' not in trim_info or not trim_info['maps']:
        print("[Error] No stitching metadata found in file. Please re-bake the dataset using interactive_stitching_trim().")
        return None

    maps_data = []
    for key, mdata in trim_info['maps'].items():
        kx_min, kx_max = mdata['kept_x_range']
        ky_min, ky_max = mdata['kept_y_range']
        xc = (kx_min + kx_max) / 2.0
        yc = (ky_min + ky_max) / 2.0
        maps_data.append({
            'key': key,
            'scan_name': mdata['scan_name'],
            'xc': xc, 'yc': yc,
            'x_min': kx_min, 'x_max': kx_max,
            'y_min': ky_min, 'y_max': ky_max
        })

    # Sort maps into spatial rows (group by similar Y center)
    maps_data.sort(key=lambda m: m['yc'], reverse=True) # Top to bottom
    
    # Identify unique rows by clustering Y centers within 2.5 mm
    rows = []
    for m in maps_data:
        placed = False
        for row in rows:
            if abs(row[0]['yc'] - m['yc']) < 2.5:
                row.append(m)
                placed = True
                break
        if not placed:
            rows.append([m])
            
    # Sort each row from left to right (by X center)
    for row in rows:
        row.sort(key=lambda m: m['xc'])

    if verbose:
        print(f"\n--- Stitching Boundary & Seam Diagnostic: {os.path.basename(h5_file_path)} ---")
        print(f"Detected Grid Layout: {len(rows)} Rows x {max(len(r) for r in rows)} Columns ({len(maps_data)} total images)")

    boundary_report = []

    # 1. Check Vertical Seams (Between Rows)
    for r_idx in range(len(rows) - 1):
        top_row = rows[r_idx]
        bot_row = rows[r_idx + 1]
        
        top_y_min = min(m['y_min'] for m in top_row)
        bot_y_max = max(m['y_max'] for m in bot_row)
        
        delta_y = bot_y_max - top_y_min
        
        if abs(delta_y) < 0.02:
            status = "PERFECT ALIGNMENT"
            rec = "No Y-trim adjustment needed."
        elif delta_y > 0:
            status = f"OVERLAP (+{delta_y:.3f} mm)"
            rec = f"Increase Top/Bottom trim by ~{delta_y/2.0:.3f} mm to eliminate row overlap seam."
        else:
            status = f"GAP ({delta_y:.3f} mm)"
            rec = f"Decrease Top/Bottom trim by ~{abs(delta_y)/2.0:.3f} mm to close row gap."
            
        rep_line = {
            'type': 'Row Seam (Y)',
            'between': f"Row {r_idx+1} & Row {r_idx+2}",
            'delta_mm': delta_y,
            'status': status,
            'recommendation': rec
        }
        boundary_report.append(rep_line)
        
        if verbose:
            print(f"\n  [ROW SEAM: Row {r_idx+1} / Row {r_idx+2}]")
            print(f"    Status: {status}")
            print(f"    Action: {rec}")

    # 2. Check Horizontal Seams (Between Columns in each Row)
    for r_idx, row in enumerate(rows):
        for c_idx in range(len(row) - 1):
            left_img = row[c_idx]
            right_img = row[c_idx + 1]
            
            delta_x = left_img['x_max'] - right_img['x_min']
            
            if abs(delta_x) < 0.02:
                status = "PERFECT ALIGNMENT"
                rec = "No X-trim adjustment needed."
            elif delta_x > 0:
                status = f"OVERLAP (+{delta_x:.3f} mm)"
                rec = f"Increase Left/Right trim by ~{delta_x/2.0:.3f} mm to eliminate column overlap seam."
            else:
                status = f"GAP ({delta_x:.3f} mm)"
                rec = f"Decrease Left/Right trim by ~{abs(delta_x)/2.0:.3f} mm to close column gap."
                
            rep_line = {
                'type': 'Column Seam (X)',
                'between': f"Row {r_idx+1} Col {c_idx+1} & Col {c_idx+2}",
                'delta_mm': delta_x,
                'status': status,
                'recommendation': rec
            }
            boundary_report.append(rep_line)
            
            if verbose:
                print(f"\n  [COLUMN SEAM: Row {r_idx+1} Col {c_idx+1} / Col {c_idx+2}]")
                print(f"    Status: {status}")
                print(f"    Action: {rec}")

    return boundary_report


