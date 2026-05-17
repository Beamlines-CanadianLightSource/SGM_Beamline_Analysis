import numpy as np
import h5py
import sys
import os
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from analyze_sgm_bsky_data import analyze_sgm_bsky_data

def get_user_file_action(filename, output_path):
    """
    Pops up a GUI dialog to ask the user for action if a file exists.
    Returns 'overwrite', 'rename', 'skip', or None (if dialog is closed).
    """
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Try to bring the window to the front

    action = None
    msg = f"The file '{filename}' already exists.\n\nDo you want to overwrite it?"
    response = messagebox.askyesnocancel("File Exists", msg)

    if response is True: # Yes (Overwrite)
        action = 'overwrite'
    elif response is False: # No (Rename)
        new_filename = simpledialog.askstring("Rename File", "Enter new filename:", initialvalue=filename, parent=root)
        if new_filename:
            if not new_filename.lower().endswith(('.h5', '.hdf5')):
                new_filename += '.h5'
            action = ('rename', os.path.join(os.path.dirname(output_path), new_filename))
        else: # User cancelled rename dialog
            action = 'skip'
    else: # Cancel (or dialog closed)
        action = 'skip'
    
    root.destroy() # Clean up the Tkinter window
    return action

def load_external_i0(i0_path, target_energies):
    """
    Loads an external I0 CSV file and interpolates it to match target energies.
    Expects CSV with columns like 'Energy' and 'I0' or 'Intensity'.
    """
    try:
        import pandas as pd
        df = pd.read_csv(i0_path, comment='#')
        # Try to find energy and intensity columns
        e_col = next((c for c in df.columns if 'energy' in c.lower()), df.columns[0])
        i_col = next((c for c in df.columns if any(k in c.lower() for k in ['i0', 'intensity', 'norm', 'tey'])), df.columns[1])
        
        src_e = df[e_col].values
        src_i = df[i_col].values
        
        # Interpolate to target energies
        # use np.interp for basic linear interpolation
        interp_i0 = np.interp(target_energies, src_e, src_i)
        return interp_i0
    except Exception as e:
        print(f"Error loading external I0 from {i0_path}: {e}", file=sys.stderr)
        return None

def save_pymca_stack_h5(path_pack, output_path=None, channel_roi=None, map_roi=None, external_i0_path=None):
    """
    Converts the analyzed stack data into a PyMca-compatible HDF5 file.
    
    Saves each energy step as a separate NXentry (entry_0000, entry_0001, ...).
    This structure mimics a series of individual scans, which PyMca handles robustly as a stack.
    """
    if not path_pack or not path_pack.get('sdd_files'):
        print("Error: Could not retrieve data paths or no SDD files found.", file=sys.stderr)
        return

    # --- Determine the final save path with user interaction ---
    final_save_path = output_path
    
    # If no output path is provided, or if we want to give the user the chance to rename/pick folder
    if not final_save_path:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Get source folder from path_pack (h5_file_path is stored there in analyze_sgm_bsky_data)
        h5_src = path_pack.get('h5_file_path', '')
        initial_dir = os.path.dirname(h5_src) if h5_src else os.getcwd()
        initial_name = os.path.splitext(os.path.basename(h5_src))[0] + "_PCA-CA.h5" if h5_src else "pca_ca_stack.h5"
        
        final_save_path = filedialog.asksaveasfilename(
            title="Save PyMca-compatible HDF5 Stack",
            initialdir=initial_dir,
            initialfile=initial_name,
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        root.destroy()
        
        if not final_save_path:
            print("    -> Save operation cancelled by user.")
            return

    # --- Check if file exists and prompt user (only if not already handled by filedialog) ---
    # Note: asksaveasfilename already handles the overwrite prompt, but we keep this for CLI safety
    if os.path.exists(final_save_path) and final_save_path == output_path:
        action = get_user_file_action(os.path.basename(final_save_path), final_save_path)
        if action == 'overwrite':
            print(f"    -> Overwriting {os.path.basename(final_save_path)}.")
        elif action == 'skip':
            print(f"    -> Skipping save operation for {os.path.basename(final_save_path)}.")
            return
        elif isinstance(action, tuple) and action[0] == 'rename':
            final_save_path = action[1]
            print(f"    -> Saving as {os.path.basename(final_save_path)}.")
        else:
            print("    -> Save operation cancelled.")
            return

    x_raw = path_pack.get('x', np.array([]))
    y_raw = path_pack.get('y', np.array([]))
    if x_raw.size == 0 or y_raw.size == 0:
        print("Error: Coordinate data not found.", file=sys.stderr)
        return

    # --- Alignment and Trim Extraction ---
    roll_shift = path_pack.get('roll_shift', 0)
    x_trim = path_pack.get('x_trim', 0.0)
    y_trim = path_pack.get('y_trim', 0.0)
    
    if roll_shift != 0:
        print(f"    -> Applying Roll Shift: {roll_shift}")
    if x_trim > 0 or y_trim > 0:
        print(f"    -> Applying Spatial Trim: X={x_trim}, Y={y_trim}")
        
    # --- Spatial Masking (ROI + Trim) ---
    # Calculate full scan bounds first (needed for trim and fallback)
    x_min_full, x_max_full = np.min(x_raw), np.max(x_raw)
    y_min_full, y_max_full = np.min(y_raw), np.max(y_raw)

    # Get map_roi from argument, then path_pack, then fall back to full scan range
    if map_roi is None:
        map_roi = path_pack.get('map_roi')

    if map_roi:
        x1_req, x2_req = sorted([map_roi[0], map_roi[1]])
        y1_req, y2_req = sorted([map_roi[2], map_roi[3]])
        print(f"    -> Using map_roi: X=[{x1_req:.3f}, {x2_req:.3f}], Y=[{y1_req:.3f}, {y2_req:.3f}]")
    else:
        x1_req, x2_req = x_min_full, x_max_full
        y1_req, y2_req = y_min_full, y_max_full
        print(f"    -> No map_roi provided; using full scan range.")

    if channel_roi is None:
        channel_roi = path_pack.get('channel_roi', (0, 255))
        print(f"    -> Inherited channel_roi from context: {channel_roi}")
    else:
        print(f"    -> Using provided channel_roi: {channel_roi}")

    # Trim the effective area based on x_trim/y_trim from alignment
    eb_x1, eb_x2 = x_min_full + x_trim, x_max_full - x_trim
    eb_y1, eb_y2 = y_min_full + y_trim, y_max_full - y_trim
    
    # Final ROI is intersection of requested map_roi and the alignment valid area
    final_x1 = max(x1_req, eb_x1)
    final_x2 = min(x2_req, eb_x2)
    final_y1 = max(y1_req, eb_y1)
    final_y2 = min(y2_req, eb_y2)
    
    print(f"    -> Target Export ROI: X=[{final_x1:.3f}, {final_x2:.3f}], Y=[{final_y1:.3f}, {final_y2:.3f}]")
    
    mask = (x_raw >= final_x1) & (x_raw <= final_x2) & (y_raw >= final_y1) & (y_raw <= final_y2)
    valid_points = np.where(mask)[0]
    
    if valid_points.size == 0:
        print("Error: No data points found within the specified ROI and trim boundaries.", file=sys.stderr)
        return
        
    x_filtered = x_raw[mask]
    y_filtered = y_raw[mask]

    all_energies = np.array(sorted(path_pack['energies']))
    detector_names = sorted(path_pack['sdd_files'].keys())
    scan_name = path_pack.get('scan_name', 'stack_scan')
    
    # --- Grid Determination (on filtered data) ---
    dx = np.abs(np.diff(x_filtered))
    valid_dx = dx[dx > 1e-5]
    step_x = np.median(valid_dx) if valid_dx.size > 0 else 1.0
    
    dy = np.abs(np.diff(y_filtered))
    valid_dy = dy[dy > 1e-5]
    step_y = np.median(valid_dy) if valid_dy.size > 0 else 1.0
    
    min_xf, max_xf = np.min(x_filtered), np.max(x_filtered)
    min_yf, max_yf = np.min(y_filtered), np.max(y_filtered)
    
    nx = int(np.round((max_xf - min_xf) / step_x)) + 1
    ny = int(np.round((max_yf - min_yf) / step_y)) + 1
    
    # Re-calculate indices relative to filtered bounds
    ix = np.round((x_raw - min_xf) / step_x).astype(int)
    iy = np.round((y_raw - min_yf) / step_y).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # --- Pre-allocate 3D Stacks (Y, X, Energy) ---
    num_energies = len(all_energies)
    detector_stacks = {det: np.zeros((ny, nx, num_energies), dtype=np.float32) for det in detector_names}
    sum_stack = np.zeros((ny, nx, num_energies), dtype=np.float32)
    
    # --- Create HDF5 File ---
    try:
        if not isinstance(final_save_path, (str, bytes, os.PathLike)):
            print(f"Error: output_path must be a string or path-like object, got {type(final_save_path)}: {final_save_path}", file=sys.stderr)
            return

        with h5py.File(final_save_path, 'w') as f:
            f.attrs['NX_class'] = 'NXroot'
            f.attrs['default'] = 'entry/measurement'
            f.attrs['creator'] = 'save_pymca_stack_h5.py'
            
            # Create the single Nexus-compliant Entry
            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'
            entry.attrs['default'] = 'measurement'
            entry.attrs['exit_slit_gap'] = path_pack.get('exit_slit_gap', 'N/A') # Placeholder for future data
            entry.create_dataset('title', data=f"{scan_name} Multi-Detector Stack")


            measurement = entry.create_group('measurement')
            measurement.attrs['NX_class'] = 'NXdata'
            measurement.attrs['signal'] = 'average' # Default to the Average of all detectors
            
            # Define Axes
            measurement.attrs['axes'] = [b'y', b'x', b'energy']
            measurement.attrs['y_indices'] = np.array([0], dtype=np.int32)
            measurement.attrs['x_indices'] = np.array([1], dtype=np.int32)
            measurement.attrs['energy_indices'] = np.array([2], dtype=np.int32)

            # Create Axes datasets (shared by all stacks)
            x_axis = np.linspace(min_xf, max_xf, nx).astype(np.float32)
            y_axis = np.linspace(max_yf, min_yf, ny).astype(np.float32)
            
            y_ds = measurement.create_dataset('y', data=y_axis)
            y_ds.attrs['units'] = 'mm'
            y_ds.attrs['long_name'] = 'Y'
            y_ds.attrs['axis'] = 1 
            
            x_ds = measurement.create_dataset('x', data=x_axis)
            x_ds.attrs['units'] = 'mm'
            x_ds.attrs['long_name'] = 'X'
            x_ds.attrs['axis'] = 2
            
            # Final Energy Axis
            final_energies = all_energies.astype(np.float32)
            if path_pack.get('i0_calib_enabled') and "Internal" not in i0_source and "mcc" not in i0_source.lower():
                final_energies += path_pack.get('i0_energy_shift', 0.0)
            
            en_ds = measurement.create_dataset('energy', data=final_energies)
            en_ds.attrs['units'] = 'eV'
            en_ds.attrs['long_name'] = 'Energy (eV)'
            en_ds.attrs['axis'] = 3 
            
            # --- Handle I0 Normalization ---
            i0_values = None
            i0_source = "mcc1"
            
            if 'ext_i0_values' in path_pack and path_pack['ext_i0_values'] is not None:
                i0_values = path_pack['ext_i0_values']
                i0_source = path_pack.get('i0_source', 'External')
                print(f"    -> Using user-selected I0 for normalization: {i0_source}")
            elif external_i0_path and os.path.exists(external_i0_path):
                print(f"    -> Using External I0 for normalization: {os.path.basename(external_i0_path)}")
                i0_values = load_external_i0(external_i0_path, all_energies)
                i0_source = f"External: {os.path.basename(external_i0_path)}"
            
            if i0_values is None:
                # Use internal mcc1
                mcc1_means = []
                mcc_data_dict = path_pack.get('mcc_data', {})
                mcc_channels = path_pack.get('mcc_channel_names', [])
                
                mcc1_idx = -1
                for idx, name in enumerate(mcc_channels):
                    if any(k == name.lower().strip() for k in ['ch1', 'mcc1', '1']):
                        mcc1_idx = idx
                        break
                        
                if mcc1_idx != -1 and mcc_data_dict:
                    for energy in all_energies:
                        energy_mcc = mcc_data_dict.get(energy)
                        if energy_mcc is not None and energy_mcc.shape[0] > 0:
                            m_mcc = mask[:energy_mcc.shape[0]]
                            if np.any(m_mcc):
                                # Check if it's 1D or 2D. Sometimes if only 1 pixel, it's 1D
                                if len(energy_mcc.shape) > 1:
                                    mcc1_col = energy_mcc[:, mcc1_idx]
                                else:
                                    mcc1_col = np.array([energy_mcc[mcc1_idx]])
                                mean_val = mcc1_col[m_mcc].mean()
                                mcc1_means.append(0.0 if np.isnan(mean_val) else mean_val)
                            else:
                                mcc1_means.append(0.0)
                        else:
                            mcc1_means.append(0.0)
                
                if np.any(mcc1_means):
                    i0_values = np.array(mcc1_means)
                    print("    -> Using Internal I0 (mcc1) for normalization.")
                else:
                    print("    -> Warning: No I0 found (mcc1 or external). Results will not be normalized.", file=sys.stderr)
                    i0_values = np.ones(len(all_energies))
                    i0_source = "None"
            
            # Ensure I0 has no zeros
            i0_values = np.where(i0_values <= 0, 1.0, i0_values)
            
            # Apply manual energy calibration if enabled (External I0 Only)
            if path_pack.get('i0_calib_enabled') and "Internal" not in i0_source and "mcc" not in i0_source.lower():
                shift = path_pack.get('i0_energy_shift', 0.0)
                if shift != 0:
                    i0_source += f" (Energy Shifted by {shift:+.2f} eV)"
                    print(f"    -> Applied manual energy shift: {shift:+.2f} eV")
            
            i0_ds = measurement.create_dataset('i0', data=i0_values.astype(np.float32))

            # Store ROI and I0 metadata
            measurement.attrs['channel_roi'] = channel_roi
            ipfy_val = path_pack.get('ipfy_mode', False)
            measurement.attrs['ipfy_mode'] = ipfy_val
            if ipfy_val:
                print(f"    -> [IPFY] IPFY Mode FLAG saved to HDF5 metadata: True")
            measurement.attrs['i0_source'] = i0_source

            # --- Metadata Group ---
            meta_group = f.create_group('stack_metadata')
            
            # Recalculate Energy Regions based on the actual energies being saved
            energy_regions = "N/A"
            try:
                from analyze_sgm_bsky_data import detect_energy_regions
                energy_regions = detect_energy_regions(all_energies)
            except Exception:
                pass

            meta_keys = ['scan_name', 'project', 'date', 'grating', 'harmonic', 'strip', 
                         'polarization', 'exit_slit_gap', 'command', 'coordinates', 'xps_z', 
                         'time_per_map', 'number_of_points']
            for key in meta_keys:
                if key in path_pack:
                    val = path_pack[key]
                    if val is None: val = 'N/A'
                    meta_group.attrs[key] = str(val)
            
            # Add processing metadata
            meta_group.attrs['Energy Regions'] = energy_regions
            meta_group.attrs['i0_source'] = i0_source
            meta_group.attrs['nx'] = nx
            meta_group.attrs['ny'] = ny
            meta_group.attrs['normalized'] = "Yes" if i0_source != "None" else "No"
            meta_group.attrs['roll_shift'] = roll_shift
            meta_group.attrs['x_trim'] = x_trim
            meta_group.attrs['y_trim'] = y_trim

            # Iterate through energies to populate the stacks
            print(f"Processing {num_energies} energy steps for {len(detector_names)} detectors...")
            
            for en_idx, energy in enumerate(all_energies):
                # Total summed grid for this energy (for sum/average)
                total_sum_grid = np.zeros((ny, nx), dtype=np.float32)
                
                for det_name in detector_names:
                    sdd_filepath = path_pack['sdd_files'][det_name].get(energy)
                    
                    if not sdd_filepath or not os.path.exists(sdd_filepath):
                        continue
                        
                    try:
                        data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
                        n_spec = len(data_1d) // 256
                        
                        if n_spec == 0:
                            continue
                            
                        limit = min(n_spec, len(ix))
                        spectra = data_1d[:limit*256].reshape((limit, 256))
                        
                        # Apply roll_shift if present
                        if roll_shift != 0:
                            spectra = np.roll(spectra, shift=roll_shift, axis=0)
                        
                        # Sum across the specified channels for each pixel
                        ch_start, ch_end = channel_roi
                        spectra_roi = np.sum(spectra[:, ch_start:ch_end+1], axis=1).astype(np.float32)
                        
                        # Map to grid, ONLY for points within the final mask
                        grid_2d = np.zeros((ny, nx), dtype=np.float32)
                        
                        # Extract ix/iy/roi for the valid masked points within this file's limit
                        limit_mask = mask.copy()
                        limit_mask[limit:] = False
                        
                        # Points that are both in the file AND in the spatial mask
                        valid_map_indices = np.where(limit_mask)[0]
                        
                        if valid_map_indices.size > 0:
                            # Map 1D filtered data onto 2D grid
                            grid_2d[iy[valid_map_indices], ix[valid_map_indices]] = spectra_roi[valid_map_indices]
                        
                        # Normalize by I0 for this energy step
                        grid_2d = grid_2d / i0_values[en_idx]
                        
                        flipped_grid = np.flipud(grid_2d)
                        
                        # Store in individual detector stack
                        detector_stacks[det_name][:, :, en_idx] = flipped_grid
                        
                        # Add to overall total for this energy
                        total_sum_grid += flipped_grid
                            
                    except Exception as e:
                        print(f"    -> Error processing {det_name} at {energy} eV: {e}", file=sys.stderr)
                
                # Store the combined total in the sum stack
                sum_stack[:, :, en_idx] = total_sum_grid
                
                if (en_idx + 1) % 10 == 0 or en_idx == num_energies - 1:
                    print(f"    -> Progress: {en_idx + 1}/{num_energies} energies processed.")

            # Save Individual Detector Stacks
            for det_name, stack_data in detector_stacks.items():
                ds = measurement.create_dataset(det_name, data=stack_data, compression="gzip")
                ds.attrs['interpretation'] = 'spectrum'
                ds.attrs['long_name'] = f"{det_name} ROI Sum"

            # Save the Overall Sum
            sum_ds = measurement.create_dataset('sum', data=sum_stack, compression="gzip")
            sum_ds.attrs['interpretation'] = 'spectrum'
            sum_ds.attrs['long_name'] = "Sum of All Detectors"

            # Calculate and Save the Average
            avg_stack = sum_stack / max(1, len(detector_names))
            avg_ds = measurement.create_dataset('average', data=avg_stack, compression="gzip")
            avg_ds.attrs['interpretation'] = 'spectrum'
            avg_ds.attrs['long_name'] = "Average of Detectors"

        print(f"\nSuccessfully saved PyMca-compatible HDF5 stack (multi-detector) to: {final_save_path}")
        print(f"Datasets: {', '.join(detector_names)}, sum, average")
        print(f"Default Signal: average")
        return final_save_path
        
    except Exception as e:
        print(f"Error creating HDF5 file: {e}", file=sys.stderr)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert analyzed stack data to a PyMca-compatible HDF5 file.")
    parser.add_argument("h5_file_path", help="Path to the source HDF5 file from a stack scan.")
    parser.add_argument("output_h5_path", nargs='?', default=None, help="Optional: Path to the output HDF5 file. If omitted, a dialog will appear.")
    parser.add_argument("channel_roi_start", type=int, help="The start channel for the ROI sum.")
    parser.add_argument("channel_roi_end", type=int, help="The end channel for the ROI sum.")
    parser.add_argument("--map_roi", type=float, nargs=4, metavar=('X1', 'X2', 'Y1', 'Y2'),
                        help="Define a rectangular spatial ROI for spectrum calculations, e.g., --map_roi -1 1 -0.5 0.5")

    args = parser.parse_args()
    
    print(f"Analyzing stack file: {args.h5_file_path}...")
    path_pack = analyze_sgm_bsky_data(args.h5_file_path)
    
    if path_pack:
        # For stack data, map_roi is not directly used for the 4D cube, but it's part of the context
        # We'll pass it along, though it's not directly used in the current implementation of save_pymca_stack_h5
        # If map_roi is not provided, we can default to the full scan range
        current_map_roi = args.map_roi
        if current_map_roi is None:
            x_coords = path_pack.get('x', np.array([]))
            y_coords = path_pack.get('y', np.array([]))
            if x_coords.size > 0 and y_coords.size > 0:
                current_map_roi = [np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)]
            else:
                current_map_roi = [0, 0, 0, 0] # Default empty if no coords
            print(f"Using full scan range as map_roi: {current_map_roi}")

        save_pymca_stack_h5(path_pack, 
                            output_path=args.output_h5_path, 
                            channel_roi=(args.channel_roi_start, args.channel_roi_end),
                            map_roi=current_map_roi) # Pass map_roi, even if not directly used for 4D cube
    else:
        print("Failed to analyze stack. Aborting save to PyMca HDF5.", file=sys.stderr)