import numpy as np
import h5py
import sys
import os
import tkinter as tk
from tkinter import messagebox, filedialog
from analyze_sgm_bsky_data import analyze_sgm_bsky_data

def save_pymca_4d_stack_h5(path_pack, output_path=None, normalize=True):
    """
    Converts an entire analyzed stack into a single 4D PyMca-compatible HDF5 file.
    
    Structure: (n_energies, ny, nx, n_channels)
    
    Allows PyMca to:
    1. Scroll through energy points.
    2. Pick any pixel and see the full XRF spectrum.
    3. Perform PCA on XANES or XRF.
    4. Batch fit elemental distributions across the whole stack.
    """
    if not path_pack or not path_pack.get('sdd_files'):
        print("Error: Could not retrieve data paths or no SDD files found.", file=sys.stderr)
        return

    # --- Determine the final save path ---
    final_save_path = output_path
    if not final_save_path:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        h5_src = path_pack.get('h5_file_path', '')
        initial_dir = os.path.dirname(h5_src) if h5_src else os.getcwd()
        initial_name = os.path.splitext(os.path.basename(h5_src))[0] + "_PyMca_4D.h5" if h5_src else "stack_4d_pymca.h5"
        
        final_save_path = filedialog.asksaveasfilename(
            title="Save 4D PyMca HDF5 Stack",
            initialdir=initial_dir,
            initialfile=initial_name,
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        root.destroy()
        if not final_save_path:
            print("    -> Save operation cancelled.")
            return

    # --- Extraction of Metadata ---
    x_raw = path_pack.get('x', np.array([]))
    y_raw = path_pack.get('y', np.array([]))
    if x_raw.size == 0 or y_raw.size == 0:
        print("Error: Coordinate data not found.", file=sys.stderr)
        return

    all_energies = np.array(sorted(path_pack['energies']))
    detector_names = sorted(path_pack['sdd_files'].keys())
    num_energies = len(all_energies)
    n_channels = 256
    
    # Alignment and Trim
    roll_shift = path_pack.get('roll_shift', 0)
    x_trim = path_pack.get('x_trim', 0.0)
    y_trim = path_pack.get('y_trim', 0.0)
    
    # Grid Determination
    min_x, max_x = np.min(x_raw), np.max(x_raw)
    min_y, max_y = np.min(y_raw), np.max(y_raw)
    
    # Simple grid inference
    dx = np.abs(np.diff(x_raw))
    valid_dx = dx[dx > 1e-5]
    step_x = np.median(valid_dx) if valid_dx.size > 0 else 1.0
    
    dy = np.abs(np.diff(y_raw))
    valid_dy = dy[dy > 1e-5]
    step_y = np.median(valid_dy) if valid_dy.size > 0 else 1.0
    
    nx = int(np.round((max_x - min_x) / step_x)) + 1
    ny = int(np.round((max_y - min_y) / step_y)) + 1
    
    print(f"Detected Grid: {ny} x {nx}")
    print(f"Energies: {num_energies}")
    
    # Index mapping
    ix = np.round((x_raw - min_x) / step_x).astype(int)
    iy = np.round((y_raw - min_y) / step_y).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # --- Handle Normalization (I0) ---
    i0_values = np.ones(num_energies)
    i0_source = "None"
    
    if normalize:
        if 'ext_i0_values' in path_pack and path_pack['ext_i0_values'] is not None:
            i0_values = path_pack['ext_i0_values']
            i0_source = path_pack.get('i0_source', 'External')
        else:
            # Try internal mcc1
            mcc1_means = []
            mcc_data_dict = path_pack.get('mcc_data', {})
            mcc_channels = path_pack.get('mcc_channel_names', [])
            mcc1_idx = -1
            for idx, name in enumerate(mcc_channels):
                if any(k in name.lower().strip() for k in ['ch1', 'mcc1', '1']):
                    mcc1_idx = idx
                    break
            
            if mcc1_idx != -1 and mcc_data_dict:
                for energy in all_energies:
                    energy_mcc = mcc_data_dict.get(energy)
                    if energy_mcc is not None and energy_mcc.shape[0] > 0:
                        val = energy_mcc[:, mcc1_idx].mean()
                        mcc1_means.append(val if not np.isnan(val) else 1.0)
                    else:
                        mcc1_means.append(1.0)
                if np.any(np.array(mcc1_means) != 1.0):
                    i0_values = np.array(mcc1_means)
                    i0_source = "mcc1"
        
        # Avoid division by zero
        i0_values = np.where(i0_values == 0, 1.0, i0_values)
        print(f"Normalizing by {i0_source}...")

    # --- Process and Save to HDF5 ---
    try:
        with h5py.File(final_save_path, 'w') as f:
            f.attrs['NX_class'] = 'NXroot'
            f.attrs['default'] = 'entry/measurement'
            f.attrs['creator'] = 'save_pymca_4d_stack_h5.py'
            
            entry = f.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'
            entry.attrs['default'] = 'xrf_measurement'
            
            x_axis = np.linspace(min_x, max_x, nx).astype(np.float32)
            y_axis = np.linspace(max_y, min_y, ny).astype(np.float32)
            chan_axis = np.arange(n_channels).astype(np.float32)
            en_axis = all_energies.astype(np.float32)

            # --- 1. XRF 3D Measurement (Y, X, Channels) ---
            xrf_meas = entry.create_group('xrf_measurement')
            xrf_meas.attrs['NX_class'] = 'NXdata'
            xrf_meas.attrs['axes'] = ['y', 'x', 'channels']
            xrf_meas.attrs['signal'] = 'sdd_xrf_stack_3d'
            xrf_meas.attrs['y_indices'] = np.array([0], dtype=np.int32)
            xrf_meas.attrs['x_indices'] = np.array([1], dtype=np.int32)
            xrf_meas.attrs['channels_indices'] = np.array([2], dtype=np.int32)
            xrf_meas.attrs['i0_source'] = i0_source
            xrf_meas.create_dataset('y', data=y_axis)
            xrf_meas.create_dataset('x', data=x_axis)
            xrf_meas.create_dataset('channels', data=chan_axis)

            # --- 2. XANES 3D Measurement (Y, X, Energy) ---
            xanes_meas = entry.create_group('xanes_measurement')
            xanes_meas.attrs['NX_class'] = 'NXdata'
            xanes_meas.attrs['axes'] = ['y', 'x', 'energy']
            xanes_meas.attrs['signal'] = 'sdd_xanes_stack_3d'
            xanes_meas.attrs['y_indices'] = np.array([0], dtype=np.int32)
            xanes_meas.attrs['x_indices'] = np.array([1], dtype=np.int32)
            xanes_meas.attrs['energy_indices'] = np.array([2], dtype=np.int32)
            xanes_meas.attrs['i0_source'] = i0_source
            xanes_meas.create_dataset('y', data=y_axis)
            xanes_meas.create_dataset('x', data=x_axis)
            xanes_meas.create_dataset('energy', data=en_axis)

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
                         'polarization', 'exit_slit_gap', 'command', 'coordinates']
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
            meta_group.attrs['normalized'] = "Yes" if normalize else "No"
            meta_group.attrs['roll_shift'] = roll_shift
            meta_group.attrs['x_trim'] = x_trim
            meta_group.attrs['y_trim'] = y_trim

            # --- 3. Full 4D Measurement (Y, X, Energy, Channels) ---
            full_meas = entry.create_group('full_4d_measurement')
            full_meas.attrs['NX_class'] = 'NXdata'
            full_meas.attrs['axes'] = ['y', 'x', 'energy', 'channels']
            full_meas.attrs['y_indices'] = np.array([0], dtype=np.int32)
            full_meas.attrs['x_indices'] = np.array([1], dtype=np.int32)
            full_meas.attrs['energy_indices'] = np.array([2], dtype=np.int32)
            full_meas.attrs['channels_indices'] = np.array([3], dtype=np.int32)
            full_meas.attrs['i0_source'] = i0_source
            full_meas.create_dataset('y', data=y_axis)
            full_meas.create_dataset('x', data=x_axis)
            full_meas.create_dataset('energy', data=en_axis)
            full_meas.create_dataset('channels', data=chan_axis)

            has_multiple = len(detector_names) > 1

            # Create Datasets
            # XRF 3D
            xrf_dataset_3d = xrf_meas.create_dataset('sdd_xrf_stack_3d', (ny, nx, n_channels), 
                                                     dtype=np.float32, compression="gzip")
            xrf_dataset_3d.attrs['interpretation'] = 'spectrum'
            xrf_dataset_3d.attrs['long_name'] = "XRF Stack (Energy Sum)"

            # XANES 3D
            roi_ch = path_pack.get('channel_roi', (0, 255))
            sum_dataset_3d = xanes_meas.create_dataset('sdd_xanes_stack_3d', (ny, nx, num_energies), 
                                                       dtype=np.float32, compression="gzip")
            sum_dataset_3d.attrs['interpretation'] = 'spectrum' 
            sum_dataset_3d.attrs['long_name'] = f"XANES Stack (Ch {roi_ch[0]}-{roi_ch[1]})"

            # FULL 4D
            if has_multiple:
                sum_dataset_4d = full_meas.create_dataset('sdd_sum_4d', (ny, nx, num_energies, n_channels), 
                                                          dtype=np.float32, compression="gzip")
                sum_dataset_4d.attrs['interpretation'] = 'spectrum'
                full_meas.attrs['signal'] = 'sdd_sum_4d'
            else:
                full_meas.attrs['signal'] = detector_names[0]

            det_datasets_4d = {}
            for det in detector_names:
                ds = full_meas.create_dataset(det, (ny, nx, num_energies, n_channels), 
                                              dtype=np.float32, compression="gzip")
                ds.attrs['interpretation'] = 'spectrum'
                det_datasets_4d[det] = ds

            # In-memory accumulator for XRF 3D stack
            xrf_sum_accum = np.zeros((ny, nx, n_channels), dtype=np.float32)

            # Loop through energies to populate
            print(f"Building stacks (XRF 3D, XANES 3D, Full 4D Hypercube)...")
            
            for en_idx, energy in enumerate(all_energies):
                norm_factor = i0_values[en_idx]
                energy_sum_4d = np.zeros((ny, nx, n_channels), dtype=np.float32)
                
                for det in detector_names:
                    sdd_path = path_pack['sdd_files'][det].get(energy)
                    if not sdd_path or not os.path.exists(sdd_path):
                        continue
                    
                    try:
                        data_1d = np.fromfile(sdd_path, dtype=np.uint32)
                        limit = min(len(data_1d) // n_channels, len(ix))
                        if limit == 0: continue
                        
                        spectra = data_1d[:limit*n_channels].reshape((limit, n_channels)).astype(np.float32)
                        
                        if roll_shift != 0:
                            spectra = np.roll(spectra, shift=roll_shift, axis=0)
                        
                        # Map to grid
                        grid_3d = np.zeros((ny, nx, n_channels), dtype=np.float32)
                        grid_3d[iy[:limit], ix[:limit], :] = spectra
                        grid_3d = np.flipud(grid_3d) # Match coordinate convention
                        
                        if normalize:
                            grid_3d /= norm_factor
                        
                        # Write to 4D individual detector (Energy is index 2)
                        det_datasets_4d[det][:, :, en_idx, :] = grid_3d
                        
                        if has_multiple:
                            energy_sum_4d += grid_3d
                            
                    except Exception as e:
                        print(f"Error at {energy} eV ({det}): {e}")

                if has_multiple:
                    sum_dataset_4d[:, :, en_idx, :] = energy_sum_4d
                    xrf_sum_accum += energy_sum_4d
                    # Calculate ROI sum for 3D stack
                    sum_dataset_3d[:, :, en_idx] = np.sum(energy_sum_4d[:, :, roi_ch[0]:roi_ch[1]], axis=2)
                else:
                    # Single detector fallback
                    det0 = detector_names[0]
                    single_4d = det_datasets_4d[det0][:, :, en_idx, :]
                    xrf_sum_accum += single_4d
                    sum_dataset_3d[:, :, en_idx] = np.sum(single_4d[:, :, roi_ch[0]:roi_ch[1]], axis=2)

                if (en_idx + 1) % 20 == 0 or en_idx == num_energies - 1:
                    print(f"    -> Progress: {en_idx + 1}/{num_energies} energies.")

            # Save XRF Accumulator to dataset
            xrf_dataset_3d[:] = xrf_sum_accum

        print(f"\n[SUCCESS] Multi-mode HDF5 saved to: {final_save_path}")
        print("1. For XRF ROI Imaging: Open 'xrf_measurement/sdd_xrf_stack_3d'")
        print("2. For XANES PCA: Open 'xanes_measurement/sdd_xanes_stack_3d' -> 'Load as 1D Stack'")
        print("3. For Full Hypercube: Open 'full_4d_measurement'")
        return final_save_path

    except Exception as e:
        print(f"Error creating 4D HDF5: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_pymca_4d_stack_h5.py [h5_stack_file]")
    else:
        src_h5 = sys.argv[1]
        print(f"Analyzing stack: {src_h5}")
        pp = analyze_sgm_bsky_data(src_h5)
        if pp:
            save_pymca_4d_stack_h5(pp)
