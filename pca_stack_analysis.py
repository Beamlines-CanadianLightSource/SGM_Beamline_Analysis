import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import traceback
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
import mplcursors
import dask.array as da
from dask import delayed, compute

def pca_stack_analysis(path_pack, n_components=5):
    """
    Performs PCA on the full hyperspectral stack for each detector using Dask for memory efficiency.
    For each spatial point, it concatenates the spectra from all energies.
    Plots the score maps and 2D loading maps for the principal components.

    Args:
        path_pack (dict): The output from analyze_stack.
        n_components (int): The number of principal components to compute and plot.
    """
    try:
        if not path_pack or not path_pack.get('sdd_files'):
            print("Error: Could not retrieve data paths or no SDD files found.", file=sys.stderr)
            return
            
        x_coords = path_pack.get('x', np.array([]))
        y_coords = path_pack.get('y', np.array([]))
        if x_coords.size == 0 or y_coords.size == 0:
            print("Error: Coordinate data not found.", file=sys.stderr)
            return

        all_energies = np.array(sorted(path_pack['energies']))
        detector_names = sorted(path_pack['sdd_files'].keys())
        if not detector_names:
            print("No SDD detectors found to analyze.", file=sys.stderr)
            return

        # Helper function for lazy loading
        def load_energy_slice(filepath, n_pixels, pixels_per_spectrum=256):
            if not filepath or not os.path.exists(filepath):
                return np.full((n_pixels, pixels_per_spectrum), np.nan, dtype=np.float32)
            
            try:
                data_1d = np.fromfile(filepath, dtype=np.uint32)
                num_spectra = len(data_1d) // pixels_per_spectrum
                
                if num_spectra != n_pixels:
                    # Handle mismatch
                    spectra_2d = np.full((n_pixels, pixels_per_spectrum), np.nan, dtype=np.float32)
                    limit = min(num_spectra, n_pixels)
                    spectra_2d[:limit, :] = data_1d[:limit * pixels_per_spectrum].reshape((limit, pixels_per_spectrum))
                    return spectra_2d
                else:
                    return data_1d.reshape((num_spectra, pixels_per_spectrum)).astype(np.float32)
            except Exception:
                return np.full((n_pixels, pixels_per_spectrum), np.nan, dtype=np.float32)

        # --- Main Loop for each Detector ---
        for det_name in detector_names:
            print(f"\n--- Starting PCA for Detector: {det_name} across all energies (Dask Accelerated) ---")
            
            # --- 1. Load and Assemble Hyperspectral Data Lazily ---
            dask_arrays = []
            pixels_per_spectrum = 256
            
            for energy in all_energies:
                sdd_filepath = path_pack['sdd_files'][det_name].get(energy)
                
                # Create delayed object
                d = delayed(load_energy_slice)(sdd_filepath, x_coords.size, pixels_per_spectrum)
                # Create dask array from delayed
                # We know the shape is (n_pixels, 256)
                d_arr = da.from_delayed(d, shape=(x_coords.size, pixels_per_spectrum), dtype=np.float32)
                dask_arrays.append(d_arr)

            # Stack along a new axis (axis=1) -> (num_points, num_energies, num_channels)
            hyperspectral_stack = da.stack(dask_arrays, axis=1)
            
            # Reshape to (num_points, num_energies * num_channels)
            num_points, num_energies, num_channels = hyperspectral_stack.shape
            flattened_hyperspectral = hyperspectral_stack.reshape(num_points, -1)
            
            # Rechunk to ensure efficient processing
            # Chunking along the first dimension (samples) allows us to process subsets of pixels
            # while keeping all features for those pixels together.
            flattened_hyperspectral = flattened_hyperspectral.rechunk({0: 'auto', 1: -1})
            
            print(f"    -> Assembled dask array shape: {flattened_hyperspectral.shape}")

            # --- 2. Preprocess and Perform PCA ---
            # Handle NaNs: remove any spatial point (row) that has any NaN value
            print("    -> Computing valid data mask...")
            # We compute the mask eagerly to filter the data
            is_nan = da.isnan(flattened_hyperspectral).any(axis=1)
            valid_mask_computed = (~is_nan).compute()
            
            if np.sum(valid_mask_computed) == 0:
                print("    -> Error: No valid data points found after handling missing files. Skipping PCA.", file=sys.stderr)
                continue
            
            final_data = flattened_hyperspectral[valid_mask_computed]
            final_x = x_coords[valid_mask_computed]
            final_y = y_coords[valid_mask_computed]
            print(f"    -> Found {final_data.shape[0]} valid data points for PCA.")

            print("    -> Scaling data (StandardScaler equivalent)...")
            # Compute mean and std for scaling
            mean = final_data.mean(axis=0)
            std = final_data.std(axis=0)
            # Avoid division by zero
            std = da.where(std == 0, 1.0, std)
            
            scaled_spectra = (final_data - mean) / std

            print("    -> Performing Randomized SVD via Dask...")
            # svd_compressed is efficient for tall/fat matrices
            u, s, v = da.linalg.svd_compressed(scaled_spectra, k=n_components)
            
            # Calculate explained variance ratio
            n_samples = final_data.shape[0]
            n_features = final_data.shape[1]
            explained_variance = (s**2) / (n_samples - 1)
            total_variance = n_features # Since data is standardized
            explained_variance_ratio_dask = explained_variance / total_variance
            
            # Compute results
            print("    -> Computing final PCA results...")
            # u: (n_samples, k), s: (k,), v: (k, n_features)
            # Scores = u * s
            scores_dask = u * s
            
            scores_val, loadings_val, evr_val = compute(scores_dask, v, explained_variance_ratio_dask)
            
            print(f"    -> Explained variance ratio: {evr_val}")

            # --- 3. Visualize Results ---
            print("    -> Generating plots...")
            
            fig, axes = plt.subplots(n_components, 2, figsize=(12, 3.5 * n_components), squeeze=False)
            scan_name = path_pack.get('scan_name', 'N/A')
            fig.suptitle(f"Hyperspectral PCA Results for {det_name} - Scan: {scan_name}", fontsize=16)

            for i in range(n_components):
                ax_map = axes[i, 0]
                ax_spec = axes[i, 1]

                # --- Plot Score Map (Left) ---
                score_map = scores_val[:, i]
                
                trip = ax_map.tripcolor(final_x, final_y, score_map, shading='gouraud', cmap='viridis')
                fig.colorbar(trip, ax=ax_map)
                ax_map.set_title(f"Score Map - Component {i+1}\n(Variance: {evr_val[i]:.2%})")
                ax_map.set_xlabel("Hexapod X")
                ax_map.set_ylabel("Hexapod Y")
                ax_map.set_aspect('equal', adjustable='box')

                # --- Plot Component Loading (Right) ---
                loading_vector = loadings_val[i, :]
                # Reshape back to (num_energies, num_channels)
                loading_map = loading_vector.reshape(num_energies, num_channels)
                
                im = ax_spec.imshow(loading_map, aspect='auto', cmap='coolwarm', 
                                    extent=[0, num_channels, all_energies[-1], all_energies[0]]) # Flipped y-axis
                fig.colorbar(im, ax=ax_spec, label="Component Weight")
                ax_spec.set_title(f"Loading Map - Component {i+1}")
                ax_spec.set_xlabel("Channel/Bin Number")
                ax_spec.set_ylabel("Energy (eV)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    except Exception as e:
        print(f"A critical error occurred in pca_stack_analysis: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Perform PCA on hyperspectral data from a stack scan.")
    parser.add_argument("h5_file_path", help="Path to the HDF5 file from a stack scan.")
    parser.add_argument("--n_components", type=int, default=5, help="Number of principal components to compute (default: 5).")

    args = parser.parse_args()
    
    print("Analyzing stack file to find data paths...")
    path_pack = analyze_sgm_bsky_data(args.h5_file_path)
    
    if path_pack:
        pca_stack_analysis(path_pack,
                           n_components=args.n_components)
    else:
        print("Failed to analyze stack. Aborting PCA.", file=sys.stderr)