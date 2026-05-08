import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import traceback
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from analyze_sgm_bsky_data import analyze_sgm_bsky_data

def cluster_pca_spectra(path_pack, n_clusters=4, n_components=10):
    """
    Performs PCA on the full hyperspectral stack, then uses K-Means clustering
    on the PCA scores to segment the data. Finally, it plots the mean spectrum
    for each cluster.

    Args:
        path_pack (dict): The output from analyze_stack.
        n_clusters (int): The number of clusters to find.
        n_components (int): The number of principal components to use for clustering.
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

        # --- Main Loop for each Detector ---
        for det_name in detector_names:
            print(f"\n--- Starting Cluster Analysis for Detector: {det_name} ---")
            
            # --- 1. Load and Assemble Hyperspectral Data ---
            hyperspectral_data_list = []
            
            for energy in all_energies:
                sdd_filepath = path_pack['sdd_files'][det_name].get(energy)
                
                if not sdd_filepath or not os.path.exists(sdd_filepath):
                    pixels_per_spectrum = 256
                    nan_spectra = np.full((x_coords.size, pixels_per_spectrum), np.nan)
                    hyperspectral_data_list.append(nan_spectra)
                    continue

                try:
                    pixels_per_spectrum = 256
                    data_1d = np.fromfile(sdd_filepath, dtype=np.uint32)
                    num_spectra = len(data_1d) // pixels_per_spectrum

                    if num_spectra != x_coords.size:
                        spectra_2d = np.full((x_coords.size, pixels_per_spectrum), np.nan)
                        limit = min(num_spectra, x_coords.size)
                        spectra_2d[:limit, :] = data_1d[:limit * pixels_per_spectrum].reshape((limit, pixels_per_spectrum))
                    else:
                        spectra_2d = data_1d[:num_spectra * pixels_per_spectrum].reshape((num_spectra, pixels_per_spectrum))
                    
                    hyperspectral_data_list.append(spectra_2d)

                except Exception as e:
                    print(f"    -> Error loading data for energy {energy} eV: {e}", file=sys.stderr)
                    pixels_per_spectrum = 256
                    nan_spectra = np.full((x_coords.size, pixels_per_spectrum), np.nan)
                    hyperspectral_data_list.append(nan_spectra)

            hyperspectral_stack = np.stack(hyperspectral_data_list, axis=1)
            num_points, num_energies, num_channels = hyperspectral_stack.shape
            flattened_hyperspectral = hyperspectral_stack.reshape(num_points, -1)

            # --- 2. Preprocess Data ---
            valid_mask = ~np.isnan(flattened_hyperspectral).any(axis=1)
            if np.sum(valid_mask) == 0:
                print("    -> Error: No valid data points found. Skipping analysis.", file=sys.stderr)
                continue
            
            original_data_valid = flattened_hyperspectral[valid_mask]
            final_x = x_coords[valid_mask]
            final_y = y_coords[valid_mask]
            
            print(f"    -> Found {original_data_valid.shape[0]} valid data points.")
            print("    -> Scaling data and performing PCA...")
            scaler = StandardScaler()
            scaled_spectra = scaler.fit_transform(original_data_valid)

            pca = PCA(n_components=n_components)
            pca_scores = pca.fit_transform(scaled_spectra)
            
            # --- 3. K-Means Clustering ---
            print(f"    -> Performing K-Means clustering with k={n_clusters}...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pca_scores)

            # --- 4. Visualize Results ---
            print("    -> Generating plots...")
            
            # Create a figure with two subplots: Cluster Map and Mean Spectra
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            scan_name = path_pack.get('scan_name', 'N/A')
            fig.suptitle(f"Cluster Analysis for {det_name} - Scan: {scan_name}", fontsize=16)

            # --- Plot Cluster Map (Left) ---
            ax_map = axes[0]
            scatter = ax_map.scatter(final_x, final_y, c=cluster_labels, cmap='viridis', marker='s', s=15, edgecolors='none')
            ax_map.set_title(f"Spatial Cluster Map (k={n_clusters})")
            ax_map.set_xlabel("Hexapod X")
            ax_map.set_ylabel("Hexapod Y")
            ax_map.set_aspect('equal', adjustable='box')
            # Add a colorbar with discrete ticks
            bounds = np.arange(n_clusters + 1) - 0.5
            norm = plt.cm.colors.BoundaryNorm(bounds, plt.cm.viridis.N)
            cbar = fig.colorbar(scatter, ax=ax_map, boundaries=bounds, ticks=np.arange(n_clusters))
            cbar.set_label("Cluster ID")

            # --- Plot Mean Spectra (Right) ---
            ax_spec = axes[1]
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                cluster_mask = (cluster_labels == i)
                if np.any(cluster_mask):
                    # Calculate mean over the original, non-scaled, non-PCA data
                    mean_spectrum_flat = original_data_valid[cluster_mask].mean(axis=0)
                    # Reshape back to 2D for plotting
                    mean_spectrum_2d = mean_spectrum_flat.reshape(num_energies, num_channels)
                    
                    # For simplicity, let's plot the sum over all energies for each channel
                    # This gives a representative 1D spectrum for the cluster
                    total_mean_spectrum = mean_spectrum_2d.sum(axis=0)
                    
                    ax_spec.plot(total_mean_spectrum, label=f'Cluster {i}', color=colors[i])

            ax_spec.set_title("Mean Spectra for Each Cluster")
            ax_spec.set_xlabel("Channel/Bin Number")
            ax_spec.set_ylabel("Average Intensity")
            ax_spec.legend()
            ax_spec.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    except Exception as e:
        print(f"A critical error occurred in cluster_pca_spectra: {e}", file=sys.stderr)
        traceback.print_exc()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Perform PCA and K-Means clustering on hyperspectral data from a stack scan.")
    parser.add_argument("h5_file_path", help="Path to the HDF5 file from a stack scan.")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters to find (default: 4).")
    parser.add_argument("--n_components", type=int, default=10, help="Number of principal components to use for clustering (default: 10).")

    args = parser.parse_args()
    
    print("Analyzing stack file to find data paths...")
    path_pack = analyze_sgm_bsky_data(args.h5_file_path)
    
    if path_pack:
        cluster_pca_spectra(path_pack,
                            n_clusters=args.n_clusters,
                            n_components=args.n_components)
    else:
        print("Failed to analyze stack. Aborting cluster analysis.", file=sys.stderr)