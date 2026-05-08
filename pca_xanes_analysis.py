import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys

def pca_xanes_analysis(h5_path, dataset_name='average', n_components=5, show_plot=True, return_dict=False):
    """
    Performs PCA on a specific ROI-summed stack within an HDF5 file and exports all results.
    """
    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return None

    print(f"Loading dataset '{dataset_name}' from {h5_path}...")
    
    try:
        with h5py.File(h5_path, 'r+') as f:
            # Navigate to the measurement group
            if 'entry/measurement' not in f:
                print("Error: Could not find 'entry/measurement' in HDF5 file.")
                return None
            
            meas = f['entry/measurement']
            
            if dataset_name not in meas:
                available = [k for k in meas.keys() if isinstance(meas[k], h5py.Dataset)]
                print(f"Error: Dataset '{dataset_name}' not found. Available datasets: {available}")
                return None
            
            # Load Data
            stack = meas[dataset_name][()] # (ny, nx, n_energies)
            energy = meas['energy'][()]
            y_axis = meas['y'][()]
            x_axis = meas['x'][()]
            
            ny, nx, n_energies = stack.shape
            print(f"    -> {dataset_name} stack shape: {stack.shape}")
            
            # 1. Prepare data for PCA
            data_flat = stack.reshape(-1, n_energies)
            
            # Filter out dead pixels
            pixel_sums = np.sum(np.abs(data_flat), axis=1)
            valid_mask = (pixel_sums > 0) & (~np.isnan(pixel_sums))
            
            data_valid = data_flat[valid_mask]
            
            if data_valid.shape[0] < n_components:
                print(f"Warning: Not enough valid pixels ({data_valid.shape[0]}) for {n_components} components in {dataset_name}. Adjusting n_components.")
                n_components = max(1, data_valid.shape[0])

            # 2. Scale and Perform PCA
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_valid)
            
            pca = PCA(n_components=n_components)
            pca_scores = pca.fit_transform(data_scaled) 
            pca_loadings = pca.components_ 
            
            explained_var = pca.explained_variance_ratio_

            # 3. Reconstruct Eigenimages
            eigenimages = np.zeros((ny, nx, n_components), dtype=np.float32)
            for i in range(n_components):
                full_scores = np.zeros(ny * nx, dtype=np.float32)
                full_scores[valid_mask] = pca_scores[:, i]
                eigenimages[:, :, i] = full_scores.reshape(ny, nx)

            # 4. Save CSV
            base_dir = os.path.dirname(os.path.abspath(h5_path))
            scan_name = os.path.splitext(os.path.basename(h5_path))[0]
            csv_path = os.path.join(base_dir, f"{scan_name}_pca_vectors_{dataset_name}.csv")
            
            cols = {'Energy_eV': energy}
            for i in range(n_components):
                cols[f'PC{i+1}_Loading (Var:{explained_var[i]:.2%})'] = pca_loadings[i]
            
            pd.DataFrame(cols).to_csv(csv_path, index=False)

            # 5. Save back to HDF5
            pca_group_path = f"entry/pca_results/{dataset_name}"
            if pca_group_path in f:
                del f[pca_group_path]
            
            pca_group = f.create_group(pca_group_path)
            pca_group.attrs['n_components'] = n_components
            pca_group.attrs['dataset_source'] = dataset_name
            pca_group.create_dataset('eigenimages', data=eigenimages, compression="gzip")
            pca_group.create_dataset('eigenvectors', data=pca_loadings)
            pca_group.create_dataset('explained_variance', data=explained_var)
            
            print(f"    -> {dataset_name} results saved to HDF5.")

        if show_plot:
            plot_results(energy, x_axis, y_axis, eigenimages, pca_loadings, explained_var, dataset_name, h5_path)
        
        results = {
            'dataset': dataset_name,
            'eigenimages': eigenimages,
            'loadings': pca_loadings,
            'variance': explained_var,
            'energy': energy,
            'x': x_axis,
            'y': y_axis
        }
        
        return results if return_dict else h5_path

    except Exception as e:
        print(f"An error occurred during PCA on {dataset_name}: {e}")
        return None

def run_pca_all_detectors(h5_path, n_components=3):
    """
    Performs PCA on sdd1, sdd2, sdd3, sdd4 and average, then plots them side-by-side.
    """
    print(f"\n{'='*60}\nRunning Multi-Detector PCA Analysis\n{'='*60}")
    
    detectors = ['sdd1', 'sdd2', 'sdd3', 'sdd4', 'average']
    all_results = []
    
    for det in detectors:
        res = pca_xanes_analysis(h5_path, dataset_name=det, n_components=n_components, show_plot=False, return_dict=True)
        if res:
            all_results.append(res)
    
    if not all_results:
        print("Error: No datasets were successfully analyzed.")
        return

    plot_multi_detector_results(all_results, h5_path)
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

def plot_multi_detector_results(all_results, h5_path):
    """
    Plots comparative grids of eigenimages and eigenvectors for all detectors.
    """
    n_det = len(all_results)
    n_comp = all_results[0]['eigenimages'].shape[2]
    
    energy = all_results[0]['energy']
    x_axis = all_results[0]['x']
    y_axis = all_results[0]['y']
    
    scan_name = os.path.splitext(os.path.basename(h5_path))[0]
    base_dir = os.path.dirname(os.path.abspath(h5_path))

    # --- Figure 1: Eigenimages Comparison ---
    fig_img, axes_img = plt.subplots(n_comp, n_det, figsize=(2.8*n_det, 2.8*n_comp), squeeze=False)
    fig_img.suptitle(f"PCA Eigenimage Comparison: {scan_name}", fontsize=16)
    
    for c in range(n_comp):
        for d in range(n_det):
            res = all_results[d]
            ax = axes_img[c, d]
            data = res['eigenimages'][:, :, c]
            var = res['variance'][c]
            
            im = ax.imshow(data, extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]], cmap='viridis')
            if c == 0:
                ax.set_title(f"{res['dataset']}\nPC{c+1} ({var:.1%})")
            else:
                ax.set_title(f"PC{c+1} ({var:.1%})")
            
            if d == 0:
                ax.set_ylabel(f"Component {c+1}\nY (mm)")
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    img_plot_path = os.path.join(base_dir, f"{scan_name}_pca_comparison_images.png")
    plt.savefig(img_plot_path, dpi=150)
    
    # --- Figure 2: Eigenvectors Comparison ---
    fig_vec, axes_vec = plt.subplots(n_comp, n_det, figsize=(2.8*n_det, 2.2*n_comp), squeeze=False)
    fig_vec.suptitle(f"PCA Eigenvector (Loading) Comparison: {scan_name}", fontsize=16)
    
    for c in range(n_comp):
        for d in range(n_det):
            res = all_results[d]
            ax = axes_vec[c, d]
            loading = res['loadings'][c, :]
            
            ax.plot(energy, loading, color='red', linewidth=1.5)
            if c == 0:
                ax.set_title(f"{res['dataset']}")
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.set_ylabel(f"PC{c+1} Loading")
            if c == n_comp - 1:
                ax.set_xlabel("Energy (eV)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    vec_plot_path = os.path.join(base_dir, f"{scan_name}_pca_comparison_vectors.png")
    plt.savefig(vec_plot_path, dpi=150)
    
    print(f"\nMulti-detector plots saved to:\n  -> {img_plot_path}\n  -> {vec_plot_path}")
    
    scrolled_img = _display_scrollable_figure(fig_img)
    scrolled_vec = _display_scrollable_figure(fig_vec)
    
    if not (scrolled_img or scrolled_vec):
        plt.show()

def plot_results(energy, x_axis, y_axis, eigenimages, loadings, variance, dataset_name, h5_path):
    """
    Generates a multi-panel plot previewing the PCA results for a single dataset.
    """
    n_comp = loadings.shape[0]
    fig, axes = plt.subplots(n_comp, 2, figsize=(10, 2.5 * n_comp))
    if n_comp == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(f"PCA Results for Stack: {dataset_name}", fontsize=16)

    for i in range(n_comp):
        # Eigenimage (Left)
        ax_img = axes[i, 0]
        im = ax_img.imshow(eigenimages[:, :, i], extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]], cmap='viridis')
        fig.colorbar(im, ax=ax_img)
        ax_img.set_title(f"PC{i+1} Eigenimage (Var: {variance[i]:.2%})")
        ax_img.set_xlabel("X (mm)")
        ax_img.set_ylabel("Y (mm)")

        # Eigenvector (Right)
        ax_vec = axes[i, 1]
        ax_vec.plot(energy, loadings[i, :], 'r-', linewidth=1.5)
        ax_vec.set_title(f"PC{i+1} Eigenvector (Loading)")
        ax_vec.set_xlabel("Energy (eV)")
        ax_vec.set_ylabel("Weight")
        ax_vec.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    base_dir = os.path.dirname(os.path.abspath(h5_path))
    scan_name = os.path.splitext(os.path.basename(h5_path))[0]
    plot_path = os.path.join(base_dir, f"{scan_name}_{dataset_name}_pca_preview.png")
    plt.savefig(plot_path, dpi=150)
    print(f"    -> Preview plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pca_xanes_analysis.py [h5_file_path] [dataset_name] [n_components]")
        print("Example: python pca_xanes_analysis.py my_stack.h5 average 5")
        print("To run for all detectors: python pca_xanes_analysis.py my_stack.h5 all 3")
    else:
        h5_file = sys.argv[1]
        ds_name = sys.argv[2] if len(sys.argv) > 2 else 'average'
        comp = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        if ds_name.lower() == 'all':
            run_pca_all_detectors(h5_file, n_components=comp)
        else:
            pca_xanes_analysis(h5_file, ds_name, comp)

