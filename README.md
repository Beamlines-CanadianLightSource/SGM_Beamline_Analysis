# SGM Beamline Data Analysis

A collection of Jupyter notebooks and Python utility scripts for the end-to-end processing of Synchrotron XRF and XANES stack data collected at the SGM beamline, Canadian Light Source.

## Features

- **Data Parsing:** Extract metadata, spatial coordinates, and spectral data from raw HDF5 files.
- **Interactive Alignment:** Align spatial drift and select spectral ROIs via integrated GUIs.
- **4D Hypercube Exploration:** Real-time XANES extraction and visualization.
- **Normalization:** Support for internal `mcc1` reference signals and external I0 standards.
- **Advanced Analysis:** 
    - Principal Component Analysis (PCA) for dimensionality reduction.
    - K-Means Clustering for spatial mapping of chemical species.
    - Interactive cluster merging for high-quality XANES extraction.
- **Nexus Export:** Generate PyMca-compatible 4D HDF5 files.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Notebooks

1. **`SGM_BSky_Data_Analysis-N.ipynb`**: The primary workflow for processing a single data stack.
2. **`SGM_BSky_Data_Analysis-4quad.ipynb`**: Specialized workflow for stitching together 4 quadrant maps into a single master map.

Detailed instructions for each component can be found in the [User Guide](SGM_BSky_Data_Analysis_Guide.md).

## Project Structure

- `SGM_BSky_Data_Analysis-N.ipynb`: Main analysis notebook.
- `SGM_BSky_Data_Analysis-4quad.ipynb`: Quadrant stitching notebook.
- `SGM_BSky_Data_Analysis_Guide.md`: Comprehensive user documentation.
- `analyze_sgm_bsky_data.py`: Core data parsing logic.
- `plot_sgm_bsky_data.py`: Interactive dashboard engine.
- `stitching_utils.py`: Quadrant alignment and stitching tools.
- `pca_xanes_analysis.py` / `cluster_xanes_analysis.py`: Statistical analysis modules.

## Acknowledgments

This software is developed for use at the SGM Beamline of the Canadian Light Source.
