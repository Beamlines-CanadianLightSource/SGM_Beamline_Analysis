# SGM_BSky Data Analysis User Guide

This guide provides an overview of the `SGM_BSky_Data_Analysis.ipynb` notebook and its associated python modules. It outlines the overall workflow, details the purpose of each script, and explains how to interact with the various GUIs that pop up during analysis.

## Overview & Purpose

The `SGM_BSky_Data_Analysis.ipynb` notebook is an end-to-end processing pipeline for Synchrotron XRF and XANES stack data collected at the SGM beamline. It is designed to:
1. Parse and extract metadata, spatial coordinates, and spectral data from raw HDF5 files.
2. Interactively align spatial drift and select spectral ROIs.
3. Explore the 4D hypercube interactively, isolate regions of interest (ROI), and view real-time XANES extraction.
4. Normalize the raw data using `mcc1` reference signals or external I0 standards.
5. Perform Principal Component Analysis (PCA) to extract the most significant spectral variations.
6. Run K-Means Clustering on the PCA scores to spatially map chemical species.
7. Allow interactive merging of these clusters to extract high-quality, averaged XANES spectra.
8. Export the processed stacks into a fully Nexus-compliant 4D HDF5 format for advanced visualization in PyMca.

---

## Quick Start: Running the Notebook

To use the pipeline, open `SGM_BSky_Data_Analysis.ipynb` in your Jupyter environment and run the cells sequentially (using `Shift + Enter`). 

For datasets involving multi-quadrant stitching, refer to the example notebook: `SGM_BSky_Data_Analysis-4quad.ipynb`.

As you progress through the notebook, various file dialogs and interactive GUIs will pop up. **Keep an eye on your taskbar**, as some pop-ups (especially the Matplotlib/Tkinter windows) might open behind your browser window.

> [!TIP]
> Make sure you are using the `.venv` environment configured for this project to avoid any missing dependency errors.

---

## Core Modules & GUI Instructions

  - Changes made here are saved to the `path_pack` and applied to all downstream steps.

### 2. `calibrate_sdd_xrf.py` (Energy Calibration)
**Purpose:** Precisely maps the SDD detector "channels" (0-255) to physical "energy" (eV). This is essential for identifying elements in your XRF spectra and ensuring your XANES plots have an accurate X-axis.

**When to Run:** 
- **Stable Setup:** Energy calibration is typically very stable. You only need to run this tool once every **six months**, provided there have been no major changes to the beamline electronics or detector settings.
- **Precision Work:** For high-precision studies, you may choose to recalibrate for every new set of spectra.

**How to Launch:** 
In a new cell in your Jupyter Notebook, run:
```python
from calibrate_sdd_xrf import run_calibration
run_calibration()
```

**Using the Interactive Calibration GUI:**
1. **Load a Standard:** Start by loading a scan of a known material (e.g., AlPO4 for Al and P, or MgO for Mg and O).
2. **Adjust Sensitivity Sliders:**
   - **Threshold:** This is a height filter. A value of 0.1 means the tool will only label peaks that are at least 10% as tall as the strongest peak in the spectrum. Lower this to find weak signals (like Boron or Carbon).
   - **Distance:** This sets the minimum horizontal space (in channels) between peaks. Use this to prevent the tool from misidentifying noise or "bumps" on the side of a large peak as separate elements.
3. **Select Number of Points:** Choose **2, 3, or 4 points** for your fit. Using more points (e.g., C, O, and Al together) creates a more robust calibration line across the entire energy range.
4. **Assign Peaks:** 
   - The tool labels every detected peak with an **ID** (e.g., ID:0, ID:1) on the plot.
   - For each detector, use the dropdowns to match an **ID** to your target energy (e.g., "Assign ID:1 to Al-K").
   - If a peak isn't detected automatically, select **"Manual"** and type the channel number directly.
5. **Save:** Click **Calculate & Save**. The tool performs a linear regression and updates your global calibration file.

### 2.1 `stitching_utils.py` (Multi-Quadrant Maps)
**Purpose:** Used when a sample is too large to fit in a single scan and was collected as 4 overlapping "quadrants" (SW, SE, NE, NW). This tool joins them into a single, seamless master map. 

**Example Workflow:** For a hands-on example of this process, use the `SGM_BSky_Data_Analysis-4quad.ipynb` notebook.
**User Interaction:**
- **Interactive Trimming (`interactive_stitching_trim`):** A specialized widget that displays all selected quadrants on a single plot.
  
> [!WARNING]
> **Mandatory Trimming for Hexapod Overshoots**  
> When the SGM beamline scans a map, the hexapod stage physically overshoots the target area on the left and right edges to accelerate and decelerate. The data recorded during these turnarounds contains stretched coordinates and noise. **You must use the Left and Right sliders to trim off these edges (usually about 0.1mm on both sides needs to be trimmed) before stitching.** If you do not trim the maps, the overshoot regions will physically overlap with adjacent maps, creating massive noise, artifacts, and distorting the aspect ratio of the final stitched image.

- **Handling Missing Quadrants:** If you only have 2 or 3 quadrants, you can explicitly skip a slot by clicking **Cancel** in the file selection dialog for that quadrant. The tool will handle the gap automatically.
- **Duplicate Prevention:** A warning will appear if you select the same file for multiple quadrants. Avoid this, as it causes spatial data overlap and visualization artifacts.
- **Automatic Gap Masking:** The tool now automatically masks the space between quadrants. This prevents "smearing" artifacts (where data appears to stretch across missing areas) in the dashboard and alignment tools.
- **Asymmetric Sliders:** Use the **Left, Right, Top, Bottom** sliders for each quadrant (now labeled Map 1, Map 2, etc.) to remove bad edge data caused by stage acceleration or to eliminate overlaps.
- **Contrast Control:** Includes a dedicated contrast slider to help you see fine details while aligning the boundaries.
- **Data Preservation:** The stitched dataset now preserves all 4 MCC reference channels (`mcc1` through `mcc4`), ensuring complete metadata for normalization.
- **Bake Stitched Map:** Once the maps join perfectly without visible "seams" or overshoots, click this button to generate a new master HDF5 file and data directory. The resulting "baked" map is fully compatible with the rest of the pipeline.

### 3. `plot_sgm_bsky_data.py` (The Main Interactive Dashboard)
**Purpose:** This is the most critical component of the notebook. It acts as the "Command Center" where you visually explore the 4D hypercube, correlate spatial maps with XANES spectra, and manage normalization.

**Dashboard Layout:**
- **Detector Rows:** Each SDD detector gets its own 3-panel row:
  - **Left Panel (Energy Map):** Displays the spatial XRF intensity at the currently selected energy step.
  - **Middle Panel (Average Map):** Displays the spatial XRF intensity averaged across the entire energy stack.
  - **Right Panel (Spectrum):** Displays the 256-channel emission spectrum for the spatial region currently selected.
- **Summary Dashboard:** At the bottom, a global overview plots the extracted XANES spectra (Energy vs Intensity) for the selected spatial ROI across all detectors.

**How to Use the Interactive Features:**
1. **Navigating Energy:** Use the slider at the bottom of the dashboard to scan through the energy stack. The "Energy Map" (left panels) will update in real-time.
2. **Spatial ROI Selection (Map to Spectrum):** 
   - Click and drag a rectangle on any map (left or middle panels) to select a spatial Region of Interest (ROI). 
   - Alternatively, click the "Switch to Polygon" button at the bottom to draw custom freehand shapes.
   - *Result:* As soon as you draw the ROI, the right-hand panel updates to show the emission spectrum of *only those pixels*. The Summary Dashboard at the bottom also updates to show the full XANES scan for that specific physical area.
    - *Result:* The dashboard will trigger a **GLOBAL REFRESH**. It will re-integrate the entire 4D stack across all energies for your newly selected channels, instantly updating every map in the dashboard.
4. **Global Contrast Slider:**
   - Located right below the energy slider, the **"Image Contrast %"** range slider allows you to adjust the min/max percentiles for all map displays simultaneously.
   - **Tip:** Drag the handles to roughly **[2.0, 98.0]** to ignore noisy "hot pixels" and instantly reveal the high-contrast features of your sample. This affects the visualization only and does not change the underlying data values.
5. **Synchronization:** Everything is synced! Drawing an ROI on `sdd1` will automatically apply the exact same ROI to `sdd2`, `sdd3`, and `sdd4`.
5. **Double-Click to Copy (Quick Export):**
   - You can instantly copy any plot to your clipboard by **double-clicking** it.
   - **Single Plot:** Double-click inside a map or spectrum area to copy just that specific plot.
   - **Full Window:** Double-click in the gray margin area to copy the entire dashboard layout (e.g., all 3 plots in a row).
   - *Result:* High-resolution images are ready to be pasted directly into PowerPoint or other documents.

6. **Exporting Processed Data:**
   - The Summary Dashboard at the bottom contains several yellow action buttons for saving your results:
   - **Save XANES Spectra for PCA/CA:** (Formerly "Save PyMca Stack"). This exports a compact **3D HDF5 stack** (`_PCA-CA.h5`) where the spectrum is reduced to a single intensity value (from your ROI) per pixel. This is the format required for **PCA and Clustering** analysis.
   - **Save XRF Spectra for Elemental Analysis using PyMca:** (Formerly "Save 4D PyMca Stack"). This exports a massive **4D HDF5 hypercube** (`_Elemental_PyMca.h5`) containing the full raw spectrum for every pixel. Use this for **elemental peak fitting** in PyMca.
   - **Save XRD/XANES Spectra:** Exports your currently extracted 1D spectra (Intensity vs Energy) to a CSV file.
   - **Save XRD Spectra:** Exports the 1D fluorescence spectrum (Intensity vs Channels) for the selected spatial area to a CSV file.

7. **Sync Map ROI:**
   - The light green **"Sync Map ROI"** button is located directly above the energy maps on each individual detector's dashboard row. If you draw an ROI on one map and the summary spectrum doesn't automatically update, or if the visual regions get out of sync, click this button on the active map to force all other maps to align perfectly with your drawn ROI.

8. **Enable SDD Energy Calib:**
   - **How it Works:** Located in the dashboard settings area, checking the **"Enable SDD Energy Calib"** checkbox applies your saved calibration parameters (`gain` and `offset` stored in `sdd_calibration.json`) to all detectors in real-time.
   - **Transitioning from Bin (Channels) to Energy (eV):** 
     - **ROI Inputs:** Checking this box dynamically changes the **Spectral ROI Limits** input fields at the bottom from channel bin coordinates (e.g. `ROI Start (Ch)` / `ROI End (Ch)`) to physical energy values (e.g. `ROI Start (eV)` / `ROI End (eV)`). Unchecking it instantly reverts them back to channel bins.
     - **Span Selector Sync:** The horizontal Span Selector overlay on your spectral plots shifts its scale accordingly. When calibration is enabled, selecting a region on the plot defines physical energy boundaries.
     - **X-Axis Update:** The X-axis labels on the emission spectra update from raw channels (`0` to `255`) to calibrated energy levels in electron-volts (`eV`).
     - **ROI Syncing:** When enabled, ROI selection perfectly synchronizes across all detectors by **Energy (eV)** rather than raw channels. This ensures that selecting an element's characteristic emission peak (e.g., Al-K at ~1486.7 eV) is accurate for all four SDDs, even if they have slight hardware offset differences.

**I0 Normalization and Smoothing:**
- At the beginning of plotting, you will be prompted to select the normalization source:
  - **Internal I0 (`mcc1`):** Uses the current from the **Au mesh** collected during your scan.
  - **External I0 CSV:** Uses a previously collected standard. 
    - *Example:* For Carbon (**C K-edge**) analysis, it is standard practice to use a spectrum from **BN** (Boron Nitride) as your external I0.
- Regardless of your choice, an **I0 Preview Dialog** will open. For external files, you will first use dropdowns to select the Energy (X) and Intensity (Y) columns.
- **Smoothing:** You can enable Savitzky-Golay smoothing to remove noise from your chosen I0 standard (internal or external) before applying it to your data. Adjust the "Window Size" and see the preview update live compared to the raw I0 spectrum.

**Exporting Data (Adding Sample Specific Information):**
- When saving spectra or data from the summary dashboard, you have the option to toggle the **"Add Sample Specific Information to File Header"** checkbox (located just above the Save buttons).
- **Unchecked (Default):** This is the default setting. It skips the metadata prompt and simply includes a standard header containing all relevant beamline parameters that the data was collected by.
- **Checked:** This opens a dialog generating an expanded header where the User can add additional information such as compound name, chemical formula, authors, sample preparation details, etc. It is expected to be used primarily for **reference compounds**. We expect to submit these reference compounds to the [Canadian Light Source X-ray Absorption database](https://xasdb.lightsource.ca/). This information will subsequently be submitted to the [Federated Research Data Repository (FRDR)](https://www.frdr-dfdr.ca/repo/).

### 4. `pca_xanes_analysis.py`
**Purpose:** Performs Principal Component Analysis (PCA) to reduce the dimensionality of the XANES stack and isolate the primary chemical variations (components) while filtering out background noise.
**How it works:**
- Flattens the 3D stack, scales the data, and computes eigenvectors (Loadings) and eigenimages (Scores).
- Automatically filters out dead/empty pixels to prevent math errors.
- Saves eigenvectors to a CSV and the eigenimages back into the HDF5 file under `entry/pca_results`.
**Multi-Detector "Run All" Mode:**
- You can process individual datasets (like `average`) or you can tell the script to run **all** datasets at once (`sdd1`, `sdd2`, `sdd3`, `sdd4`, and `average`).
- Running them all at once automatically generates comparative, side-by-side grids (one for eigenimages, one for eigenvectors) so you can easily spot detector-specific artifacts or verify consistency across all SDDs.
- *Note: To maintain high visibility in Jupyter, these large comparative grids are rendered with a built-in horizontal scrollbar, preventing them from overflowing or shrinking too small.*

### 5. `cluster_xanes_analysis.py`
**Purpose:** Uses K-Means clustering on the PCA scores (eigenimages) to group pixels that share similar XANES signatures.
**How it works:**
- You specify the number of clusters ($k$).
- The algorithm assigns each valid pixel to a cluster, producing a spatial map of distinct "zones" or species.
- It then computes the average XANES spectrum for all pixels within each cluster.
- Saves the cluster maps and extracted spectra to CSVs, PNG previews, and back into the HDF5 file.
**Multi-Detector "Run All" Mode:**
- Just like the PCA analysis, you can cluster all SDD detectors and the `average` simultaneously.
- This outputs a comparative grid of cluster maps and a grid of the extracted cluster XANES spectra. It is extremely useful for validating that the chemical zones detected are real sample features and not just anomalies occurring on a single detector.
- *Note: Similar to PCA, these multi-detector outputs are rendered as horizontally scrollable figures within Jupyter Notebooks.*
- **Cluster Sums:** In multi-detector mode, the tool also calculates and saves the **Sum** of all spectra in each cluster (in addition to the Mean), providing the total integrated signal per detector.

**Metadata & CSV Export:**
- All clustering CSV exports now include a standard header with beamline metadata.
- **Pass `metadata` dict:** To ensure headers are complete (Project, Date, etc.), pass the `pp` (path_pack) dictionary from `analyze_sgm_bsky_data` to the `metadata` parameter of the clustering functions.
- **Full Metadata Mode:** You can enable `use_full_metadata=True` to prompt for detailed research metadata (Authors, Affiliation, etc.) before saving.

### 6. `interactive_cluster_merger.py`
**Purpose:** A powerful interactive dashboard that lets you review the K-Means clustering results, combine similar clusters, and extract a single, high signal-to-noise XANES spectrum.
**User Interaction (GUI):**
- The dashboard displays the Cluster Map on the left and the Merged Spectrum on the right.
- **Detector Dropdown:** Switch between `sdd1`, `sdd2`, `sdd3`, `sdd4`, or `average` to see how clustering performed across different detectors.
- **Checkboxes:** Check or uncheck clusters ($C1, C2, C3...$) to see their combined average spectrum update in real-time. The map will also mask out unselected clusters so you can verify exactly which spatial regions are contributing to your spectrum.
- **Save Button:** Click "Save Merged Spectrum" to export the live spectrum to a CSV file (includes all beamline metadata in the file header).

### 7. `save_pymca_4d_stack_h5.py`
**Purpose:** Packages the final normalized and aligned data into a multi-mode HDF5 file specifically formatted for PyMca.
**How it works:**
- It creates three distinct datasets: `sdd_xrf_stack_3d`, `sdd_xanes_stack_3d`, and a `full_4d_measurement`.
- It injects strict **Nexus (`NXdata`) metadata** (`x_indices`, `y_indices`, `channels_indices`) so that PyMca automatically understands the difference between spatial axes (the map) and spectral axes (the energy/channels).

### 8. `sum_mcc1_spectra.py`
**Purpose:** Extracts and aggregates the internal `mcc1` current across multiple HDF5 scans. Used when internal normalization is preferred over external standards.
**User Interaction (GUI):**
- When run, a window titled **"MCC1 Spectra Summator"** will appear.
- Click **"Load Stack HDF5 Files"** to select one or multiple `.h5` stack files.
- On the left panel, you will see checkboxes for every loaded file, along with **"Select All"** and **"Deselect All"** buttons for quick management. Unchecking a file will immediately remove it from the **Selected Average**.
- The right panel displays the plot, showing individual active spectra (faded). It overlays two summary lines:
  - **Global Average (All Files):** A black dashed line showing the average of all loaded files, regardless of checkbox state.
  - **Selected Average:** A thick red solid line showing the average of only the checked files.
- Click **"Save Average to CSV"** to export the *Selected Average* data, which can then be used in the normalization steps.
- Click **"Close Dashboard"** (or the window 'X') to cleanly exit the application loop.

---

## Exporting & External Analysis

### Exporting the Notebook to HTML and PDF
If you need to share your analysis, save your progress, or print the notebook to a PDF for record-keeping, the most reliable method is to export it to HTML first.

**Method A: Using the Terminal (Recommended for PyCharm/VSCode users)**
Since IDEs often hide or remove the classic Jupyter export menus, the command line is the easiest way:
1. Open the Terminal in your project (e.g., the "Terminal" tab at the bottom of PyCharm).
2. Run the following command:
   ```bash
   .venv\Scripts\python -m nbconvert --to html SGM_BSky_Data_Analysis.ipynb
   ```
3. A file named `SGM_BSky_Data_Analysis.html` will be generated in your project folder.

**Method B: Using the Jupyter Web Interface**
If you are running Jupyter in a web browser:
1. Navigate to the top menu bar.
2. Click **File > Save and Export Notebook As...** (or "Download as").
3. Select **HTML** from the dropdown menu to download the file.

**Converting to PDF:** 
Once you have the HTML file, open it in any modern web browser (Edge, Chrome, Firefox). Press `Ctrl + P` (or right-click and select Print). Change the printer destination to **"Save as PDF"** and save the file. 

*Note: Exporting to HTML first and then printing to PDF is highly recommended over direct PDF export, as it prevents wide code blocks and interactive plots from being awkwardly cut off across printed pages.*

### PyMca Analysis (XANES vs XRF)
When you export data from the dashboard for use in PyMca, pay attention to the file suffix:

1. **_PCA-CA.h5 (XANES Stack):**
   - **Format:** 3D (Energy, Y, X).
   - **Use case:** Speciation, PCA, and Clustering.
   - **PyMca Step:** Right-click the dataset and select **"Load as 1D Stack"**. Go to **Plugins > Multivariate Analysis > PCA**.

2. **_Elemental_PyMca.h5 (XRF Hypercube):**
   - **Format:** 4D (Energy, Y, X, Channels).
   - **Use case:** Elemental peak fitting and quantification.
   - **PyMca Step:** Open the file in PyMca and navigate to the `full_4d_measurement` or `xrf_measurement`. This preserves all 256 channels so you can fit overlapping elements.

### Exporting to PowerPoint/Excel
The application supports direct clipboard exports for all interactive dashboards and clustering results. 
- **Images:** Simply **double-click** on any interactive plot (Map or Spectrum) to copy a high-resolution bitmap directly to your Windows clipboard. You can then paste it instantly into PowerPoint or Word.
- **Data:** Many modules also support copying the raw buffer data into Excel via dedicated "Copy Data" buttons (where available).
