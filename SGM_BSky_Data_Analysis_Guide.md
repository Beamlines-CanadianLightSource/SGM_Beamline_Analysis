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

To use the pipeline, we highly recommend running the notebook inside **VS Code** configured to use the project's virtual environment:

1. **Open the Project in VS Code:** Open the cloned repository directory (the root folder containing the notebook files).
2. **Open the Notebook:** Double-click on `SGM_BSky_Data_Analysis.ipynb`.
3. **Select the Kernel:** In the top-right corner of the notebook editor in VS Code, click **"Select Kernel"** -> **"Python Environments"** and choose the virtual environment located at `.venv/Scripts/python` (the project's virtual environment).
4. **Run Cells:** Execute the cells sequentially (using `Shift + Enter`). 

> [!NOTE]
> **Why VS Code + `.venv`?**  
> Running the notebook in VS Code with the local `.venv` environment ensures that all necessary dependencies (including `scikit-image` and standard Matplotlib backends) load correctly. It also handles interactive widgets (like ROI selectors and the deconvolution panel) with much better performance and lower interface lag than standard browser-based Jupyter setups.

For datasets involving multi-quadrant stitching, refer to the example notebook: `SGM_BSky_Data_Analysis-4quad.ipynb`.

As you progress through the notebook, various file dialogs and interactive GUIs will pop up. **Keep an eye on your taskbar**, as some pop-ups (especially the Matplotlib/Tkinter windows) might open behind your browser or IDE window.

---

## Core Modules & GUI Instructions

### 1. `analyze_sgm_bsky_data.py` (Data Loading & Output Routing)
**Purpose:** Reads raw HDF5 beamline files, parses motor positions (X/Y coordinates), extracts energy vectors, reads SDD spectra and MCC scaler channels, and structures everything into a standard `path_pack` dictionary.
- **Automatic Loki Server Output Redirection (Raw $\to$ Preprocessed):**
  - If data is loaded from a read-only beamline directory containing `\raw\` (e.g. `V:\projects\<project>\raw\<date>\<scan>\...`), all saved outputs (XANES CSVs, XRD CSVs, exported images, and PyMca stacks) are automatically redirected to `V:\projects\<project>\preprocessed\<date>\<scan>\`.
  - Folders are created on-demand when saving (Option A), preventing empty ghost folders on the server.
  - Offline local datasets (e.g. on `C:\` or `D:\`) continue saving in their scan directory as usual.
- **Automatic Date-Dependent SDD Detector Position Mapping (August 12, 2026 Cutoff):**
  - After **August 12, 2026**, the physical positions of the SDD detectors around the sample stage changed (`sdd1`, `sdd3`, and `sdd4` were repositioned, while `sdd2` remained in the same position).
  - `analyze_sgm_bsky_data.py` automatically extracts the scan acquisition date (`YYYY-MM-DD`, `YYYYMMDD`, or `YYMMDD` from HDF5 attributes, stitched `source_files` metadata, or file/directory names) and applies the appropriate detector position labels across all figures, plot legends, saved XANES/XRD CSV headers, PyMca HDF5 attributes, and PCA/Cluster outputs:

| Detector | On or Before Aug 12, 2026 ($\le \text{2026-08-12}$) | After Aug 12, 2026 ($> \text{2026-08-12}$) |
| :--- | :--- | :--- |
| **`sdd1`** | `sdd1 (90, top)` | **`sdd1 (90, side)`** |
| **`sdd2`** | `sdd2 (45, front, in-plane)` | **`sdd2 (45, front, in-plane)`** *(unchanged)* |
| **`sdd3`** | `sdd3 (90, side)` | **`sdd3 (45, front, out of plane)`** |
| **`sdd4`** | `sdd4 (45, front, out-plane)` | **`sdd4 (90, top)`** |

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
- **Automatic Gap Masking:** The tool automatically masks empty space between quadrants or staggered L-shaped tiles (even for wide stitched maps $>500$ pixels across). This prevents Delaunay triangulation (`as_scatter_plot=False`) from stretching triangles across missing areas in the dashboard and alignment tools.
- **Asymmetric Sliders (`0.001 mm` / `1 µm` Precision):** Use the **Left, Right, Top, Bottom** sliders for each quadrant (labeled Map 1, Map 2, etc.) with `0.001 mm` resolution to remove bad edge data caused by stage acceleration or to eliminate overlaps without over-trimming seam boundaries (`calculate_initial_zero_overlap_trims` automatically leaves a half-pixel shared boundary so `interactive_rgb_overlay` renders seamless joints without white gap lines).
- **Staggered / Non-Aligned Tile Trimming Modes:** When stitching maps that are staggered in $Y$ (top/bottom not aligned) or $X$ (left/right not aligned), only part of an edge physically overlaps its neighbor:
  - **Trim Scope Toggle (`Overlap-Only (L-shape)` vs. `Full Edge (Box)`):**
    - **`Overlap-Only (L-shape)`:** Trims *only* the spatial segment where a neighboring map actually overlaps, preserving the non-overlapping top/bottom or left/right extensions of each tile.
    - **`Full Edge (Box)` (Default):** Trims the entire straight edge of the tile across its full span.
  - **`Align Flush Bounds` Button:** Automatically calculates and sets the Top/Bottom (or Left/Right) sliders across a row or column so all tiles share a common flush rectangular bounding edge.
- **Contrast Control:** Includes a dedicated contrast slider to help you see fine details while aligning the boundaries.
- **Data Preservation:** The stitched dataset now preserves all 4 MCC reference channels (`mcc1` through `mcc4`), ensuring complete metadata for normalization.
- **Bake Stitched Map:** Once the maps join perfectly without visible "seams" or overshoots, click this button to generate a new master HDF5 file and data directory. The resulting "baked" map is fully compatible with the rest of the pipeline.
  - **HPC save destination:** On headless JupyterHub/HPC, **Bake Stitched Image** opens an in-notebook folder selector instead of saving immediately. It starts at the preprocessed destination derived from the first selected HDF5 file. Navigate with **Up** and **Open Selected Folder**, or enter a folder path directly. Optionally enter a **New subfolder** name. **Save and Bake** creates missing folders and starts stitching; **Cancel** returns to trimming without writing anything. A selected `raw` path is still redirected to its matching `preprocessed` path, shown under **Save under**. The generated stitched dataset retains its own automatically named folder inside your chosen destination. Desktop behavior is unchanged.

### 2.2 `sf.py` (Scattering Factors & Compound Optical Density Correction)
**Purpose:** Calculates anomalous atomic scattering factors ($f_1, f_2$), attenuation lengths, and X-ray Optical Density ($\text{OD}$) for any chemical compound using the Henke database (`henke.xdr`).
- **Core Functionality:** Given a compound formula (e.g., `BN`, `SiO2`, `Al2O3`), photon energy grid (eV), density $\rho$ ($\text{g/cm}^3$), and sample thickness $t$ ($\mu\text{m}$), `sf.py` computes:
  $$\text{OD}(E) = \mu(E) \cdot \rho \cdot t$$
- **Primary Use Case:** Modifying raw or reference $I_0$ signals by dividing out the compound's absorption profile ($I_{0,\text{modified}} = I_{0,\text{raw}} / \text{OD}$) to isolate true sample absorption edges during normalization.

### 3. `plot_sgm_bsky_data.py` (The Main Interactive Dashboard)
**Purpose:** This is the most critical component of the notebook. It acts as the "Command Center" where you visually explore the 4D hypercube, correlate spatial maps with XANES spectra, and manage normalization.

**Dashboard Layout:**
The dashboard is organized vertically to provide immediate access to global spectra and seamless spatial map exploration without excessive scrolling:
1. **Control Bar:** Features the global `REFRESH PLOTS` button, `IPFY Normalization` toggle, and status indicators.
2. **Summary Dashboard (XRD / XANES Spectra):** Plots the extracted XANES spectra (Energy vs Intensity) for the selected spatial ROI across all detectors, with its own horizontal scrollbar right below it.
3. **Settings & Calibration Controls:**
   - **Image Energy Selector (Top):** `Select Image Energy for Display:` slider and calibrated energy button.
   - **Image Contrast % & Log Toggles:** Global contrast percentile slider (`[0.0, 100.0]`), `Use Log Scale (Images)`, and `Use Log Scale (Spectra)`.
   - **Hist Scope Dropdown:** Choose the pixel population for intensity distribution histograms across both MCC and SDD rows:
     - `Full Map` (default): Computes statistics and distribution over all valid pixels of the active map.
     - `Spatial ROI`: Restricts histogram and statistics exclusively to pixels inside the currently drawn spatial rectangle or polygon ROI.
     - `Both (Overlay)`: Overlays both Full Map (skyblue) and Spatial ROI (coral) histograms onto aligned bins with comparative Mean and Median reference lines.
   - **Calibration Toggles:** `Enable I0 Shift` with numerical shift input (eV), and `Enable SDD Energy Calibration`.
   - **Spectral ROI Limits:** Numerical inputs for `ROI Start (eV)` and `ROI End (eV)`.
   - **Deconvolution (Live):** PSF FWHM and iterations controls for deblurring.
4. **Top Horizontal View Scrollbar:** A dedicated horizontal scrollbar ("simple grey slider") located directly above the detector images with quick buttons (`◄ Show Maps` / `Show XRD Spectra ►`). This lets you slide between the maps (left) and the XRD emission spectra (right) without scrolling down to the bottom of the page.
5. **Detector & MCC Rows (`sdd1` to `sdd4`, `mcc1` Au Mesh $I_0$, & `mcc4` TEY):**
   - **Multi-Energy Stack Mode (`len(all_energies) > 1`):** SDD rows display a 4-panel layout: `[Energy Map, Stack Average, XRD Emission Spectrum, Pixel Intensity Histogram]`. MCC rows display a 3-panel layout: `[Energy Map, Stack Average, Pixel Intensity Histogram]`.
   - **Single-Energy Map / Image Mode (`len(all_energies) <= 1`):** When plotting a single map/image, the redundant `Stack Average` panel is automatically omitted: SDD rows display a 3-panel layout `[Energy Map, XRD Emission Spectrum, Pixel Intensity Histogram]`, and MCC rows display a clean 2-panel layout `[Energy Map, Pixel Intensity Histogram]`. Furthermore, the top control bar automatically hides XANES-specific buttons (**Save XANES Spectra**, **Save Normalized XANES Spectra for PCA/CA**) and the **Average `sdd1..sdd4`** checkboxes.
   - **Pixel Intensity Distribution Histogram (MCC & SDD Rows):** Every row features a 50-bin **Pixel Intensity Distribution Histogram** reflecting strictly valid pixels of the scanned area (ignoring empty whitespace in stitched scans). It includes:
     - Vertical dashed line for **Average / Mean** intensity
     - Shaded band for **Standard Deviation ($\pm 1\sigma$)**
     - Vertical dotted line for **Median** intensity
     - Total **Pixel Count ($N$)**
     - Dynamic updates when changing the **Hist Scope** (`Full Map`, `Spatial ROI`, or `Both (Overlay)`) or drawing/moving a spatial ROI.
   - **Automatic Date-Dependent SDD Position Labels:** Because the physical positions of the SDD detectors changed after **August 12, 2026**, the pipeline automatically detects the scan date (`YYYY-MM-DD` / `YYMMDD` from HDF5 metadata or file paths) and labels all figure titles, legends, exported CSV headers, and HDF5 datasets accordingly:
     - **On or Before Aug 12, 2026 ($\le \text{2026-08-12}$):** `sdd1 (90, top)`, `sdd2 (45, front, in-plane)`, `sdd3 (90, side)`, `sdd4 (45, front, out-plane)`
     - **After Aug 12, 2026 ($> \text{2026-08-12}$):** `sdd1 (90, side)`, `sdd2 (45, front, in-plane)`, `sdd3 (45, front, out of plane)`, `sdd4 (90, top)`
6. **Bottom Controls:** The native bottom horizontal scrollbar directly above the `Select Image Energy for Display (Bottom):` slider.

**How to Use the Interactive Features:**
1. **Horizontal Scrolling (Viewing Maps vs. XRD Spectra):**
   - The figures have been scaled down by ~30% (~1015–1050px width), fitting standard screens much more comfortably.
   - **Top Horizontal Scrollbar:** Located immediately above the SDD1 row, you can drag the grey scrollbar or click `◄ Show Maps (Left)` / `Show XRD Spectra (Right) ►` to slide smoothly between the images and spectra right at the top without scrolling to the bottom.
   - **Bottom Horizontal Scrollbar:** Located directly above the bottom energy selector, fully synchronized with the top scrollbar.
2. **Navigating Excitation Energy:**
   - Synchronized energy sliders are located both in the settings panel above the images (`Select Image Energy for Display:`) and at the very bottom of the dashboard (`Select Image Energy for Display (Bottom):`).
   - Moving either slider (or clicking on the summary plot) updates all detectors and maps in real time.
2. **Spatial ROI Selection (Map to Spectrum):** 
   - Click and drag a rectangle on any map (left or middle panels) to select a spatial Region of Interest (ROI). 
   - Alternatively, click the "Switch to Polygon" button at the bottom to draw custom freehand shapes.
   - *Result:* As soon as you draw the ROI, the right-hand panel updates to show the emission spectrum of *only those pixels*. The Summary Dashboard at the bottom also updates to show the full XANES scan for that specific physical area.
3. **Spectral Channel ROI Selection (Spectrum to Map):**
   - Click and drag on the right-hand spectrum plot (or use the `ROI Start` / `ROI End` input boxes) to define an emission channel range (e.g. channels `21-35` or energy `1470-1500 eV`).
   - *Result:* The dashboard triggers a **GLOBAL REFRESH**, re-integrating the 4D stack across all energies for your selected channels and updating every spatial map.
4. **Global Contrast Slider:**
   - Located in the controls panel between the summary spectra and detector maps, the **"Image Contrast %"** range slider allows you to adjust the min/max percentiles for all map displays simultaneously (e.g. `[2.0, 98.0]`).
5. **Synchronization:** Everything is synced! Drawing an ROI on `sdd1` will automatically apply the exact same ROI to `sdd2`, `sdd3`, and `sdd4`.
   - *Result:* High-resolution images are ready to be pasted directly into PowerPoint or other documents.

6. **Exporting Processed Data:**
   - The Summary Dashboard at the bottom contains several yellow action buttons for saving your results:
   - **Save Normalized XANES Spectra for PCA/CA:** (Formerly "Save PyMca Stack"). This exports a compact **3D HDF5 stack** (`_PCA-CA.h5`) where the spectrum is reduced to a single intensity value (from your ROI) per pixel. This is the format required for **PCA and Clustering** analysis.
   - **Save XRF Spectra for Elemental Analysis using PyMca:** (Formerly "Save 4D PyMca Stack"). This exports a massive **4D HDF5 hypercube** (`_Elemental_PyMca.h5`) containing the full raw spectrum for every pixel. Use this for **elemental peak fitting** in PyMca.
   - **Save XRD/XANES Spectra:** Exports your currently extracted 1D spectra (Intensity vs Energy) to a CSV file.
   - **Save XRD Spectra:** Exports the consolidated 1D fluorescence spectrum for the selected spatial area to a CSV file. **Column 1** is **`Energy_eV`** (Calibrated Energy in eV) for direct X-axis plotting in Excel/PyMca, **Column 2** is **`Channel`** (0-255), followed by individual detector counts (`sdd1`, `sdd2`, `sdd3`, `sdd4`) and **`Average_SDD`**. The header dynamically displays calibrated energy range `# SDD ROI Energy: min-max eV (ChStart-ChEnd)` when active.

7. **Sync Map ROI:**
   - The light green **"Sync Map ROI"** button is located directly above the energy maps on each individual detector's dashboard row. If you draw an ROI on one map and the summary spectrum doesn't automatically update, or if the visual regions get out of sync, click this button on the active map to force all other maps to align perfectly with your drawn ROI.

8. **Enable SDD Energy Calib:**
   - **How it Works:** Located in the dashboard settings area, checking the **"Enable SDD Energy Calib"** checkbox applies your saved calibration parameters (`gain` and `offset` stored in `sdd_calibration.json`) to all detectors in real-time.
   - **X-Axis and Units:** By default, the emission spectra X-axis, span selectors, and input boxes always display in **Energy (eV)**. When uncalibrated (box unchecked), a default approximate scale of `10 eV` per channel is used (mapping channel range `0-255` to `0-2550` eV). 
   - **Exact Calibration Mode:** Checking the box switches the scale from the approximate 10 eV/channel mapping to the **exact calibration parameters** calculated for each specific detector.
   - **ROI Inputs & Sync:** The **Spectral ROI Limits** input fields at the bottom are always labeled as `ROI Start (eV)` / `ROI End (eV)`. When exact calibration is enabled, selecting a region on any plot or typing an energy range defines physical energy boundaries. ROI selection perfectly synchronizes across all detectors by **Energy (eV)** rather than raw channels. This ensures that selecting an element's characteristic emission peak (e.g., Al-K at ~1486.7 eV) is accurate for all four SDDs, even if they have slight hardware offset differences.

**I0 Normalization, Smoothing & OD Division:**
- At the beginning of plotting, you will be prompted to select the normalization source:
  - **Internal I0 (`mcc1`):** Uses the Au mesh reference current collected during your scan.
  - **External I0 CSV:** Uses a previously collected standard (e.g., BN for Carbon C K-edge analysis).
- An **I0 Preview Dialog** opens to configure processing parameters:
  - **Column Selection:** On the desktop, select Energy (X) and Intensity (Y) columns. On the HPC, the external CSV columns are selected automatically and their names are printed.
  - **Smoothing:** Enable Savitzky-Golay smoothing with an adjustable window size to filter out noise in your $I_0$ standard.
  - **Energy Shift:** Apply an energy calibration offset (eV) to align the $I_0$ standard with your scan energies.
  - **Compound Optical Density (SF) Correction:** Optionally divide the $I_0$ signal by the Optical Density of a compound (e.g., BN) using the Henke database (`sf.py`), with custom density (g/cm³), thickness (µm), and auto-scaling.
- **HPC notebook preview:** For either source, the original and modified I0 curves are overlaid and update when the controls change. Use an odd smoothing window of at least 3 points and no longer than the spectrum. Click **Apply** to build the dashboard using the modified I0, or **Cancel** to build it using the original I0. Invalid settings disable Apply and show an error. The plotting call returns before this choice is made; the dashboard appears after the button is clicked.
- **Normalized Plot Titles:** $I_0$ normalization details are formatted into compact, 2-line small print (`fontsize=8.5`) above normalized subplots so long parameter strings never overlap or crowd the spectral curves.

**IPFY Mode (Inverse Partial Fluorescence Yield):**
- **Purpose & Inversion Math:** When **"IPFY Mode (Invert for PCA)"** is checked, the normalized spectra $N(E) = \text{RAW}(E) / I_0(E)$ are inverted by multiplying by $-1$ and shifted by a baseline constant:
  $$\text{Offset} = |\min(-N(E))| + 500 = \max(N(E)) + 500$$
  $$\text{IPFY}(E) = -N(E) + \text{Offset}$$
  This transforms inverse fluorescence dips into positive absorption-like XANES peaks with a 500-unit baseline offset above zero, making them directly ready for PCA, K-Means clustering, and speciation.
- **CSV Export & TEY Behavior:** Clicking **"Save XANES Spectra"** while IPFY Mode is active exports the inverted and baseline-shifted IPFY normalized fluorescence spectra into `NORM_IPFY_sdd1` through `NORM_IPFY_Average_SDD`. Total Electron Yield (TEY / `mcc4`) is already a direct upright absorption spectrum, so it is preserved as standard $I_0$-normalized ($\text{TEY} / I_0$) without inversion to overlay accurately with the IPFY spectra.
- **Header Metadata Transparency:** The saved CSV header includes complete IPFY mathematical details:
  - `# IPFY Mode: Active`
  - `# IPFY Math Operation: Normalized_IPFY_SDD = (-1 * (RAW / I0)) + Offset`
  - `# IPFY Baseline Formula: Offset = abs(min(-1 * (RAW / I0))) + 500`
  - `# IPFY Detector Offsets: sdd1: +<offset1>, sdd2: +<offset2>, sdd3: +<offset3>, sdd4: +<offset4>`

**Exporting Data & Research Metadata Caching:**
- When saving spectra or data from the summary dashboard, you can toggle **"Add Sample Specific Information to File Header"** or click the green **"Edit Metadata"** button located directly on the control panel.
- **Unchecked (Default):** Skips the metadata prompt and includes a standard header containing all relevant beamline parameters (`Endstation:`, `# Grid Dimensions: Nx x Ny (# points)`, `# Time Per Image:`, and `# SDD Calibration: Active (Scan: ..., Edges: ...)`).
- **Checked / Edit Metadata:** Opens an expanded dialog where you can add or edit sample-specific research metadata (Compound Name, Chemical Formula, Authors, Affiliation, Element, Edge, Preparation Method, Calibrated To, Calibration Reference, Temperature, Scan Mode, Chamber Conditions, Spot Size, Comments). This information is expected to be used primarily for **reference compounds** submitted to the [Canadian Light Source X-ray Absorption database](https://xasdb.lightsource.ca/) and the [Federated Research Data Repository (FRDR)](https://www.frdr-dfdr.ca/repo/).
- **Persistent Memory & Auto-Filling Across Scans:** All research metadata entries are automatically cached in Python memory and atomically saved to disk (`~/.sgm_last_metadata.json`). When loading a new scan, opening the dialog pre-fills all your previous research parameters while automatically substituting **Sample Name** (`Name`) with the current scan's name, allowing single-click confirmation or quick on-the-fly editing without re-typing.
- **Re-Editing Anytime:** You can click **Edit Metadata** or save again at any time to review or modify the pre-filled metadata fields before or after exporting.

- **Live Spatial Deconvolution:**
  - **Purpose:** Restores spatial resolution lost to the finite size of the X-ray beam spot (PSF) and scanning motion blur.
  - **How to Use:** Located in the dashboard controls panel under the label **Deconvolution (Live)**. Check **"Enable Deconv"**, select the algorithm (**Richardson-Lucy** or **Wiener Filter**), and adjust the **PSF FWHM (µm)** slider to match your estimated beam size (e.g. $10\,\mu\text{m}$).
  - **Iterative Optimization:** For Richardson-Lucy, use the **Iterations** slider to tune the deblurring intensity (typically 15 to 30 is optimal).
  - **Responsiveness:** Sliders use delayed updates (`continuous_update=False`) to avoid CPU lag during dragging. The calculation runs immediately once you release the slider handle.
  - **Stability Notes:** Utilizes a custom division-by-zero epsilon check ($10^{-12}$) and intensity normalization to prevent NaNs and solid-color blank plots on masked/trimmed margins.

**Map Rendering Mode: `as_scatter_plot=False` vs. `as_scatter_plot=True`**

When calling `plot_sgm_bsky_data(sgm_data, as_scatter_plot=...)`, you can choose between two spatial map rendering modes:

| Feature | `as_scatter_plot=False` (Default — `tripcolor`) | `as_scatter_plot=True` (Pixel-Matched `scatter`) |
| :--- | :--- | :--- |
| **How It Works** | Builds a **Delaunay triangulation** (`ax.tripcolor`) connecting exact $(x, y)$ stage coordinates into a continuous, gap-free polygonal mesh (`shading='flat'`), automatically masking stretched triangles across missing quadrants or L-shaped gaps. | Plots each $(x, y)$ measurement as an individual **square marker** (`marker='s'`, `linewidths=0`) whose area ($s$ in $\text{pt}^2$) is dynamically sized to match the physical grid spacing on screen. |
| **Advantages** | - **True edge-to-edge geometry:** Adjacent points meet at exact shared polygon boundaries with **zero overplotting**, regardless of zoom level or figure DPI.<br>- **Zero directional/serpentine bias:** Completely immune to marker draw-order effects on bidirectional (snake) fly-scans.<br>- **Smooth zoom & vector export:** Polygons scale naturally in physical $(x, y)$ data coordinates. | - **Discrete raw point inspection:** Shows each raw measurement point independently without triangulating between neighbors.<br>- **Fast point-cloud rendering:** Useful for sparse, irregularly sampled, or non-gridded coordinate sets where triangulation is undesirable.<br>- **Crisp L-shaped / staggered outer edges:** Never attempts to bridge concave boundary corners with triangles. |
| **Disadvantages** | - **Slightly slower initial mesh build** on very large stitched point clouds ($>250\text{k}$ points).<br>- Requires edge-length masking (`max_edge`) so triangles do not stretch across wide empty gaps between non-contiguous tiles. | - **Draw-order overlap sensitivity if markers exceed 1 pixel:** Because SGM fly-scans run bidirectionally (even rows left $\rightarrow$ right, odd rows right $\rightarrow$ left), oversized markers cause dark background points to overwrite opposite sides of thin bright features on alternating rows.<br>- *(Note: `plot_sgm_bsky_data` now attaches a dynamic pixel-matched resizer `compute_optimal_scatter_size` / `attach_dynamic_scatter_resizer` that automatically scales $s$ to ~1 physical pixel step, eliminating shifted-shadow artifacts while zooming or exporting).* |
| **When to Use** | **Recommended for most maps**, especially high-resolution fly-scans ($200 \times 200$ to $500 \times 500+$), samples with fine/thin high-contrast features (e.g., thin ridges, cell walls, punctate particles), and publication figures. | Useful when inspecting **low-resolution/coarse maps**, verifying **raw discrete sampling locations** (e.g., checking for dropped points or irregular step spacing), or viewing **staggered multi-tile layouts** as discrete point clouds. |

  - **Why Bidirectional (Serpentine) Fly-Scans Can Show a "Shifted Shadow" with Fixed Scatter Markers (and Why It Appeared on Stitched High-Res Maps):**
    - **Did the beamline scanning method change? No.** Both older scans (e.g., July 2026 Butterfly Wing maps) and newer scans (e.g., October 2026 Wheat maps) use the exact same **bidirectional (serpentine / snake) hexapod fly-scan** (even rows scan Left $\rightarrow$ Right, odd rows scan Right $\rightarrow$ Left).
    - **How serpentine order interacts with oversized markers:** Matplotlib renders scatter points sequentially in array order. When Matplotlib's fixed default marker size (`s = 36` $\text{pt}^2$, a $6\text{ pt} \times 6\text{ pt}$ square) was larger than a single pixel step, dark background points immediately following a narrow bright feature overwrote the **right half** of the square on even rows (`L → R`) and the **left half** on odd rows (`R → L`), visibly splitting thin ridges or dots into two shifted copies.
    - **Why it wasn't noticeable before:**
      1. **Default is `as_scatter_plot=False` (`tripcolor`):** `interactive_roll_align`, `interactive_stitching_trim`, and the default `plot_sgm_bsky_data(..., as_scatter_plot=False)` all use `tripcolor` (no overlapping markers). Only notebooks explicitly passing `as_scatter_plot=True` used `scatter`.
      2. **Sample Morphology:** Samples with **very thin, high-contrast ridges or punctate spots (~2–4 pixels wide) against a dark background** (like the wheat Si K-edge ridge) get split in half by trailing dark pixels, whereas broader continuous regions (like butterfly wing veins/scales or $30 \times 30$ to $60 \times 60$ XANES stacks) blend interior overlaps smoothly.
      3. **Wide Staggered Multi-Tile Span & Compact 30%-Smaller Subplots:** Fitting a wide staggered 3-tile span (~900 pixels across $5.8\text{ mm}$) into the compact $\sim 157\text{ pt}$ subplot shrunk each pixel to $\sim 0.17\text{ pt}$, making a $6\text{ pt}$ marker span $>30$ pixels until dynamic pixel-matched sizing (`compute_optimal_scatter_size`) was introduced.
    - *(See [`SGM_Map_Rendering_and_Scatter_Plot_Guide.md`](SGM_Map_Rendering_and_Scatter_Plot_Guide.md) for the complete standalone reference).*

### 3.1 `plot_sgm_bsky_data_batch.py` (Batch Processing & Comparison of Multiple Scans)
**Purpose:** Enables batch processing and side-by-side comparison of XANES spectra across multiple loaded stack scans (`SGM_BSky_Data_Analysis-batch.ipynb`). It processes all scans using unified spatial and spectral ROI parameters, normalizes the spectra ($I_0$ internal/external with optional smoothing, energy shift, and optical density correction), and exports summary CSV files for each scan to its original directory.

**Batch Workflow & Features:**
- **HPC batch selection:** Run the `sgm_data_batch = analyze_sgm_bsky_data_batch()` cell and navigate to the project in the in-notebook browser. Add individual `.h5` files from one or more folders with **Add Selected File**, or use **Add Current Folder** to include `.h5` files recursively from the current folder and its subfolders. Duplicate files are added only once. Remove selected entries or clear the selection as needed, then click **Load Batch**. Wait for the loaded-scan summary before running the batch plotting cell. The returned list is initially empty and is populated when loading finishes; Cancel before loading leaves it empty. Desktop selection and explicit file-path lists continue to work as before.
- **Scan and spectrum identification:** The batch loading summary shows **Sample Name** and a single **Scan Name**, without a duplicate filename entry. Each raw/normalized figure has one shared **Sample Name** legend to the right, outside the spectra; entries omit scan timestamps, `stack_data`, and waterfall offset suffixes. Colors are consistent across detector and MCC subplots. Waterfall mode still adds display-only vertical offsets to separate curves; these are not measured intensities. Raw and normalized plot titles display the selected **SDD Energy ROI in eV**, including after manual refresh. Without SDD calibration, channel units are retained and explicitly marked as energy unavailable.
 - **Automatic element ROI:** Use `plot_sgm_bsky_data_batch(sgm_data_batch, auto_element_roi=True)` and select **Yes** when asked to apply SDD calibration. The first multi-energy spectrum stack (skipping survey maps) is fitted using the same element-detection and Gaussian-fitting helper as the single-scan dashboard; its energy window is applied to every scan with detector-specific calibration. Optional `element="Si"` overrides element detection and `n_sigma=2.0` controls the fit width. A successful fit overrides `xrf_roi`; if fitting or calibration is unavailable, an explicit warning is printed and the supplied ROI (or default channel ROI) is retained. The ROI inspector remains available for manual changes.
1. **Interactive XRF ROI Inspector (Tab 0):** The batch dashboard includes an integrated XRF spectrum inspector. You can click and drag the mouse on the peak (or use the interactive span selector) to dynamically adjust your spectral ROI (`xrf_roi` in eV or `channel_roi`), or type exact bounds into the `ROI Start` and `ROI End` input fields.
2. **Live Batch Refresh:** Click the green **REFRESH BATCH PLOTS** button to instantly re-calculate and update all raw and normalized XANES spectra across every loaded scan.
3. **$I_0$ Selection & Interactive Preview Dialog:** Prompts whether to use Internal $I_0$ (`mcc1`) or load an External $I_0$ CSV file. For external files, the interactive **I0 Preview Dialog** lets you:
   - Select Energy (X) and Intensity (Y) columns.
   - Apply **Savitzky-Golay smoothing** with an adjustable window size.
   - Apply an **Energy Shift (eV)** to align the $I_0$ standard with your scan energies.
   - Divide by **Compound Optical Density (OD)** using the Henke database (`sf.py`), with configurable chemical formula, density (g/cm³), thickness (µm), and auto-scaling.
4. **SDD Energy Calibration:** Prompts whether to apply `sdd_calibration.json` calibration. When active, it uses energy-equivalent physical bounds (eV) to ensure precise cross-detector alignment.
5. **Waterfall Offset Option:** Offers an optional vertical offset to separate and stagger spectral lines for clear comparison.
6. **Comparative Figures:** Generates multi-panel comparative figures for both **RAW** and **NORMALIZED** XANES spectra across all scans. Normalized titles use compact multi-line formatting so long $I_0$ parameters never overlap the plots.
7. **Automatic CSV Export:** Click the yellow **Save XANES Spectra CSVs** button to export individual XANES summary CSV files (`<ScanName>_Rect_<ROI>_summary.csv`) with complete metadata headers directly into each scan's original folder.

### 3.2 `plot_2d_rixs_map.py` (2D Incident vs. Emission RIXS Maps)
**Purpose:** Generates 2D heatmaps of Incident Energy (X-axis) vs. Emission Energy/Channel (Y-axis) for RIXS and 2D XRF map visualization.
- **Orientation:** Incident energy is plotted on the X-axis (eV) and Emission energy (eV) or channels (0-255) on the Y-axis.
- **Grid Layout:** Plots are displayed compactly side-by-side (1x2 for 2 detectors, 2x2 grid for 4 detectors).
- **Interactive Save Button:** Includes a yellow **"Save 2D Map"** button on the figure canvas so you can inspect the image on screen first before choosing to save to the original data folder with overwrite confirmation.

### 3.3 `plot_selected_emission_spectra.py` (Overlaid Emission / XRD Spectra & Peak Dispersion Analysis)
**Purpose:** Extracts and overlays 1D emission/XRD spectra for selected incident energies and derives quantitative peak shifts as a function of incident energy.
- **Detector Selection (`detector`):** Accepts individual detectors (`'sdd1'`, `'sdd2'`, `'sdd3'`, `'sdd4'`), `'average'` (averaging all detectors), or **`'all'`** to automatically plot every detector present in the stack.
- **Dual-Panel Display (`plot_mode='both'`):**
  - Rendered in a compact, viewport-friendly figure size (30% smaller, $10.0 \times 4.2$ inches) so both panels are fully visible simultaneously on any screen without horizontal scrolling.
  - **Left Panel (Spectra Overlay):** Overlays 1D emission spectra with individual curve colors, optional waterfall vertical offsets (`apply_waterfall=True`), and marks fitted peak positions directly on each spectrum.
  - **Right Panel (Peak Dispersion Curve):** Plots detected Emission Peak Position ($E_{\text{emission}}$ in eV) vs. Incident Energy ($E_{\text{inc}}$ in eV).
- **Peak Tracking Capabilities (`track_peak='both'`, `'scatter'`, or `'fluorescence'`):**
  - **Elastic Scatter Peak ($E_{\text{emission}} \approx E_{\text{inc}}$):** Follows the diagonal reference line with an expected slope of $1.000$, validating detector energy calibration and linearity.
  - **Characteristic Fluorescence Peak ($E_{\text{emission}} \approx E_{\text{fluor}}$):** Follows a horizontal reference line (e.g. Mg K-edge at $1253.6\text{ eV}$), revealing subtle resonant Raman shifts or verifying peak stability.
- **Deconvolution:** Employs a joint double-Gaussian model when the scatter peak and fluorescence peak overlap near the absorption threshold ($<160\text{ eV}$ separation) to completely prevent peak pulling or cross-talk.
- **Automatic Element Lookup:** Automatically detects the target element and edge from scan/sample names (e.g. `TyMg807` -> Mg K-edge at $1253.6\text{ eV}$) or energy range proximity, or accepts explicit `element='Mg'` and `edge='K'`.
- **Interactive Save Buttons:**
  - **"Save Plot":** Exports the dual-panel figure (PNG/PDF/SVG).
  - **"Save Spectra & Shifts (CSV)":** Exports a comprehensive CSV file containing a **Peak Dispersion Summary Table** (`Incident_Energy_eV`, `Scatter_Peak_eV`, `Scatter_FWHM_eV`, `Scatter_Amp`, `Fluor_Peak_eV`, `Fluor_FWHM_eV`, `Fluor_Amp`) followed by the full 256-channel spectra.

### 4. `pca_xanes_analysis.py`
**Purpose:** Performs Principal Component Analysis (PCA) to reduce the dimensionality of the XANES stack and isolate the primary chemical variations (components) while filtering out background noise.
**How it works:**
- Flattens the 3D stack, scales the data, and computes eigenvectors (Loadings) and eigenimages (Scores).
- Automatically filters out dead/empty pixels to prevent math errors.
- Saves eigenvectors to a CSV and the eigenimages back into the HDF5 file under `entry/pca_results`.
- Displays spatial map axes with physical units **`X (mm)`** and **`Y (mm)`**.

> [!IMPORTANT]
> **Data Normalization & Scaling in PCA/CA:**
> - **$I_0$ Intensity Normalization:** The input HDF5 stack (e.g. `_PCA-CA.h5`) contains data already normalized by the incident beam intensity $I_0$ (via `mcc1 (Au Mesh)` or external standards) during the saving stage in the main dashboard.
> - **Metadata & Header Transparency:** PCA execution logs, plot suptitles, and saved vector CSVs (`*_pca_vectors_<dataset>.csv`) include explicit normalization info:
>   `# Normalized: Yes (I0 Source: Internal (mcc1 / Au Mesh))`
> - **Standard Scaling (Z-score Normalization):** Before performing the PCA decomposition, the data is automatically centered (mean subtracted) and scaled (divided by standard deviation) using `scikit-learn`'s `StandardScaler` to prevent high-intensity features/energies from disproportionately dominating the variance.
> - **Automatic HDF5 File Redirection:** If an `_Elemental_PyMca.h5` 4D hypercube file path is selected by mistake, `resolve_pca_h5_path` automatically detects it and redirects to the corresponding 3D XANES stack (`_PCA-CA.h5`).

**Multi-Detector "Run All" Mode:**
- You can process individual datasets (like `average`) or you can tell the script to run **all** datasets at once (`sdd1`, `sdd2`, `sdd3`, `sdd4`, and `average`).
- Running them all at once automatically generates comparative, side-by-side grids (one for eigenimages, one for eigenvectors) so you can easily spot detector-specific artifacts or verify consistency across all SDDs. All figure titles display $I_0$ normalization metadata (e.g. `[Normalized: Yes (I0 Source: Internal (mcc1 / Au Mesh))]`).
- *Note: To maintain high visibility in Jupyter, these large comparative grids are rendered with a built-in horizontal scrollbar, preventing them from overflowing or shrinking too small.*

### 5. `cluster_xanes_analysis.py`
**Purpose:** Uses K-Means clustering on the PCA scores (eigenimages) to group pixels that share similar XANES signatures.
**How it works:**
- You specify the number of clusters ($k$).
- The algorithm assigns each valid pixel to a cluster, producing a spatial map of distinct "zones" or species.
- It then computes the average XANES spectrum for all pixels within each cluster.
- Saves the cluster maps and extracted spectra to CSVs, PNG previews, and back into the HDF5 file.
- All exported CSV headers (`*_cluster_spectra_summary.csv` and `*_all_detectors_cluster_sums.csv`) and figure suptitles document $I_0$ normalization details (e.g., `# Normalized: Yes (I0 Source: Internal (mcc1 / Au Mesh))`).
- Automatically redirects `_Elemental_PyMca.h5` inputs to `_PCA-CA.h5`.

**Multi-Detector "Run All" Mode:**
- Just like the PCA analysis, you can cluster all SDD detectors and the `average` simultaneously.
- This outputs a comparative grid of cluster maps and a grid of the extracted cluster XANES spectra. It is extremely useful for validating that the chemical zones detected are real sample features and not just anomalies occurring on a single detector.
- *Note: Similar to PCA, these multi-detector outputs are rendered as horizontally scrollable figures within Jupyter Notebooks.*
- **Cluster Sums:** In multi-detector mode, the tool also calculates and saves the **Sum** of all spectra in each cluster (in addition to the Mean), providing the total integrated signal per detector.

**Metadata & CSV Export:**
- All clustering CSV exports include a standard header with beamline metadata and $I_0$ normalization details (`# Normalized: Yes (I0 Source: Internal (mcc1 / Au Mesh))`).
- **Pass `metadata` dict:** To ensure headers are complete (Project, Date, etc.), pass the `pp` (path_pack) dictionary from `analyze_sgm_bsky_data` to the `metadata` parameter of the clustering functions.
- **Full Metadata Mode:** You can enable `use_full_metadata=True` to prompt for detailed research metadata (Authors, Affiliation, etc.) before saving.

### 6. `interactive_cluster_merger.py`
**Purpose:** A powerful interactive dashboard that lets you review the K-Means clustering results, combine similar clusters, and extract a single, high signal-to-noise XANES spectrum.
**User Interaction (GUI):**
- The dashboard displays the Cluster Map on the left and the Merged Spectrum on the right. Both display physical units (`X (mm)`, `Y (mm)`) and $I_0$ normalization metadata.
- **Detector Dropdown:** Switch between `sdd1`, `sdd2`, `sdd3`, `sdd4`, or `average` to see how clustering performed across different detectors.
- **Checkboxes:** Check or uncheck clusters ($C1, C2, C3...$) to see their combined average spectrum update in real-time. The map will also mask out unselected clusters so you can verify exactly which spatial regions are contributing to your spectrum.
- **Normalization Info:** Displays $I_0$ normalization source details (e.g. `Data Normalized: Yes (I0 Source: Internal (mcc1 / Au Mesh))`) in the UI header and saved CSV headers.
- **Save Button:** Click "Save Merged Spectrum" to export the live spectrum to a CSV file (includes beamline metadata and normalization info in the file header).
- **Auto-Redirection:** Automatically resolves `_Elemental_PyMca.h5` paths to `_PCA-CA.h5`.

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

### Exporting Images and Histograms
When you click **"Save All Images"** on the dashboard toolbar:
- High-resolution PNG and TIFF files are saved for every detector's representative energy map and stack average map.
- In addition, a standalone high-resolution **Pixel Intensity Distribution Histogram** PNG is automatically saved for each detector (`<scan_name>_<detector>_Histogram.png`).
- On headless HPC/JupyterHub, **Save All Images** first shows an editable **Save folder** inside the notebook. The suggested destination is `<save_dir>/<scan_name>/Images`, with `raw` redirected to `preprocessed`; the scan-name component is not repeated if the save directory already has that name. This separates maps even when their HDF5 files share a directory. Change the path or folder name, then click **Save Images**; **Cancel** writes nothing. Missing folders are created only on confirmation, and the final absolute destination is shown in the export status and notebook output. `save_images=True` retains automatic export without prompting, using this map-specific default.
- Desktop exports retain the original `<save_dir>/Images` destination and one-click behavior.

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
