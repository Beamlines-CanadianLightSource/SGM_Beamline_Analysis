# PyMCA Elemental Mapping Guide: HDF5 Dataset Selection

This guide explains which dataset group to select in **PyMCA** when opening saved 4D/3D HDF5 (`.h5`) files created by the SGM Beamline Analysis pipeline, depending on whether you are doing single-energy elemental mapping, energy-summed mapping, or XANES spectral analysis.

---

## Quick Reference Summary

| Dataset Group in HDF5 | Data Shape / Dimensions | Intended Use Case |
| :--- | :--- | :--- |
| **`entry/full_4d_measurement`** | `(Y, X, Energy, Channels)` | **Elemental mapping at a specific single excitation energy** (or energy-resolved XRF mapping). |
| **`entry/xrf_measurement`** | `(Y, X, Channels)` | Elemental mapping **summed across all excitation energies** (highest signal-to-noise ratio). |
| **`entry/xanes_measurement`** | `(Y, X, Energy)` | XANES spectrum extraction, total yield maps, and PCA across excitation energies (**NOT** for XRF elemental mapping). |

---

## 1. Single-Energy Elemental Mapping: `full_4d_measurement`

### What it is
* **Dataset Signal**: `sdd_sum_4d` (or individual detector group like `sdd1`)
* **Dimensions**: `(ny, nx, n_energies, n_channels)` — 4D Hypercube

### Why use it for single-energy maps
* Preserves all **256 MCA detector channels** for **every individual excitation energy step** in the scan.
* Allows you to isolate a specific beam energy (e.g., above or below an absorption edge) and perform ROI integration or peak fitting across detector channels for that exact excitation condition.

### Step-by-Step in PyMCA:
1. Open PyMCA and select **File $\rightarrow$ Open**.
2. Expand the file hierarchy: `entry` $\rightarrow$ **`full_4d_measurement`**.
3. Open the dataset in **ROI Imaging / Stack Tool**.
4. Use the **Energy Axis / Index slider** to select your target single excitation energy.
5. Define your ROI window(s) over the 256 detector channels (e.g., Fe $K_\alpha$, Ca $K_\alpha$, Si $K_\alpha$) or run PyMCA Matrix/Batch fitting to generate spatial elemental maps at that single energy.

---

## 2. Energy-Summed Elemental Mapping: `xrf_measurement`

### What it is
* **Dataset Signal**: `sdd_xrf_stack_3d`
* **Dimensions**: `(ny, nx, n_channels)` — 3D Stack

### Why use it
* Sums the 256 MCA channels across **all excitation energy points** in the scan.
* Produces maximum signal-to-noise ratio for detecting trace elements when excitation energy dependence is not restricted to one specific energy step.

### Step-by-Step in PyMCA:
1. Expand `entry` $\rightarrow$ **`xrf_measurement`**.
2. Click on `sdd_xrf_stack_3d` and open with **ROI Imaging / Stack Tool**.
3. Select elemental energy channel regions directly on the summed 256-channel spectrum.

---

## 3. XANES Spectrum & PCA Analysis: `xanes_measurement`

### What it is
* **Dataset Signal**: `sdd_xanes_stack_3d`
* **Dimensions**: `(ny, nx, n_energies)` — 3D Stack

### Why **NOT** to use it for XRF Elemental Mapping
* The 256 MCA detector channels have **already been integrated/collapsed** into a single intensity value per energy step.
* Because MCA energy channel information is collapsed, you cannot separate individual elemental fluorescence emission peaks (e.g., separating Fe vs. Ca) from this group.
* **Primary Use**: Loading as a 1D Stack for XANES spectrum extraction at specific pixels, PCA (Principal Component Analysis), or k-means clustering across excitation energies.

---

## 4. Grid Dimensions (Rows & Columns) & Bypassing the Prompt

When opening 4D hypercubes or 3D stacks, PyMCA needs to know how spatial pixels are arranged on the 2D grid (`ny` Rows $\times$ `nx` Columns).

### How to Bypass the "Number of Rows" Prompt Automatically
Instead of right-clicking or opening an individual 4D raw dataset array (`sdd_sum_4d`), **double-click the group container itself** (e.g., **`entry/full_4d_measurement`** or **`entry/xrf_measurement`**):
* PyMCA identifies the group as a NeXus `NXdata` structure.
* It automatically reads the spatial `y` axis length as the number of rows (`ny`) and `x` as columns (`nx`) without popping up the dimension prompt dialog.

### How to Find the Row Count if Prompted
If PyMCA prompts you for **"Number of rows:"**:
1. **PyMCA HDF5 File Tree**:
   * Expand `entry/full_4d_measurement/y` (or `entry/xrf_measurement/y`). The number of array elements in **`y`** is the exact number of **Rows** (`ny`).
   * Alternatively, click on **`stack_metadata`** or **`full_4d_measurement`**. Look for the **`ny`** / **`rows`** attribute in the attribute panel.
2. **Console / Notebook Output**:
   * When `analyze_sgm_bsky_data` or `save_pymca_4d_stack_h5` runs, it prints the grid dimensions directly in the console:
     `Detected Grid: 41 x 41` $\rightarrow$ `Detected Grid: [ny (Rows)] x [nx (Columns)]`.
