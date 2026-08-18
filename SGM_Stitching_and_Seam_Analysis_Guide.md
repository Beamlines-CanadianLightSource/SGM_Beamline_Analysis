# SGM Beamline Map Stitching & Seam Analysis Guide

This guide provides a comprehensive explanation of spatial map stitching, automated zero-overlap alignment, HDF5 metadata traceability, and physical/spectroscopic causes of seam lines across stitched synchrotron XRF maps.

---

## 1. Overview of Map Stitching Workflow

When collecting large-area X-ray Fluorescence (XRF) and XANES maps at the SGM beamline, datasets are often acquired as a grid of smaller spatial scan quadrants or sub-maps (e.g., $2 \times 2$ or $3 \times 3$ grid arrays).

The `stitching_utils.py` module stitches multiple spatial scan HDF5 files into a single unified master HDF5 file and processes the associated energy-resolved raw detector binary files (`.bin`) and counter files (`.csv`).

### Spatial File & Directory Naming
Stitched datasets automatically analyze physical stage coordinates $(X, Y)$ across all input images to identify the **Top-Left** scan and **Bottom-Right** scan, regardless of the order in which files were selected:

- **Top-Left Scan**: Image with maximum $(Y - X)$ center coordinate (highest $Y$, lowest $X$).
- **Bottom-Right Scan**: Image with minimum $(Y - X)$ center coordinate (lowest $Y$, highest $X$).

**Naming Output**:
- **Master HDF5 File**: `Stitched_<TopLeftScan>_to_<BottomRightScan>.h5`  
  *(e.g., `Stitched_173717_S1_map-xy_to_173725_S9_map-xy.h5`)*
- **Energy Data Directories**: `Stitched_<TopLeftScan>_to_<BottomRightScan>_<energy>eV`

---

## 2. Interactive Trimming and Auto-Alignment

To eliminate boundary overlaps and seam artifacts, `interactive_stitching_trim()` provides live interactive visualization and automated alignment.

### 2.1 How Trimming & Joining Works (Direct Edge Trimming vs. Averaging)

#### Direct Edge Trimming (No Pixel Averaging)
- **We trim right to the physical spatial edge in mm**.
- **We do NOT average pixels at the boundaries**.

#### Why We Do Not Average Pixels:
In synchrotron XRF and XANES mapping, every pixel represents raw photon counts and 256-channel MCA spectra tied to exact physical stage coordinates $(x, y)$. Averaging overlapping pixels from separate scan acquisitions taken minutes apart would distort raw photon counts, corrupt energy spectra, and smear fine chemical features.

#### The Auto-Trimming Edge Split:
When two adjacent images overlap spatially (e.g., Image 1 and Image 2 overlap horizontally by $0.30\text{ mm}$):
1. **Splitting the Overlap**: The algorithm splits the $0.30\text{ mm}$ overlap evenly ($0.15\text{ mm}$ each).
2. **Image 1**: Trimmed by $0.15\text{ mm}$ on its Right edge.
3. **Image 2**: Trimmed by $0.15\text{ mm}$ on its Left edge.
4. **Result**: Both images meet precisely at the midpoint boundary with zero spatial overlap and zero gap.

#### Joining the Datasets:
- **Coordinates**: The kept physical $(x, y)$ coordinates from all images are concatenated into unified master arrays (`master_x`, `master_y`).
- **Raw Detector Data**: Kept pixel rows are extracted directly from raw binary detector files (`.bin`) and counter files (`.csv`) and concatenated without altering any raw photon values.

---

## 3. HDF5 Metadata Traceability

All stitched `.h5` files store full trimming metadata inside the `stitching_metadata` group:

- **Attributes**: `num_source_maps`, `source_files`, `top_left_scan`, `bottom_right_scan`, `stitched_tag`.
- **Per-Map Datasets (`map_1`, `map_2`, ...)**:
  - `trim_left_mm`, `trim_right_mm`, `trim_top_mm`, `trim_bottom_mm`
  - `orig_x_min`, `orig_x_max`, `orig_y_min`, `orig_y_max`
  - `kept_x_min`, `kept_x_max`, `kept_y_min`, `kept_y_max`
  - `total_points`, `kept_points`, `trimmed_points`
  - `trimmed_x`, `trimmed_y`, `kept_x`, `kept_y` coordinate arrays

### Reading Metadata
Use `read_stitching_trim_info("path/to/Stitched_File.h5")` to display the full metadata summary.

---

## 4. Stitching Seams: Physical & Spectroscopic Causes

When plotting stitched maps (e.g., $3 \times 3$ grid), seam lines between rows or columns can arise from physical stage alignment, Delaunay triangulation, or spectroscopic self-absorption effects.

```
       +-------------------+-------------------+-------------------+
       |     Image 1       |     Image 2       |     Image 3       |
       |                   |                   |                   |
Row 1  + - - - - - - - - - + - - - - - - - - - + - - - - - - - - - + <--- Row Seam Y1
       |     Image 4       |     Image 5       |     Image 6       |
       |                   |                   |                   |
Row 2  + - - - - - - - - - + - - - - - - - - - + - - - - - - - - - + <--- Row Seam Y2
       |     Image 7       |     Image 8       |     Image 9       |
       |                   |                   |                   |
Row 3  +-------------------+-------------------+-------------------+
                             ^                   ^
                        Col Seam X1         Col Seam X2
```

### Cause 1: Physical Overlap or Gap in Stage Coordinates
- **Overlap (+mm)**: If adjacent images overlap in $(X, Y)$, data points from both images intermingle along the boundary. Matplotlib's Delaunay triangulation (`tripcolor`) connects triangles across overlapping points, producing zig-zag seam lines.
- **Gap (-mm)**: If images cut off too much, a blank strip of missing coordinates appears.

### Cause 2: Spectroscopic Energy Dependence (Soft vs. Hard X-rays)
A key observation in XRF mapping is that **seams may appear strongly at low energy (Carbon K-edge, 200–350 eV) but disappear completely at high energy (1600–1850 eV)**.

| Feature | Low Energy (Carbon K-edge, 200–350 eV) | High Energy (1600–1850 eV) |
| :--- | :--- | :--- |
| **Photon Escape Depth** | Extremely shallow ($\sim 0.5 - 1.0\,\mu\text{m}$) | Deep penetration ($> 10\,\mu\text{m}$) |
| **Self-Absorption Sensitivity** | Very high; sensitive to micro-tilt ($Z$-axis) | Very low; uniform isotropic emission |
| **Detector Geometry Effects** | Strong takeoff-angle dependence across SDD heads | Negligible directional dependence |
| **Seam Visibility** | **Visible along specific SDD channels** | **Seamless / Invisible** |

### Cause 3: SDD Multi-Element Array Orientation
The 4 Silicon Drift Detector channels (SDD1, SDD2, SDD3, SDD4) sit at different physical orientation angles around the sample stage:
- **SDD3**: Positioned at an angle sensitive to both $X$ (column) and $Y$ (row) stage translation, showing seams in both axes at $284\text{ eV}$.
- **SDD1 & SDD4**: Positioned at angles sensitive primarily to $Y$ (row) stage elevation, showing seams mainly along $Y$ row boundaries at $284\text{ eV}$.
- **SDD Sum (`sdd_sum`)**: Summing all 4 detector elements averages out directional takeoff-angle biases, significantly reducing visible seam steps.

### Cause 4: Single-Energy Scans & $I_0$ Normalization Behavior
- In single-energy map scans, incoming $I_0$ (`mcc1`) current is constant per scan (a single scalar value applied across all pixels).
- Because $I_0$ acts as a uniform constant multiplier for each scan, $I_0$ normalization does **not** alter relative spatial steps or fix takeoff-angle seams within a single-energy map.
- This confirms that seam visibility at low energy (Carbon K-edge, $200\text{--}350\text{ eV}$) is **100% a physical/geometric takeoff-angle & sample micro-tilt effect**, rather than a normalization or code artifact.

---

## 5. Seam Diagnostics & Best Practices

### Diagnostic Tool: `check_stitching_gaps()`
Run `check_stitching_gaps()` to analyze spatial boundaries in a baked stitched `.h5` file:

```python
from stitching_utils import check_stitching_gaps

# Analyze seam boundaries of a stitched dataset:
check_stitching_gaps("path/to/Stitched_173717_to_173725.h5")
```

**Output Example**:
```text
--- Stitching Boundary & Seam Diagnostic: Stitched_173717_to_173725.h5 ---
Detected Grid Layout: 3 Rows x 3 Columns (9 total images)

  [ROW SEAM: Row 1 / Row 2]
    Status: OVERLAP (+0.120 mm)
    Action: Increase Top/Bottom trim by ~0.060 mm to eliminate row overlap seam.

  [ROW SEAM: Row 2 / Row 3]
    Status: PERFECT ALIGNMENT
    Action: No Y-trim adjustment needed.
```

### Summary of Best Practices for Seamless Maps
1. **Pre-populate Trims**: Use `interactive_stitching_trim()` to leverage automatic zero-overlap calculation on load.
2. **Diagnose Seams**: Run `check_stitching_gaps()` on baked `.h5` files to verify physical boundary alignment.
3. **Use `sdd_sum`**: For soft X-ray maps (e.g. Carbon K-edge at $284\text{ eV}$), plot `sdd_sum` (`SDD1 + SDD2 + SDD3 + SDD4`) to average out directional detector takeoff-angle variations.
4. **Enable $I_0$ Normalization**: Normalize raw fluorescence counts by incoming beam mesh current (`mcc1`) to eliminate synchrotron ring current drift across map scans.
