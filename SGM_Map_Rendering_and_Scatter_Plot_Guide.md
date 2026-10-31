# SGM Beamline Map Rendering Guide: `as_scatter_plot=False` vs. `as_scatter_plot=True`

This document explains the two spatial map rendering modes in [`plot_sgm_bsky_data.py`](plot_sgm_bsky_data.py), their advantages and disadvantages, when to use one over the other, and why bidirectional (serpentine) fly-scans can exhibit a "shifted shadow / double image" artifact if scatter markers are not pixel-matched.

---

## 1. Overview of Map Rendering Modes

When calling `plot_sgm_bsky_data(sgm_data, as_scatter_plot=...)`, you can choose how the 2D spatial fluorescence maps (`SDD1`–`SDD4` and `TEY / MCC4`) are rendered on screen and in exported images:

```python
# Option A: Triangulated continuous mesh (Default & Recommended for high-res maps)
plot_sgm_bsky_data(sgm_data, as_scatter_plot=False)

# Option B: Discrete pixel-matched square markers
plot_sgm_bsky_data(sgm_data, as_scatter_plot=True)
```

| Feature | `as_scatter_plot=False` (Default — `tripcolor`) | `as_scatter_plot=True` (Pixel-Matched `scatter`) |
| :--- | :--- | :--- |
| **How It Works** | Builds a **Delaunay triangulation** (`ax.tripcolor`) connecting exact $(x, y)$ stage coordinates into a continuous, gap-free polygonal mesh (`shading='flat'`), automatically masking stretched triangles across missing quadrants or L-shaped gaps. | Plots each $(x, y)$ measurement as an individual **square marker** (`marker='s'`, `linewidths=0`) whose area ($s$ in $\text{pt}^2$) is dynamically sized to match the physical grid spacing on screen. |
| **Advantages** | - **True edge-to-edge geometry:** Adjacent points meet at exact shared polygon boundaries with **zero overplotting**, regardless of zoom level or figure DPI.<br>- **Zero directional/serpentine bias:** Completely immune to marker draw-order effects on bidirectional (snake) fly-scans.<br>- **Smooth zoom & vector export:** Polygons scale naturally in physical $(x, y)$ data coordinates. | - **Discrete raw point inspection:** Shows each raw measurement point independently without triangulating between neighbors.<br>- **Fast point-cloud rendering:** Useful for sparse, irregularly sampled, or non-gridded coordinate sets where triangulation is undesirable.<br>- **Crisp L-shaped / staggered outer edges:** Never attempts to bridge concave boundary corners with triangles. |
| **Disadvantages** | - **Slightly slower initial mesh build** on very large stitched point clouds ($>250\text{k}$ points).<br>- Requires edge-length masking (`max_edge`) so triangles do not stretch across wide empty gaps between non-contiguous tiles. | - **Draw-order overlap sensitivity if markers exceed 1 pixel:** Because SGM fly-scans run bidirectionally (even rows left $\rightarrow$ right, odd rows right $\rightarrow$ left), oversized markers cause dark background points to overwrite opposite sides of thin bright features on alternating rows.<br>- *(Resolved by dynamic pixel-matched marker sizing in `plot_sgm_bsky_data.py` — see Section 2 below).* |
| **When to Use** | **Recommended for most maps**, especially high-resolution fly-scans ($200 \times 200$ to $500 \times 500+$), samples with fine/thin high-contrast features (e.g., thin ridges, cell walls, punctate particles), and publication figures. | Useful when inspecting **low-resolution/coarse maps**, verifying **raw discrete sampling locations** (e.g., checking for dropped points or irregular step spacing), or viewing **staggered multi-tile layouts** as discrete point clouds. |

---

## 2. How Pixel-Matched Scatter Sizing Works (`as_scatter_plot=True`)

In Matplotlib, `ax.scatter(x, y, c=z, marker='s')` defaults to a fixed marker area of **`s = 36` $\text{pt}^2$** (a $6\text{ pt} \times 6\text{ pt}$ square on screen), regardless of how many data points are in the map.

To prevent overlapping markers on high-density maps, `plot_sgm_bsky_data.py` uses two helper functions:
1. **`compute_optimal_scatter_size(ax, x, y)`**:
   - Estimates the median physical step size $(\Delta x, \Delta y)$ between consecutive scan points.
   - Projects $(\Delta x, \Delta y)$ into figure display points ($\text{pt}$, where $1\text{ pt} = 1/72\text{ inch}$) for the current subplot bounding box and axis limits.
   - Sets the square marker area $s = (\text{pixel\_step\_pt} \times 1.05)^2$ so each square marker covers **~1 physical pixel** with no gaps and no neighboring overlap.
2. **`attach_dynamic_scatter_resizer(ax, scatter_coll, x, y)`**:
   - Hooks into Matplotlib's `xlim_changed`, `ylim_changed`, and `resize_event` callbacks so that if you zoom into a region or resize the figure, the scatter marker size $s$ automatically recalculates to match the zoomed pixel scale.

---

## 3. FAQ: Why Did the "Shifted Shadow / Double Image" Artifact Appear Now, and Did Our Scanning Method Change?

### Did the SGM beamline scanning method change?
**No.** Inspection of both older July 2026 maps (e.g., Butterfly Wing `S5_map_2026-07-29_001935_1195.06eV.h5`, $389 \times 389$) and recent October 2026 maps (e.g., Wheat `sample_1_A_map_2026-10-28_184252_1850.00eV.h5`, $302 \times 302$) confirms that the beamline has always used the same **bidirectional (serpentine / snake) hexapod fly-scan**:
- **Even rows (Row 0, 2, 4, ...):** Stage scans **Left $\rightarrow$ Right** ($+X$).
- **Odd rows (Row 1, 3, 5, ...):** Stage scans **Right $\rightarrow$ Left** ($-X$).

### How does serpentine scanning interact with oversized scatter markers?
When Matplotlib draws a scatter plot, it renders markers **sequentially in array order** (later points paint directly on top of earlier points):

```text
Even Row (Left -> Right):   ... [Dark] -> [BRIGHT] -> [Dark] -> [Dark] ...
                                            ^
                             Subsequent dark points overwrite the RIGHT side of the bright square
                             => Apparent bright center shifts LEFT (-3 pt)

Odd Row  (Right -> Left):   ... [Dark] <- [Dark] <- [BRIGHT] <- [Dark] ...
                                                       ^
                             Subsequent dark points overwrite the LEFT side of the bright square
                             => Apparent bright center shifts RIGHT (+3 pt)
```

When the marker size was fixed at $6\text{ pt} \times 6\text{ pt}$ (`s = 36`) while the true pixel spacing was only $0.17\text{--}0.5\text{ pt}$, every marker was **12 to 33 pixels wide**. As a result, a narrow bright feature was shifted left by ~6–15 pixels on even rows and right by ~6–15 pixels on odd rows, visually splitting any thin line or spot into a **shifted double image / ghost shadow**.

### Why was this not noticeable in earlier datasets?
Four specific factors converged on the stitched wheat dataset to make this artifact stand out starkly:

1. **Other tools and the default setting use `tripcolor` (`as_scatter_plot=False`):**
   - The alignment utility (`interactive_roll_align`), the stitching utility (`interactive_stitching_trim`), and the default signature of `plot_sgm_bsky_data(..., as_scatter_plot=False)` all use `tripcolor`, which never overlaps markers.
   - In `SGM_BSky_Data_Analysis-4quad.ipynb` (Cell 5), `as_scatter_plot=True` was explicitly passed in the function call.
2. **Sample Morphology (Thin High-Contrast Ridge & Punctate Spots vs. Broad Regions):**
   - The wheat sample at the Si K-edge ($1850\text{ eV}$) contains a **very thin, bright diagonal silica ridge and tiny isolated phytolith/trichome spots (~2–4 pixels wide) surrounded by near-zero dark background**.
   - When a 2-pixel-wide bright ridge is surrounded by dark background, dark pixels immediately after the ridge completely erase the trailing half of an oversized marker on alternating rows, splitting the ridge into two parallel lines.
   - In contrast, on samples with **broad, continuous features** (such as butterfly wing veins/scales) or **coarser XANES stacks** ($30 \times 30$ to $60 \times 60$ pixels, where each pixel is already $3\text{--}5\text{ pt}$ wide), overlapping markers blend into the interior of the feature rather than splitting thin isolated points into double images.
3. **Staggered Multi-Tile Aspect Ratio ($5.8\text{ mm} \times 3.4\text{ mm}$):**
   - Because the three wheat tiles (`A`, `B`, `C`) were staggered diagonally across a wide horizontal span (~900 pixels across 3 tiles), each pixel occupied only $\sim 0.17\text{ pt}$ on screen—so a $6\text{ pt}$ marker spanned **over 30 pixels** at once.
4. **30% Compact Dashboard Subplot Size:**
   - The recent UI update scaled the dashboard row figures down by ~30% (`figsize=(10.5, 3.36)`) to fit narrower screens without scrolling. Shrinking the subplot width to $\sim 157\text{ pt}$ while Matplotlib's default `s = 36` marker remained fixed in points increased the marker-to-pixel footprint by an additional ~40%.

