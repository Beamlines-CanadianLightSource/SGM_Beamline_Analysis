# `lmfit` Automated Peak & Position Guessing Guide

This guide explains how `lmfit` handles initial parameter estimation and how to combine `lmfit` with automated peak detection algorithms (such as `scipy.signal.find_peaks`) to automatically guess the number of peaks and their centroid positions prior to non-linear curve fitting.

---

## 1. Native `lmfit` Model Guessing (`model.guess()`)

Individual `lmfit` model classes (such as `GaussianModel`, `LorentzianModel`, `VoigtModel`, `PseudoVoigtModel`) include a built-in `.guess(y, x=x)` method.

- **Capabilities**: Automatically estimates initial starting values for `center`, `sigma`, and `amplitude` for a **single isolated peak** based on data statistics (max height, mean, variance).
- **Limitations**: `model.guess()` operates on a **single peak model**. It cannot automatically determine how many peaks exist in a multi-component spectrum, nor can it unmix overlapping peaks without user specification.

---

## 2. Automated Peak Detection with `scipy.signal.find_peaks`

To automatically discover both the **number of peaks** ($N$) and their **centroid positions** ($x_1, x_2, \dots, x_N$), standard practice in Python spectroscopy uses `scipy.signal.find_peaks` (or `scipy.signal.find_peaks_cwt` for wavelets / heavily overlapped features) prior to building `lmfit` models.

### Algorithm Steps:
1. **Smoothing (Optional)**: Apply Savitzky-Golay filtering (`scipy.signal.savgol_filter`) to suppress high-frequency noise spikes that could be misidentified as peaks.
2. **Peak Discovery**: Run `find_peaks(y, height=..., prominence=..., distance=...)`.
   - `prominence`: Minimum relative height of a peak above surrounding baseline (prevents noise false-positives).
   - `distance`: Minimum separation between adjacent peak centroids (in data point index steps or physical energy units).
   - `height`: Absolute threshold minimum.
3. **Dynamic Model & Parameter Construction**: Feed detected positions directly into `lmfit` model creation routines or `peak_fitting_utils.py`.

---

## 3. Python Implementation Workflow

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import lmfit

def auto_detect_and_fit_peaks(x, y, prominence=0.05, distance=5, model_type='gaussian'):
    """
    Automatically detects peaks, constructs an lmfit composite model, and performs curve fitting.
    """
    # 1. Smooth data to prevent noise false-positives
    y_smooth = savgol_filter(y, window_length=11, polyorder=2) if len(y) > 11 else y

    # 2. Automatically find peak indices and locations
    peak_indices, properties = find_peaks(y_smooth, prominence=prominence, distance=distance)
    peak_centers = x[peak_indices]
    peak_heights = y[peak_indices]

    print(f"Auto-detected {len(peak_centers)} peaks at positions (x): {np.round(peak_centers, 2)}")

    # 3. Dynamically construct lmfit Composite Model & Parameters
    composite_model = None
    params = lmfit.Parameters()

    for i, (center, height) in enumerate(zip(peak_centers, peak_heights)):
        prefix = f"peak{i+1}_"
        model_cls = getattr(lmfit.models, f"{model_type.capitalize()}Model")
        p_model = model_cls(prefix=prefix)

        # Initial parameter bounds based on detected centroids
        params.add(f"{prefix}center", value=center, min=center - 1.5, max=center + 1.5)
        params.add(f"{prefix}amplitude", value=height * 1.5, min=0.0)
        params.add(f"{prefix}sigma", value=0.4, min=0.05, max=2.0)

        composite_model = p_model if composite_model is None else (composite_model + p_model)

    # 4. Perform non-linear fit
    if composite_model is not None:
        result = composite_model.fit(y, params, x=x)
        return result, peak_centers
    return None, []
```

---

## 4. XANES & Spectroscopy Considerations

| Feature | Behavior with Auto-Detection | Recommendation |
| :--- | :--- | :--- |
| **XRF / Raman / XRD / XPS** | Distinct, sharp local maxima | Works exceptionally well out-of-the-box. |
| **XANES / NEXAFS** | Strong overlapping white-lines, edge steps (`erf`/`atan`), & post-edge resonances | Use `prominence` thresholding and include step background models (`StepModel`) to prevent false peak detection near edge jumps. |

---

## 5. Integrating with `peak_fitting_utils.py`

You can pass auto-detected peak centers directly into the existing `peak_configs` list in `peak_fitting_utils.py`:

```python
from scipy.signal import find_peaks
from peak_fitting_utils import build_xanes_composite_model, create_initial_parameters, fit_single_spectrum

# Detect peak centers automatically
peaks, _ = find_peaks(intensity_arr, prominence=0.05, distance=5)
auto_centers = energy_arr[peaks]

# Dynamically construct peak_configs
peak_configs = [
    {
        'prefix': f'peak{i+1}_',
        'type': 'gaussian',
        'center': {'value': c, 'min': c - 1.0, 'max': c + 1.0},
        'sigma':  {'value': 0.4, 'min': 0.1, 'max': 1.5},
        'amplitude': {'value': 1.0, 'min': 0.0, 'max': 100.0}
    }
    for i, c in enumerate(auto_centers)
]

# Fit spectrum using peak_fitting_utils
model = build_xanes_composite_model(peak_configs)
params = create_initial_parameters(peak_configs)
result = fit_single_spectrum(energy_arr, intensity_arr, model, params)
```
