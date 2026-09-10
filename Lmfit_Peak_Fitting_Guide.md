# `lmfit` Peak Fitting & Fityk-Style Deconvolution Guide

This guide describes how to perform multi-component curve fitting and spectral deconvolution using `lmfit`, mirroring the methodology established in `Hongjie_Combined_Jay.ipynb` (cells 12–23) and incorporating key workflows from **Fityk**.

---

## 1. Overview & Methodology

In X-ray Absorption Spectroscopy (XANES / NEXAFS), X-ray Fluorescence (XRF), XPS, and Raman spectroscopy, experimental spectra consist of overlapping peak transitions superimposed on a step or baseline background.

To extract quantitative peak areas, height intensities, and energy shifts, spectra must be deconvolved into individual mathematical components:

$$\text{Spectrum}(E) = \text{StepBackground}(E) + \sum_{k=1}^{N} \text{Peak}_k(E)$$

```
  Intensity
     ^
     |         /\         Total Best Fit (Step + Peaks)
     |        /  \   /\
     |  _    /    \_/  \     <--- Peak Deconvolution
     | / \__/            \___
     |________________________\___  Step / Edge Background
     +-----------------------------> Energy (eV)
```

---

## 2. Fityk Parallels in `lmfit`

**Fityk** is a widely used standalone desktop program for non-linear curve fitting. The Python package `lmfit` provides direct equivalents for all of Fityk's core capabilities:

| Fityk Capability | `lmfit` Equivalent | Function in `peak_fitting_utils.py` |
| :--- | :--- | :--- |
| **Edge / Baseline** | `StepModel(form='erf')`, `LinearModel()` | `build_xanes_composite_model()` |
| **Peak Profiles** | `GaussianModel`, `LorentzianModel`, `VoigtModel` | `build_xanes_composite_model()` |
| **Parameter Bounds** | `min`, `max`, initial `value` | `create_initial_parameters()` |
| **Parameter Binding** | `expr='peak1_sigma'` (tying parameters) | `create_initial_parameters()` |
| **Batch Spectrum Fitting** | Iterative loop over spectral stacks | `fit_spectrum_stack()` |
| **Two-Pass Refinement** | Freeze centers/widths (`vary=False`), fit amplitudes | `refine_two_pass_stack_fit()` |
| **Residual Analysis** | Data minus Fit comparison plot | `plot_fit_deconvolution()` |

---

## 3. Basic Workflow & Code Examples

### Step 1: Define Model & Parameter Configurations

```python
from peak_fitting_utils import (
    build_xanes_composite_model,
    create_initial_parameters,
    fit_single_spectrum,
    plot_fit_deconvolution,
    extract_spectrum_from_sgm_data
)
from analyze_sgm_bsky_data import analyze_sgm_bsky_data

# 1. Load scan stack/map using analyze_sgm_bsky_data
sgm_data = analyze_sgm_bsky_data("path/to/stack.h5")

# 2. Extract energy_arr and intensity_arr (average TEY / MCC1 spectrum across the map)
energy_arr, intensity_arr = extract_spectrum_from_sgm_data(sgm_data, channel_idx=0, mode='mean')

# 3. Step background configuration (e.g. Carbon K-edge jump at 288 eV)
step_config = {
    'prefix': 'step_',
    'form': 'erf',
    'center': {'value': 288.0, 'min': 287.5, 'max': 288.5},
    'sigma':  {'value': 2.0,   'min': 1.0,   'max': 5.0},
    'amplitude': {'value': 1.0, 'min': 0.0, 'max': 50.0}
}

# 4. Peak configurations
peak_configs = [
    {
        'prefix': 'peak1_', 'type': 'gaussian',
        'center':    {'value': 284.1, 'min': 284.0, 'max': 284.3},
        'sigma':     {'value': 0.40,  'min': 0.15,  'max': 0.60},
        'amplitude': {'value': 1.0,   'min': 0.0,   'max': 100.0}
    },
    {
        'prefix': 'peak2_', 'type': 'gaussian',
        'center':    {'value': 285.75, 'min': 285.65, 'max': 285.85},
        'sigma':     {'value': 0.40,   'min': 0.15,   'max': 0.60},
        'amplitude': {'value': 1.0,    'min': 0.0,    'max': 100.0}
    },
    {
        'prefix': 'peak3_', 'type': 'gaussian',
        'center':    {'value': 286.5, 'min': 286.4, 'max': 286.6},
        'sigma':     {'value': 0.40,  'min': 0.15,  'max': 0.60},
        'amplitude': {'value': 1.0,   'min': 0.0,   'max': 100.0}
    }
]

# 5. Construct Composite Model and Parameters
model = build_xanes_composite_model(peak_configs, step_config=step_config)
params = create_initial_parameters(peak_configs, step_config=step_config)
```

---

### Step 2: Fit a Single Spectrum

```python
# 1. Fit single spectrum extracted from sgm_data
fit_out = fit_single_spectrum(energy_arr, intensity_arr, model, params)

# 2. Print detailed lmfit fit report
print(fit_out.fit_report())

# 3. Plot deconvolution and residuals
fig, axes = plot_fit_deconvolution(
    energy_arr, intensity_arr, fit_out,
    title="Carbon K-Edge Single Spectrum Fit",
    xlim=(280, 305)
)
```

---

### Step 3: Fityk Two-Pass Refinement across Spectrum Stacks

When fitting noisy dataset stacks or spatial maps:
1. **Pass 1**: Fit selected high-SNR spectra to establish average peak positions and widths.
2. **Pass 2**: Freeze peak shapes (`vary=False`) and fit only amplitudes across all spectra.

```python
from peak_fitting_utils import refine_two_pass_stack_fit

# energy_arr: 1D array of energy values (n_energy_points,)
# y_stack shape: (n_spectra, n_energy_points)
pass1_res, pass2_res, summary_df, frozen_params = refine_two_pass_stack_fit(
    energy_arr,
    y_stack,
    model,
    params,
    select_indices=[0, 1, 2, 5]  # Index of representative high SNR spectra
)

# summary_df contains peak areas, heights, and chi-squared for every spectrum
summary_df.to_csv("peak_fitting_summary.csv", index=False)
```

---

## 4. Advanced Features

### Parameter Binding / Constraints
To force `peak2` to always share the exact same width ($\sigma$) as `peak1`:

```python
params['peak2_sigma'].expr = 'peak1_sigma'
```

To maintain a fixed relative energy spacing (e.g. Peak 2 is always exactly $1.65\,\text{eV}$ above Peak 1):

```python
params['peak2_center'].expr = 'peak1_center + 1.65'
```

### Supported Model Line Shapes
- `GaussianModel`: Standard symmetric Gaussian peak ($e^{-(x-\mu)^2 / 2\sigma^2}$).
- `LorentzianModel`: Cauchy-Lorentz shape with heavy tail broadening.
- `VoigtModel`: Convolution of Gaussian (instrumental) and Lorentzian (core-hole lifetime) broadening.
- `StepModel`: Error function (`erf`), arctan (`atan`), or logistic edge jump.
- `LinearModel` / `PolynomialModel`: Smooth slope baseline removal.
