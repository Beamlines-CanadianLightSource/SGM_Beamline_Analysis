# Spatial Deconvolution Guide for SGM Synchrotron Mapping Data

This guide describes how to apply deconvolution techniques to clean up spatial maps generated from X-ray fluorescence (XRF) and X-ray absorption near edge structure (XANES) stacks at the SGM beamline.

Deconvolution recovers the true spatial features of your sample by mathematically reversing the blurring caused by the X-ray beam spot size and stage scanning dynamics.

---

## 1. Why Deconvolve?

In synchrotron micro-probe and nano-probe mapping, spatial resolution is fundamentally limited by:
1. **The Finite Beam Spot size:** The beam profile acts as a spatial low-pass filter (Point Spread Function, or PSF) that blurs details smaller than the spot diameter.
2. **Fly-Scanning Motion Blur:** When the sample is continuously scanned while the detector integrates, the beam spot effectively smears along the axis of movement (typically horizontal, $X$-axis).

```
   [True Sharp Map]  --->  Convolved with PSF (Beam Spot + Motion)  --->  [Blurred Observed Map]
          ^                                                                        |
          |_______________________ [ Deconvolution ] <_____________________________|
```

---

## 2. Modeling the Point Spread Function (PSF)

To deconvolve an image, you must define or estimate a PSF. In synchrotron setups, the PSF is commonly modeled as a **2D Gaussian**:

$$\text{PSF}(x, y) = A \exp\left( -\left( \frac{x^2}{2\sigma_x^2} + \frac{y^2}{2\sigma_y^2} \right) \right)$$

Where the standard deviation $\sigma$ is calculated from the Full Width at Half Maximum (FWHM) of the beam focus:
$$\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}} \approx \frac{\text{FWHM}}{2.355}$$

### Converting Physical Dimensions to Pixel Units
If your beam spot FWHM is $10\,\mu\text{m} \times 10\,\mu\text{m}$ and your interpolated map pixel resolution is $2\,\mu\text{m}/\text{pixel}$, the FWHM in pixel units is:
$$\text{FWHM}_{\text{pixels}} = \frac{10\,\mu\text{m}}{2\,\mu\text{m}/\text{pixel}} = 5\,\text{pixels}$$
$$\sigma_{\text{pixels}} = \frac{5}{2.355} \approx 2.12\,\text{pixels}$$

---

## 3. Recommended Algorithms

### Richardson-Lucy (RL) Deconvolution
Richardson-Lucy is the **gold standard** for synchrotron X-ray maps.
* **Poisson Noise Assumption:** It was specifically designed for signal-dependent Poisson noise (photon counting), which fits XRF and XANES counts perfectly.
* **Non-Negativity Constraint:** Through iterative multiplication, the pixel intensities can never become negative, preventing physically impossible negative counts.
* **Iterative Process:** Re-blurs the estimated image and compares it to the original, adjusting the estimate at each step.

### Wiener Filter
A fast, non-iterative frequency-domain deconvolution.
* **FFT-based:** Extremely fast (one-step calculation).
* **Gaussian Noise Assumption:** It assumes additive white noise rather than Poisson noise.
* **Artifacts:** Prone to "ringing" (Gibbs phenomenon) around sharp edges and can produce negative values.

---

## 4. Code Implementations

Below are two implementations. You must apply these algorithms **after** gridding/interpolating your scattered data (e.g., after the triangulation steps in [alignment_utils.py](file:///c:/Users/dynesj/PycharmProjects/PythonProject/Gemini_CLI2/alignment_utils.py)).

### Option A: Pure NumPy / SciPy Implementation (No installation required)

This implementation uses only standard library components already available in your virtual environment.

```python
import numpy as np
from scipy.signal import convolve2d

def generate_gaussian_psf(fwhm_x, fwhm_y, shape=(15, 15)):
    """
    Generates a normalized 2D Gaussian PSF.
    
    Parameters:
    - fwhm_x, fwhm_y: FWHM of the beam spot in pixel units.
    - shape: Size of the PSF kernel (should be odd, typically ~3x to 5x the FWHM).
    """
    sigma_x = fwhm_x / 2.355
    sigma_y = fwhm_y / 2.355
    
    y, x = np.mgrid[-shape[0]//2 + 1 : shape[0]//2 + 1,
                    -shape[1]//2 + 1 : shape[1]//2 + 1]
    
    psf = np.exp(-((x**2)/(2*sigma_x**2) + (y**2)/(2*sigma_y**2)))
    return psf / np.sum(psf)

def richardson_lucy_deconv(image, psf, num_iter=30, eps=1e-12):
    """
    Richardson-Lucy deconvolution of a 2D image.
    
    Parameters:
    - image: 2D numpy array (gridded XRF/XANES intensity map).
    - psf: 2D numpy array representing the PSF.
    - num_iter: Number of iterations (typically 15 to 50).
    - eps: Small constant to avoid division by zero.
    """
    # Enforce float type and non-negativity
    im_deconv = np.copy(image).astype(float)
    im_deconv[im_deconv < 0] = 0.0
    
    # The mirror (adjoint) of the PSF is needed for back-projection
    psf_mirror = np.flip(psf)
    
    for i in range(num_iter):
        # 1. Re-blur current estimate (symmetric boundary handles edges well)
        blur = convolve2d(im_deconv, psf, mode='same', boundary='symm')
        
        # 2. Calculate deviation ratio between raw image and re-blurred estimate
        ratio = image / (blur + eps)
        
        # 3. Project the deviation back by convolving with the flipped PSF
        correction = convolve2d(ratio, psf_mirror, mode='same', boundary='symm')
        
        # 4. Update estimate
        im_deconv *= correction
        im_deconv[im_deconv < 0] = 0.0
        
    return im_deconv
```

### Option B: Scikit-Image Restoration Module (Requires Installation)

If you install `scikit-image` (`pip install scikit-image`), you can use its optimized routines:

```python
from skimage import restoration

# Richardson-Lucy Deconvolution
deconvolved_rl = restoration.richardson_lucy(image, psf, num_iter=30)

# Unsupervised Wiener Filter (automatically estimates the noise parameters)
deconvolved_wiener, _ = restoration.unsupervised_wiener(image, psf)
```

---

## 5. Step-by-Step Example: Deblurring a Grid Map

Here is how you can load an interpolated grid map, run the Richardson-Lucy algorithm, and plot a side-by-side comparison:

```python
import numpy as np
import matplotlib.pyplot as plt
from alignment_utils import grid_interpolate_map

# 1. Assume you have your scattered coordinates (x, y) and counts (z)
# Interpolate onto a regular grid
grid_x, grid_y, grid_z = grid_interpolate_map(x_coords, y_coords, intensity_values, resolution=150)

# Fill any NaNs resulting from interpolation with 0
grid_z = np.nan_to_num(grid_z, nan=0.0)

# 2. Define the PSF
# Assume beam FWHM is 4.5 pixels wide (isotropic beam spot)
psf = generate_gaussian_psf(fwhm_x=4.5, fwhm_y=4.5, shape=(19, 19))

# 3. Run Richardson-Lucy Deconvolution (e.g., 30 iterations)
clean_map = richardson_lucy_deconv(grid_z, psf, num_iter=30)

# 4. Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(grid_z, origin='lower', cmap='viridis')
axes[0].set_title("Original Blurred Map")
axes[0].axis('off')

axes[1].imshow(clean_map, origin='lower', cmap='viridis')
axes[1].set_title("Deconvolved Map (Richardson-Lucy)")
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

---

## 6. Best Practices & Tips

* **Edge Ringing:** If your map has sharp transitions at the outer border, the deconvolution can cause ringing. Using `boundary='symm'` in the 2D convolutions helps mitigate this, but trimming off a 2-to-3 pixel border post-deconvolution is a robust practice.
* **Stop Iterations Early:** Richardson-Lucy will eventually start deconvolving random background noise if run too long, resulting in a speckled "noise-grain" artifact. Keep iterations between **15 and 40**.
* **Zero Counts and NaN Prevention:** In raw synchrotron maps, exact zero values are common (especially outside the sample boundary or in trimmed edges). Traditional Richardson-Lucy algorithms (including scikit-image's default `richardson_lucy`) will divide by zero and generate `NaN` values, which quickly propagate through convolution to blank out the entire map. To fix this, always use a custom deconvolution implementation that adds a small epsilon ($10^{-12}$) to the blur denominator, clips negative intensities to zero, and normalizes the input range to `[0.0, 1.0]` before deblurring (scaling it back afterward).
* **UI Slider Responsiveness (Lag Prevention):** When hooking deconvolution parameters up to GUI sliders (like in Jupyter Notebooks), set `continuous_update=False` on the sliders (e.g. `FloatSlider` or `IntSlider`). This ensures deconvolution runs only when you *release* the slider handle, preventing the UI from lagging or freezing due to continuous CPU calculation queues during dragging.
* **Jupyter IDE Setup (VS Code and Virtual Environments):** Running interactive notebooks directly in VS Code linked to the project's virtual environment (`.venv`) is highly recommended over standalone browser-based Jupyter instances. This ensures all compiled C-extensions (like those in `scipy` and `scikit-image`) load with matching DLL dependencies without system conflicts. Make sure to select `.venv/Scripts/python` as your active Jupyter kernel in VS Code.
