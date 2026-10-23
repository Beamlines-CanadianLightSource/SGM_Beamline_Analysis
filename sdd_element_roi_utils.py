"""
sdd_element_roi_utils.py

Automated Element Region of Interest (ROI) Selection and Gaussian Peak Fitting
for SGM Beamline SDD XRF Stacks and Maps.
"""

import os
import re
import numpy as np
from scipy.optimize import curve_fit
import sdd_calibration_utils as sdd_calib

# Comprehensive reference emission lines (in eV)
# Covers K-alpha, L-alpha, and M-alpha lines commonly analyzed at SGM / CLS.
ELEMENT_EMISSION_LINES = {
    # --- K-Edges (K-alpha emission in eV) ---
    ("B", "K"): 183.3,
    ("C", "K"): 277.0,
    ("N", "K"): 392.4,
    ("O", "K"): 524.9,
    ("F", "K"): 677.0,
    ("NE", "K"): 848.6,
    ("NA", "K"): 1041.0,
    ("MG", "K"): 1253.6,
    ("AL", "K"): 1486.7,
    ("SI", "K"): 1739.8,
    ("P", "K"): 2013.7,
    ("S", "K"): 2307.8,
    ("CL", "K"): 2622.4,
    ("AR", "K"): 2957.7,
    ("K", "K"): 3313.8,
    ("CA", "K"): 3691.7,
    ("SC", "K"): 4090.6,
    ("TI", "K"): 4510.8,
    ("V", "K"): 4952.2,
    ("CR", "K"): 5414.7,
    ("MN", "K"): 5898.8,
    ("FE", "K"): 6403.8,
    ("CO", "K"): 6930.3,
    ("NI", "K"): 7478.2,
    ("CU", "K"): 8046.3,
    ("ZN", "K"): 8638.9,

    # --- L-Edges (L-alpha emission in eV) ---
    ("K", "L"): 294.6,
    ("CA", "L"): 346.4,
    ("SC", "L"): 398.8,
    ("TI", "L"): 452.2,
    ("V", "L"): 511.3,
    ("CR", "L"): 572.8,
    ("MN", "L"): 637.4,
    ("FE", "L"): 705.0,
    ("CO", "L"): 776.2,
    ("NI", "L"): 851.5,
    ("CU", "L"): 929.7,
    ("ZN", "L"): 1011.7,
    ("GA", "L"): 1097.9,
    ("GE", "L"): 1188.0,
    ("AS", "L"): 1282.0,
    ("SE", "L"): 1379.1,
    ("BR", "L"): 1480.4,
    ("RB", "L"): 1694.1,
    ("SR", "L"): 1806.5,
    ("Y", "L"): 1922.6,
    ("ZR", "L"): 2042.4,
    ("NB", "L"): 2165.9,
    ("MO", "L"): 2293.2,
    ("RU", "L"): 2558.5,
    ("RH", "L"): 2696.8,
    ("PD", "L"): 2838.6,
    ("AG", "L"): 2984.3,
    ("CD", "L"): 3133.7,
    ("IN", "L"): 3286.9,
    ("SN", "L"): 3444.0,

    # --- M-Edges (M-alpha emission in eV) ---
    ("MO", "M"): 229.3,
    ("RU", "M"): 280.0,
    ("BA", "M"): 780.5,
    ("LA", "M"): 833.3,
    ("CE", "M"): 883.3,
    ("PR", "M"): 928.8,
    ("ND", "M"): 977.7,
    ("SM", "M"): 1080.9,
    ("EU", "M"): 1130.9,
    ("GD", "M"): 1185.2,
    ("TB", "M"): 1241.1,
    ("DY", "M"): 1292.8,
    ("HO", "M"): 1351.1,
    ("ER", "M"): 1405.7,
    ("TM", "M"): 1462.3,
    ("YB", "M"): 1521.3,
    ("LU", "M"): 1581.2,
    ("TA", "M"): 1709.6,
    ("W", "M"): 1775.4,
    ("PT", "M"): 2050.3,
    ("AU", "M"): 2122.9,
    ("PB", "M"): 2345.5,
    ("BI", "M"): 2422.6,
    ("U", "M"): 3171.0,
}

# Default edges for elements when edge is not explicitly specified in filename
DEFAULT_ELEMENT_EDGE = {
    "B": "K", "C": "K", "N": "K", "O": "K", "F": "K", "NE": "K",
    "NA": "K", "MG": "K", "AL": "K", "SI": "K", "P": "K", "S": "K", "CL": "K", "AR": "K",
    "CA": "L", "SC": "L", "TI": "L", "V": "L", "CR": "L", "MN": "L", "FE": "L",
    "CO": "L", "NI": "L", "CU": "L", "ZN": "L", "GA": "L", "GE": "L", "AS": "L", "SE": "L",
    "MO": "M", "CE": "M", "ND": "M", "GD": "M", "LA": "M"
}


def extract_element_and_edge(name_or_path):
    """
    Parses element and absorption edge from a filename, scan name, sample name, or path.
    
    Examples:
        '141438_Ca_silicate_Ctape_si_k_fine_stack' -> ('Si', 'K')
        '032433_TiON_sheet_30min_o_k_fine_stack'  -> ('O', 'K')
        '083357_LMFP82_nocarbon_fe_l3l2_fine_stack' -> ('Fe', 'L')
        '061857_600_N_stack'                     -> ('N', 'K')
        '185933_S5_SiO_si_k_fine_stack'          -> ('Si', 'K')
        'Si'                                     -> ('Si', 'K')
        'Si-K'                                   -> ('Si', 'K')
    """
    if not name_or_path:
        return None, None
        
    text = str(name_or_path).strip()
    if os.path.exists(text) or ('/' in text or '\\' in text):
        # Extract filename and parent folder
        p_base = os.path.basename(text)
        p_parent = os.path.basename(os.path.dirname(os.path.abspath(text)))
        text = f"{p_parent}_{p_base}"
        
    text_clean = text.replace('-', '_').replace(' ', '_').lower()

    # 1. Exact match for input like 'si' or 'si_k' or 'si-k'
    m_simple = re.fullmatch(r'([a-z]{1,2})(?:_([klmno]\d*))?', text_clean)
    if m_simple:
        elem = m_simple.group(1).capitalize()
        edge = m_simple.group(2).upper() if m_simple.group(2) else DEFAULT_ELEMENT_EDGE.get(elem.upper(), 'K')
        if (elem.upper(), edge) in ELEMENT_EMISSION_LINES:
            return elem, edge

    # 2. Look for pattern: _<elem>_<edge>_ (e.g. _si_k_, _fe_l_, _fe_l3l2_, _o_k_, _mo_m_)
    edge_regex = r'(?:^|_)([a-z]{1,2})_([klmno]\d*(?:[klmno]\d*)*)(?:_|$)'
    matches = list(re.finditer(edge_regex, text_clean))
    if matches:
        # Check from right to left (closest to suffix like _fine_stack)
        for m in reversed(matches):
            elem = m.group(1).capitalize()
            edge_raw = m.group(2).upper()
            edge_primary = edge_raw[0] # Take 'L' from 'L3' or 'L3L2'
            if (elem.upper(), edge_primary) in ELEMENT_EMISSION_LINES:
                return elem, edge_primary
            if (elem.upper(), edge_raw) in ELEMENT_EMISSION_LINES:
                return elem, edge_raw

    # 3. Look for concatenated edge: _<elem><edge>_ (e.g. _sik_, _fel_, _ok_)
    edge_concat_regex = r'(?:^|_)([a-z]{1,2})([klmno]\d*)(?:_|$)'
    matches2 = list(re.finditer(edge_concat_regex, text_clean))
    if matches2:
        for m in reversed(matches2):
            elem = m.group(1).capitalize()
            edge_raw = m.group(2).upper()
            edge_primary = edge_raw[0]
            if (elem.upper(), edge_primary) in ELEMENT_EMISSION_LINES:
                return elem, edge_primary

    # 4. Fallback: single element symbol bounded by underscores (e.g. _600_n_stack -> N K-edge)
    for elem_sym, def_edge in DEFAULT_ELEMENT_EDGE.items():
        pattern = rf'(?:^|_){elem_sym.lower()}(?:_|$)'
        if re.search(pattern, text_clean):
            return elem_sym.capitalize(), def_edge

    return None, None


def get_theoretical_emission_energy(element, edge=None):
    """
    Returns theoretical emission energy (eV) for the given element and edge.
    """
    if not element:
        return None
    elem_up = str(element).strip().upper()
    if edge:
        edge_up = str(edge).strip().upper()[0] # e.g. L3 -> L
        val = ELEMENT_EMISSION_LINES.get((elem_up, edge_up))
        if val is not None:
            return float(val)
            
    # Try default edge for this element
    def_edge = DEFAULT_ELEMENT_EDGE.get(elem_up, 'K')
    val = ELEMENT_EMISSION_LINES.get((elem_up, def_edge))
    if val is not None:
        return float(val)
        
    # Search any edge matching this element
    for (el, ed), energy in ELEMENT_EMISSION_LINES.items():
        if el == elem_up:
            return float(energy)
            
    return None


def gaussian_with_baseline(x, y0, a, mu, sigma):
    """Gaussian peak function with constant baseline."""
    sig = np.maximum(sigma, 1e-4)
    return y0 + a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_gaussian_xrf_roi(x_energy_ev, y_counts, theoretical_energy_ev, 
                         search_half_window_ev=40.0, n_sigma=2.0):
    """
    Fits a Gaussian peak to an XRF spectrum near theoretical_energy_ev.
    Accommodates detector calibration offsets (typically +-5 eV).
    
    Parameters
    ----------
    x_energy_ev : np.ndarray
        Energy values in eV.
    y_counts : np.ndarray
        Spectral counts corresponding to x_energy_ev.
    theoretical_energy_ev : float
        Target theoretical emission line in eV.
    search_half_window_ev : float, optional
        Half-width of energy window to search around target (default 40 eV).
    n_sigma : float, optional
        ROI integration half-width multiplier in units of standard deviation (default 2.0).
        2.0 * sigma covers ~95.4% of Gaussian area.
        
    Returns
    -------
    dict with keys:
        'roi_min_ev': float
        'roi_max_ev': float
        'peak_center_ev': float
        'sigma_ev': float
        'fwhm_ev': float
        'amplitude': float
        'baseline': float
        'fit_success': bool
    """
    x = np.asarray(x_energy_ev, dtype=float)
    y = np.asarray(y_counts, dtype=float)
    
    # Restrict to search window
    w_min = theoretical_energy_ev - search_half_window_ev
    w_max = theoretical_energy_ev + search_half_window_ev
    mask = (x >= w_min) & (x <= w_max)
    
    sub_x = x[mask]
    sub_y = y[mask]
    
    # Fallback default if no points or zero counts
    default_fwhm = 90.0
    default_sigma = default_fwhm / 2.355
    fallback_res = {
        'roi_min_ev': round(theoretical_energy_ev - n_sigma * default_sigma, 1),
        'roi_max_ev': round(theoretical_energy_ev + n_sigma * default_sigma, 1),
        'peak_center_ev': round(theoretical_energy_ev, 1),
        'sigma_ev': round(default_sigma, 1),
        'fwhm_ev': round(default_fwhm, 1),
        'amplitude': 0.0,
        'baseline': 0.0,
        'fit_success': False,
    }
    
    if len(sub_x) < 5 or np.max(sub_y) <= 0:
        return fallback_res

    # Initial parameter guesses
    max_idx = np.argmax(sub_y)
    mu_guess = float(sub_x[max_idx])
    y0_guess = float(np.min(sub_y))
    a_guess = float(np.max(sub_y) - y0_guess)
    sigma_guess = 38.0 # Typical SDD detector resolution (~90 eV FWHM)

    # Allow peak center to shift up to +-18 eV from theoretical to accommodate calibration drifts
    bounds = (
        [0.0, 0.0, theoretical_energy_ev - 18.0, 8.0],
        [np.inf, np.inf, theoretical_energy_ev + 18.0, 120.0]
    )

    try:
        popt, _ = curve_fit(
            gaussian_with_baseline, 
            sub_x, sub_y, 
            p0=[y0_guess, a_guess, mu_guess, sigma_guess], 
            bounds=bounds, 
            maxfev=2500
        )
        y0_fit, a_fit, mu_fit, sigma_fit = popt
        fwhm_fit = 2.355 * sigma_fit
        
        roi_min = round(mu_fit - n_sigma * sigma_fit, 1)
        roi_max = round(mu_fit + n_sigma * sigma_fit, 1)
        
        return {
            'roi_min_ev': round(float(mu_fit - n_sigma * sigma_fit), 1),
            'roi_max_ev': round(float(mu_fit + n_sigma * sigma_fit), 1),
            'peak_center_ev': round(float(mu_fit), 2),
            'sigma_ev': round(float(sigma_fit), 2),
            'fwhm_ev': round(float(fwhm_fit), 2),
            'amplitude': round(float(a_fit), 2),
            'baseline': round(float(y0_fit), 2),
            'fit_success': True
        }
    except Exception:
        # Fallback using numerical centroid and variance if non-linear fit did not converge
        try:
            y_clean = np.maximum(sub_y - np.min(sub_y), 0)
            if np.sum(y_clean) > 0:
                mu_num = float(np.sum(sub_x * y_clean) / np.sum(y_clean))
                var_num = float(np.sum(((sub_x - mu_num) ** 2) * y_clean) / np.sum(y_clean))
                sigma_num = float(np.clip(np.sqrt(var_num), 12.0, 80.0))
                return {
                    'roi_min_ev': round(float(mu_num - n_sigma * sigma_num), 1),
                    'roi_max_ev': round(float(mu_num + n_sigma * sigma_num), 1),
                    'peak_center_ev': round(float(mu_num), 2),
                    'sigma_ev': round(float(sigma_num), 2),
                    'fwhm_ev': round(float(2.355 * sigma_num), 2),
                    'amplitude': float(np.max(y_clean)),
                    'baseline': float(np.min(sub_y)),
                    'fit_success': True
                }
        except Exception:
            pass

    return fallback_res


def auto_detect_and_fit_stack_roi(path_pack, representative_energy=None, 
                                  map_roi=None, element_override=None, 
                                  edge_override=None, n_sigma=2.0, verbose=True):
    """
    High-level entry point to automatically determine the XRF ROI from stack metadata.
    
    1. Extracts element & edge from scan/sample/path name (or override).
    2. Retrieves theoretical emission line energy.
    3. Loads SDD calibration and builds the calibrated XRF spectrum for active pixels.
    4. Fits a Gaussian peak.
    5. Returns (roi_min_ev, roi_max_ev) and per-detector channel boundaries.
    """
    # 1. Identify element and edge
    elem = element_override
    edge = edge_override
    if not elem:
        for key in ['scan_name', 'sample_name', 'h5_file_path', 'h5_dir']:
            val = path_pack.get(key)
            if val and val != 'N/A':
                e, ed = extract_element_and_edge(val)
                if e:
                    elem = e
                    edge = ed
                    break

    if not elem:
        if verbose:
            print("  [Auto Element ROI] Could not detect element from file/scan name.")
        return None

    # 2. Lookup theoretical energy
    th_energy = get_theoretical_emission_energy(elem, edge)
    if th_energy is None:
        if verbose:
            print(f"  [Auto Element ROI] No emission line found for element: {elem}-{edge}")
        return None

    # 3. Load SDD calibration
    calib_data = sdd_calib.load_calibration()
    if not calib_data:
        if verbose:
            print("  [Auto Element ROI] Warning: SDD calibration file not found. Auto-fitting requires calibrated SDDs.")
        return None

    detector_names = sorted(path_pack.get('sdd_files', {}).keys())
    if not detector_names:
        return None

    # Determine representative energy
    all_energies = np.array(sorted(path_pack.get('energies', [])))
    if representative_energy is None:
        if len(all_energies) > 0:
            representative_energy = all_energies[len(all_energies) // 2]
        else:
            return None

    # 4. Build master calibrated spectrum across all detectors for active pixels
    x_coords = path_pack.get('x', np.array([]))
    y_coords = path_pack.get('y', np.array([]))
    x_trim = path_pack.get('x_trim', 0.0)
    y_trim = path_pack.get('y_trim', 0.0)

    # Grid for combining calibrated spectra (1 eV resolution)
    e_grid = np.arange(max(0, th_energy - 100), th_energy + 100, 1.0)
    combined_counts = np.zeros_like(e_grid, dtype=float)
    detectors_used = 0

    for det in detector_names:
        p = path_pack['sdd_files'][det].get(representative_energy)
        if not p or not os.path.exists(p) or det not in calib_data:
            continue
        try:
            d1d = np.fromfile(p, dtype=np.uint32)
            num_s = min(len(d1d) // 256, x_coords.size)
            s2d = d1d[:num_s * 256].reshape((num_s, 256))
            roll_shift = path_pack.get('roll_shift', 0)
            if roll_shift != 0:
                s2d = np.roll(s2d, shift=roll_shift, axis=0)

            # Spatial mask: if map_roi is provided, use it; else full trimmed image
            cx = x_coords[:num_s]
            cy = y_coords[:num_s]
            xmin, xmax = np.min(cx), np.max(cx)
            ymin, ymax = np.min(cy), np.max(cy)
            mask = (cx >= xmin + x_trim) & (cx <= xmax - x_trim) & (cy >= ymin + y_trim) & (cy <= ymax - y_trim)
            if map_roi:
                x1, x2 = sorted(map_roi[0:2])
                y1, y2 = sorted(map_roi[2:4])
                mask &= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)

            if not np.any(mask):
                mask = slice(None)

            spec_sum = np.sum(s2d[mask], axis=0)
            gain = calib_data[det].get('gain', 1.0)
            offset = calib_data[det].get('offset', 0.0)
            e_axis_det = sdd_calib.channel_to_energy(np.arange(256), gain, offset)

            # Interpolate onto master grid
            interp_counts = np.interp(e_grid, e_axis_det, spec_sum, left=0.0, right=0.0)
            combined_counts += interp_counts
            detectors_used += 1
        except Exception:
            continue

    if detectors_used == 0 or np.max(combined_counts) <= 0:
        if verbose:
            print(f"  [Auto Element ROI] Insufficient data counts to fit emission peak for {elem}.")
        return None

    # 5. Fit Gaussian
    fit_res = fit_gaussian_xrf_roi(e_grid, combined_counts, th_energy, search_half_window_ev=40.0, n_sigma=n_sigma)
    roi_min = fit_res['roi_min_ev']
    roi_max = fit_res['roi_max_ev']

    if verbose:
        shift = fit_res['peak_center_ev'] - th_energy
        print(f"  [Auto Element ROI] Element: {elem} ({edge}-edge, theoretical: {th_energy:.1f} eV)")
        print(f"  [Auto Element ROI] Fitted Peak: {fit_res['peak_center_ev']:.1f} eV (Calibration offset: {shift:+.1f} eV)")
        print(f"  [Auto Element ROI] Resolution: FWHM = {fit_res['fwhm_ev']:.1f} eV (sigma = {fit_res['sigma_ev']:.1f} eV)")
        print(f"  [Auto Element ROI] Selected ROI (+/- {n_sigma:.1f} sigma): {roi_min:.1f} - {roi_max:.1f} eV")

    return {
        'element': elem,
        'edge': edge,
        'theoretical_ev': th_energy,
        'xrf_roi': (roi_min, roi_max),
        'fit_details': fit_res
    }
