import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
try:
    from tkinter import filedialog
except (ImportError, ModuleNotFoundError):
    filedialog = None
from scipy.optimize import curve_fit

from analyze_sgm_bsky_data import analyze_sgm_bsky_data
import sdd_calibration_utils as sdd_calib
import sdd_element_roi_utils as elem_utils
from alignment_utils import safe_filedialog_call, HAS_GUI


# -----------------------------------------------------------------------------
# Fitting Models & Helpers
# -----------------------------------------------------------------------------

def _double_gaussian(x, y0, m_slope, a1, mu1, sig1, a2, mu2, sig2):
    """Two Gaussians with a shared linear baseline."""
    sig1 = np.maximum(sig1, 1e-4)
    sig2 = np.maximum(sig2, 1e-4)
    return (y0 + m_slope * (x - mu1) +
            a1 * np.exp(-0.5 * ((x - mu1) / sig1) ** 2) +
            a2 * np.exp(-0.5 * ((x - mu2) / sig2) ** 2))


def _single_gaussian_with_slope(x, y0, m_slope, a, mu, sig):
    """Single Gaussian with linear baseline."""
    sig = np.maximum(sig, 1e-4)
    return y0 + m_slope * (x - mu) + a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def fit_single_peak(x_axis, y_counts, center_target_ev, window_half_width=45.0):
    """
    Fits a single Gaussian peak around center_target_ev.
    Falls back to a 3-point parabolic interpolation if curve_fit fails.
    """
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_counts, dtype=float)
    mask = (x >= center_target_ev - window_half_width) & (x <= center_target_ev + window_half_width)
    sub_x, sub_y = x[mask], y[mask]

    if len(sub_x) < 4 or np.max(sub_y) <= 0:
        return {'center': np.nan, 'amplitude': 0.0, 'fwhm': np.nan, 'baseline': 0.0, 'success': False}

    idx_max = np.argmax(sub_y)
    peak_x = float(sub_x[idx_max])
    peak_y = float(sub_y[idx_max])
    base_guess = float(np.min(sub_y))
    amp_guess = max(float(peak_y - base_guess), 1.0)
    sig_guess = 35.0  # Nominal ~82 eV FWHM for SDD

    try:
        bounds = (
            [0.0, -10.0, 0.0, center_target_ev - window_half_width, 6.0],
            [np.inf, 10.0, np.inf, center_target_ev + window_half_width, 80.0]
        )
        popt, _ = curve_fit(
            _single_gaussian_with_slope,
            sub_x, sub_y,
            p0=[base_guess, 0.0, amp_guess, peak_x, sig_guess],
            bounds=bounds,
            maxfev=1500
        )
        return {
            'center': float(popt[3]),
            'amplitude': float(popt[2]),
            'fwhm': float(2.355 * popt[4]),
            'baseline': float(popt[0]),
            'success': True
        }
    except Exception:
        # 3-point parabolic centroid interpolation
        if 0 < idx_max < len(sub_y) - 1:
            y1, y2, y3 = sub_y[idx_max - 1], sub_y[idx_max], sub_y[idx_max + 1]
            denom = 2 * (y1 - 2 * y2 + y3)
            delta = (y1 - y3) / denom if denom != 0 else 0.0
            dx = sub_x[1] - sub_x[0] if len(sub_x) > 1 else 1.0
            return {
                'center': float(sub_x[idx_max] + delta * dx),
                'amplitude': amp_guess,
                'fwhm': 82.0,
                'baseline': base_guess,
                'success': True
            }
        return {
            'center': peak_x,
            'amplitude': amp_guess,
            'fwhm': 82.0,
            'baseline': base_guess,
            'success': False
        }


def fit_spectrum_peaks(x_axis, y_counts, en_inc, en_fluor=None, track_peak='both'):
    """
    Deconvolves and tracks both the Elastic Scatter peak (near en_inc)
    and the Characteristic Fluorescence peak (near en_fluor).
    Uses a joint double-Gaussian fit when peaks are close and overlapping (<160 eV).
    """
    results = {'scatter': None, 'fluor': None}

    # 1. Joint Double Gaussian if both requested and nearby
    if track_peak == 'both' and en_fluor is not None:
        sep = abs(en_inc - en_fluor)
        if 15.0 <= sep <= 160.0:
            w_min = min(en_fluor, en_inc) - 45.0
            w_max = max(en_fluor, en_inc) + 45.0
            mask = (x_axis >= w_min) & (x_axis <= w_max)
            sub_x, sub_y = x_axis[mask], y_counts[mask]

            if len(sub_x) >= 8 and np.max(sub_y) > 0:
                base_guess = float(np.min(sub_y))
                idx_f = np.argmin(np.abs(sub_x - en_fluor))
                idx_s = np.argmin(np.abs(sub_x - en_inc))
                a_f_guess = max(float(sub_y[idx_f] - base_guess), 1.0)
                a_s_guess = max(float(sub_y[idx_s] - base_guess), 1.0)

                p0 = [base_guess, 0.0, a_f_guess, en_fluor, 35.0, a_s_guess, en_inc, 32.0]
                bounds = (
                    [0.0, -10.0, 0.0, en_fluor - 25.0, 6.0, 0.0, en_inc - 25.0, 6.0],
                    [np.inf, 10.0, np.inf, en_fluor + 25.0, 80.0, np.inf, en_inc + 25.0, 80.0]
                )
                try:
                    popt, _ = curve_fit(_double_gaussian, sub_x, sub_y, p0=p0, bounds=bounds, maxfev=2500)
                    results['fluor'] = {
                        'center': float(popt[3]),
                        'amplitude': float(popt[2]),
                        'fwhm': float(2.355 * popt[4]),
                        'success': True
                    }
                    results['scatter'] = {
                        'center': float(popt[6]),
                        'amplitude': float(popt[5]),
                        'fwhm': float(2.355 * popt[7]),
                        'success': True
                    }
                    return results
                except Exception:
                    pass

    # 2. Independent single-peak fits (if joint fit not applicable or failed)
    if track_peak in ('scatter', 'both'):
        w_half = 45.0
        if en_fluor is not None:
            sep = abs(en_inc - en_fluor)
            if sep < 90.0:
                w_half = max(10.0, 0.45 * sep)
        results['scatter'] = fit_single_peak(x_axis, y_counts, en_inc, window_half_width=w_half)

    if track_peak in ('fluorescence', 'both') and en_fluor is not None:
        w_half = 45.0
        sep = abs(en_inc - en_fluor)
        if sep < 90.0:
            w_half = max(10.0, 0.45 * sep)
        results['fluor'] = fit_single_peak(x_axis, y_counts, en_fluor, window_half_width=w_half)

    return results


def auto_detect_element_and_edge(data_pack=None, element=None, edge=None):
    """
    Identifies element, absorption edge, and theoretical emission energy (eV)
    from explicit inputs, scan/sample names, or incident beam energy.
    """
    if element is not None:
        elem = element.capitalize()
        ed = (edge or elem_utils.DEFAULT_ELEMENT_EDGE.get(elem.upper(), 'K')).upper()
        theor = elem_utils.get_theoretical_emission_energy(elem, ed)
        return elem, ed, theor

    if data_pack is None:
        return None, None, None

    # 1. Standard scan name / sample name parsing
    scan_name = data_pack.get('scan_name', '')
    sample_name = data_pack.get('sample_name', '') or data_pack.get('sample', '')
    h5_path = data_pack.get('h5_file_path', '') or data_pack.get('h5_dir', '')

    for candidate in [scan_name, sample_name, h5_path]:
        if candidate:
            elem, ed = elem_utils.extract_element_and_edge(candidate)
            if elem:
                theor = elem_utils.get_theoretical_emission_energy(elem, ed)
                return elem, ed, theor

    # 2. Match element symbols inside compound names (e.g. TyMg807 -> Mg)
    combined_str = f"{scan_name}_{sample_name}_{os.path.basename(str(h5_path))}"
    found_symbols = re.findall(r'([A-Z][a-z]?)', combined_str)
    energies = data_pack.get('energies', [])
    avg_e = float(np.mean(energies)) if len(energies) > 0 else 0.0

    best_elem, best_edge, best_theor = None, None, None
    min_dist = float('inf')

    for sym in found_symbols:
        sym_up = sym.upper()
        if sym_up in elem_utils.DEFAULT_ELEMENT_EDGE:
            ed = elem_utils.DEFAULT_ELEMENT_EDGE[sym_up]
            theor = elem_utils.get_theoretical_emission_energy(sym, ed)
            if theor is not None and avg_e > 0:
                dist = abs(avg_e - theor)
                if dist < min_dist:
                    min_dist = dist
                    best_elem = sym.capitalize()
                    best_edge = ed
                    best_theor = theor

    if best_elem and min_dist < 400.0:
        return best_elem, best_edge, best_theor

    # 3. Energy proximity fallback
    if avg_e > 0:
        for (el, ed), em_energy in elem_utils.ELEMENT_EMISSION_LINES.items():
            if 0 < (avg_e - em_energy) < 160.0:
                return el.capitalize(), ed, em_energy

    return None, None, None


# -----------------------------------------------------------------------------
# Main Visualization Function
# -----------------------------------------------------------------------------

def plot_selected_emission_spectra(
    data_pack=None,
    selected_energies=None,
    detector='sdd1',
    track_peak='both',
    element=None,
    edge=None,
    plot_mode='both',
    use_energy_calib=True,
    apply_waterfall=False,
    waterfall_offset=None,
    show_markers=True,
    map_roi=None,
    figsize=None,
    save_plot=False,
    save_csv=False,
    output_dir=None,
    file_format='png',
    dpi=300
):
    """
    Plots overlaid 1D Emission / XRD Spectra and derives peak shifts as a function
    of incident energy (both counts/intensity vs emission energy AND peak position vs incident energy).

    Features:
      - Tracks the Elastic Scatter Peak (shifting with E_inc: E_em ~ E_inc, slope ~ 1.0).
      - Tracks Characteristic Fluorescence Peak (e.g. Mg-K at 1253.6 eV, slope ~ 0.0).
      - Supports detector='all' to automatically plot all available detectors ('sdd1'..'sdd4').
      - Dual-panel side-by-side display ('both'), spectra overlay ('spectra'), or dispersion curve ('dispersion').
      - Default 30% smaller compact layout (10.0 x 4.2 in) to fit all screen sizes without clipping.
      - Joint two-component deconvolution when peaks overlap near absorption threshold.
      - Exports peak shift tables (center, FWHM, amplitude) + full spectra in CSV.

    Args:
        data_pack (dict, optional): Data dictionary returned by analyze_sgm_bsky_data().
        selected_energies (list or array, optional): List of incident energy values (eV) or indices.
                                                    Defaults to 5 evenly spaced energies across stack.
        detector (str or list): Detector name ('sdd1'..'sdd4', 'average', or 'all').
        track_peak (str or None): 'both' (default), 'scatter', 'fluorescence', or None.
        element (str, optional): Target element (e.g. 'Mg', 'Si'). Auto-detected if None.
        edge (str, optional): Absorption edge (e.g. 'K', 'L'). Auto-detected if None.
        plot_mode (str): 'both' (dual-panel side-by-side), 'spectra', or 'dispersion'.
        use_energy_calib (bool): If True, converts emission channels to calibrated eV.
        apply_waterfall (bool): If True, applies vertical offset between spectra.
        waterfall_offset (float, optional): Custom vertical offset.
        show_markers (bool): If True, displays markers on spectral lines.
        map_roi (list, optional): [x1, x2, y1, y2] spatial ROI for filtering.
        figsize (tuple, optional): Custom figure dimensions (width, height) in inches.
        save_plot (bool): If True, opens save dialog for image plot after display.
        save_csv (bool): If True, opens save dialog for spectra CSV after display.
        output_dir (str, optional): Target folder for save dialog.
        file_format (str): Default image format ('png', 'pdf', 'svg', 'jpg').
        dpi (int): Image resolution (DPI).

    Returns:
        list of str or None: List of saved file paths if saved, else None.
    """
    if data_pack is None:
        data_pack = analyze_sgm_bsky_data(verbose=True)

    if not data_pack or not data_pack.get('sdd_files'):
        print("Error: No valid SDD stack data found.", file=sys.stderr)
        return None

    all_energies = np.array(sorted(data_pack['energies']))
    if len(all_energies) == 0:
        print("Error: Stack contains no incident energies.", file=sys.stderr)
        return None

    # Handle energy selection
    if selected_energies is None:
        if len(all_energies) <= 5:
            target_energies = list(all_energies)
        else:
            idx_choices = np.linspace(0, len(all_energies) - 1, 5, dtype=int)
            target_energies = list(all_energies[idx_choices])
    else:
        target_energies = []
        for item in selected_energies:
            if isinstance(item, (int, np.integer)) and 0 <= item < len(all_energies):
                target_energies.append(all_energies[item])
            else:
                try:
                    val = float(item)
                    closest_idx = np.abs(all_energies - val).argmin()
                    target_energies.append(all_energies[closest_idx])
                except (ValueError, TypeError):
                    continue
        target_energies = sorted(list(dict.fromkeys(target_energies)))

    if not target_energies:
        print("Error: No valid incident energies selected.", file=sys.stderr)
        return None

    scan_title = data_pack.get('scan_name', 'Stack Scan')
    available_dets = sorted(data_pack['sdd_files'].keys())

    # Handle multi-detector 'all' or list of detectors
    det_str = str(detector).strip().lower()
    if det_str == 'all' or isinstance(detector, (list, tuple)):
        dets_to_process = available_dets if det_str == 'all' else list(detector)
        all_saved = []
        for d in dets_to_process:
            if d in available_dets:
                res = plot_selected_emission_spectra(
                    data_pack=data_pack,
                    selected_energies=selected_energies,
                    detector=d,
                    track_peak=track_peak,
                    element=element,
                    edge=edge,
                    plot_mode=plot_mode,
                    use_energy_calib=use_energy_calib,
                    apply_waterfall=apply_waterfall,
                    waterfall_offset=waterfall_offset,
                    show_markers=show_markers,
                    map_roi=map_roi,
                    figsize=figsize,
                    save_plot=save_plot,
                    save_csv=save_csv,
                    output_dir=output_dir,
                    file_format=file_format,
                    dpi=dpi
                )
                if res:
                    all_saved.extend(res)
            else:
                print(f"Warning: Detector '{d}' not found in stack. Available: {available_dets}")
        return all_saved if all_saved else None

    # Determine detectors to read for single or average
    if detector == 'average':
        target_dets = available_dets
    elif detector in available_dets:
        target_dets = [detector]
    else:
        print(f"Warning: Detector '{detector}' not found. Available: {available_dets}")
        target_dets = available_dets[:1]

    # Spatial mask setup
    x_raw = data_pack.get('x', np.array([]))
    y_raw = data_pack.get('y', np.array([]))
    spatial_mask = None
    if map_roi and x_raw.size > 0 and y_raw.size > 0:
        x1, x2 = sorted(map_roi[0:2])
        y1, y2 = sorted(map_roi[2:4])
        spatial_mask = (x_raw >= x1) & (x_raw <= x2) & (y_raw >= y1) & (y_raw <= y2)

    # Calibration setup
    calib_data = sdd_calib.load_calibration() if use_energy_calib else None
    ref_det = target_dets[0]
    gain = 1.0
    offset = 0.0
    if use_energy_calib:
        if calib_data and ref_det in calib_data:
            gain = calib_data[ref_det].get('gain', 1.0)
            offset = calib_data[ref_det].get('offset', 0.0)

        if gain != 1.0 or offset != 0.0:
            emission_axis = sdd_calib.channel_to_energy(np.arange(256), gain, offset)
        else:
            emission_axis = np.arange(256) * 10.0
        xlabel = "Emission Energy (eV)"
    else:
        emission_axis = np.arange(256)
        xlabel = "Emission Channel"

    # Identify element & theoretical emission line
    detected_elem, detected_edge, theoretical_fluor_ev = auto_detect_element_and_edge(
        data_pack=data_pack,
        element=element,
        edge=edge
    )
    if detected_elem and theoretical_fluor_ev is not None:
        print(f"  [Auto Element Detection] Target: {detected_elem} {detected_edge}-edge "
              f"(Theoretical Emission: {theoretical_fluor_ev:.1f} eV)")

    # Extract 1D spectra for selected energies
    spectra_data = []
    for en in target_energies:
        det_spectra = []
        for det_name in target_dets:
            p = data_pack['sdd_files'][det_name].get(en)
            if p and os.path.exists(p):
                d1d = np.fromfile(p, dtype=np.uint32)
                num_s = len(d1d) // 256
                if num_s > 0:
                    s2d = d1d[:num_s * 256].reshape((num_s, 256))
                    if spatial_mask is not None and spatial_mask.size >= num_s:
                        m = spatial_mask[:num_s]
                        spec = np.sum(s2d[m], axis=0) if np.any(m) else np.sum(s2d, axis=0)
                    else:
                        spec = np.sum(s2d, axis=0)
                    det_spectra.append(spec)

        if det_spectra:
            avg_spec = np.mean(det_spectra, axis=0)
            spectra_data.append({'energy': en, 'spectrum': avg_spec})

    if not spectra_data:
        print("Error: Could not extract spectral data for selected energies.", file=sys.stderr)
        return None

    # Perform Peak Tracking (if requested and calibrated in eV)
    if track_peak and use_energy_calib:
        for item in spectra_data:
            en_val = item['energy']
            raw_y = item['spectrum']
            fits = fit_spectrum_peaks(
                x_axis=emission_axis,
                y_counts=raw_y,
                en_inc=en_val,
                en_fluor=theoretical_fluor_ev,
                track_peak=track_peak
            )
            item['peak_fits'] = fits
    else:
        for item in spectra_data:
            item['peak_fits'] = {'scatter': None, 'fluor': None}

    # Calculate Waterfall offset
    if apply_waterfall:
        if waterfall_offset is None:
            max_amp = max(np.ptp(item['spectrum']) for item in spectra_data)
            offset_step = 0.5 * max_amp if max_amp > 0 else 1.0
        else:
            offset_step = waterfall_offset
    else:
        offset_step = 0.0

    # Colors
    n_specs = len(spectra_data)
    if n_specs <= 5:
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    else:
        palette = plt.cm.tab10(np.linspace(0, 1, n_specs))

    # Configure subplots based on plot_mode (default 30% smaller layout)
    if plot_mode == 'both' and track_peak:
        actual_figsize = figsize if figsize is not None else (10.0, 4.2)
        fig, (ax_spec, ax_disp) = plt.subplots(1, 2, figsize=actual_figsize)
        plt.subplots_adjust(bottom=0.20, top=0.88, left=0.08, right=0.96, wspace=0.28)
    elif plot_mode == 'dispersion' and track_peak:
        actual_figsize = figsize if figsize is not None else (6.2, 4.2)
        fig, ax_disp = plt.subplots(figsize=actual_figsize)
        plt.subplots_adjust(bottom=0.20, top=0.88, left=0.12, right=0.94)
        ax_spec = None
    else:
        actual_figsize = figsize if figsize is not None else (6.8, 4.2)
        fig, ax_spec = plt.subplots(figsize=actual_figsize)
        plt.subplots_adjust(bottom=0.20, top=0.88, left=0.12, right=0.94)
        ax_disp = None

    # -------------------------------------------------------------------------
    # Panel 1: Emission Spectra Overlay
    # -------------------------------------------------------------------------
    if ax_spec is not None:
        for idx, item in enumerate(spectra_data):
            en_val = item['energy']
            raw_y = item['spectrum']
            off_val = idx * offset_step
            y_plot = raw_y + off_val
            color = palette[idx % len(palette)]
            fmt = 'o-' if show_markers else '-'

            lbl = f"E_inc = {en_val:.2f} eV" + (f" (+{off_val:.1f})" if apply_waterfall and off_val > 0 else "")
            line, = ax_spec.plot(
                emission_axis, y_plot, fmt,
                color=color, label=lbl, alpha=0.85, linewidth=1.5, markersize=3
            )

            # Mark fitted peak centers on spectra curves
            pf = item.get('peak_fits', {})
            if pf.get('scatter') and pf['scatter']['success']:
                sc_c = pf['scatter']['center']
                sc_y = np.interp(sc_c, emission_axis, y_plot)
                ax_spec.plot(sc_c, sc_y, 'o', color=color, markeredgecolor='black',
                             markersize=6, zorder=5)

            if pf.get('fluor') and pf['fluor']['success']:
                fl_c = pf['fluor']['center']
                fl_y = np.interp(fl_c, emission_axis, y_plot)
                ax_spec.plot(fl_c, fl_y, 's', color=color, markeredgecolor='darkred',
                             markersize=6, zorder=5)

        ax_spec.set_xlabel(xlabel, fontsize=8.5, fontweight='semibold')
        ax_spec.set_ylabel('Counts / Intensity' + (' (Waterfall Offset)' if apply_waterfall else ''),
                           fontsize=8.5, fontweight='semibold')
        ax_spec.tick_params(labelsize=8)
        title_suffix = f" ({len(spectra_data)} Energies)\n{scan_title} ({detector})"
        ax_spec.set_title(f'Emission / XRD Spectra' + title_suffix, fontsize=9.5, fontweight='bold')
        ax_spec.grid(True, linestyle=':', alpha=0.6)
        ax_spec.legend(loc='best', fontsize=7)

    # -------------------------------------------------------------------------
    # Panel 2: Peak Position vs. Incident Energy (Dispersion Curve)
    # -------------------------------------------------------------------------
    if ax_disp is not None:
        inc_energies = np.array([item['energy'] for item in spectra_data])
        scat_centers = np.array([
            item['peak_fits']['scatter']['center']
            if (item['peak_fits'].get('scatter') and item['peak_fits']['scatter']['success'])
            else np.nan for item in spectra_data
        ])
        fluor_centers = np.array([
            item['peak_fits']['fluor']['center']
            if (item['peak_fits'].get('fluor') and item['peak_fits']['fluor']['success'])
            else np.nan for item in spectra_data
        ])

        # 1. Theoretical Elastic Scatter Reference Line (y = x)
        if track_peak in ('scatter', 'both'):
            e_min = np.min(inc_energies) - 15.0
            e_max = np.max(inc_energies) + 15.0
            ref_x = np.linspace(e_min, e_max, 100)
            ax_disp.plot(ref_x, ref_x, 'k--', alpha=0.5, linewidth=1.3,
                         label='Ideal Scatter (E_em = E_inc, slope=1.0)', zorder=1)

        # 2. Theoretical Fluorescence Reference Line
        if theoretical_fluor_ev is not None and track_peak in ('fluorescence', 'both'):
            elem_lbl = f"{detected_elem} {detected_edge}-edge" if detected_elem else "Fluorescence"
            ax_disp.axhline(theoretical_fluor_ev, color='red', linestyle=':', alpha=0.7, linewidth=1.4,
                            label=f'Theor. {elem_lbl} ({theoretical_fluor_ev:.1f} eV)', zorder=1)

        # 3. Plot Scatter Peak Centers & Calculate Measured Slope
        if track_peak in ('scatter', 'both'):
            valid_s = ~np.isnan(scat_centers)
            if np.sum(valid_s) >= 2:
                slope, intercept = np.polyfit(inc_energies[valid_s], scat_centers[valid_s], 1)
                lbl_sc = f'Scatter Peak (slope = {slope:.3f})'
            else:
                lbl_sc = 'Scatter Peak'
            if np.any(valid_s):
                ax_disp.plot(inc_energies[valid_s], scat_centers[valid_s], 'o-', color='#1f77b4',
                             linewidth=1.8, markersize=7, markeredgecolor='black',
                             label=lbl_sc, zorder=3)

        # 4. Plot Fluorescence Peak Centers
        if track_peak in ('fluorescence', 'both'):
            valid_f = ~np.isnan(fluor_centers)
            if np.any(valid_f):
                mean_f = np.mean(fluor_centers[valid_f])
                elem_lbl = f"{detected_elem} {detected_edge}" if detected_elem else "Fluor"
                lbl_fl = f'{elem_lbl} Peak (mean = {mean_f:.1f} eV)'
                ax_disp.plot(inc_energies[valid_f], fluor_centers[valid_f], 's-', color='#d62728',
                             linewidth=1.8, markersize=7, markeredgecolor='black',
                             label=lbl_fl, zorder=3)

        ax_disp.set_xlabel('Incident Energy (eV)', fontsize=8.5, fontweight='semibold')
        ax_disp.set_ylabel('Peak Position (Emission eV)', fontsize=8.5, fontweight='semibold')
        ax_disp.tick_params(labelsize=8)
        ax_disp.set_title(f'Peak Shift vs. Incident Energy\n{scan_title} ({detector})',
                          fontsize=9.5, fontweight='bold')
        ax_disp.grid(True, linestyle=':', alpha=0.6)
        ax_disp.legend(loc='best', fontsize=7)

    saved_files = []

    # -------------------------------------------------------------------------
    # Interactive Buttons
    # -------------------------------------------------------------------------
    if plot_mode == 'both' and track_peak:
        ax_save_btn = fig.add_axes([0.24, 0.02, 0.22, 0.06])
        ax_save_csv_btn = fig.add_axes([0.54, 0.02, 0.26, 0.06])
    else:
        ax_save_btn = fig.add_axes([0.20, 0.02, 0.26, 0.06])
        ax_save_csv_btn = fig.add_axes([0.52, 0.02, 0.28, 0.06])

    btn_save = Button(ax_save_btn, 'Save Plot', color='yellow', hovercolor='khaki')
    btn_save.label.set_fontsize(8)
    btn_save.label.set_fontweight('bold')

    btn_save_csv = Button(ax_save_csv_btn, 'Save Spectra & Shifts (CSV)', color='gold', hovercolor='khaki')
    btn_save_csv.label.set_fontsize(8)
    btn_save_csv.label.set_fontweight('bold')

    def trigger_save(event=None):
        if output_dir:
            save_directory = os.path.abspath(output_dir)
        elif data_pack.get('save_dir') and data_pack['save_dir'] != 'N/A':
            save_directory = data_pack['save_dir']
        else:
            h5_path = data_pack.get('h5_file_path') or data_pack.get('h5_dir')
            if h5_path and os.path.isfile(h5_path):
                from analyze_sgm_bsky_data import resolve_preprocessed_save_dir
                save_directory = resolve_preprocessed_save_dir(os.path.dirname(os.path.abspath(h5_path)))
            elif h5_path and os.path.isdir(h5_path):
                from analyze_sgm_bsky_data import resolve_preprocessed_save_dir
                save_directory = resolve_preprocessed_save_dir(os.path.abspath(h5_path))
            else:
                save_directory = os.getcwd()

        os.makedirs(save_directory, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in scan_title)
        ext = file_format.lstrip('.')
        initial_name = f"{safe_title}_Emission_Peak_Shift_{detector}.{ext}"

        ax_save_btn.set_visible(False)
        ax_save_csv_btn.set_visible(False)
        fig.canvas.draw_idle()

        func_plot = getattr(filedialog, 'asksaveasfilename', 'asksaveasfilename') if filedialog else 'asksaveasfilename'
        out_filename = safe_filedialog_call(
            func_plot,
            title="Save Emission Spectra & Peak Shifts Plot",
            initialdir=save_directory,
            initialfile=initial_name,
            defaultextension=f".{ext}",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("SVG Vector Image", "*.svg"),
                ("JPEG Image", "*.jpg;*.jpeg"),
                ("All Files", "*.*")
            ]
        )

        if out_filename:
            fig.savefig(out_filename, dpi=dpi, bbox_inches='tight')
            print(f"Saved plot to: {out_filename}")
            saved_files.append(out_filename)
        else:
            print("Save plot operation cancelled.")

        ax_save_btn.set_visible(True)
        ax_save_csv_btn.set_visible(True)
        fig.canvas.draw_idle()

    def trigger_save_csv(event=None):
        if output_dir:
            save_directory = os.path.abspath(output_dir)
        elif data_pack.get('save_dir') and data_pack['save_dir'] != 'N/A':
            save_directory = data_pack['save_dir']
        else:
            h5_path = data_pack.get('h5_file_path') or data_pack.get('h5_dir')
            if h5_path and os.path.isfile(h5_path):
                from analyze_sgm_bsky_data import resolve_preprocessed_save_dir
                save_directory = resolve_preprocessed_save_dir(os.path.dirname(os.path.abspath(h5_path)))
            elif h5_path and os.path.isdir(h5_path):
                from analyze_sgm_bsky_data import resolve_preprocessed_save_dir
                save_directory = resolve_preprocessed_save_dir(os.path.abspath(h5_path))
            else:
                save_directory = os.getcwd()

        os.makedirs(save_directory, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in scan_title)
        initial_csv_name = f"{safe_title}_Emission_Peak_Shifts_{detector}.csv"

        func_csv = getattr(filedialog, 'asksaveasfilename', 'asksaveasfilename') if filedialog else 'asksaveasfilename'
        out_csv = safe_filedialog_call(
            func_csv,
            title="Save Selected Emission Spectra & Peak Shifts (CSV)",
            initialdir=save_directory,
            initialfile=initial_csv_name,
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if not out_csv:
            print("Save CSV operation cancelled.")
            return

        calib_gain = gain if (gain != 1.0 or offset != 0.0) else 10.0
        calib_offset = offset

        # Metadata Header
        rows = [
            "# ========================================================",
            "# Canadian Light Source (CLS) - SGM Beamline (11ID-1)",
            "# Selected Emission Spectra & Peak Dispersion Export",
            "# ========================================================",
            "# Facility: Canadian Light Source (CLS)",
            "# Beamline: Spherical Grating Monochromator (SGM) (11ID-1)",
            "# Application: plot_selected_emission_spectra",
            f"# Scan Name: {scan_title}",
            f"# Date: {data_pack.get('date', 'N/A')}",
            f"# Project: {data_pack.get('project', 'N/A')}",
            f"# Scan Type: {data_pack.get('scan_type', 'N/A')}",
            f"# Source File: {data_pack.get('h5_file_path', 'N/A')}",
            f"# Grating: {data_pack.get('grating', 'N/A')}",
            f"# Harmonic: {data_pack.get('harmonic', 'N/A')}",
            f"# Mirror Stripe: {data_pack.get('strip', 'N/A')}",
            f"# Polarization: {data_pack.get('polarization', 'N/A')}",
            f"# Exit Slit Gap: {data_pack.get('exit_slit_gap', 'N/A')}",
            f"# XPS Z: {data_pack.get('xps_z', 'N/A')}",
            f"# Time Per Map: {data_pack.get('time_per_map', 'N/A')}",
            f"# Coordinates: {data_pack.get('coordinates', 'N/A')}",
            f"# Command: {data_pack.get('command', 'N/A')}",
            f"# Target Detector: {detector} ({'+'.join(target_dets)})",
        ]

        if map_roi:
            rows.append(f"# Spatial ROI: X=[{map_roi[0]}, {map_roi[1]}], Y=[{map_roi[2]}, {map_roi[3]}]")
        else:
            rows.append("# Spatial ROI: Full Map")

        if use_energy_calib:
            rows.append(f"# SDD Energy Calibration: Enabled (Reference: {ref_det}, Gain={calib_gain:.4f}, Offset={calib_offset:.4f})")
        else:
            rows.append("# SDD Energy Calibration: Disabled (Default Linear Scaling)")

        if detected_elem and theoretical_fluor_ev is not None:
            rows.append(f"# Detected Element: {detected_elem} ({detected_edge}-edge, Theoretical Emission: {theoretical_fluor_ev:.2f} eV)")

        energies_str = ", ".join([f"{e:.2f} eV" for e in target_energies])
        rows.append(f"# Selected Incident Energies ({len(target_energies)}): {energies_str}")

        if apply_waterfall and offset_step > 0:
            rows.append(f"# Note: Raw spectra saved (Waterfall display offset of {offset_step:.2f} counts not applied to CSV)")

        # Section 1: Peak Shift Dispersion Table
        if any(item.get('peak_fits') for item in spectra_data):
            rows.append("#")
            rows.append("# ========================================================")
            rows.append("# Peak Shift Dispersion Summary Table")
            rows.append("# ========================================================")
            rows.append("# Incident_Energy_eV,Scatter_Peak_eV,Scatter_FWHM_eV,Scatter_Amp,Fluor_Peak_eV,Fluor_FWHM_eV,Fluor_Amp")
            for item in spectra_data:
                pf = item.get('peak_fits', {})
                sc = pf.get('scatter') or {}
                fl = pf.get('fluor') or {}

                sc_c = f"{sc.get('center', np.nan):.2f}" if sc.get('success') else "NaN"
                sc_w = f"{sc.get('fwhm', np.nan):.2f}" if sc.get('success') else "NaN"
                sc_a = f"{sc.get('amplitude', np.nan):.2f}" if sc.get('success') else "NaN"

                fl_c = f"{fl.get('center', np.nan):.2f}" if fl.get('success') else "NaN"
                fl_w = f"{fl.get('fwhm', np.nan):.2f}" if fl.get('success') else "NaN"
                fl_a = f"{fl.get('amplitude', np.nan):.2f}" if fl.get('success') else "NaN"

                rows.append(f"# PEAK_DISPERSION,{item['energy']:.2f},{sc_c},{sc_w},{sc_a},{fl_c},{fl_w},{fl_a}")

        # Section 2: Full 1D Spectra
        rows.append("#")
        rows.append("# ========================================================")
        rows.append("# Full 1D Emission Spectra (256 channels)")
        rows.append("# ========================================================")
        c_idx = 1
        rows.append(f"# Column {c_idx}: Calibrated Emission Energy (eV)"); c_idx += 1
        rows.append(f"# Column {c_idx}: Channel (0-255)"); c_idx += 1
        for item in spectra_data:
            en_val = item['energy']
            rows.append(f"# Column {c_idx}: Emission_Counts_Einc_{en_val:.2f}eV ({detector})"); c_idx += 1
        rows.append("#")

        for ch in range(256):
            e_val = sdd_calib.channel_to_energy(ch, calib_gain, calib_offset)
            row_items = [f"{e_val:.2f}", str(ch)]
            for item in spectra_data:
                spec_val = item['spectrum'][ch]
                row_items.append(f"{spec_val:.2f}")
            rows.append(",".join(row_items))

        try:
            with open(out_csv, 'wb') as f:
                f.write("\r\n".join(rows).encode('utf-8-sig'))
            print(f"  -> [SAVE] Selected emission spectra CSV exported to: {out_csv}")
            saved_files.append(out_csv)
        except Exception as e:
            print(f"Error saving CSV file {out_csv}: {e}", file=sys.stderr)

    btn_save.on_clicked(trigger_save)
    btn_save_csv.on_clicked(trigger_save_csv)
    fig._btn_save_ref = btn_save
    fig._btn_save_csv_ref = btn_save_csv

    if save_plot:
        trigger_save()
    if save_csv:
        trigger_save_csv()

    plt.show()
    return saved_files if saved_files else None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Plot overlaid emission spectra and peak dispersion curves.")
    parser.add_argument("h5", nargs="?", default=None, help="HDF5 stack file path (optional)")
    parser.add_argument("--detector", default="sdd1", help="Detector name (sdd1-sdd4, average, or all)")
    parser.add_argument("--track_peak", default="both", choices=["both", "scatter", "fluorescence", "none"], help="Peak to track")
    parser.add_argument("--element", default=None, help="Target element (e.g. Mg, Si)")
    parser.add_argument("--edge", default=None, help="Target absorption edge (e.g. K, L)")
    parser.add_argument("--plot_mode", default="both", choices=["both", "spectra", "dispersion"], help="Display layout")
    parser.add_argument("--waterfall", action="store_true", help="Apply waterfall offset")
    args = parser.parse_args()

    track_choice = None if args.track_peak == "none" else args.track_peak
    if args.h5:
        dp = analyze_sgm_bsky_data(args.h5)
    else:
        dp = None

    plot_selected_emission_spectra(
        data_pack=dp,
        detector=args.detector,
        track_peak=track_choice,
        element=args.element,
        edge=args.edge,
        plot_mode=args.plot_mode,
        apply_waterfall=args.waterfall
    )
