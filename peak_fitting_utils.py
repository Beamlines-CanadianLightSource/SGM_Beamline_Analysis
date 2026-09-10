"""
peak_fitting_utils.py

Modular Peak Fitting and Curve Deconvolution Module using lmfit.
Designed for XANES/NEXAFS, XRF, XPS, Raman, and general spectroscopy data analysis.
Implements Fityk-style single-spectrum and batch stack deconvolution workflows.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
import sdd_calibration_utils as sdd_calib
from lmfit import Parameters
from lmfit.models import (
    GaussianModel,
    LorentzianModel,
    VoigtModel,
    PseudoVoigtModel,
    StepModel,
    LinearModel,
    PolynomialModel,
    ConstantModel,
)


def get_model_class(model_type_str):
    """
    Map model type string to lmfit model class.
    """
    model_map = {
        'gaussian': GaussianModel,
        'lorentzian': LorentzianModel,
        'voigt': VoigtModel,
        'pseudovoigt': PseudoVoigtModel,
        'step': StepModel,
        'linear': LinearModel,
        'polynomial': PolynomialModel,
        'constant': ConstantModel,
    }
    key = str(model_type_str).lower().strip()
    if key not in model_map:
        raise ValueError(
            f"Unsupported model type '{model_type_str}'. Supported types: {list(model_map.keys())}"
        )
    return model_map[key]


def build_xanes_composite_model(peak_configs, step_config=None, bg_config=None):
    """
    Dynamically constructs a composite lmfit Model.

    Parameters
    ----------
    peak_configs : list of dict
        List of peak specifications. Each dict should have at least:
        - 'prefix': str (e.g., 'peak1_')
        - 'type': str (e.g., 'gaussian', 'lorentzian', 'voigt'), default 'gaussian'
    step_config : dict or None
        Step/edge configuration dict. Example:
        - 'prefix': 'step_'
        - 'form': 'erf' or 'atan' or 'logistic', default 'erf'
    bg_config : dict or None
        Baseline background configuration dict. Example:
        - 'prefix': 'bg_'
        - 'type': 'linear' or 'constant' or 'polynomial'

    Returns
    -------
    composite_model : lmfit.Model
    """
    composite_model = None

    # 1. Step background (if provided)
    if step_config:
        prefix = step_config.get('prefix', 'step_')
        form = step_config.get('form', 'erf')
        step_model = StepModel(form=form, prefix=prefix)
        composite_model = step_model

    # 2. Linear / Baseline background (if provided)
    if bg_config:
        prefix = bg_config.get('prefix', 'bg_')
        bg_type = bg_config.get('type', 'linear')
        bg_cls = get_model_class(bg_type)
        bg_model = bg_cls(prefix=prefix)
        composite_model = (
            bg_model if composite_model is None else (composite_model + bg_model)
        )

    # 3. Spectral Peaks
    for peak_cfg in peak_configs:
        prefix = peak_cfg.get('prefix', 'peak1_')
        peak_type = peak_cfg.get('type', 'gaussian')
        peak_cls = get_model_class(peak_type)
        p_model = peak_cls(prefix=prefix)

        if composite_model is None:
            composite_model = p_model
        else:
            composite_model = composite_model + p_model

    return composite_model


def create_initial_parameters(peak_configs, step_config=None, bg_config=None):
    """
    Builds an lmfit.Parameters object with values, min/max bounds, vary flags, and expressions.

    Parameters
    ----------
    peak_configs : list of dict
        Each dict specifies peak parameter constraints. Key names expected:
        - 'prefix': str (e.g. 'peak1_')
        - 'center': dict {'value': float, 'min': float, 'max': float, 'vary': bool, 'expr': str}
        - 'sigma': dict {'value': float, 'min': float, 'max': float, 'vary': bool, 'expr': str}
        - 'amplitude': dict {'value': float, 'min': float, 'max': float, 'vary': bool, 'expr': str}
        - 'gamma': dict (for VoigtModel/PseudoVoigtModel)
    step_config : dict or None
        Step edge parameter bounds ('center', 'sigma', 'amplitude').
    bg_config : dict or None
        Background parameter bounds (e.g. 'intercept', 'slope').

    Returns
    -------
    params : lmfit.Parameters
    """
    params = Parameters()

    def _add_param(prefix, param_name, param_dict):
        if not param_dict:
            return
        full_name = f"{prefix}{param_name}"
        val = param_dict.get('value', None)
        min_v = param_dict.get('min', -np.inf)
        max_v = param_dict.get('max', np.inf)
        vary = param_dict.get('vary', True)
        expr = param_dict.get('expr', None)
        params.add(
            full_name,
            value=val,
            min=min_v,
            max=max_v,
            vary=vary,
            expr=expr,
        )

    # Step config
    if step_config:
        prefix = step_config.get('prefix', 'step_')
        for pname in ['center', 'sigma', 'amplitude']:
            if pname in step_config:
                _add_param(prefix, pname, step_config[pname])

    # Background config
    if bg_config:
        prefix = bg_config.get('prefix', 'bg_')
        for pname in ['intercept', 'slope', 'c0', 'c1', 'c2']:
            if pname in bg_config:
                _add_param(prefix, pname, bg_config[pname])

    # Peak configs
    for peak_cfg in peak_configs:
        prefix = peak_cfg.get('prefix', 'peak1_')
        for pname in ['center', 'sigma', 'amplitude', 'gamma', 'fraction', 'height']:
            if pname in peak_cfg:
                _add_param(prefix, pname, peak_cfg[pname])

    return params


def fit_single_spectrum(x_data, y_data, model, params, method='leastsq', **kwargs):
    """
    Fits a single 1D spectrum.

    Parameters
    ----------
    x_data : array-like
        Independent variable (e.g., energy in eV).
    y_data : array-like
        Dependent intensity variable.
    model : lmfit.Model
        Composite model.
    params : lmfit.Parameters
        Starting parameters.
    method : str
        Fitting algorithm (default 'leastsq' / Levenberg-Marquardt).

    Returns
    -------
    fit_out : lmfit.model.ModelResult
    """
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)

    fit_out = model.fit(y_arr, params, x=x_arr, method=method, **kwargs)
    return fit_out


def fit_spectrum_stack(x_data, y_stack, model, initial_params, fix_shapes=False, method='leastsq'):
    """
    Fits a dataset/stack of spectra (2D array of shape [n_spectra, n_points]).

    Parameters
    ----------
    x_data : array-like
        1D array of energy/x points of length n_points.
    y_stack : array-like
        2D array of shape (n_spectra, n_points) or list of 1D arrays.
    model : lmfit.Model
        Composite model.
    initial_params : lmfit.Parameters
        Base parameter set.
    fix_shapes : bool
        If True, locks peak center and sigma parameters ('vary=False') and only fits amplitudes/intensities.
    method : str
        Fitting method (default 'leastsq').

    Returns
    -------
    results_list : list of lmfit.model.ModelResult
    summary_df : pandas.DataFrame
        DataFrame summarizing peak parameter values, uncertainties, and fit statistics across all spectra.
    """
    x_arr = np.asarray(x_data, dtype=float)
    y_stack_arr = np.asarray(y_stack, dtype=float)

    params_to_use = initial_params.copy()
    if fix_shapes:
        for pname, p in params_to_use.items():
            if 'center' in pname or 'sigma' in pname or 'gamma' in pname:
                p.vary = False

    results_list = []
    summary_records = []

    for i in range(y_stack_arr.shape[0]):
        y_curr = y_stack_arr[i]
        fit_out = model.fit(y_curr, params_to_use, x=x_arr, method=method)
        results_list.append(fit_out)

        rec = {'spectrum_index': i, 'redchi': fit_out.redchi, 'chisqr': fit_out.chisqr, 'nfev': fit_out.nfev}
        for pname, pval in fit_out.params.items():
            rec[pname] = pval.value
            rec[f"{pname}_stderr"] = pval.stderr if pval.stderr is not None else np.nan

        summary_records.append(rec)

    summary_df = pd.DataFrame(summary_records)
    return results_list, summary_df


def refine_two_pass_stack_fit(x_data, y_stack, model, initial_params, select_indices=None, method='leastsq'):
    """
    Executes a Fityk-style two-pass refinement strategy:
    Pass 1: Fits all or selected spectra with unconstrained/free shape parameters.
    Pass 2: Computes mean peak centers and widths, freezes them ('vary=False'),
            and re-fits amplitudes across all spectra.

    Parameters
    ----------
    x_data : array-like
        1D x points.
    y_stack : array-like
        2D array of shape (n_spectra, n_points).
    model : lmfit.Model
        Composite model.
    initial_params : lmfit.Parameters
        Initial parameter guess and bounds.
    select_indices : list of int or None
        Indices of high signal-to-noise spectra to use for shape parameter averaging.
        If None, uses all spectra in y_stack.

    Returns
    -------
    pass1_results : list of ModelResult
    pass2_results : list of ModelResult
    pass2_summary_df : pandas.DataFrame
    refined_params : lmfit.Parameters
        The frozen parameter set used for Pass 2.
    """
    x_arr = np.asarray(x_data, dtype=float)
    y_stack_arr = np.asarray(y_stack, dtype=float)
    n_spectra = y_stack_arr.shape[0]

    # Pass 1: Initial unconstrained fit
    pass1_results, _ = fit_spectrum_stack(
        x_arr, y_stack_arr, model, initial_params, fix_shapes=False, method=method
    )

    if select_indices is None:
        select_indices = list(range(n_spectra))

    # Calculate average shape parameters (centers, sigmas, gammas)
    refined_params = initial_params.copy()
    shape_keys = [
        k for k in initial_params.keys()
        if ('center' in k or 'sigma' in k or 'gamma' in k)
    ]

    for key in shape_keys:
        vals = [pass1_results[idx].values[key] for idx in select_indices if key in pass1_results[idx].values]
        if len(vals) > 0:
            mean_val = float(np.mean(vals))
            refined_params[key].value = mean_val
            refined_params[key].vary = False

    # Ensure amplitude/height/intercept parameters remain free (vary=True)
    for key, p in refined_params.items():
        if 'amplitude' in key or 'height' in key or 'intercept' in key or 'c0' in key:
            p.vary = True

    # Pass 2: Re-fit with fixed shape parameters
    pass2_results, pass2_summary_df = fit_spectrum_stack(
        x_arr, y_stack_arr, model, refined_params, fix_shapes=True, method=method
    )

    return pass1_results, pass2_results, pass2_summary_df, refined_params


def plot_fit_deconvolution(
    x_data,
    y_data,
    fit_out,
    title=None,
    show_components=True,
    xlim=None,
    xlabel="Energy (eV)",
    ylabel="Normalized Intensity",
    save_path=None,
    figsize=(9, 6)
):
    """
    Plots raw data, total best fit, individual deconvolved peak components, and residual panel.

    Parameters
    ----------
    x_data : array-like
        1D x values.
    y_data : array-like
        1D y data.
    fit_out : lmfit.model.ModelResult
        Model fit result object.
    title : str or None
        Plot title.
    show_components : bool
        Whether to draw individual step & peak curves.
    xlim : tuple or None
        (x_min, x_max) range.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    save_path : str or None
        If provided, saves figure to specified filepath.
    figsize : tuple
        Figure width and height.

    Returns
    -------
    fig, (ax_main, ax_res) : matplotlib Figure and Axes
    """
    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # Main spectrum & fit plot
    ax_main.plot(x_arr, y_arr, 'o', color='#2b5c8f', markersize=3, alpha=0.7, label='Data')
    ax_main.plot(x_arr, fit_out.best_fit, '-', color='#d95f02', linewidth=2.0, label='Best Fit')

    if show_components:
        comps = fit_out.eval_components(x=x_arr)
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(comps), 10)))
        idx = 0
        for comp_name, comp_y in comps.items():
            clean_name = comp_name.rstrip('_')
            if 'step' in comp_name.lower():
                ax_main.plot(x_arr, comp_y, '--', color='#7570b3', linewidth=1.5, label=f'Step ({clean_name})')
            elif 'bg' in comp_name.lower():
                ax_main.plot(x_arr, comp_y, ':', color='#666666', linewidth=1.5, label=f'BG ({clean_name})')
            else:
                ax_main.plot(x_arr, comp_y, '-', color=colors[idx % 10], alpha=0.85, linewidth=1.2, label=clean_name)
                idx += 1

    ax_main.set_ylabel(ylabel, fontsize=11)
    if title:
        ax_main.set_title(title, fontsize=12, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_main.grid(True, linestyle=':', alpha=0.5)

    # Residual plot
    residuals = y_arr - fit_out.best_fit
    ax_res.plot(x_arr, residuals, '-', color='#e7298a', linewidth=1.0)
    ax_res.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax_res.set_xlabel(xlabel, fontsize=11)
    ax_res.set_ylabel("Residual", fontsize=10)
    ax_res.grid(True, linestyle=':', alpha=0.5)

    if xlim:
        ax_main.set_xlim(xlim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, (ax_main, ax_res)


def extract_spectrum_from_sgm_data(
    sgm_data,
    channel_idx=0,
    use_sdd=False,
    detector_name='sdd1',
    roi_window=None,
    mode='mean',
    pixel_idx=None
):
    """
    Extracts energy_arr (1D array of energies in eV) and intensity_arr (1D intensity spectrum)
    from an sgm_data dictionary returned by analyze_sgm_bsky_data().

    Parameters
    ----------
    sgm_data : dict
        Data dictionary returned by analyze_sgm_bsky_data().
    channel_idx : int or str, default 0
        MCC channel index (e.g. 0 for MCC1/TEY) or channel name string (used when use_sdd=False).
    use_sdd : bool, default False
        If True, extracts Partial Fluorescence Yield (PFY) spectrum from SDD files in sgm_data['sdd_files'].
    detector_name : str, default 'sdd1'
        Name of SDD detector (e.g. 'sdd1', 'sdd2', 'sdd3', 'sdd4') to extract from when use_sdd=True.
    roi_window : tuple (start_ch, end_ch) or None
        Channel range (min_channel, max_channel) to integrate for SDD PFY. If None, sums across all MCA channels.
    mode : str, default 'mean'
        Reduction mode across spatial pixels: 'mean', 'sum', or 'pixel'.
    pixel_idx : int or None
        If mode=='pixel', index of spatial pixel to extract.

    Returns
    -------
    energy_arr : np.ndarray
        1D array of energies (in eV).
    intensity_arr : np.ndarray
        1D array of intensity values corresponding to energy_arr.
    """
    if not isinstance(sgm_data, dict) or 'energies' not in sgm_data:
        raise ValueError("Invalid sgm_data object. Expected dict returned by analyze_sgm_bsky_data().")

    energy_arr = np.asarray(sgm_data['energies'], dtype=float)
    if len(energy_arr) == 0:
        raise ValueError("No energy points found in sgm_data['energies'].")

    intensity_list = []

    if use_sdd:
        import os
        sdd_files = sgm_data.get('sdd_files', {})
        if not sdd_files or detector_name not in sdd_files:
            available_dets = list(sdd_files.keys())
            if available_dets:
                detector_name = available_dets[0]
            else:
                raise ValueError("No SDD files found in sgm_data['sdd_files'].")

        sdd_dict = sdd_files[detector_name]

        for energy in energy_arr:
            fpath = sdd_dict.get(energy)
            if fpath and os.path.exists(fpath):
                try:
                    d1d = np.fromfile(fpath, dtype=np.uint32)
                    if roi_window is not None:
                        start_ch, end_ch = roi_window
                        if d1d.ndim > 1:
                            val_arr = np.sum(d1d[:, start_ch:end_ch], axis=1)
                        else:
                            val_arr = np.array([np.sum(d1d[start_ch:end_ch])])
                    else:
                        val_arr = np.array([np.sum(d1d)])

                    if mode == 'mean':
                        val = np.nanmean(val_arr)
                    elif mode == 'sum':
                        val = np.nansum(val_arr)
                    elif mode == 'pixel' and pixel_idx is not None:
                        val = val_arr[pixel_idx] if pixel_idx < len(val_arr) else np.nanmean(val_arr)
                    else:
                        val = np.nanmean(val_arr)
                    intensity_list.append(0.0 if np.isnan(val) else float(val))
                except Exception:
                    intensity_list.append(0.0)
            else:
                intensity_list.append(0.0)
    else:
        mcc_channels = sgm_data.get('mcc_channel_names', [])
        if isinstance(channel_idx, str):
            target_name = channel_idx.lower().strip()
            found_idx = -1
            for idx, ch_name in enumerate(mcc_channels):
                if target_name in ch_name.lower().strip():
                    found_idx = idx
                    break
            if found_idx != -1:
                channel_idx = found_idx
            else:
                channel_idx = 0

        mcc_data_dict = sgm_data.get('mcc_data', {})

        for energy in energy_arr:
            table = mcc_data_dict.get(energy)
            if table is not None and len(table) > 0:
                if len(table.shape) > 1 and channel_idx < table.shape[1]:
                    col = table[:, channel_idx]
                elif len(table.shape) == 1:
                    col = table
                else:
                    col = table.flatten()

                if mode == 'mean':
                    val = np.nanmean(col)
                elif mode == 'sum':
                    val = np.nansum(col)
                elif mode == 'pixel' and pixel_idx is not None:
                    val = col[pixel_idx] if pixel_idx < len(col) else np.nanmean(col)
                else:
                    val = np.nanmean(col)

                intensity_list.append(0.0 if np.isnan(val) else val)
            else:
                intensity_list.append(0.0)

    return energy_arr, np.array(intensity_list)


def _get_last_directory():
    """Helper to get last accessed directory from analyze_sgm_bsky_data config."""
    try:
        from analyze_sgm_bsky_data import get_last_dir
        return get_last_dir()
    except Exception:
        import os
        return os.getcwd()


def _save_last_directory(filepath):
    """Helper to save last accessed directory to analyze_sgm_bsky_data config."""
    try:
        import os
        from analyze_sgm_bsky_data import save_last_dir
        if filepath:
            folder = os.path.dirname(os.path.abspath(filepath))
            save_last_dir(folder)
    except Exception:
        pass


def save_spectrum_for_fitting(energy_arr, intensity_arr, filepath=None):
    """
    Saves energy_arr and intensity_arr to a clean CSV file formatted for lmfit modeling.

    Parameters
    ----------
    energy_arr : array-like
        1D array of energy values (eV).
    intensity_arr : array-like
        1D array of intensity values.
    filepath : str or None
        Destination file path. If None, opens a Tk file dialog in the last used folder.

    Returns
    -------
    str or None : Saved file path.
    """
    if filepath is None:
        import tkinter as tk
        from tkinter import filedialog
        last_dir = _get_last_directory()
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.asksaveasfilename(
            title="Save Spectrum CSV for Peak Fitting",
            initialdir=last_dir,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()

    if not filepath:
        print("Save operation cancelled.")
        return None

    import os
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    df = pd.DataFrame({
        'Energy': np.asarray(energy_arr, dtype=float),
        'Intensity': np.asarray(intensity_arr, dtype=float)
    })
    df.to_csv(filepath, index=False)
    _save_last_directory(filepath)
    print(f"Successfully exported spectrum for peak fitting: {filepath}")
    return filepath


def select_csv_columns_gui(df, filepath=None, default_detector=None):
    """
    Opens an interactive GUI popup with live plot preview allowing the user to select
    which Energy (X) column and Intensity (Y) column to load for peak fitting.
    """
    import os
    import numpy as np
    import pandas as pd
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    columns = [c for c in df.columns if not str(c).startswith('#')]
    if not columns:
        return None, None

    e_candidates = [c for c in columns if any(k in c.lower() for k in ['calibrated energy', 'energy', 'ev', 'x'])]
    default_e = e_candidates[0] if e_candidates else columns[0]

    default_i = None
    if default_detector:
        det_str = str(default_detector).lower().strip()
        for col in columns:
            if det_str in col.lower() and ('norm_' in col.lower() or 'intensity' in col.lower() or 'raw_' in col.lower()):
                default_i = col
                break

    if not default_i:
        i_candidates = [c for c in columns if any(k in c.lower() for k in ['norm_selected_average', 'norm_average', 'norm_sdd1', 'norm_sdd2', 'norm_sdd3', 'norm_sdd4', 'norm_mcc4', 'norm_tey', 'intensity'])]
        default_i = i_candidates[0] if i_candidates else (columns[1] if len(columns) > 1 else columns[0])

    root = tk.Tk()
    fname = os.path.basename(filepath) if filepath else "CSV File"
    root.title(f"Select Columns for Peak Fitting - {fname}")
    root.geometry("820x600")
    root.attributes("-topmost", True)
    try:
        root.deiconify()
        root.lift()
        root.focus_force()
    except Exception:
        pass

    res = {'energy_col': default_e, 'intensity_col': default_i, 'cancelled': False}

    # Control Frame (Top)
    ctrl_frame = tk.LabelFrame(root, text="Select Spectrum Channels (X & Y Data)", font=('Arial', 10, 'bold'), padx=10, pady=8)
    ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=10)

    tk.Label(ctrl_frame, text="Energy Axis (X-Data):", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=4, sticky='w')
    combo_e = ttk.Combobox(ctrl_frame, values=columns, state="readonly", width=42)
    combo_e.set(default_e)
    combo_e.grid(row=0, column=1, padx=5, pady=4, sticky='w')

    tk.Label(ctrl_frame, text="Intensity Signal (Y-Data):", font=('Arial', 9, 'bold')).grid(row=1, column=0, padx=5, pady=4, sticky='w')
    combo_i = ttk.Combobox(ctrl_frame, values=columns, state="readonly", width=42)
    combo_i.set(default_i)
    combo_i.grid(row=1, column=1, padx=5, pady=4, sticky='w')

    # Buttons Frame (Right of controls)
    btn_frame = tk.Frame(ctrl_frame)
    btn_frame.grid(row=0, column=2, rowspan=2, padx=15, sticky="e")

    def on_ok():
        res['energy_col'] = combo_e.get()
        res['intensity_col'] = combo_i.get()
        try:
            root.withdraw()
            root.update()
        except Exception: pass
        try: root.destroy()
        except Exception: pass
        try: root.quit()
        except Exception: pass

    def on_cancel():
        res['cancelled'] = True
        try:
            root.withdraw()
            root.update()
        except Exception: pass
        try: root.destroy()
        except Exception: pass
        try: root.quit()
        except Exception: pass

    btn_ok = tk.Button(btn_frame, text="Load Selected Spectrum", command=on_ok, bg='#d4edda', fg='#155724', font=('Arial', 9, 'bold'), padx=12, pady=4)
    btn_ok.pack(pady=3)
    btn_cancel = tk.Button(btn_frame, text="Cancel", command=on_cancel, padx=12, pady=3)
    btn_cancel.pack(pady=3)

    # Plot Frame (Center)
    fig = Figure(figsize=(7.5, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=(0, 5))

    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()

    def update_plot(event=None):
        e_col = combo_e.get()
        i_col = combo_i.get()
        if e_col in df.columns and i_col in df.columns:
            try:
                x_val = pd.to_numeric(df[e_col], errors='coerce').values
                y_val = pd.to_numeric(df[i_col], errors='coerce').values
                valid_mask = ~np.isnan(x_val) & ~np.isnan(y_val)
                x_clean, y_clean = x_val[valid_mask], y_val[valid_mask]

                ax.clear()
                ax.plot(x_clean, y_clean, 'o-', color='#1f77b4', markersize=3, alpha=0.85, label=i_col)
                ax.set_xlabel(e_col, fontsize=10, fontweight='bold')
                ax.set_ylabel(i_col, fontsize=10, fontweight='bold')
                ax.set_title(f"Spectrum Preview: {i_col} vs {e_col}", fontsize=11, fontweight='bold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='best', fontsize='small')
                fig.tight_layout()
                canvas.draw_idle()
            except Exception:
                pass

    combo_e.bind("<<ComboboxSelected>>", update_plot)
    combo_i.bind("<<ComboboxSelected>>", update_plot)
    update_plot()

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    if res['cancelled']:
        return None, None
    return res['energy_col'], res['intensity_col']


def read_summary_csv(filepath):
    """
    Reads summary CSV files exported by plot_sgm_bsky_data, parsing
    # Column N: <Name> metadata comments to extract full human-readable column headers.
    """
    import re
    import pandas as pd
    
    comment_lines = []
    first_data_line = None
    delimiter = ','
    
    with open(filepath, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith('#'):
                comment_lines.append(line_str)
            else:
                first_data_line = line_str
                break
                
    if not first_data_line:
        return pd.read_csv(filepath, comment='#')
        
    if '\t' in first_data_line:
        delimiter = '\t'
    else:
        delimiter = ','
            
    data_cols = [x.strip() for x in first_data_line.split(delimiter)]
    num_cols = len(data_cols)
    
    column_names = [None] * num_cols
    col_pattern = re.compile(r'#\s*Column\s+(\d+)\s*:\s*(.*)', re.IGNORECASE)
    has_column_meta = False
    
    for line in comment_lines:
        match = col_pattern.match(line)
        if match:
            col_idx = int(match.group(1))
            col_name = match.group(2).strip()
            if 1 <= col_idx <= num_cols:
                column_names[col_idx - 1] = col_name
                has_column_meta = True
                
    if has_column_meta:
        for idx in range(num_cols):
            if column_names[idx] is None:
                column_names[idx] = data_cols[idx] if idx < len(data_cols) else f"Column_{idx + 1}"
        try:
            df = pd.read_csv(filepath, comment='#', sep=delimiter, header=None, names=column_names, skiprows=len(comment_lines))
            # If first row duplicates column names, drop it
            if len(df) > 0 and str(df.iloc[0, 0]).strip() == str(column_names[0]).strip():
                df = df.iloc[1:].reset_index(drop=True)
            return df
        except Exception:
            pass

    return pd.read_csv(filepath, comment='#')


def load_spectrum_for_fitting(filepath=None, detector=None, raw=False, interactive=True, plot_preview=True):
    """
    Loads energy_arr and intensity_arr from a CSV file (e.g. summary CSV exported by plot_sgm_bsky_data)
    for lmfit modeling, displaying a clean summary and plot preview in the console/notebook output.

    Parameters
    ----------
    filepath : str or None
        Path to CSV file. If None, opens an interactive Tk file dialog pre-navigated to the last used folder.
    detector : str or None
        Detector preference (e.g., 'sdd1', 'sdd2', 'sdd3', 'sdd4', 'selected_average', 'average', 'tey', 'mcc4').
        If specified, bypasses column GUI and auto-selects matching column.
    raw : bool, default False
        If True, loads raw unnormalized intensity column instead of normalized.
    interactive : bool, default True
        If True and detector is None, opens a column selection GUI popup for visual column selection.
    plot_preview : bool, default True
        If True, renders an inline matplotlib spectrum plot preview in the notebook output cell.

    Returns
    -------
    energy_arr : np.ndarray
        1D energy array (eV).
    intensity_arr : np.ndarray
        1D normalized (or raw) spectrum array.
    """
    import os
    if filepath is None:
        import tkinter as tk
        from tkinter import filedialog
        last_dir = _get_last_directory()
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.askopenfilename(
            title="Select Spectrum CSV File for Fitting",
            initialdir=last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()

    if not filepath or not os.path.exists(filepath):
        print("No valid file selected.")
        return None, None

    _save_last_directory(filepath)

    # Load dataframe using metadata comments parser
    df = read_summary_csv(filepath)

    energy_col_name = None
    chosen_col = None

    # If interactive mode and detector not hardcoded, prompt GUI column selector
    if interactive and detector is None:
        try:
            energy_col_name, chosen_col = select_csv_columns_gui(df, filepath=filepath, default_detector=detector)
        except Exception as _gui_err:
            print(f"  [GUI Info] Interactive column dialog failed or closed: {_gui_err}")

    # Fallback / Auto-detection if interactive cancelled or non-interactive
    if not energy_col_name or not chosen_col:
        # 1. Identify Energy column
        e_col = [c for c in df.columns if any(k in c.lower() for k in ['calibrated energy', 'energy', 'ev', 'x'])]
        energy_col_name = e_col[0] if e_col else df.columns[0]

        # 2. Identify Intensity column based on detector preference and norm/raw mode
        prefix = "RAW_" if raw else "NORM_"

        if detector:
            det_str = str(detector).lower().strip()
            for col in df.columns:
                if col.startswith("#") or col == energy_col_name:
                    continue
                if det_str in col.lower() and (prefix in col or (not raw and "RAW_" not in col)):
                    chosen_col = col
                    break

        if not chosen_col:
            priorities = [
                "NORM_Selected_Average", "NORM_Average", "NORM_sdd1", "NORM_sdd2", "NORM_sdd3", "NORM_sdd4",
                "RAW_Selected_Average", "RAW_Average", "RAW_sdd1"
            ]
            for prio in priorities:
                for col in df.columns:
                    if prio.lower() in col.lower():
                        chosen_col = col
                        break
                if chosen_col:
                    break

        if not chosen_col:
            i_cols = [c for c in df.columns if c != energy_col_name and not c.startswith("#")]
            chosen_col = i_cols[0] if i_cols else df.columns[-1]

    energy_arr = np.asarray(df[energy_col_name].values, dtype=float)
    intensity_arr = np.asarray(df[chosen_col].values, dtype=float)

    # Clean NaNs if present
    valid_mask = ~np.isnan(energy_arr) & ~np.isnan(intensity_arr)
    energy_arr = energy_arr[valid_mask]
    intensity_arr = intensity_arr[valid_mask]

    print("\n" + "="*70)
    print(f"  [SPECTRUM LOADED FOR PEAK FITTING]")
    print("="*70)
    print(f"  File         : {os.path.basename(filepath)}")
    print(f"  Energy Axis  : '{energy_col_name}' ({len(energy_arr)} pts, {energy_arr.min():.2f} - {energy_arr.max():.2f} eV)")
    print(f"  Intensity Y  : '{chosen_col}' (Range: {intensity_arr.min():.4e} - {intensity_arr.max():.4e})")
    print("="*70)

    if plot_preview and len(energy_arr) > 0:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 3.6), dpi=100)
            ax.plot(energy_arr, intensity_arr, 'o-', color='#1f77b4', markersize=3, alpha=0.85, label=chosen_col)
            ax.set_xlabel(f"{energy_col_name}", fontsize=10, fontweight='bold')
            ax.set_ylabel(chosen_col, fontsize=10, fontweight='bold')
            ax.set_title(f"Loaded Spectrum: {chosen_col}\nSource File: {os.path.basename(filepath)}", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize='small')
            plt.tight_layout()
            plt.show()
        except Exception as _p_err:
            pass

    return energy_arr, intensity_arr


def extract_gaussian_pfy_xanes(
    path_pack,
    target_emission_e,
    window_ev=2.0,
    fit_crop_ev=100.0,
    detectors=None,
    map_roi=None,
    energy_shift=0.0,
    i0_values=None,
    plot_summary=True,
    export_csv=True,
    output_filename=None,
    figsize=(9.2, 3.8)
):
    """
    Fits a Gaussian peak (+ linear baseline) to the specified target XRF emission line 
    at each incident energy step for every SDD detector to extract high-fidelity 
    Gaussian-fitted Partial Fluorescence Yield (PFY-XANES) spectra.

    Parameters
    ----------
    path_pack : dict
        Data pack dictionary returned by analyze_sgm_bsky_data() or plot_sgm_bsky_data().
    target_emission_e : float
        Target emission line energy in eV (e.g., 1486.7 eV for Al-K, 1739.0 eV for Si-K).
    window_ev : float, default 2.0
        Centroid tolerance constraint (+/- eV around target_emission_e).
    fit_crop_ev : float, default 100.0
        Crop window (+/- eV around target_emission_e) used to fit the peak + baseline.
    detectors : list of str or None
        List of SDD detector names to process (defaults to all available SDDs).
    map_roi : list or None
        Optional spatial ROI [x1, x2, y1, y2] in mm.
    energy_shift : float, default 0.0
        Energy calibration shift (eV) applied to the incident monochromator energies.
    i0_values : array-like or None
        Optional custom I0 normalization array. If None, uses mcc1 or internal I0.
    plot_summary : bool, default True
        If True, displays matplotlib summary figures of the extracted PFY-XANES spectra and sample fit.
    export_csv : bool, default True
        If True, exports the resulting DataFrame to a CSV file.
    output_filename : str or None
        Custom CSV output filename. If None, auto-generates a descriptive filename.

    Returns
    -------
    df_pfy : pandas.DataFrame
        DataFrame containing incident energies, raw fitted Gaussian areas per SDD,
        normalized PFY-XANES spectra, and average PFY-XANES curves.
    """
    if not path_pack:
        raise ValueError("path_pack is empty or invalid.")

    scan_name = path_pack.get('scan_name', 'sdd_stack')
    save_dir = path_pack.get('save_dir') or os.getcwd()
    all_energies = np.array(sorted(path_pack.get('energies', [])))
    num_energies = len(all_energies)

    if num_energies == 0:
        raise ValueError("No energy data found in path_pack.")

    calibrated_energies = all_energies + energy_shift

    # Determine active detectors
    available_dets = sorted(path_pack.get('sdd_files', {}).keys())
    if not available_dets and 'stack_maps' in path_pack:
        available_dets = sorted(path_pack['stack_maps'].keys())

    if detectors is None:
        detectors = available_dets
    else:
        detectors = [d for d in detectors if d in available_dets]

    if not detectors:
        raise ValueError("No matching SDD detectors found in path_pack.")

    # Spatial ROI mask setup
    x_coords = path_pack.get('x', np.array([]))
    y_coords = path_pack.get('y', np.array([]))
    
    if map_roi is not None and len(x_coords) > 0 and len(y_coords) > 0:
        x1, x2 = sorted(map_roi[0:2])
        y1, y2 = sorted(map_roi[2:4])
        spatial_mask = (x_coords >= x1) & (x_coords <= x2) & (y_coords >= y1) & (y_coords <= y2)
    else:
        spatial_mask = np.ones(len(x_coords), dtype=bool) if len(x_coords) > 0 else None

    # SDD Calibration data
    calib_data = sdd_calib.load_calibration() or {}

    raw_pfy_data = {det: np.zeros(num_energies) for det in detectors}
    sample_fits = {}

    mid_e_idx = num_energies // 2

    print("\n" + "="*75)
    print(f"  [GAUSSIAN-FITTED PFY-XANES EXTRACTION]")
    print("="*75)
    print(f"  Scan Name        : {scan_name}")
    print(f"  Target Emission  : {target_emission_e:.2f} eV (Tolerance: +/- {window_ev:.2f} eV)")
    print(f"  Fit Range        : {target_emission_e - fit_crop_ev:.1f} to {target_emission_e + fit_crop_ev:.1f} eV")
    print(f"  Detectors        : {', '.join(detectors)}")
    print(f"  Energy Range     : {calibrated_energies[0]:.2f} - {calibrated_energies[-1]:.2f} eV ({num_energies} pts)")
    print("="*75)

    for det in detectors:
        gain = calib_data.get(det, {}).get('gain', 10.0)
        offset = calib_data.get(det, {}).get('offset', 0.0)
        
        emission_energies = sdd_calib.channel_to_energy(np.arange(256), gain, offset)
        
        crop_mask = (emission_energies >= target_emission_e - fit_crop_ev) & (emission_energies <= target_emission_e + fit_crop_ev)
        x_fit = emission_energies[crop_mask]
        
        if len(x_fit) < 3:
            x_fit = np.arange(256)
            crop_mask = np.ones(256, dtype=bool)

        model = GaussianModel(prefix='peak_') + LinearModel(prefix='bg_')

        for e_idx, energy in enumerate(all_energies):
            if 'stack_maps' in path_pack and det in path_pack['stack_maps']:
                raw_pixels = path_pack['stack_maps'][det][e_idx]
                if raw_pixels.ndim == 2:
                    if spatial_mask is not None and spatial_mask.size == raw_pixels.shape[0]:
                        spec_256 = np.sum(raw_pixels[spatial_mask], axis=0)
                    else:
                        spec_256 = np.sum(raw_pixels, axis=0)
                else:
                    spec_256 = raw_pixels
            elif 'sdd_files' in path_pack and det in path_pack['sdd_files']:
                fpath = path_pack['sdd_files'][det].get(energy)
                spec_256 = np.zeros(256)
                if fpath and os.path.exists(fpath):
                    try:
                        d1d = np.fromfile(fpath, dtype=np.uint32)
                        num_pix = len(d1d) // 256
                        if num_pix > 0:
                            s2d = d1d[:num_pix*256].reshape((num_pix, 256))
                            if spatial_mask is not None and spatial_mask.size >= num_pix:
                                spec_256 = np.sum(s2d[spatial_mask[:num_pix]], axis=0)
                            else:
                                spec_256 = np.sum(s2d, axis=0)
                    except Exception:
                        pass

            y_fit = spec_256[crop_mask]

            if len(y_fit) == 0 or np.all(y_fit == 0) or np.max(y_fit) == 0:
                raw_pfy_data[det][e_idx] = 0.0
                continue

            params = model.make_params()
            params['peak_center'].set(value=target_emission_e, min=target_emission_e - window_ev, max=target_emission_e + window_ev)
            params['peak_sigma'].set(value=25.0, min=5.0, max=80.0)
            params['peak_amplitude'].set(value=max(np.max(y_fit) * 20.0, 1.0), min=0.0)
            params['bg_slope'].set(value=0.0)
            params['bg_intercept'].set(value=np.min(y_fit))

            try:
                fit_res = model.fit(y_fit, params, x=x_fit)
                area = fit_res.params['peak_amplitude'].value
                raw_pfy_data[det][e_idx] = max(0.0, float(area))
                
                if e_idx == mid_e_idx:
                    sample_fits[det] = (x_fit, y_fit, fit_res)
            except Exception:
                raw_pfy_data[det][e_idx] = max(0.0, float(np.sum(y_fit)))

    # Normalization Setup
    if i0_values is None:
        if 'mcc_data' in path_pack and 'mcc1' in path_pack['mcc_data']:
            i0_values = np.array(path_pack['mcc_data']['mcc1'])
        elif 'mcc_maps' in path_pack and 'mcc1' in path_pack['mcc_maps']:
            mcc1_arr = path_pack['mcc_maps']['mcc1']
            i0_values = np.nanmean(mcc1_arr, axis=1)
        else:
            i0_values = np.ones(num_energies)

    i0_safe = np.where(i0_values <= 0, 1.0, np.abs(i0_values))

    # Construct DataFrame
    df_pfy = pd.DataFrame({
        'Calibrated_Energy_eV': calibrated_energies,
        'Original_Energy_eV': all_energies,
        'I0': i0_values
    })

    norm_pfy_dict = {}
    for det in detectors:
        df_pfy[f'RAW_{det}_Gaussian_PFY'] = raw_pfy_data[det]
        norm_val = raw_pfy_data[det] / i0_safe
        df_pfy[f'NORM_{det}_Gaussian_PFY'] = norm_val
        norm_pfy_dict[det] = norm_val

    # Averages
    raw_avg = np.nanmean([raw_pfy_data[d] for d in detectors], axis=0)
    norm_avg = np.nanmean([norm_pfy_dict[d] for d in detectors], axis=0)

    df_pfy['RAW_Average_Gaussian_PFY'] = raw_avg
    df_pfy['NORM_Average_Gaussian_PFY'] = norm_avg

    # Export CSV if requested
    if export_csv:
        if output_filename is None:
            output_filename = f"{scan_name}_Gaussian_PFY_{target_emission_e:.1f}eV.csv"
        csv_path = os.path.join(save_dir, output_filename)
        df_pfy.to_csv(csv_path, index=False)
        print(f"  -> Exported Gaussian PFY-XANES to CSV: {csv_path}")

    # Plot summary visualization
    if plot_summary and len(detectors) > 0:
        try:
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=100)
            
            # Subplot 1: Sample Gaussian fit at middle energy step
            ax_fit = axes[0]
            first_det = detectors[0]
            if first_det in sample_fits:
                xf, yf, res = sample_fits[first_det]
                ax_fit.plot(xf, yf, 'o', color='gray', alpha=0.6, label=f"Raw Spectrum ({first_det})")
                ax_fit.plot(xf, res.best_fit, 'r-', lw=2, label="Gaussian + Baseline Fit")
                ax_fit.plot(xf, res.eval_components(x=xf)['peak_'], 'b--', label="Gaussian Peak")
                ax_fit.plot(xf, res.eval_components(x=xf)['bg_'], 'g:', label="Baseline")
                ax_fit.set_title(f"Sample Fit ({first_det} @ {calibrated_energies[mid_e_idx]:.2f} eV)\nTarget Line: {target_emission_e:.1f} eV", fontsize=10, fontweight='bold')
                ax_fit.set_xlabel("Emission Energy (eV)", fontsize=9)
                ax_fit.set_ylabel("Counts", fontsize=9)
                ax_fit.legend(fontsize=8)
                ax_fit.grid(True, linestyle=':', alpha=0.6)
            
            # Subplot 2: Extracted PFY-XANES Spectra
            ax_xanes = axes[1]
            for det in detectors:
                ax_xanes.plot(calibrated_energies, norm_pfy_dict[det], 'o-', label=f"{det} PFY", alpha=0.7)
            ax_xanes.plot(calibrated_energies, norm_avg, 'k-', lw=2.5, label="Average PFY")
            ax_xanes.set_title(f"Extracted Gaussian PFY-XANES: {scan_name}\nTarget: {target_emission_e:.1f} +/- {window_ev:.1f} eV", fontsize=10, fontweight='bold')
            ax_xanes.set_xlabel("Incident Monochromator Energy (eV)", fontsize=9)
            ax_xanes.set_ylabel("Normalized Gaussian Intensity (I/I0)", fontsize=9)
            ax_xanes.legend(fontsize=8)
            ax_xanes.grid(True, linestyle=':', alpha=0.6)

            plt.tight_layout()
            plt.show()
        except Exception as p_err:
            print(f"  [Plot Warning] Summary plot failed: {p_err}")

    return df_pfy



