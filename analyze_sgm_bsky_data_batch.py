import os
import sys
import glob
import traceback
from html import escape
try:
    from tkinter import filedialog
except (ImportError, ModuleNotFoundError):
    filedialog = None
from analyze_sgm_bsky_data import analyze_sgm_bsky_data, browse_beamline_data
from alignment_utils import safe_filedialog_call, show_custom_dialog, HAS_GUI


def _browse_batch_scans(data_packs, verbose):
    """Display a non-blocking notebook selector and populate data_packs on load."""
    import ipywidgets as widgets
    from IPython.display import display

    selected = []
    selection = widgets.SelectMultiple(
        options=[], description="Selected:", rows=6,
        layout=widgets.Layout(width="100%"),
    )
    status = widgets.HTML(value="Add files or a folder, then click Load Batch.")
    output = widgets.Output()
    load_button = widgets.Button(description="Load Batch", button_style="success")
    remove_button = widgets.Button(description="Remove Selected")
    clear_button = widgets.Button(description="Clear Selection")
    cancel_button = widgets.Button(description="Cancel")

    def update_selection():
        selection.options = list(selected)

    def add_paths(paths):
        count = 0
        existing = set(selected)
        for path in paths:
            path = os.path.abspath(path)
            if path not in existing:
                selected.append(path)
                existing.add(path)
                count += 1
        update_selection()
        status.value = f"Added {count} file(s). {len(selected)} file(s) selected."

    def add_folder(folder):
        def raise_walk_error(error):
            raise error

        paths = []
        try:
            for root, dirs, files in os.walk(folder, onerror=raise_walk_error):
                dirs[:] = [name for name in dirs if not name.startswith(".")]
                paths.extend(os.path.join(root, name) for name in files
                             if not name.startswith(".") and name.endswith(".h5"))
        except OSError as error:
            status.value = f"<b style='color:red;'>Error reading folder: {escape(str(error))}</b>"
            return
        paths.sort()
        if not paths:
            status.value = "<b style='color:red;'>No HDF5 (.h5) files found in this folder or its subfolders.</b>"
            return
        add_paths(paths)

    def add_file(path):
        if not os.path.isfile(path):
            status.value = "<b style='color:red;'>The selected file no longer exists.</b>"
            return
        add_paths([path])

    def remove_selected(button):
        to_remove = set(selection.value)
        selected[:] = [path for path in selected if path not in to_remove]
        update_selection()
        status.value = f"{len(selected)} file(s) selected."

    def clear_selection(button):
        selected.clear()
        update_selection()
        status.value = "Selection cleared."

    def load_selected(button):
        if not selected:
            status.value = "<b style='color:red;'>Add at least one HDF5 file before loading.</b>"
            return
        load_button.disabled = True
        status.value = "Loading batch. Please wait..."
        try:
            with output:
                output.clear_output(wait=True)
                loaded = analyze_sgm_bsky_data_batch(list(selected), verbose=verbose)
            data_packs[:] = loaded
            if data_packs:
                status.value = f"Loaded {len(data_packs)} scan(s). You can now run the batch plotting cell."
            else:
                status.value = "<b style='color:red;'>No scans were loaded. Check the messages below.</b>"
        except Exception as error:
            status.value = f"<b style='color:red;'>Batch loading failed: {escape(str(error))}</b>"
            with output:
                traceback.print_exc()
        finally:
            load_button.disabled = False

    container = widgets.VBox()

    def cancel(button):
        message = ("Batch browser closed; previously loaded scans remain available."
                   if data_packs else "Batch selection cancelled; no scans loaded.")
        container.children = (widgets.HTML(message),)

    remove_button.on_click(remove_selected)
    clear_button.on_click(clear_selection)
    load_button.on_click(load_selected)
    cancel_button.on_click(cancel)
    browser = browse_beamline_data(
        on_load=add_file,
        on_select_folder=add_folder,
        title="Select HDF5 scans for batch analysis",
        file_button_label="Add Selected File",
        auto_display=False,
    )
    if browser is None:
        raise RuntimeError("The batch browser requires ipywidgets and IPython.")
    container.children = (
        widgets.HTML("Navigate to a project, add individual files or add the current folder "
                     "(including its subfolders), then click <b>Load Batch</b>."),
        browser, selection, widgets.HBox([remove_button, clear_button]),
        widgets.HBox([load_button, cancel_button]), status, output,
    )
    display(container)


def analyze_sgm_bsky_data_batch(file_paths=None, verbose=True):
    """
    Loads multiple HDF5 scans in a batch, either from a list of file paths,
    by selecting multiple files, or by selecting a directory containing scans.

    Args:
        file_paths (list of str, optional): Pre-selected HDF5 file paths.
                                            If None, prompts the user.
        verbose (bool): If True, prints a consolidated summary of all loaded scans.

    Returns:
        list of dict: A list of loaded data_pack dictionaries, sorted by scan name.
        In a headless Jupyter notebook, the returned list is initially empty and
        is populated when Load Batch is clicked in the browser.
    """
    if file_paths is None or len(file_paths) == 0:
        if not HAS_GUI:
            from IPython import get_ipython
            shell = get_ipython()
            if shell is not None and getattr(shell, "kernel", None) is not None:
                data_packs = []
                _browse_batch_scans(data_packs, verbose)
                return data_packs
        # Prompt user to choose between directory scanning or multi-file selection
        choice = show_custom_dialog(
            title="Batch Mode Selection",
            message="Would you like to select a directory containing all HDF5 scans?\n\n(Select 'No' to select multiple individual files instead)",
            dialog_type="yesno"
        )
        
        if choice is True:
            # Select Directory
            func = getattr(filedialog, 'askdirectory', 'askdirectory') if filedialog else 'askdirectory'
            dir_path = safe_filedialog_call(
                func,
                title="Select Directory containing HDF5 scans"
            )
            if not dir_path:
                if verbose:
                    print("No directory selected.", file=sys.stderr)
                return None
            
            # Find all .h5 files in the directory recursively (handling nested folder structures)
            file_paths = glob.glob(os.path.join(dir_path, "**", "*.h5"), recursive=True)
            # Filter out hidden files and jupyter checkpoints
            file_paths = [
                p for p in file_paths 
                if ".ipynb_checkpoints" not in p 
                and not os.path.basename(p).startswith(".")
            ]
            if not file_paths:
                if verbose:
                    print(f"No HDF5 (.h5) files found in directory: {dir_path}", file=sys.stderr)
                return None
        elif choice is False:
            # Select Multiple Files
            func = getattr(filedialog, 'askopenfilenames', 'askopenfilenames') if filedialog else 'askopenfilenames'
            file_paths = safe_filedialog_call(
                func,
                title="Select Multiple HDF5 Stack/Map Files",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            if not file_paths or len(file_paths) == 0:
                if verbose:
                    print("No files selected.", file=sys.stderr)
                return None
        else:
            # User closed/cancelled the dialog
            if verbose:
                print("Batch selection cancelled.", file=sys.stderr)
            return None

    # Ensure files are in a predictable, sorted order
    file_paths = sorted(list(file_paths))
    data_packs = []

    if verbose:
        print(f"\nBatch Mode: Processing {len(file_paths)} scans...")

    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}, skipping.", file=sys.stderr)
            continue
        
        try:
            # Load the single scan silently
            data_pack = analyze_sgm_bsky_data(path, verbose=False)
            if data_pack:
                data_packs.append(data_pack)
        except Exception as e:
            print(f"Error loading scan at {path}: {e}", file=sys.stderr)

    # Sort loaded scans alphabetically by scan_name
    data_packs = sorted(data_packs, key=lambda x: x.get('scan_name', ''))

    if verbose:
        if len(data_packs) > 0:
            print("\n==========================================================================")
            print(f"Loaded Batch of Scans ({len(data_packs)} scans successfully loaded):")
            print("==========================================================================")
            for idx, dp in enumerate(data_packs, 1):
                nx = dp.get('nx', 'N/A')
                ny = dp.get('ny', 'N/A')
                grid_str = f"{nx} x {ny}" if (nx != 'N/A' and ny != 'N/A') else "N/A"
                print(f"  {idx:2d}. Sample Name: {dp.get('sample_name', 'N/A')}")
                print(f"      Scan Name:  {dp.get('scan_name', 'N/A')}")
                print(f"      Grid:       {grid_str} ({dp.get('Number of images', 0)} images)")
                print(f"      Representative Energy: {dp.get('representative_energy', -1.0):.2f} eV")
                print(f"      Date:       {dp.get('date', 'N/A')}")
                print(f"      Project:    {dp.get('project', 'N/A')}")
                print("--------------------------------------------------------------------------")
            print("==========================================================================\n")
        else:
            print("No scans were successfully loaded in batch.", file=sys.stderr)

    return data_packs
