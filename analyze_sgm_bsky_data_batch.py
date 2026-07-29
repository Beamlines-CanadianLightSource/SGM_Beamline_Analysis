import os
import sys
import glob
from tkinter import filedialog
from analyze_sgm_bsky_data import analyze_sgm_bsky_data
from alignment_utils import safe_filedialog_call, show_custom_dialog

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
    """
    if file_paths is None or len(file_paths) == 0:
        # Prompt user to choose between directory scanning or multi-file selection
        choice = show_custom_dialog(
            title="Batch Mode Selection",
            message="Would you like to select a directory containing all HDF5 scans?\n\n(Select 'No' to select multiple individual files instead)",
            dialog_type="yesno"
        )
        
        if choice is True:
            # Select Directory
            dir_path = safe_filedialog_call(
                filedialog.askdirectory,
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
            file_paths = safe_filedialog_call(
                filedialog.askopenfilenames,
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
                print(f"  {idx:2d}. {dp.get('scan_name', 'N/A')}")
                print(f"      File:       {os.path.basename(dp.get('h5_file_path', ''))}")
                print(f"      Grid:       {grid_str} ({dp.get('Number of images', 0)} images)")
                print(f"      Representative Energy: {dp.get('representative_energy', -1.0):.2f} eV")
                print(f"      Date:       {dp.get('date', 'N/A')}")
                print(f"      Project:    {dp.get('project', 'N/A')}")
                print("--------------------------------------------------------------------------")
            print("==========================================================================\n")
        else:
            print("No scans were successfully loaded in batch.", file=sys.stderr)

    return data_packs
