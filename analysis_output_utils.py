from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import h5py


def create_analysis_run(h5_path, analysis_name, parameters):
    """Create an independent HDF5 copy in a new, exclusively reserved run folder."""
    source = Path(h5_path).resolve()
    parameters_json = json.dumps(parameters)
    with h5py.File(source, "r"):
        pass

    scan_dir = source.parent
    if (scan_dir.parent.name in ("PCA Analysis", "Cluster Analysis")
            and re.fullmatch(r"Run_\d+", scan_dir.name)):
        scan_dir = scan_dir.parent.parent
    analysis_dir = scan_dir / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    run_number = 1
    while True:
        run_dir = analysis_dir / f"Run_{run_number:03d}"
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            run_number += 1

    destination = run_dir / source.name
    print(f"\n[Analysis Run] Copying input HDF5 to: {destination}")
    try:
        shutil.copyfile(source, destination)
        with h5py.File(destination, "r+") as handle:
            handle.attrs["analysis_type"] = analysis_name
            handle.attrs["analysis_source_h5"] = str(source)
            handle.attrs["analysis_created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            handle.attrs["analysis_parameters"] = parameters_json
    except (OSError, ValueError):
        destination.unlink(missing_ok=True)
        run_dir.rmdir()
        raise
    print(f"[Analysis Run] Images, CSVs, and analysis HDF5 will be saved in: {run_dir}")
    return str(destination)
