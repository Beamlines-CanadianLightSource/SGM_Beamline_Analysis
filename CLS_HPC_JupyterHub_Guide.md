# Running SGM Data Analysis on the CLS Planck HPC with JupyterHub

This guide details how to run the Canadian Light Source (CLS) Spherical Grating Monochromator (SGM) Bluesky data analysis suite and Jupyter notebooks on the **CLS Planck High-Performance Computing (HPC)** cluster using **JupyterHub**.

---

## 1. Overview & Motivation

Processing multi-dimensional SGM datasets—such as 4D RIXS/XRF stacks (Energy × SDD Channel × X × Y), large multi-detector emission series, PCA decompositions, and k-means clustering—can place substantial demands on local workstations.

Offloading computations to the **Planck HPC Cluster** provides:
* **High-Memory Compute Nodes**: 256 GB to 512 GB RAM per node, preventing out-of-memory errors on large 4D datasets.
* **Multi-Core Parallelism**: Multi-core AMD Epyc CPUs for parallel stack processing, deconvolution, and fitting.
* **High-Speed Storage**: Direct access to GPFS high-throughput filesystems (`$SCRATCH`), drastically reducing file I/O latency.
* **Interactive Notebooks**: Execution via JupyterHub, delivering the standard interactive JupyterLab interface through your web browser while running computations on dedicated HPC hardware.

---

## 2. Infrastructure & Architecture

As outlined in the CLS HPC architecture (and presented in the CLS HPC In-House Workshop):

```
       [ User Web Browser ]
                 │
                 │ (CLS SSO / Entra ID OAuth2)
                 ▼
     [ Configurable HTTP Proxy ]
                 │
                 ▼
           [ JupyterHub ]
                 │
                 ▼ (via sudo + Active Directory / LDAP & Munge)
       [ Slurm Form Spawner ]
                 │
                 ▼ (sbatch / salloc)
    [ Planck Head Node: planck.clsi.ca ]
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
[ Compute Node ]    [ Compute Node ]
 (plancke01..)       (planckw01..)
       │
       └── Runs JupyterLab Server inside Slurm job
       └── Direct access to $HOME, $SCRATCH, and Beamline Storage
```

* **Authentication**: Integrated with CLS Active Directory / Entra ID single sign-on.
* **Job Scheduler**: Managed by **Slurm**. Compute jobs run exclusively on compute nodes (`plancke*`, `planckw*`), keeping the login head node (`planck.clsi.ca`) responsive.
* **Spawner**: **Slurm Form Spawner** (a subclass of `batchspawner`) provides a web form to select compute resources (CPUs, RAM, wall time, GPU) before spawning your notebook container or environment.

---

## 3. Account & Network Prerequisites

1. **Planck HPC Account**:
   * If you are a new user to Planck, submit a **Jira ticket** to the Scientific Computing / IT group to grant your CLS user account permissions to submit Slurm jobs.
2. **Network Connection**:
   * You must be either directly connected to the CLS internal network (on-site) or connected via the **CLS GlobalProtect VPN**.
3. **Storage Quotas**:
   * `$HOME`: Fixed quota for scripts, configuration files, and custom environments.
   * `$SCRATCH`: High-speed GPFS scratch space with large quotas, recommended for active data files, stack generation, and temporary outputs.

---

## 4. Method 1: Web-Based JupyterHub Portal (Recommended)

When using the CLS JupyterHub web interface:

### Step 1: Log In
1. Open your browser and navigate to the CLS JupyterHub portal (e.g., `https://jupyter.clsi.ca` or the internal URL provided by the HPC team).
2. Authenticate using your standard CLS username and password.

### Step 2: Select Compute Resources (Slurm Form Spawner)
In the resource request form:
* **Partition**: `batch` (or `gpu` if utilizing accelerated machine learning workflows).
* **Number of Cores / CPUs**: `4` to `8` cores (sufficient for parallelizing detector summation and stack fits).
* **Memory**: `16 GB` to `64 GB` (depending on whether you are analyzing single scans or full 4D maps).
* **Time**: e.g., `04:00:00` (4 hours) or the duration of your analysis session.
* Click **Start**. JupyterHub will queue and start a Slurm job on a compute node, then redirect your browser into a full JupyterLab workspace.

---

## 5. Part 1: Initial One-Time Setup on Planck (First-Time Only)

> [!IMPORTANT]
> **Run this section ONLY ONCE** when setting up your account on Planck for the first time.
> Once the virtual environment and Jupyter kernel are registered and the repository is cloned, you do **not** need to re-run this setup for future sessions.

### 5.1 Open a Terminal on Planck
Open a terminal directly inside **JupyterLab** (File → New → Terminal) or connect via SSH from your local workstation:
```bash
ssh <username>@planck.clsi.ca
```
*(Optional: if you need to run compute-intensive terminal commands, you can request an interactive Slurm allocation with `salloc --cpus-per-task=4 --mem=16G --time=02:00:00`)*.

### 5.2 Load Modules and Build the Virtual Environment
Planck uses the Compute Canada / Alliance Lmod software stack:

```bash
# 1. Load the modern environment and Python
module load StdEnv/2023
module load python/3.13  # or python/3.11
module list              # inspect loaded modules

# 2. Create the virtual environment in your home directory (ONCE ONLY)
mkdir -p ~/envs
python -m venv ~/envs/sgm_analysis

# 3. Activate the environment
source ~/envs/sgm_analysis/bin/activate

# 4. Upgrade pip and install standard scientific packages
pip install --upgrade pip
pip install numpy scipy matplotlib h5py pandas scikit-image sklearn

# 5. Install Bluesky, Databroker, PyMca, and UI dependencies
pip install bluesky databroker
pip install PyMca5
pip install mplcursors ipykernel
```

### 5.3 Register the Kernel for JupyterHub
Register the virtual environment with JupyterHub so it appears permanently in your kernel dropdown:
```bash
python -m ipykernel install --user --name=sgm_analysis --display-name="Python (SGM Analysis)"
```

### 5.4 Initial Clone of the Repository
Clone the latest `SGM_Beamline_Analysis` code into your home directory:
```bash
cd ~
git clone https://github.com/Beamlines-CanadianLightSource/SGM_Beamline_Analysis.git
```

---

## 6. Part 2: Returning Workflow & Everyday Analysis (Already Initialized)

> [!TIP]
> **Use this section for your everyday analysis sessions!**
> You do **NOT** need to create the virtual environment (`venv`) or reinstall packages (`numpy`, `scipy`, etc.). Your **`Python (SGM Analysis)`** kernel is already configured and ready to use.

### Step 1: Start Your JupyterHub Session
1. Navigate to `https://jupyter.clsi.ca` in your web browser.
2. Authenticate using your CLS Active Directory username and password.
3. In the **Slurm Form Spawner**, choose your compute resources:
   * **Partition**: `batch`
   * **Cores / CPUs**: `4` to `8` cores
   * **Memory**: `16 GB` to `32 GB` (or up to `64 GB` for large 4D RIXS datasets)
   * **Time**: e.g., `04:00:00` (4 hours)
4. Click **Start**. JupyterHub will allocate a Slurm compute node and open your JupyterLab workspace.

### Step 2: Pull the Latest Code Updates (1 Command)
Whenever new features or bug fixes are pushed to GitHub, simply open a terminal in JupyterLab (File → New → Terminal) and run:
```bash
cd ~/SGM_Beamline_Analysis
git pull origin main
```
*(Note: You only need to `cd ~/SGM_Beamline_Analysis` once, followed by `git pull origin main`).*

### Step 3: Open Notebook & Select the Kernel
1. In the JupyterLab file browser on the left, navigate into `SGM_Beamline_Analysis/`.
2. Open [`SGM_BSky_Data_Analysis.ipynb`](file:///c:/Users/dynesj/PycharmProjects/PythonProject/Gemini_CLI2/SGM_Beamline_Analysis/SGM_BSky_Data_Analysis.ipynb) (or [`SGM_BSky_Data_Analysis-4quad.ipynb`](file:///c:/Users/dynesj/PycharmProjects/PythonProject/Gemini_CLI2/SGM_Beamline_Analysis/SGM_BSky_Data_Analysis-4quad.ipynb)).
3. In the top-right corner of the notebook, make sure the kernel is set to **`Python (SGM Analysis)`**.

### Step 4: Browse and Load Your Beamline Scan
In [`SGM_BSky_Data_Analysis.ipynb`](file:///c:/Users/dynesj/PycharmProjects/PythonProject/Gemini_CLI2/SGM_Beamline_Analysis/SGM_BSky_Data_Analysis.ipynb), run the interactive file browser cell:

```python
from analyze_sgm_bsky_data import analyze_sgm_bsky_data, browse_beamline_data

# Launch the interactive Loki/GPFS file explorer:
browse_beamline_data()
```
* Project folders are automatically sorted naturally by cycle number (`prj43...`, `prj42...`, `prj41...`).
* Select your project folder, select your `.h5` file, and click **Select File**.
* The scan is immediately parsed into `sgm_data`.

*(Alternative: You can also specify direct file paths with `sgm_data = analyze_sgm_bsky_data("/gpfs/Primary/beamlinedata/sgm/.../scan.h5")`).*

### Step 5: Execute Dashboard & Export Data
Run the `plot_sgm_bsky_data(...)` cell:
* **Interactive $I_0$ Preview**: Select internal Au mesh or an external $I_0$ file. The notebook overlays original and modified curves as you adjust smoothing, constant arithmetic, optical density division, or the external energy shift. Click **Apply** to use the changes or **Cancel** to use the original; the dashboard appears after your choice.
* **Native Jupyter Export Toolbar**: A row of export buttons is embedded directly above the plots in the notebook:
  - **`💾 Save XANES (CSV)`**: Exports active ROI XANES spectra.
  - **`💾 Save XRF (CSV)`**: Exports active ROI Consolidated XRF/XRD spectra.
  - **`🖼️ Save All Images`**: Opens an editable destination in the notebook, defaulting to a scan-specific `Images` folder in preprocessed storage. Click **Save Images** to export detector maps and histograms or **Cancel** to write nothing.
  - **`📦 Save 4D XRF (PyMca)`**: Exports full 4D stack for elemental analysis in PyMca.
  - **`📊 Save 3D XANES (PCA)`**: Exports 3D normalized stack for PCA/CA.
* **Anti-Overwrite Protection**: All export operations verify if a file with the same name already exists. In headless mode, filenames automatically increment (`_1`, `_2`, `_3`, etc.) so that **your previously saved data is never overwritten**.

### Batch analysis on HPC
1. Open `SGM_BSky_Data_Analysis-batch.ipynb` and run the setup and batch-loading cells. In the browser, navigate to your project, add individual HDF5 files or **Add Current Folder** to include its subfolders, then click **Load Batch**. Wait for the loaded-scan summary before running the plotting cell.
2. The plotting setup shows saved SDD calibration details and a representative spectrum labeled by sample, scan, and detector. Choose whether to apply calibration and select internal or external $I_0$. For internal $I_0$, inspect individual scans in the preview; the same processing settings are applied to each scan's own curve. Click **Apply** to use the modifications or **Cancel** to use the original $I_0$.
3. Click **Save XANES Spectra CSVs** to export. For scans under `<project>/raw/<date>/<scan>/`, files are saved in the matching `<project>/preprocessed/<date>/<scan>/` folder, created if needed. The full paths appear below the button; existing files receive numbered suffixes.

For details, see the batch section of [SGM_BSky_Data_Analysis_Guide.md](SGM_BSky_Data_Analysis_Guide.md).

---

## 7. Method 2: Slurm Batch Job with SSH Tunneling (Fallback)

If the web-based JupyterHub portal is undergoing maintenance, you can start your own dedicated JupyterLab server via an `sbatch` script and tunnel the connection to your local browser.

### Step 1: Create a Slurm Submission Script (`start_jupyter.sh`)
Create this file in your home directory on Planck:

```bash
#!/bin/bash
#SBATCH --job-name=jupyter_sgm
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jupyter_%j.log

# Load modules and activate the virtual environment
module load StdEnv/2023 python/3.10
source ~/envs/sgm_analysis/bin/activate

# Detect allocated node and assign an open port
NODE_NAME=$(hostname)
PORT=8888

echo "=================================================================="
echo "Jupyter server running on node: $NODE_NAME"
echo "To connect from your local PC, run the following SSH tunnel command:"
echo ""
echo "  ssh -N -L ${PORT}:${NODE_NAME}:${PORT} ${USER}@planck.clsi.ca"
echo ""
echo "Then navigate to http://localhost:${PORT} in your web browser."
echo "=================================================================="

# Start JupyterLab without opening a local browser window
jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0
```

### Step 2: Submit the Job
On `planck.clsi.ca`:
```bash
sbatch start_jupyter.sh
```

### Step 3: Check the Output Log for the Connection Details
```bash
cat jupyter_*.log
```
The log will show the exact node (e.g. `plancke02`) and the tokenized URL:
```text
http://127.0.0.1:8888/lab?token=abcdef123456...
```

### Step 4: Open the Tunnel on Your Local PC
Open PowerShell on your Windows workstation and execute:
```powershell
ssh -N -L 8888:plancke02:8888 <username>@planck.clsi.ca
```
Then paste the tokenized link (`http://localhost:8888/lab?token=...`) into your browser.

---

## 8. Best Practices for HPC Workflow

1. **Use `$SCRATCH` for Intermediate Files**:
   * Never write multi-gigabyte HDF5 stacks directly to `$HOME`, which has limited quota and lower throughput. Use your `$SCRATCH` directory (`/home/<username>/scratch/`) for caching, stack generation, and temporary plots.
2. **Monitor Queues & Allocations**:
   * Run `squeue -u $USER` in the terminal to inspect the status of your running job.
3. **Shut Down Sessions When Done**:
   * In JupyterHub: Click **File → Hub Control Panel → Stop My Server**.
   * Releasing the Slurm allocation returns the compute cores to the cluster queue for other beamline scientists.

---

## 9. Contacts & Resources

* **Scientific Computing Support**: `#st-general` internal channel or CLS IT Helpdesk / Jira.
* **Planck HPC Documentation**: [Planck HPC for Windows Users](file:///C:/Users/dynesj/Downloads/Planck+HPC+for+Windows+Users.doc)
* **JHub-HPC Interface Reference**: [GitHub Repository: pc2/JHub-HPC-Interface](https://github.com/pc2/JHub-HPC-Interface)
* **SGM Analysis Repository**: [Beamlines-CanadianLightSource/SGM_Beamline_Analysis](https://github.com/Beamlines-CanadianLightSource/SGM_Beamline_Analysis)
