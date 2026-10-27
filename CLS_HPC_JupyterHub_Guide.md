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

## 5. Setting Up the SGM Analysis Environment on Planck

To run [`SGM_BSky_Data_Analysis.ipynb`](file:///c:/Users/dynesj/PycharmProjects/PythonProject/Gemini_CLI2/SGM_Beamline_Analysis/SGM_BSky_Data_Analysis.ipynb), you need a Python environment containing the required beamline libraries, registered as a Jupyter kernel.

### 5.1 Open a Terminal on Planck
You can open a terminal directly inside **JupyterLab** (File → New → Terminal) or connect via SSH:
```bash
ssh <username>@planck.clsi.ca
```

### 5.2 Load Modules and Build the Virtual Environment
Planck uses the Compute Canada / Alliance Lmod software stack:

```bash
# 1. Load an updated environment and Python
module load StdEnv/2023
module load python/3.10  # or python/3.11

# 2. Create a virtual environment in your home directory
mkdir -p ~/envs
python -m venv ~/envs/sgm_analysis

# 3. Activate the environment
source ~/envs/sgm_analysis/bin/activate

# 4. Upgrade pip and install standard scientific packages
pip install --upgrade pip
pip install numpy scipy matplotlib h5py pandas

# 5. Install Bluesky / Databroker & PyMca (if needed for stack analysis)
pip install bluesky databroker
pip install PyMca5

# 6. Install ipykernel and register the kernel for JupyterHub
pip install ipykernel
python -m ipykernel install --user --name=sgm_analysis --display-name="Python (SGM Analysis)"
```

### 5.3 Verify the Kernel in JupyterHub
* Refresh your JupyterLab page in your browser.
* In the launcher or top-right kernel dropdown, select **`Python (SGM Analysis)`**.
* Any notebook running with this kernel will now have access to all installed libraries.

---

## 6. Deploying the Analysis Code on Planck

Clone the latest `SGM_Beamline_Analysis` repository directly into your Planck `$HOME` or scientific project workspace:

```bash
cd $HOME
git clone https://github.com/Beamlines-CanadianLightSource/SGM_Beamline_Analysis.git
cd SGM_Beamline_Analysis
```

### Adjusting Notebook Data Paths for Planck
In [`SGM_BSky_Data_Analysis.ipynb`](file:///c:/Users/dynesj/PycharmProjects/PythonProject/Gemini_CLI2/SGM_Beamline_Analysis/SGM_BSky_Data_Analysis.ipynb):
* Update the scan catalog location or raw data directory path:
  ```python
  # Local workstation example:
  # data_dir = r"C:\Data\SGM_Scans"

  # Planck HPC example:
  data_dir = "/home/<username>/scratch/sgm_data"
  # or direct mount to beamline raw data:
  # data_dir = "/beamline/sgm/data/2026/..."
  ```
* Save generated files (such as PyMca 4D stacks or large 2D RIXS exports) to `$SCRATCH`:
  ```python
  output_stack_path = "/home/<username>/scratch/output_stack.h5"
  ```

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
