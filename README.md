# 🖥️ HPC ML Setup — Digital Research Alliance Clusters

> **Clean, reproducible ML environment setup for Narval, Cedar, Fir, Béluga, and other Alliance clusters.**
> Install once. Run everywhere: JupyterHub, `salloc`, `sbatch`, GPU nodes.

---

## 📋 Table of Contents

- [First Login Checklist](#-first-login-checklist)
- [Module Reference](#-module-reference)
- [Required Modules Before Installation](#-required-modules-before-installation)
- [1. Create a Permanent Virtual Environment](#1️⃣-create-one-permanent-virtual-environment)
- [2. JupyterHub Kernel Setup](#2️⃣-make-it-usable-in-jupyterhub)
- [3. Interactive Jobs (salloc)](#3️⃣-use-the-environment-in-interactive-jobs)
- [4. Batch Jobs (sbatch)](#4️⃣-use-the-environment-in-batch-jobs-best-practice)
- [5. Monitoring Job Output & Training Logs](#5️⃣-monitoring-job-output--training-logs)
- [6. Building a Cluster-Optimized requirements_cc.txt](#6️⃣-building-a-cluster-optimized-requirements_cctxt)
- [7. Directory Structure](#7️⃣-best-directory-structure-for-research)
- [8. Useful Commands](#8️⃣-useful-commands)
- [9. Pre-built Wheels Trick](#️-one-extremely-useful-trick-for-alliance-clusters)
- [10. Typical Workflow](#-your-typical-workflow)
- [Requirements Template](#-requirements-template)
- [Advanced Tips](#-advanced-tips)

---

## ✅ First Login Checklist

Every time you SSH into the cluster, run this sequence before doing anything:

```bash
# 1. Check your available storage quotas
quota -s

# 2. Purge any pre-loaded default modules
module purge

# 3. Load your standard environment
module load CCconfig
module load StdEnv/2023 python/3.11.5

# 4. Activate your persistent virtual environment
source ~/envs/hpc_ml_env/bin/activate

# 5. Verify GPU/CUDA availability (if on a GPU node)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> 💡 **Tip:** Add the `module purge`, `module load`, and `source` lines to your `~/.bashrc` **only if** you always want the same environment on login. Otherwise, load manually to avoid conflicts.

---

## 📦 Module Reference

Alliance clusters use the **Lmod** module system. The table below lists all commonly used modules for ML workloads on Alliance clusters.

| Module | Category | Purpose |
|---|---|---|
| `CCconfig` | Base | Compute Canada base configuration — always load first |
| `gentoo/2023` | Base | OS compatibility layer for the Alliance software stack |
| `StdEnv/2023` | Base | Standard software environment — load after CCconfig |
| `gcccore/.12.3` | Compiler | GCC core libraries (auto-loaded as dependency) |
| `gcc/12.3` | Compiler | GCC C/C++/Fortran compiler suite |
| `python/3.11.5` | Language | Python 3.11 interpreter |
| `java/17.0.6` | Language | Java 17 LTS runtime (specific build) |
| `java/17` | Language | Java 17 LTS runtime (general alias) |
| `r/4.5.0` | Language | R statistical computing environment |
| `flexiblascore/.3.3.1` | Math | FlexiBLAS core (auto-loaded as dependency) |
| `flexiblas/3.3.1` | Math | Flexible BLAS wrapper for optimized linear algebra |
| `aocl-blas/5.1` | Math | AMD-optimized BLAS (used on AMD CPU nodes) |
| `aocl-lapack/5.1` | Math | AMD-optimized LAPACK (used on AMD CPU nodes) |
| `scipy-stack/2026a` | Science | Prebuilt NumPy, SciPy, Matplotlib, Pandas |
| `ipykernel/2026a` | Jupyter | Jupyter kernel support (alternative to pip install) |
| `arrow/23.0.1` | Data | Apache Arrow for fast columnar data processing |
| `opencv/4.13.0` | Vision | OpenCV computer vision library |
| `hwloc/2.9.1` | HPC | Hardware locality — CPU/memory topology for parallel jobs |
| `ucx/1.14.1` | HPC | Unified Communication X — high-speed networking layer |
| `libfabric/1.18.0` | HPC | Low-level network fabric (InfiniBand, Ethernet) |
| `pmix/4.2.4` | HPC | Process management interface for MPI jobs |
| `ucc/1.2.0` | HPC | Unified Collective Communication (used with OpenMPI) |
| `openmpi/4.1.5` | HPC | MPI for distributed/multi-node jobs |

Check all available modules:

```bash
module avail               # list all available modules
module spider python       # search for a specific module
module spider cuda         # search for CUDA modules
```

Load CUDA explicitly for GPU jobs if needed:

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5
```

---

## 🧩 Required Modules Before Installation

Before creating a virtual environment or installing any Python packages, you **must** load the correct set of modules. Loading these ensures the compiler toolchain, optimized linear algebra libraries, communication libraries, and Python interpreter all match what the cluster expects.

### Full Module List (All Available)

```bash
module purge
module load CCconfig
module load gentoo/2023
module load StdEnv/2023
module load gcccore/.12.3
module load gcc/12.3
module load python/3.11.5
module load java/17.0.6
module load java/17
module load flexiblascore/.3.3.1
module load flexiblas/3.3.1
module load aocl-blas/5.1
module load aocl-lapack/5.1
module load scipy-stack/2026a
module load ipykernel/2026a
module load r/4.5.0
module load hwloc/2.9.1
module load ucx/1.14.1
module load libfabric/1.18.0
module load pmix/4.2.4
module load ucc/1.2.0
module load openmpi/4.1.5
module load opencv/4.13.0
module load arrow/23.0.1
```

> ⚠️ **You do not need to load all of these every time.** Load only what your project needs. Many lower-level modules (`gcccore`, `flexiblascore`, `hwloc`, `ucx`, `libfabric`, `pmix`, `ucc`) are auto-loaded as dependencies when you load higher-level modules like `openmpi` or `scipy-stack`. Use the stacks below as your starting point.

---

### What Each Module Provides

| Module | Role |
|---|---|
| `CCconfig` | Compute Canada base config — always load first |
| `gentoo/2023` | Base OS layer used by the Alliance software stack |
| `StdEnv/2023` | Core standard environment — sets compiler and toolchain defaults |
| `gcccore/.12.3` | GCC core libraries, auto-loaded as dependency by most modules |
| `gcc/12.3` | Full GCC compiler suite — required before building or installing compiled packages |
| `python/3.11.5` | Python 3.11 interpreter — sets the baseline for your virtual environment |
| `java/17.0.6` | Specific Java 17 LTS build — needed for Spark, some data pipeline tools |
| `java/17` | Java 17 general alias — interchangeable with `java/17.0.6` in most cases |
| `r/4.5.0` | R statistical environment — useful for mixed Python/R workflows |
| `flexiblascore/.3.3.1` | FlexiBLAS core libraries, auto-loaded as dependency |
| `flexiblas/3.3.1` | Flexible BLAS wrapper — routes BLAS calls to best available backend |
| `aocl-blas/5.1` | AMD-optimized BLAS — used automatically on AMD CPU nodes for faster linear algebra |
| `aocl-lapack/5.1` | AMD-optimized LAPACK — complements `aocl-blas` for decomposition routines |
| `scipy-stack/2026a` | Prebuilt, cluster-optimized NumPy, SciPy, Matplotlib, Pandas — do not reinstall via pip |
| `ipykernel/2026a` | Jupyter kernel support — can use instead of `pip install ipykernel` |
| `hwloc/2.9.1` | Hardware locality library — detects CPU/memory topology for parallel scheduling |
| `ucx/1.14.1` | Unified Communication X — high-performance networking layer (InfiniBand, RDMA) |
| `libfabric/1.18.0` | Low-level network fabric abstraction for InfiniBand and Ethernet |
| `pmix/4.2.4` | Process Management Interface — coordinates process startup in MPI jobs |
| `ucc/1.2.0` | Unified Collective Communication — optimizes collective MPI operations |
| `openmpi/4.1.5` | Full MPI implementation — required for distributed multi-node training |
| `opencv/4.13.0` | Computer vision library — image I/O, transforms, video processing |
| `arrow/23.0.1` | Apache Arrow — fast columnar data processing and Parquet file support |

---

### Minimal Baseline (Most ML Projects)

```bash
module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load scipy-stack/2026a
```

### Extended Stack (Deep Learning + Distributed + Vision)

```bash
module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load scipy-stack/2026a
module load ipykernel/2026a
module load openmpi/4.1.5
module load arrow/23.0.1
module load opencv/4.13.0
```

### Save Your Module Stack as a Script

Save your module loading commands in a shell script for easy reuse across sessions and job scripts:

```bash
# file: load_modules.sh
#!/bin/bash
module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load scipy-stack/2026a
module load ipykernel/2026a
module load openmpi/4.1.5
module load arrow/23.0.1
module load opencv/4.13.0
echo "✅ Modules loaded successfully"
module list
```

Source it any time:

```bash
source load_modules.sh
```

---

## 1️⃣ Create ONE Permanent Virtual Environment

Do this **once** after your first login. The environment lives in `~/envs/` permanently.

```bash
module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load scipy-stack/2026a

python -m venv --no-download ~/envs/hpc_ml_env
source ~/envs/hpc_ml_env/bin/activate
```

Upgrade pip and install your packages from `requirements.txt`:

```bash
pip install --upgrade pip --no-index
pip install --no-index -r requirements.txt
```

> ⚠️ **Always use `--no-index`** when possible. This forces pip to use the cluster's pre-compiled wheel cache instead of downloading from PyPI, which is significantly faster and more stable on HPC nodes.

> 💡 The `--no-download` flag when creating the venv prevents pip from downloading anything during venv creation — it uses the system pip instead.

If a package isn't available as a cluster wheel, install from PyPI (requires internet on login nodes):

```bash
pip install some-package   # without --no-index, falls back to PyPI
```

Your environment now lives permanently at:

```
~/envs/hpc_ml_env/
```

---

## 2️⃣ Make it Usable in JupyterHub

Run once to register your environment as a Jupyter kernel:

```bash
source ~/envs/hpc_ml_env/bin/activate
pip install ipykernel --no-index
python -m ipykernel install --user --name hpc_ml_env --display-name "Python (hpc_ml_env)"
```

Alternatively, use the preloaded module instead of pip:

```bash
module load ipykernel/2026a
python -m ipykernel install --user --name hpc_ml_env --display-name "Python (hpc_ml_env)"
```

When you open JupyterHub, select the kernel named:

```
Python (hpc_ml_env)
```

To remove a kernel later:

```bash
jupyter kernelspec remove hpc_ml_env
```

---

## 3️⃣ Use the Environment in Interactive Jobs

Request an interactive GPU session with `salloc`:

```bash
salloc --time=02:00:00 \
       --cpus-per-task=4 \
       --mem=16G \
       --gres=gpu:1 \
       --account=def-supervisor
```

Once the node is allocated, load your environment:

```bash
source load_modules.sh   # or load manually
source ~/envs/hpc_ml_env/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Run your script
python train.py
```

> 💡 Interactive sessions are ideal for **debugging, prototyping, and quick experiments**. For long training runs, use `sbatch`.

---

## 4️⃣ Use the Environment in Batch Jobs (Best Practice)

Create a file called `train_job.sh`:

```bash
#!/bin/bash
#SBATCH --account=def-supervisor
#SBATCH --job-name=ml_train
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.ca

# ── Environment Setup ──────────────────────────────────────────────
module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load scipy-stack/2026a

source ~/envs/hpc_ml_env/bin/activate

# ── Verify GPU ─────────────────────────────────────────────────────
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

# ── Run Experiment ─────────────────────────────────────────────────
cd ~/projects/my_project

python -u train.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --output_dir $SCRATCH/checkpoints/run_$(date +%Y%m%d_%H%M%S)
```

> 💡 The `-u` flag on `python -u train.py` disables output buffering so `print()` statements appear in the log file immediately rather than waiting for the buffer to flush.

Create the logs directory first:

```bash
mkdir -p ~/projects/my_project/logs
```

Submit the job:

```bash
sbatch train_job.sh
```

Monitor it:

```bash
sq                        # your running/pending jobs
squeue -u $USER           # same with username
scontrol show job JOBID   # full job details
```

---

## 5️⃣ Monitoring Job Output & Training Logs

On Slurm clusters, all `print()` statements, training logs, errors, and warnings go to the **job output file**. You can monitor them live while the job runs or inspect them after it finishes.

---

### Default Output File

When you submit with `sbatch`, Slurm automatically writes **stdout and stderr** to:

```
slurm-<JOBID>.out
```

This file contains everything your script prints — training metrics, errors, warnings, and any other output. Unless you specify custom filenames in your job script, this is where everything lands.

---

### Watch Training Live (Most Useful) ⭐

Stream the output in real time while the job is running:

```bash
tail -f slurm-<JOBID>.out
```

You will see output update live as the job writes it:

```
Epoch 1/50 — loss: 0.3241 — acc: 0.8712
Epoch 2/50 — loss: 0.2893 — acc: 0.8904
Epoch 3/50 — loss: 0.2571 — acc: 0.9031
```

Stop the stream with `CTRL + C`. The job continues running — you are only detaching from the live view.

To follow all output files matching a pattern:

```bash
tail -f slurm-*.out
```

---

### Read the Output File After the Job Ends

```bash
# Scrollable view
less slurm-<JOBID>.out

# Print entire file to terminal
cat slurm-<JOBID>.out

# Show last 50 lines
tail -n 50 slurm-<JOBID>.out
```

---

### Find the Job ID If You Don't Have It

```bash
squeue --me          # jobs currently running or pending
ls slurm-*.out       # list all output files in current directory
```

---

### Use Named Output Files (Recommended)

Control the output filenames in your job script with `%j` (auto-inserts the job ID):

```bash
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
```

This produces clean, organized files:

```
logs/train-482193.out
logs/train-482193.err
```

Separating stdout and stderr makes it much easier to spot errors without scrolling through training output.

---

### Fix: Prints Not Appearing Until Job Ends

Python **buffers output by default**, meaning `print()` calls may not appear in the log file immediately — they wait until the buffer flushes or the program exits.

**Fix 1** — Use the `-u` flag when calling Python (recommended in job scripts):

```bash
python -u train.py
```

**Fix 2** — Force flush on individual print calls:

```python
print(f"Epoch {epoch} complete — loss: {loss:.4f}", flush=True)
```

**Fix 3** — Set the environment variable in your job script:

```bash
export PYTHONUNBUFFERED=1
python train.py
```

---

### Log Metrics to a Separate File

For cleaner monitoring, write training metrics to a dedicated log file and watch it independently:

```python
import logging

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s — %(message)s"
)

logging.info(f"Epoch {epoch} — loss: {loss:.4f} — acc: {acc:.4f}")
```

Then watch it:

```bash
tail -f training.log
```

---

### Monitoring Quick Reference

| Task | Command |
|---|---|
| Watch job live | `tail -f slurm-<JOBID>.out` |
| Watch all output files | `tail -f slurm-*.out` |
| Read after job ends | `less slurm-<JOBID>.out` |
| Show last N lines | `tail -n 50 slurm-<JOBID>.out` |
| Find your job ID | `squeue --me` |
| List output files | `ls slurm-*.out` |
| Fix buffered output | `python -u train.py` or `flush=True` |

---

## 6️⃣ Building a Cluster-Optimized `requirements_cc.txt`

The Alliance wheelhouse provides pre-compiled, cluster-optimized builds of most popular packages (tagged `+computecanada`). Using them makes your installs faster, more stable, and free of network dependency inside job nodes. This section shows you how to convert your standard `requirements.txt` into one that is fully backed by the Alliance wheelhouse — with a safe fallback for packages that aren't available there.

> 💡 Do all steps in this section **on the login node**, which has internet access. Compute nodes do not.

---

### The Big Picture

```
requirements.txt          (your original, PyPI-style)
        │
        ▼
  avail_wheels check
        │
   ┌────┴────────────────────┐
   │                         │
✅ Available in              ❌ NOT in Alliance
   Alliance wheelhouse           wheelhouse
   │                         │
   pip install --no-index    uv pip download → $HOME/wheelhouse/
   │                         │
   └────────────┬────────────┘
                │
          pip freeze
                │
                ▼
        requirements_cc.txt   (pinned, cluster-ready)
                │
                ▼
         used in job scripts
```

---

### Step 1 — Identify What's Available

Use `avail_wheels` to check your entire `requirements.txt` against the Alliance wheelhouse at once:

```bash
module load python/3.11.5
avail_wheels -r requirements.txt
```

To see **only the packages that are missing** (not available as `+computecanada` wheels):

```bash
avail_wheels -r requirements.txt --not-available
```

This tells you exactly which packages need the custom wheelhouse fallback — no guessing required.

---

### Step 2 — Build a Temporary Environment from the Alliance Wheelhouse

Create a short-lived temp environment and install everything that **is** available:

```bash
module load python/3.11.5

ENVDIR=/tmp/$RANDOM
virtualenv --no-download "$ENVDIR"
source "$ENVDIR/bin/activate"
pip install --no-index --upgrade pip

# Temporarily remove or comment out packages NOT in the wheelhouse
# then run:
pip install --no-index -r requirements.txt
```

> `--no-index` forces pip to look **only** at the Alliance wheelhouse (`/cvmfs/...`). Any package not found there will raise an error — this is expected and useful. It tells you exactly which packages need to be handled in the next step.

---

### Step 3 — Handle Packages Not in the Alliance Wheelhouse

For each package that errored in Step 2, download it from PyPI into your own local wheelhouse:

```bash
mkdir -p $HOME/wheelhouse/my_project

# Download the missing package (with uv for speed)
uv pip download "somepackage==X.Y.Z" -d $HOME/wheelhouse/my_project
```

Then install it into the same temp environment:

```bash
pip install --find-links=$HOME/wheelhouse/my_project --no-index "somepackage==X.Y.Z"
```

#### Handling Dependency Conflicts (`--no-deps`)

Some packages declare version constraints on their dependencies that conflict with what the Alliance wheelhouse provides (e.g., requiring `torch<2.0` when the cluster only has `torch 2.x`). In this case, install the package **without its dependency tree**:

```bash
pip install --find-links=$HOME/wheelhouse/my_project --no-index somepackage --no-deps
```

`--no-deps` installs only the package itself, skipping dependency resolution entirely. Your environment already has the correct cluster-optimized versions of common libraries like `torch`, so this is safe in practice. The package may emit a runtime warning about version mismatches, but it will usually function correctly.

---

### Step 4 — Freeze Into `requirements_cc.txt`

Once your temp environment has everything installed — both Alliance wheels and custom packages — freeze it:

```bash
pip freeze --local > requirements_cc.txt
deactivate
rm -rf "$ENVDIR"
```

Your `requirements_cc.txt` will now contain a mix of pinned entries:

```text
numpy==2.4.2+computecanada        # ✅ from Alliance wheelhouse
pandas==3.0.0+computecanada       # ✅ from Alliance wheelhouse
scikit-learn==1.5.0+computecanada # ✅ from Alliance wheelhouse
somepackage==X.Y.Z                # 📦 from your custom wheelhouse
```

Commit this file to your project repository alongside your original `requirements.txt`.

---

### Step 5 — Use `requirements_cc.txt` in Your Job Script

Your job script can now build a fresh environment quickly on the compute node, entirely from local disk:

```bash
#!/bin/bash
#SBATCH --account=def-supervisor
#SBATCH --job-name=ml_train
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load scipy-stack/2026a

# ── Build fresh env on fast local node storage ─────────────────────
VENV_DIR="$SLURM_TMPDIR/job_env"
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip

# Install Alliance-available packages (fast, no network, optimized)
pip install --no-index -r requirements_cc.txt

# Install any remaining custom packages from your local wheelhouse
pip install --find-links=$HOME/wheelhouse/my_project --no-index -r requirements_custom.txt

# ── Run Experiment ─────────────────────────────────────────────────
cd ~/projects/my_project
python -u train.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --output_dir $SCRATCH/checkpoints/run_$(date +%Y%m%d_%H%M%S)
```

> 💡 `$SLURM_TMPDIR` is a fast, local SSD scratch directory allocated per job on the compute node. Building the venv there makes installs and imports significantly faster than using `$HOME` or `$SCRATCH`.

---

### Workflow Summary

| Step | Location | Command |
|---|---|---|
| Check availability | Login node | `avail_wheels -r requirements.txt --not-available` |
| Create temp env | Login node | `virtualenv --no-download /tmp/$RANDOM` |
| Install Alliance packages | Login node | `pip install --no-index -r requirements.txt` |
| Download missing packages | Login node | `uv pip download pkg -d $HOME/wheelhouse/` |
| Install missing packages | Login node | `pip install --find-links=... --no-index pkg` |
| Handle dep conflicts | Login node | `pip install ... --no-deps pkg` |
| Freeze | Login node | `pip freeze --local > requirements_cc.txt` |
| Build env in job | Compute node | `pip install --no-index -r requirements_cc.txt` |

---

## 7️⃣ Best Directory Structure for Research

```
$HOME/
├── envs/
│   └── hpc_ml_env/               ← virtual environment (lives here permanently)
│
├── wheelhouse/
│   └── my_project/               ← custom wheels for packages not in Alliance wheelhouse
│
├── projects/
│   └── my_project/
│       ├── train.py
│       ├── requirements.txt          ← original PyPI-style requirements
│       ├── requirements_cc.txt       ← cluster-optimized, pinned requirements
│       ├── requirements_custom.txt   ← packages sourced from custom wheelhouse only
│       ├── train_job.sh
│       ├── load_modules.sh
│       ├── configs/
│       └── logs/
│           ├── train-482193.out
│           └── train-482193.err
│
└── scratch -> /scratch/$USER      ← symlink for convenience

$SCRATCH/
├── datasets/
│   └── your_dataset/
└── checkpoints/
    └── run_20260310_140000/
```

| Storage | Path | Purpose | Quota |
|---|---|---|---|
| Home | `$HOME` | Code, environments, scripts, wheelhouse | ~50 GB |
| Project | `$PROJECT` | Shared data, long-term results | ~1 TB |
| Scratch | `$SCRATCH` | Large datasets, checkpoints (purged after 60 days) | ~20 TB |
| SLURM_TMPDIR | `$SLURM_TMPDIR` | Fast per-job local SSD — build venvs here | Varies |

> ⚠️ **Never store large datasets in `$HOME`.** Always use `$SCRATCH` for datasets and model checkpoints.

---

## 8️⃣ Useful Commands

**Environment:**

```bash
source ~/envs/hpc_ml_env/bin/activate    # activate environment
deactivate                                # deactivate environment
pip list                                  # list installed packages
pip freeze > requirements.txt            # export current packages
```

**Wheels & Packages:**

```bash
avail_wheels torch                        # check if a package has a +computecanada wheel
avail_wheels -r requirements.txt          # check your whole requirements file
avail_wheels -r requirements.txt --not-available   # show only missing packages
uv pip download "pkg==X.Y" -d $HOME/wheelhouse/my_project  # download from PyPI
```

**Job Management:**

```bash
sq                           # list your jobs (short alias)
squeue -u $USER              # list your jobs
squeue --me                  # same, shorthand
scancel JOBID                # cancel a specific job
scancel -u $USER             # cancel ALL your jobs
seff JOBID                   # efficiency report after job completes
scontrol show job JOBID      # full job details
```

**Log Monitoring:**

```bash
tail -f slurm-<JOBID>.out    # stream job output live
tail -n 50 slurm-<JOBID>.out # show last 50 lines
less slurm-<JOBID>.out       # scrollable view
ls slurm-*.out               # list all output files
```

**Storage:**

```bash
quota -s                     # check storage usage
diskusage_report             # detailed Alliance quota report
ls -lh $SCRATCH/checkpoints  # check checkpoint sizes
```

**GPU:**

```bash
nvidia-smi                   # GPU status
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s) available')"
```

**Modules:**

```bash
module list                  # currently loaded modules
module purge                 # unload all modules
module avail python          # available Python versions
module spider cuda           # search for CUDA modules
module save my_stack         # save current module set
module restore my_stack      # reload saved module set
```

---

## ⭐ One Extremely Useful Trick for Alliance Clusters

Before installing anything with pip, **check what precompiled wheels are available**:

```bash
avail_wheels torch
avail_wheels "torch*"
avail_wheels numpy --python 3.11
```

Install from the cluster's optimized wheel cache:

```bash
pip install --no-index torch torchvision torchaudio
pip install --no-index numpy scipy scikit-learn pandas matplotlib
pip install --no-index transformers accelerate
```

These wheels are compiled specifically for the cluster's CPU/GPU architecture and are **much faster to install** than downloading from PyPI. They're also tested for compatibility with the cluster's CUDA version.

---

## 🚀 Your Typical Workflow

```
┌─────────────────────────────────────────────────────────┐
│  LOCAL MACHINE                                          │
│  ─────────────────────────────────────────────────────  │
│  1. Edit code in VSCode / Cursor                        │
│  2. git add . && git commit -m "update model"           │
│  3. git push origin main                                │
└──────────────────────┬──────────────────────────────────┘
                       │  SSH
┌──────────────────────▼──────────────────────────────────┐
│  CLUSTER (login node)                                   │
│  ─────────────────────────────────────────────────────  │
│  git pull origin main                                   │
│  source load_modules.sh                                 │
│  source ~/envs/hpc_ml_env/bin/activate                  │
│                                                         │
│  ┌──────────────┐     ┌──────────────────────────────┐  │
│  │  Quick debug │     │  Full training run           │  │
│  │  salloc ...  │     │  sbatch train_job.sh         │  │
│  │  python x.py │     │  tail -f slurm-*.out         │  │
│  └──────────────┘     │  seff JOBID (after)          │  │
│                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 📄 Requirements Template

Save as `requirements.txt` in your project root:

```text
# Deep Learning
torch
torchvision
torchaudio

# Transformers & HuggingFace
transformers
datasets
accelerate
peft

# Scientific Computing
# NOTE: numpy, scipy, pandas, matplotlib are provided by scipy-stack/2026a
# Do NOT reinstall them here — it causes version conflicts with the module
scikit-learn

# Experiment Tracking
tensorboard

# Explainability
shap
lime

# Utilities
tqdm
pyyaml
```

> 💡 Do **not** re-install `numpy`, `scipy`, `pandas`, or `matplotlib` if you loaded `scipy-stack/2026a` — those are already available and cluster-optimized. Reinstalling via pip can silently break things.

Install on the cluster:

```bash
pip install --no-index -r requirements.txt
# For packages not available as cluster wheels:
pip install some-package   # falls back to PyPI
```

---

## 🔧 Advanced Tips

### Run Jupyter Lab via SSH Tunnel (Stable, No Crashes)

This is the most reliable way to run Jupyter on a compute node from your local browser. It requires two terminals open simultaneously.

#### How it works

```
[Your Laptop :8888] ──SSH tunnel──▶ [Login Node] ──▶ [Compute Node :8888]
```

You forward a local port on your laptop through the login node to the actual compute node where Jupyter is running. The key rule is:

```
ssh -L LOCAL_PORT:COMPUTE_NODE_NAME:REMOTE_PORT cluster_alias
```

> ⚠️ **Common mistake:** Using `localhost` in the tunnel command instead of the actual compute node name. `localhost` refers to the login node, not the node running Jupyter — the tunnel will fail silently.

---

#### Step-by-Step Workflow

**Terminal 1 — On the cluster:**

Request an allocation and note the compute node name. It appears in your shell prompt once the job starts (e.g., `username@fc10512`).

```bash
salloc --time=02:00:00 --cpus-per-task=4 --mem=16G --account=def-supervisor
# Your prompt changes to: (username@COMPUTE_NODE ...)
# Note the compute node name — e.g.: fc10512, gra1234, blg8801, etc.

source load_modules.sh
source ~/envs/hpc_ml_env/bin/activate

jupyter lab --no-browser --port=8888
```

Jupyter will print output like:

```
http://localhost:8888/lab?token=abc123def456...
```

Copy the full token URL — you will need it in your browser.

---

**Terminal 2 — On your local machine:**

Using the compute node name from your cluster prompt, open the SSH tunnel:

```bash
ssh -L 8888:COMPUTE_NODE_NAME:8888 username@cluster.alliancecan.ca
```

Example:

```bash
ssh -L 8888:fc10512:8888 username@fir.alliancecan.ca
```

Keep this terminal open for the entire session. Closing it kills the tunnel.

---

**Browser — On your local machine:**

Open the token URL that Jupyter printed in Terminal 1:

```
http://localhost:8888/lab?token=abc123def456...
```

You are now running Jupyter on a compute node, accessed securely through your browser.

---

#### Quick Reference

| What | Command |
|---|---|
| Find compute node name | Check your shell prompt after `salloc` |
| Correct tunnel | `ssh -L 8888:COMPUTE_NODE:8888 username@cluster.alliancecan.ca` |
| Wrong tunnel (don't use) | `ssh -L 8888:localhost:8888 ...` |
| Open in browser | `http://localhost:8888/lab?token=YOUR_TOKEN` |

---

#### ⚠️ Watch Your Allocation Time

Always request enough time for your Jupyter session. A common mistake is requesting a very short allocation (e.g., `--time=00:05:00` = **5 minutes**, not 5 hours) which kills your session before you can do meaningful work.

Recommended minimums:

```bash
# For exploration / debugging
salloc --time=02:00:00 ...

# For longer experiments
salloc --time=08:00:00 ...
```

Check whether your job is still alive at any time:

```bash
squeue -u $USER
```

If it's gone, restart from Terminal 1 with a new `salloc`. You will get a new compute node name and need to update your tunnel accordingly.

---

### Multi-GPU Training with PyTorch + Slurm

```bash
#!/bin/bash
#SBATCH --account=def-supervisor
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

module purge
module load CCconfig
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load openmpi/4.1.5
module load scipy-stack/2026a

source ~/envs/hpc_ml_env/bin/activate

srun python -m torch.distributed.run \
    --nproc_per_node=4 \
    train.py --distributed
```

### Resuming from Checkpoints

Always save checkpoints to `$SCRATCH` and write your training script to resume:

```python
import os
import torch

checkpoint_path = os.path.join(os.environ.get("SCRATCH", "."), "checkpoints", "latest.pt")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resumed from epoch {start_epoch}", flush=True)
else:
    start_epoch = 0
    print("Starting from scratch", flush=True)
```

---

## 📚 References

- [Digital Research Alliance — Running Jobs](https://docs.alliancecan.ca/wiki/Running_jobs)
- [Alliance — Python Virtual Environments](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment)
- [Alliance — Available Python Wheels](https://docs.alliancecan.ca/wiki/Available_Python_wheels)
- [Alliance — Storage and File Management](https://docs.alliancecan.ca/wiki/Storage_and_file_management)
- [Alliance — JupyterHub](https://docs.alliancecan.ca/wiki/JupyterHub)
- [Alliance — Using Modules](https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en)

---

*Maintained for research use — Queen's University ECE (AI)*
