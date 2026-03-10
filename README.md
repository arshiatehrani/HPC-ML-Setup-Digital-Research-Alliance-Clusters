# 🖥️ HPC ML Setup — Digital Research Alliance Clusters

> **Clean, reproducible ML environment setup for Narval, Cedar, Fir, Béluga, and other Alliance clusters.**
> Install once. Run everywhere: JupyterHub, `salloc`, `sbatch`, GPU nodes.

---

## 📋 Table of Contents

- [First Login Checklist](#-first-login-checklist)
- [Module Reference](#-module-reference)
- [1. Create a Permanent Virtual Environment](#1️⃣-create-one-permanent-virtual-environment)
- [2. JupyterHub Kernel Setup](#2️⃣-make-it-usable-in-jupyterhub)
- [3. Interactive Jobs (salloc)](#3️⃣-use-the-environment-in-interactive-jobs)
- [4. Batch Jobs (sbatch)](#4️⃣-use-the-environment-in-batch-jobs-best-practice)
- [5. Directory Structure](#5️⃣-best-directory-structure-for-research)
- [6. Useful Commands](#6️⃣-useful-commands)
- [7. Pre-built Wheels Trick](#️-one-extremely-useful-trick-for-alliance-clusters)
- [8. Typical Workflow](#-your-typical-workflow)
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
module load StdEnv/2023 python/3.11.5

# 4. Activate your persistent virtual environment
source ~/envs/elec888_env/bin/activate

# 5. Verify GPU/CUDA availability (if on a GPU node)
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> 💡 **Tip:** Add the `module purge`, `module load`, and `source` lines to your `~/.bashrc` **only if** you always want the same environment on login. Otherwise, load manually to avoid conflicts.

---

## 📦 Module Reference

Alliance clusters use the **Lmod** module system. Common modules for ML workloads:

| Module | Purpose |
|---|---|
| `StdEnv/2023` | Standard software environment (always load first) |
| `python/3.11.5` | Python interpreter |
| `python/3.10.13` | Alternative Python version |
| `cuda/12.2` | CUDA toolkit (loaded automatically with GPU nodes in most cases) |
| `cudnn/8.9.5.29` | cuDNN for deep learning |
| `scipy-stack/2023b` | NumPy, SciPy, Matplotlib, Pandas (prebuilt) |
| `arrow/14.0.1` | Apache Arrow / PyArrow |
| `ipykernel/2026a` | Prebuilt ipykernel module (alternative to pip install) |
| `StdEnv/2020` | Legacy environment (use only if required) |

Check all available modules:

```bash
module avail          # list all available modules
module spider python  # search for a specific module
```

Load CUDA explicitly for GPU jobs if needed:

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.11.5
```

---

## 1️⃣ Create ONE Permanent Virtual Environment

Do this **once** after your first login. The environment lives in `~/envs/` permanently.

```bash
module purge
module load StdEnv/2023 python/3.11.5

python -m venv ~/envs/elec888_env
source ~/envs/elec888_env/bin/activate
```

Upgrade pip and install your packages from `requirements.txt`:

```bash
pip install --upgrade pip --no-index
pip install --no-index -r requirements.txt
```

> ⚠️ **Always use `--no-index`** when possible. This forces pip to use the cluster's pre-compiled wheel cache instead of downloading from PyPI, which is significantly faster and more stable on HPC nodes.

If a package isn't available as a wheel, install from PyPI (requires internet on login nodes):

```bash
pip install some-package  # without --no-index
```

Your environment now lives permanently at:

```
~/envs/elec888_env/
```

---

## 2️⃣ Make it Usable in JupyterHub

Run once to register your environment as a Jupyter kernel:

```bash
source ~/envs/elec888_env/bin/activate
pip install ipykernel --no-index
python -m ipykernel install --user --name elec888_env --display-name "Python (elec888_env)"
```

When you open JupyterHub, select the kernel named:

```
Python (elec888_env)
```

To remove a kernel later:

```bash
jupyter kernelspec remove elec888_env
```

---

## 3️⃣ Use the Environment in Interactive Jobs

Request an interactive GPU session with `salloc`:

```bash
salloc --time=02:00:00 \
       --cpus-per-task=4 \
       --mem=16G \
       --gres=gpu:1 \
       --account=def-bakhshai
```

Once the node is allocated, load your environment:

```bash
module purge
module load StdEnv/2023 python/3.11.5
source ~/envs/elec888_env/bin/activate

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
#SBATCH --account=def-bakhshai
#SBATCH --job-name=elec888_train
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@queensu.ca

# ── Environment Setup ──────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11.5

source ~/envs/elec888_env/bin/activate

# ── Verify GPU ─────────────────────────────────────────────────────
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi

# ── Run Experiment ─────────────────────────────────────────────────
cd ~/projects/Survival-Analysis-Probabilistic-ML

python train_sota_models.py \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --output_dir ~/scratch/checkpoints/run_$(date +%Y%m%d_%H%M%S)
```

Create the logs directory first:

```bash
mkdir -p ~/projects/Survival-Analysis-Probabilistic-ML/logs
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

## 5️⃣ Best Directory Structure for Research

```
$HOME/
├── envs/
│   └── elec888_env/          ← virtual environment (lives here permanently)
│
├── projects/
│   └── Survival-Analysis-Probabilistic-ML/
│       ├── train_sota_models.py
│       ├── requirements.txt
│       ├── train_job.sh
│       ├── configs/
│       └── logs/
│
└── scratch -> /scratch/$USER  ← symlink for convenience

$SCRATCH/
├── datasets/
│   └── your_dataset/
└── checkpoints/
    └── run_20260310_140000/
```

| Storage | Path | Purpose | Quota |
|---|---|---|---|
| Home | `$HOME` | Code, environments, scripts | ~50 GB |
| Project | `$PROJECT` | Shared data, long-term results | ~1 TB |
| Scratch | `$SCRATCH` | Large datasets, checkpoints (purged after 60 days) | ~20 TB |

> ⚠️ **Never store large datasets in `$HOME`.** Always use `$SCRATCH` for datasets and model checkpoints.

---

## 6️⃣ Useful Commands

**Environment:**

```bash
source ~/envs/elec888_env/bin/activate    # activate environment
deactivate                                 # deactivate environment
pip list                                   # list installed packages
pip freeze > requirements.txt             # export current packages
```

**Job Management:**

```bash
sq                           # list your jobs (short alias)
squeue -u $USER              # list your jobs
scancel JOBID                # cancel a specific job
scancel -u $USER             # cancel ALL your jobs
seff JOBID                   # efficiency report after job completes
```

**Storage:**

```bash
quota -s                     # check storage usage
diskusage_report             # detailed Alliance quota report
ls -lh ~/scratch/checkpoints # check checkpoint sizes
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
│  FIR / NARVAL CLUSTER (login node)                      │
│  ─────────────────────────────────────────────────────  │
│  git pull origin main                                   │
│  module purge && module load StdEnv/2023 python/3.11.5  │
│  source ~/envs/elec888_env/bin/activate                 │
│                                                         │
│  ┌──────────────┐     ┌──────────────────────────────┐  │
│  │  Quick debug │     │  Full training run           │  │
│  │  salloc ...  │     │  sbatch train_job.sh         │  │
│  │  python x.py │     │  sq → watch job              │  │
│  └──────────────┘     │  tail -f logs/job-*.out      │  │
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
numpy
scipy
scikit-learn
pandas

# Survival Analysis / Probabilistic ML
lifelines
scikit-survival
pymc

# Experiment Tracking
tensorboard

# Explainability
shap
lime

# Utilities
matplotlib
seaborn
tqdm
pyyaml
```

Install on the cluster:

```bash
pip install --no-index -r requirements.txt
# For packages not available as wheels:
pip install lifelines pymc  # these may need PyPI
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
salloc --time=02:00:00 --cpus-per-task=4 --mem=16G --account=def-bakhshai
# Your prompt changes to: (username@COMPUTE_NODE ...)
# Note the compute node name, e.g.: fc10512, gra1234, blg8801, etc.

module purge
module load StdEnv/2023 python/3.11.5
# Alternative: module load StdEnv/2023 ipykernel/2026a

source ~/envs/elec888_env/bin/activate

jupyter lab --no-browser --port=8888
```

Jupyter will print output like:

```
http://localhost:8888/lab?token=abc123def456...
```

Copy the full token URL — you will need it in your browser.

---

**Terminal 2 — On your local machine:**

Using the compute node name from your cluster prompt (e.g., `fc10512`), open the SSH tunnel:

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
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

srun python -m torch.distributed.run \
    --nproc_per_node=4 \
    train_sota_models.py --distributed
```

### Resuming from Checkpoints

Always save checkpoints to `$SCRATCH` and write your training script to resume:

```python
checkpoint_path = os.environ.get("SCRATCH", ".") + "/checkpoints/latest.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
```

---

## 📚 References

- [Digital Research Alliance — Running Jobs](https://docs.alliancecan.ca/wiki/Running_jobs)
- [Alliance — Python Virtual Environments](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment)
- [Alliance — Available Python Wheels](https://docs.alliancecan.ca/wiki/Available_Python_wheels)
- [Alliance — Storage and File Management](https://docs.alliancecan.ca/wiki/Storage_and_file_management)
- [Alliance — JupyterHub](https://docs.alliancecan.ca/wiki/JupyterHub)

---

*Maintained by Arshia Esmail Tehrani — Queen's University ECE (AI)*
