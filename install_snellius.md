# Installing CREST-MLIP with UMA on Snellius

## 1. Install Miniforge (own conda, no module conflicts)

```bash
# Download Miniforge (conda-forge default, no Anaconda license issues)
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
rm Miniforge3-Linux-x86_64.sh

# Initialize for bash (adds to ~/.bashrc)
~/miniforge3/bin/conda init bash
source ~/.bashrc

# Prevent conda from conflicting with Snellius modules
conda config --set auto_activate_base false
```

Log out and back in. From now on, `conda activate` works without loading any Snellius module.

## 2. Create the UMA conda environment

```bash
conda create -n crest-uma python=3.11 -y
conda activate crest-uma

# PyTorch with CUDA 12.1 (matches Snellius A100 nodes)
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# UMA (fairchem-core) + dependencies
pip install fairchem-core huggingface_hub

# Build tools
conda install cmake ninja gfortran openblas -c conda-forge -y

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from fairchem.core import pretrained_mlip; print('fairchem OK')"
```

## 3. Clone and build CREST-MLIP

```bash
cd ~
git clone https://github.com/EPiCs-group/crest-mlip.git
cd crest-mlip
git submodule update --init

conda activate crest-uma

cmake -B build \
  -DWITH_PYMLIP=true \
  -DPython3_ROOT_DIR=$CONDA_PREFIX \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j8
```

Verify the binary links to conda Python:
```bash
ldd build/crest | grep python
# Should show: libpython3.11.so -> .../miniforge3/envs/crest-uma/lib/...
```

## 4. Download UMA model

```bash
mkdir -p ~/models
cd ~/models

# Download via Python (gated model, may need HuggingFace login)
python -c "
from fairchem.core import pretrained_mlip
pu = pretrained_mlip.get_predict_unit('uma-s-1', device='cpu')
print('UMA model downloaded')
"
```

If the model requires HuggingFace authentication:
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HuggingFace token
```

## 5. Test on a login node (CPU only)

```bash
cd ~/crest-mlip
cat > /tmp/test_sp.toml << 'EOF'
runtype = "sp"
threads = 1

[[calculation.level]]
method = "uma"
device = "cuda"
EOF

conda activate crest-uma
build/crest examples/ethane.xyz --input /tmp/test_sp.toml
```

## 6. SLURM job script for GPU conformer search

Create `~/submit_crest.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=crest-uma
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=crest_%j.out
#SBATCH --error=crest_%j.err

# --- Do NOT load any Snellius modules ---
# Our own conda handles everything (Python, CUDA, compilers)
module purge

# Activate our conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate crest-uma

# Prevent OpenBLAS threading conflicts with CREST's OpenMP
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run CREST
cd $SLURM_SUBMIT_DIR
~/crest-mlip/build/crest molecule.xyz --input crest.toml
```

## 7. Example TOML for conformer search

Create `crest.toml`:

```toml
runtype = "imtd-gc"
threads = 16

[[calculation.level]]
method = "uma"
device = "cuda"
```

Submit:
```bash
cp molecule.xyz crest.toml /your/workdir/
cd /your/workdir
sbatch ~/submit_crest.sh
```

## 8. Tips for Snellius

- **Do NOT mix `module load` with conda** — our Miniforge provides everything.
  If you see `libstdc++` or GLIBC errors, check that no modules are loaded: `module list`
- **Scratch storage**: Use `$TMPDIR` or `/scratch-shared/$USER` for large runs.
  CREST writes many temporary files during MTD.
- **GPU allocation**: Snellius A100 nodes have 4 GPUs. Request `--gpus=1` for single-GPU
  CREST runs. Multi-GPU is not yet supported in CREST.
- **Wall time**: A 100-atom conformer search with UMA typically takes 12-36 hours.
  Request `--time=48:00:00` for safety.
- **Memory**: UMA loads a ~500 MB model. Request at least 16 GB RAM
  (default on GPU nodes is usually sufficient).
