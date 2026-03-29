<h1 align="center">CREST</h1>
<h3 align="center">Conformer-Rotamer Ensemble Sampling Tool</h3>
<div align="center">

[![Latest Version](https://img.shields.io/github/v/release/crest-lab/crest?color=khaki)](https://github.com/crest-lab/crest/releases/latest)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/crest?color=khaki)](https://anaconda.org/conda-forge/crest)
[![DOI](https://img.shields.io/badge/DOI-10.1039%2Fc9cp06869d%20-blue)](http://dx.doi.org/10.1039/c9cp06869d)
[![DOI](https://img.shields.io/badge/DOI-10.1063%2F5.0197592-blue)](https://doi.org/10.1063/5.0197592)
[![CI workflow](https://github.com/crest-lab/crest/actions/workflows/build-CI.yml/badge.svg)](https://github.com/crest-lab/crest/actions/workflows/build-CI.yml)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPL_v3-coral.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Documentation](https://img.shields.io/badge/documentation-crest--lab.github.io%2Fcrest--docs%2F-gold)](https://crest-lab.github.io/crest-docs/)

</div>

---

> **CREST-MLIP** — This fork adds Machine Learning Interatomic Potential (MLIP)
> support to CREST via three backends: **libtorch** (direct C++ TorchScript),
> **pymlip** (embedded Python for UMA/MACE), and **ASE socket** (TCP to any
> ASE calculator). Based on CREST 3.0.2.
> See [CHANGES.md](CHANGES.md) for a detailed list of modifications.

---

## MLIP Backends

| Backend | Method keyword | When to Use |
|---------|----------------|-------------|
| `libtorch` | `libtorch` | Fastest GPU inference, no Python runtime needed |
| `pymlip` (UMA) | `uma` | Meta's Universal MLIP Accelerator (fairchem) |
| `pymlip` (MACE) | `mace` | MACE foundation models (mace-torch) |
| `ase-socket` | `ase-socket` | Any ASE-compatible calculator via TCP socket |

## Building with MLIP Support

### Prerequisites
- CMake >= 3.17 and gfortran >= 10 (same as upstream CREST)
- For **libtorch**: PyTorch C++ (libtorch) installed
- For **pymlip**: Python 3.10+ with `fairchem-core` (UMA) or `mace-torch` (MACE)

Conda environment files are provided in `environments/`:
```bash
conda env create -f environments/uma-cuda.yml
conda activate crest-uma
```

### CMake Build

For pymlip (UMA/MACE):
```bash
cmake -B build \
  -DWITH_PYMLIP=true \
  -DPython3_EXECUTABLE=$(which python) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

For libtorch:
```bash
cmake -B build \
  -DWITH_LIBTORCH=true \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

> **Note:** `WITH_GFNFF` is auto-enabled when building with MLIP support — GFN-FF provides
> lightweight topology-based WBOs needed by SHAKE bond constraints and the flexibility measure.

## Quick Start

```bash
# UMA conformer search (GPU)
crest molecule.xyz --input examples/conformer_search_uma.toml

# MACE optimization via embedded Python
crest molecule.xyz --input examples/optimization_mace_pymlip.toml

# MACE conformer search via libtorch (TorchScript)
crest molecule.xyz --input examples/conformer_search_mace_libtorch.toml
```

## Key Changes from Upstream CREST 3.0.2

- **Three MLIP calculator backends** — each targeting a different use case:
  - **libtorch**: loads TorchScript-exported models (`.pt`) and runs inference entirely in C++ via the PyTorch C API. No Python runtime needed at all. This is the fastest path for production GPU runs. Models are exported with `scripts/export_model.py` or `scripts/export_mace.py`, which bake graph construction, neighbor lists, and unit conversions into the TorchScript module.
  - **pymlip**: embeds the CPython interpreter inside the Fortran process and calls UMA (`fairchem-core`) or MACE (`mace-torch`) calculators directly in-memory. Avoids TCP socket overhead while reusing all Python model infrastructure. The GIL is acquired per-call, making it thread-safe for OpenMP.
  - **ASE socket**: connects over TCP to an external Python server (`src/python_server/crest_ase_server.py`) wrapping any ASE-compatible calculator. The most flexible option — works with any model or code that has an ASE interface, at the cost of serialization overhead.

- **GPU batched inference** — CREST's conformer search generates hundreds of independent structures per iteration. Instead of evaluating them one-by-one through OpenMP threads (which would serialize on a single GPU anyway), the batched path in `parallel.f90` collects all structures, pads them to equal length, and sends them through the model in a single batched forward pass. Two sub-paths exist: single-GPU (one batch) and multi-GPU (round-robin distribution across `ngpus` devices). Batch size is auto-tuned from atom count if not set explicitly. This typically gives 5-20x speedup over per-structure evaluation.

- **Shared model loading** — MLIP models can be 100 MB–2 GB. Without sharing, each OpenMP thread would load its own copy, quickly exhausting GPU memory. Instead, the master thread loads the model once and broadcasts the handle to all workers. For libtorch this uses a mutex-protected shared pointer; for pymlip, the GIL naturally serializes access to a single Python object. Workers that receive a shared handle skip cleanup on exit (only the owner deallocates).

- **WBO fallback cascade for MLIP calculators** — CREST's metadynamics relies on Wiberg Bond Orders (WBOs) in two places: (1) SHAKE bond constraints use WBOs to identify which bonds to constrain (threshold > 0.5), and (2) the `flexi()` function uses WBOs to estimate molecular flexibility, which controls metadynamics simulation length. MLIP calculators provide only energy and gradient — no WBOs. To fill this gap, we implemented a cascade: first try GFN2-xTB (produces continuous WBOs of 1.0/1.5/2.0, most accurate), then fall back to GFN-FF topology (binary 0/1 WBOs from neighbor list only, ~0.01s, no singlepoint needed), then fall back to a size-based default (flexibility = 0.5). SHAKE has an additional safety net: if no WBOs are available at all, it degrades from mode 2 (all-bond constraints) to mode 1 (X-H bonds only, which need no WBOs). GFN-FF is auto-enabled in the build system (`WITH_GFNFF`) whenever MLIP support is compiled in.

- **MLIP resource cleanup** — GPU memory, Python interpreter state, and TCP sockets must be released after each algorithm step (MD, optimization, singlepoint, scan, numerical Hessian). Every algorithm endpoint calls idempotent cleanup routines that free GPU tensors, close socket connections, and release Python objects. This prevents GPU memory leaks during multi-step workflows (e.g., conformer search → optimization → frequency calculation).

- **Conda environments** — ready-to-use YAML files in `environments/` for UMA and MACE, with both CPU and CUDA variants. These pin compatible versions of PyTorch, fairchem-core/mace-torch, and all dependencies.

## TOML Configuration Reference

### MLIP Keywords

All keys go inside `[[calculation.level]]` blocks.

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `method` | `uma`, `mace`, `libtorch`, `pymlip`, `ase-socket` | — | Calculator backend selection |
| `device` | `cpu`, `cuda`, `cuda:0`–`cuda:3`, `mps` | `cpu` | Compute device for inference |
| `model_path` | file path | — | Path to model checkpoint (`.pt` or `.model`) |
| `model_type` | `uma`, `mace` | — | Model family (pymlip backend only) |
| `model_format` | `generic`, `mace` | `generic` | TorchScript output format (libtorch only) |
| `cutoff` | float (Angstrom) | `6.0` | Neighbor list cutoff (libtorch only) |
| `task` | string | — | UMA task name, e.g. `omol` |
| `atom_refs` | file path | — | Per-element energy references YAML (UMA only) |
| `compile_mode` | `""`, `reduce-overhead`, `max-autotune` | `""` | torch.compile mode (pymlip only) |
| `dtype` | `float64`, `float32` | `float64` | Floating-point precision (MACE only) |
| `turbo` | `true`/`false` | `false` | UMA turbo-mode: tf32 + compile + merge_mole |
| `batch_size` | integer | `0` (auto) | Structures per GPU batch |
| `aten_threads` | integer | `0` (auto) | ATen intra-op threads |
| `shared_model` | `true`/`false` | `false` | Share one model across threads (libtorch) |
| `ngpus` | integer | `0` (auto) | GPUs for multi-GPU batching |
| `host` | string | `127.0.0.1` | ASE socket server hostname |
| `port` | integer | `6789` | ASE socket server TCP port |
| `debug` | `true`/`false` | `false` | Per-call timing output |

### Example TOML

```toml
runtype = "imtd-gc"
threads = 16

[[calculation.level]]
method     = "uma"
device     = "cuda"
batch_size = 16
```

---

CREST (abbreviated from ***C***onformer-***R***otamer ***E***nsemble ***S***ampling ***T***ool) is a program for the automated exploration of the low-energy molecular chemical space.
It functions as an OMP scheduler for calculations with efficient force-field and semiempirical quantum mechanical methods such as xTB, and provides
a variety of capabilities for creation and analysis of structure ensembles.<br> See our recent publication in *J. Chem. Phys.* for a feature overview: [**https://doi.org/10.1063/5.0197592**](https://doi.org/10.1063/5.0197592)

<div align="center">
<img src="./assets/newtoc.png" alt="CREST" width="750">
</div>


## Documentation

The CREST documentation with installation instructions and application examples is hosted at: <br>
<div align="center">

[![Documentation](https://img.shields.io/badge/documentation-crest--lab.github.io%2Fcrest--docs%2F-gold)](https://crest-lab.github.io/crest-docs/)


</div>

## Installation quick guide

There are multiple possible ways of installing CREST. Detailed build instructions can be found at <https://crest-lab.github.io/crest-docs/page/installation>.

> [!WARNING]  
> For any installation make sure that you have correctly installed and sourced the [`xtb`](https://github.com/grimme-lab/xtb) program before attempting any calculations with CREST.
> **While `xtb` is technically not needed for the primary runtypes of CREST versions >3.0 thanks to an integration of [`tblite`](https://github.com/tblite/tblite), some functionalities, like QCG, still require it!**

##

### Option 1: Precompiled binaries 
[![Latest Version](https://img.shields.io/github/v/release/crest-lab/crest?color=khaki)](https://github.com/crest-lab/crest/releases/latest)
[![Github Downloads All Releases](https://img.shields.io/github/downloads/crest-lab/crest/total)](https://github.com/crest-lab/crest/releases)

The *statically linked* binaries can be found at the [release page](https://github.com/crest-lab/crest/releases) of this repository.
The most recent program version is automatically build (both Meson/Intel and CMake/GNU) from the main branch and can be found at the [continous release page](https://github.com/crest-lab/crest/releases/tag/latest), or directly download them here:

[![Download (GNU)](https://img.shields.io/badge/download-GNU_build_binary-green)](https://github.com/crest-lab/crest/releases/download/latest/crest-gnu-12-ubuntu-latest.tar.xz)
[![Download (ifort)](https://img.shields.io/badge/download-ifort_build_binary-blue.svg)](https://github.com/crest-lab/crest/releases/download/latest/crest-intel-2023.1.0-ubuntu-latest.tar.xz)

Simply unpack the binary  and add it to your *PATH* variable.
```bash
tar -xf crest-gnu-12-ubuntu-latest.tar.xz
```
or
```bash
tar -xf crest-intel-2023.1.0-ubuntu-latest.tar.xz
```
The program should be directly executable.

##

### Option 2: Conda
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/crest?color=khaki)](https://anaconda.org/conda-forge/crest) 
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/crest.svg)](https://anaconda.org/conda-forge/crest)

A [conda-forge](https://github.com/conda-forge) feedstock is maintained at <https://github.com/conda-forge/crest-feedstock>.

Installing CREST from the `conda-forge` channel can be done via:

```
conda install conda-forge::crest
```

The conda-forge distribution is based on a *dynamically linked* CMake/GNU build.
> [!WARNING]
> When using OpenBLAS as shared library backend for the linear algebra in CREST, please set the system variable `export OPENBLAS_NUM_THREADS=1`, as there may be an ugly warning in the concurrent (nested) parallel code parts otherwise. 


##

### Option 3: Compiling from source
<h4>Tested builds</h4>
<!--blank line after HTML-->

![CI workflow](https://github.com/crest-lab/crest/actions/workflows/build.yml/badge.svg)

Working and tested builds of CREST (mostly on Ubuntu 20.04 LTS):

| Build System | Compiler | Linear Algebra Backend | Build type     | Status     | Note |
|--------------|----------|------------------------|:--------------:|:----------:|:----:|
| CMake 3.30.2 | GNU (gcc 14.1.0)  | [libopenblas 0.3.27](https://anaconda.org/conda-forge/libopenblas) | dynamic | ✅ ||
| CMake 3.30.2 | GNU (gcc 12.3.0)  | [libopenblas-dev](https://packages.debian.org/stable/libdevel/libopenblas-dev) | static  | ✅ | [![Download (GNU)](https://img.shields.io/badge/download-GNU_build_binary-green)](https://github.com/crest-lab/crest/releases/download/latest/crest-gnu-12-ubuntu-latest.tar.xz)|
| CMake 3.28.3 | [Intel (`ifort`/`icc` 2021.9.0)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)   | [MKL static (oneAPI 2023.1)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | dynamic | ⚠️  | OpenMP/MKL problem ([#285](https://github.com/crest-lab/crest/issues/285)) |
| Meson 1.2.0 | [Intel (`ifort`/`icx` 2023.1.0)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)   | [MKL static (oneAPI 2023.1)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) | static  | ✅ | [![Download (ifort)](https://img.shields.io/badge/download-ifort_build_binary-blue.svg)](https://github.com/crest-lab/crest/releases/download/latest/crest-intel-2023.1.0-ubuntu-latest.tar.xz) |


Generally, subprojects should be initialized for the *default* build options, which can be done by 
```bash
git submodule init
git submodule update
```
For more information about builds including subprojects see [here](./subprojects/README.md).

Some basic build instructions can be found in the following dropdown tabs:



<details open>
<summary><h4><code>CMake</code> build</h4></summary>
<!-- blank line to recover markdown format-->

Building CREST with CMake works with the following chain of commands (in this example with `gfortran/gcc` compilers):
```bash
export FC=gfortran CC=gcc
cmake -B _build
```
and then to build the CREST binary
```bash
make -C _build
```
*Optionally*, the build can be tested via
```bash
make test -C _build
```
The `CMake` build typically requires access to shared libraries of LAPACK and OpenMP. They must be present in the library paths at compile and runtime.
Alternatively, a static build can be selected by using `-DSTATICBUILD=true` in the CMake setup step. The current static build with GNU compilers is available from the [**continous release page**](https://github.com/crest-lab/crest/releases/tag/latest). 
</details>

<details>
<summary><h4><code>meson</code> build</h4></summary>
<!-- blank line to recover markdown format-->

For the setup an configuration of meson see also the [meson setup](https://github.com/grimme-lab/xtb/blob/master/meson/README.adoc) page hosted at the `xtb` repository.
The chain of commands to build CREST with meson is:

```bash
export FC=ifort CC=icc
meson setup _build --prefix=$PWD/_dist
meson install -C _build
```

The `meson` build of CREST is mainly focused on and tested with the Intel `ifort`/`icc` compilers.
When using newer versions of Intel's oneAPI, replacing `icc` with `icx` should work. Please refrain from using `ifx` instead of `ifort`, however.
When attempting to build with `gfortran` and `gcc`, add `-Dla_backend=mkl` to the meson setup command. Compatibility with the GNU compilers might be limited. We recommend the CMake build (see the corresponding section) in this instance.

By default the `meson` build will create a **statically** linked binary.
</details>


---

### Citations

1. P. Pracht, F. Bohle, S. Grimme, *Phys. Chem. Chem. Phys.*, **2020**, 22, 7169-7192.
  DOI: [10.1039/C9CP06869D](https://dx.doi.org/10.1039/C9CP06869D)

2. S. Grimme, *J. Chem. Theory Comput.*, **2019**, 155, 2847-2862.
  DOI: [10.1021/acs.jctc.9b00143](https://dx.doi.org/10.1021/acs.jctc.9b00143)

3. P. Pracht, S. Grimme, *Chem. Sci.*, **2021**, 12, 6551-6568.
  DOI: [10.1039/d1sc00621e](https://dx.doi.org/10.1039/d1sc00621e)

4. P. Pracht, C.A. Bauer, S. Grimme, *J. Comput. Chem.*, **2017**, *38*, 2618-2631. 
  DOI: [10.1002/jcc.24922](https://dx.doi.org/10.1002/jcc.24922)

5. S. Spicher, C. Plett, P. Pracht, A. Hansen, S. Grimme,  *J. Chem. Theory Comput.*, **2022**,
  *18*, 3174-3189. DOI: [10.1021/acs.jctc.2c00239](https://dx.doi.org/10.1021/acs.jctc.2c00239)

6. P. Pracht, C. Bannwarth, *J. Chem. Theory Comput.*, **2022**, *18 (10)*, 6370-6385. DOI: [10.1021/acs.jctc.2c00578](https://dx.doi.org/10.1021/acs.jctc.2c00578)

7. P. Pracht, S. Grimme, C. Bannwarth, F. Bohle, S. Ehlert, G. Feldmann, J. Gorges, M. Müller, T. Neudecker, C. Plett, S. Spicher, P. Steinbach, P. Wesołowski, F. Zeller, *J. Chem. Phys.*, **2024**, *160*, 114110. DOI: [10.1063/5.0197592](https://doi.org/10.1063/5.0197592)

If you use the MLIP backends in this fork, please also cite:

8. **UMA**: Meta Fundamental AI Research, *fairchem-core*, [https://github.com/FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)

9. **MACE**: I. Batatia, D.P. Kovacs, G.N.C. Simm, C. Ortner, G. Csanyi, *NeurIPS*, **2022**. DOI: [10.48550/arXiv.2206.07697](https://doi.org/10.48550/arXiv.2206.07697)

<details>
<summary><h4>BibTex entries</h4></summary>
<!-- blank line to recover markdown format-->

```
@article{Pracht2020,
  author ="Pracht, Philipp and Bohle, Fabian and Grimme, Stefan",
  title  ="Automated exploration of the low-energy chemical space with fast quantum chemical methods",
  journal  ="Phys. Chem. Chem. Phys.",
  year  ="2020",
  volume  ="22",
  issue  ="14",
  pages  ="7169-7192",
  doi  ="10.1039/C9CP06869D"
}

@article{Grimme2019,
  author = {Grimme, Stefan},
  title = {Exploration of Chemical Compound, Conformer, and Reaction Space with Meta-Dynamics Simulations Based on Tight-Binding Quantum Chemical Calculations},
  journal = {J. Chem. Theory Comput.},
  volume = {15},
  number = {5},
  pages = {2847-2862},
  year = {2019},
  doi = {10.1021/acs.jctc.9b00143}
}

@article{Pracht2021,
  author ="Pracht, Philipp and Grimme, Stefan",
  title  ="Calculation of absolute molecular entropies and heat capacities made simple",
  journal  ="Chem. Sci.",
  year  ="2021",
  volume  ="12",
  issue  ="19",
  pages  ="6551-6568",
  doi  ="10.1039/D1SC00621E",
  url  ="http://dx.doi.org/10.1039/D1SC00621E"
}

@article{Pracht2017,
  author = {Pracht, Philipp and Bauer, Christoph Alexander and Grimme, Stefan},
  title = {Automated and efficient quantum chemical determination and energetic ranking of molecular protonation sites},
  journal = {J. Comput. Chem.},
  volume = {38},
  number = {30},
  pages = {2618-2631},
  doi = {https://doi.org/10.1002/jcc.24922},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.24922},
  year = {2017}
}

@article{Spicher2022,
  author = {Spicher, Sebastian and Plett, Christoph and Pracht, Philipp and Hansen, Andreas and Grimme, Stefan},
  title = {Automated Molecular Cluster Growing for Explicit Solvation by Efficient Force Field and Tight Binding Methods},
  journal = {J. Chem. Theory Comput.},
  volume = {18},
  number = {5},
  pages = {3174-3189},
  year = {2022},
  doi = {10.1021/acs.jctc.2c00239}
}

@article{Pracht2022,
  author = {Pracht, Philipp and Bannwarth, Christoph},
  title = {Fast Screening of Minimum Energy Crossing Points with Semiempirical Tight-Binding Methods},
  journal = {J. Chem. Theory Comput.},
  volume = {18},
  number = {10},
  pages = {6370-6385},
  year = {2022},
  doi = {10.1021/acs.jctc.2c00578}
}

@article{Pracht2024,
  author = {Pracht, Philipp and Grimme, Stefan and Bannwarth, Christoph and Bohle, Fabian and Ehlert, Sebastian and Feldmann, Gereon and Gorges, Johannes and M\"uller, Marcel and Neudecker, Tim and Plett, Christoph and Spicher, Sebastian and Steinbach, Pit and Weso\{}lowski, Patryk A. and Zeller, Felix},
  title = "{CREST - A program for the exploration of low-energy molecular chemical space}",
  journal = {J. Chem. Phys.},
  volume = {160},
  number = {11},
  pages = {114110},
  year = {2024},
  month = {03},
  issn = {0021-9606},
  doi = {10.1063/5.0197592},
  url = {https://doi.org/10.1063/5.0197592}
}
```
</details>




### License

CREST is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CREST is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU Lesser General Public License for more details.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in CREST by you, as defined in the GNU Lesser General Public license, shall be licensed as above, without any additional terms or conditions
