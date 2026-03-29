# CREST-UMA: Changes from Original CREST 3.0.2

This document describes all changes made to the CREST codebase to add MLIP (Machine Learning Interatomic Potential) calculator support via three backends: **libtorch** (direct C++ inference), **pymlip** (embedded Python inference for UMA/MACE), and **ASE socket** (TCP client connecting to any ASE calculator server).

**Base version**: CREST 3.0.2 (commit `163f50a`)
**Total**: 4,636 lines of new code (10 new files) + 1,222 lines of modifications across 27 existing files.

---

## Table of Contents

1. [New Calculator Backends (New Files)](#1-new-calculator-backends-new-files)
2. [Build System Changes](#2-build-system-changes)
3. [Calculator Data Model (`calc_type.f90`)](#3-calculator-data-model-calc_typef90)
4. [Calculator Dispatch (`calculator.F90`, `subprocess_engrad.f90`)](#4-calculator-dispatch-calculatorf90-subprocess_engradf90)
5. [TOML Input Parsing (`parse_calcdata.f90`)](#5-toml-input-parsing-parse_calcdataf90)
6. [GPU Batched Inference in Parallel Loops (`parallel.f90`)](#6-gpu-batched-inference-in-parallel-loops-parallelf90)
7. [Resource Cleanup in Algorithm Entry Points](#7-resource-cleanup-in-algorithm-entry-points)
8. [SHAKE Fallback for MLIP Calculators](#8-shake-fallback-for-mlip-calculators)
9. [GFN0/GFN2 WBO Fallback for MD Length Setup](#9-gfn0gfn2-wbo-fallback-for-md-length-setup)
10. [Bug Fixes (Pre-existing)](#10-bug-fixes-pre-existing)
11. [Minor / Cosmetic Changes](#11-minor--cosmetic-changes)
12. [TOML Configuration Reference](#12-toml-configuration-reference)

---

## 1. New Calculator Backends (New Files)

### 1.1 Libtorch Direct Inference (C++ → Fortran)

Loads a TorchScript `.pt` model directly via the libtorch C++ API. Supports MACE-LAMMPS format models. No Python runtime needed.

| File | Lines | Description |
|------|-------|-------------|
| `src/calculator/libtorch_bridge.cpp` | 1,393 | C++ bridge: model loading, neighbor list construction, forward pass, energy/gradient extraction. Supports shared model (mutex-protected), batched inference, pipelined GPU execution, and multi-GPU dispatch. |
| `src/calculator/libtorch_bridge.h` | 173 | C header with `extern "C"` function declarations for Fortran interop. |
| `src/calculator/calculator_libtorch.F90` | 682 | Fortran module `calc_libtorch`: wraps C bridge functions via `iso_c_binding`. Provides `libtorch_engrad()`, `libtorch_init_shared()`, `libtorch_cleanup()`, batch/pipeline/multi-GPU entry points. Handles unit conversion (Bohr↔Å, Hartree↔eV). |

**Key features**:
- Thread-safe shared model with mutex (single model loaded into GPU memory, all OMP threads share it)
- Pipelined batch inference: overlap data transfer with GPU compute
- Multi-GPU support: round-robin dispatch across CUDA devices
- MACE-LAMMPS model format support (neighbor list + edge construction)

### 1.2 Embedded Python Inference (C → Python → Fortran)

Embeds the CPython interpreter to run UMA (fairchem-core) or MACE (mace-torch) models directly. Supports `torch.compile` and mixed precision.

| File | Lines | Description |
|------|-------|-------------|
| `src/calculator/pymlip_bridge.c` | 849 | C bridge: Python interpreter init/finalize, model loading via importlib, `atoms_to_graph` conversion, forward pass, gradient extraction. Handles GIL acquisition, reference counting, error reporting. |
| `src/calculator/pymlip_bridge.h` | 113 | C header declarations. |
| `src/calculator/calculator_pymlip.F90` | 437 | Fortran module `calc_pymlip`: wraps C bridge. Provides `pymlip_engrad()`, `pymlip_init()`, `pymlip_cleanup()`, `pymlip_finalize()`, and `pymlip_engrad_batch_f()`. |

**Key features**:
- Supports both UMA and MACE model families
- `torch.compile` modes: `reduce-overhead`, `max-autotune`
- UMA "turbo" mode: tf32 + compile + merge_mole
- Atom reference energy subtraction (YAML file)
- Batch inference for GPU singlepoint loops

### 1.3 ASE Socket Calculator (C → Fortran + Python server)

TCP socket client that connects to any ASE calculator wrapped in a Python server. Uses length-prefixed JSON protocol.

| File | Lines | Description |
|------|-------|-------------|
| `src/calculator/ase_socket_bridge.c` | 376 | C bridge: TCP socket connect/disconnect, JSON serialization of atomic numbers + positions, deserialization of energy + gradient response. |
| `src/calculator/ase_socket_bridge.h` | 69 | C header declarations. |
| `src/calculator/calculator_ase_socket.F90` | 206 | Fortran module `calc_ase_socket`: wraps C bridge. Provides `ase_socket_engrad()`, `ase_socket_init()`, `ase_socket_cleanup()`. |
| `src/python_server/crest_ase_server.py` | 338 | Python TCP server: wraps any ASE calculator class. Accepts connections, receives JSON requests, returns energy+gradient. Supports `--calculator-class`, `--port`, `--host` args. |

**Protocol**: Client sends `[4-byte length][JSON: {"atomic_numbers": [...], "positions": [[x,y,z],...]}]`, server responds with `[4-byte length][JSON: {"energy": float, "gradient": [[fx,fy,fz],...]}]`.

---

## 2. Build System Changes

### `config/CMakeLists.txt`
Two new CMake options (both default `FALSE`):
```cmake
option(WITH_LIBTORCH "Enable direct libtorch MLIP inference" FALSE)
option(WITH_PYMLIP   "Enable embedded Python MLIP inference" FALSE)
```

### `CMakeLists.txt` (root)
- **libtorch**: `enable_language(CXX)`, `find_package(Torch REQUIRED)`, sets C++17 standard, adds `WITH_LIBTORCH` compile definition
- **pymlip**: `find_package(Python3 REQUIRED COMPONENTS Development)`, adds `WITH_PYMLIP` compile definition, runtime checks for `fairchem` and `mace` Python packages
- Both link their libraries to all targets (`${TORCH_LIBRARIES}`, `Python3::Python`)
- ASE socket requires no special dependencies (POSIX sockets, always available)

### `src/calculator/CMakeLists.txt`
- Conditionally adds `libtorch_bridge.cpp` (if `WITH_LIBTORCH`) and `pymlip_bridge.c` (if `WITH_PYMLIP`)
- Always adds `ase_socket_bridge.c`
- Always adds `calculator_libtorch.F90`, `calculator_pymlip.F90`, `calculator_ase_socket.F90` (they contain stub implementations when compile flags are off)

### `meson_options.txt` + `config/meson.build`
Equivalent meson build support: two new boolean options, dependency resolution for `Torch` (cmake method) and `python3-embed`.

### `assets/template/metadata.f90`
Added `libtorchvar` and `pymlipvar` metadata strings for build info printout.

---

## 3. Calculator Data Model (`calc_type.f90`)

### New job type IDs
```fortran
integer :: libtorch   = 12
integer :: pymlip     = 13
integer :: ase_socket = 14
```

### New job descriptions
```
'MLIP direct inference via libtorch'
'MLIP inference via embedded Python'
'ASE calculator via TCP socket'
```

### New fields in `calculation_settings` type (~37 new fields)

**Libtorch fields**:
- `libtorch_handle` (c_ptr) — opaque C++ model handle
- `libtorch_model_path` — path to TorchScript `.pt` file
- `libtorch_device_id` — 0=CPU, 1=CUDA, 2=MPS, 10-13=CUDA:0-3
- `libtorch_model_format` — 0=generic, 1=MACE-LAMMPS
- `libtorch_cutoff` — neighbor list cutoff (Å, default 6.0)
- `libtorch_debug` — per-call timing output
- `libtorch_call_count`, `libtorch_total_time` — profiling counters
- `libtorch_shared_model` — share single model across threads
- `mlip_batch_size`, `mlip_aten_threads`, `mlip_ngpus` — parallelization settings

**PyMLIP fields**:
- `pymlip_handle` (c_ptr) — opaque C context handle
- `pymlip_model_type` — `'uma'` or `'mace'`
- `pymlip_model_path` — path to model checkpoint
- `pymlip_device` — `'cpu'`, `'cuda'`, `'mps'`
- `pymlip_task` — UMA task name (default `'omol'`)
- `pymlip_atom_refs` — atom reference energy YAML file
- `pymlip_compile_mode` — torch.compile mode
- `pymlip_dtype` — `'float64'` or `'float32'`
- `pymlip_turbo` — UMA turbo inference mode
- `pymlip_debug`, `pymlip_call_count`, `pymlip_total_time`

**ASE socket fields**:
- `socket_handle` (c_ptr) — opaque socket handle
- `socket_host` — server hostname (default `'localhost'`)
- `socket_port` — port number (default 6789)
- `socket_debug`, `socket_call_count`, `socket_total_time`

### Defaults logic in `autocomplete()`
For pymlip jobs: auto-sets default device (`cpu`), default model path (`uma-s-1p1` for UMA, `medium` for MACE), default task (`omol` for UMA).

### Short flags
- libtorch → `'libtorch-MACE'` or `'libtorch-MLIP'`
- pymlip → `'UMA (embed)'` or `'MACE (embed)'`

---

## 4. Calculator Dispatch (`calculator.F90`, `subprocess_engrad.f90`)

### `calculator.F90`
Three new `case` blocks in the main `engrad` dispatch (inside `engrad_core()`):
```fortran
case (jobtype%libtorch)
  call libtorch_engrad(molptr, calc%calcs(id), calc%etmp(id), ...)
case (jobtype%pymlip)
  call pymlip_engrad(molptr, calc%calcs(id), calc%etmp(id), ...)
case (jobtype%ase_socket)
  call ase_socket_engrad(molptr, calc%calcs(id), calc%etmp(id), ...)
```

Also re-exports all public routines from the three new calculator modules.

### `subprocess_engrad.f90`
Added `use` statements for `calc_libtorch`, `calc_pymlip`, `calc_ase_socket` and corresponding `public` declarations to make all routines accessible to the rest of CREST.

---

## 5. TOML Input Parsing (`parse_calcdata.f90`)

### New `method` keywords
```toml
method = "libtorch"      # or "mace-direct"
method = "pymlip"        # or "mlip-embed", "embed-mlip"
method = "uma"           # shortcut → pymlip + model_type=uma
method = "mace"          # shortcut → pymlip + model_type=mace
method = "ase-socket"    # or "socket", "ext-socket"
```

### New TOML keys (~25 new keys recognized)

| Key | Type | Applies to | Description |
|-----|------|-----------|-------------|
| `model_path` | string | libtorch, pymlip | Path to model file |
| `device` | string | libtorch, pymlip | `cpu`, `cuda`, `cuda:0`-`cuda:3`, `mps` |
| `model_type` | string | pymlip | `uma` or `mace` |
| `model_format` | string | libtorch | `generic` or `mace-lammps` |
| `cutoff` | float | libtorch | Neighbor list cutoff in Å |
| `task` | string | pymlip (UMA) | UMA task name |
| `atom_refs` | string | pymlip | Atom reference energy YAML |
| `compile_mode` | string | pymlip | `reduce-overhead`, `max-autotune` |
| `dtype` | string | pymlip | `float64` or `float32` |
| `turbo` | bool | pymlip (UMA) | Enable turbo inference |
| `batch_size` | int | libtorch, pymlip | GPU batch size (0=auto) |
| `aten_threads` | int | libtorch | ATen internal threads |
| `shared_model` | bool | libtorch | Share model across threads |
| `ngpus` | int | libtorch | Number of GPUs |
| `host` | string | ase_socket | Server hostname |
| `port` | int | ase_socket | Server port |
| `libtorch_debug` | bool | libtorch | Per-call timing |
| `pymlip_debug` | bool | pymlip | Per-call timing |
| `socket_debug` | bool | ase_socket | Per-call timing |

### Bug fix: null guard for `axis()` call
Added `if (moltmp%nat > 0 .and. allocated(moltmp%at))` guard around `axis()` call at line 92 to prevent segfault when TOML `--input` is used (structure not yet loaded when `parse_calculation_data()` is called).

---

## 6. GPU Batched Inference in Parallel Loops (`parallel.f90`)

**+606 lines** — the largest single change. Added two GPU-optimized code paths to `crest_sploop()` (the parallel singlepoint/optimization loop):

### Libtorch GPU batched path
- Packs all atomic positions into a contiguous buffer
- Calls `libtorch_engrad_batch_pipeline_f()` for single-GPU pipelined inference
- Calls `libtorch_engrad_batch_multigpu_f()` for multi-GPU round-robin inference
- Auto-selects batch size based on atom count (64 for <30 atoms, 16 for <100, 4 otherwise)
- Loads shared model once, propagates handle to all OMP threads

### PyMLIP GPU batched path
- Sequential batch processing with single GIL
- Calls `pymlip_engrad_batch_f()` in chunks
- Same auto batch-size heuristic

### Per-thread model initialization
For serial (non-batch) paths:
- Libtorch: loads shared model once, propagates `libtorch_handle` to all thread copies
- PyMLIP: loads model once (GIL-serialized), propagates `pymlip_handle` to all thread copies
- Both reset call counters per thread copy

---

## 7. Resource Cleanup in Algorithm Entry Points

Added MLIP cleanup loops (libtorch, pymlip, ase_socket) at the end of every algorithm entry point to ensure model handles and socket connections are released:

| File | Subroutine | Change |
|------|-----------|--------|
| `src/algos/singlepoint.f90` | `crest_singlepoint` | +12 lines: cleanup loop |
| `src/algos/optimization.f90` | `crest_optimization` | +12 lines: cleanup loop |
| `src/algos/dynamics.f90` | `crest_moleculardynamics` | +12 lines: cleanup loop |
| `src/algos/numhess.f90` | `crest_numhess` | +12 lines: cleanup loop |
| `src/algos/scan.f90` | `crest_scan` | +23 lines: cleanup for both `calc` and `calcclean` |
| `src/algos/setuptest.f90` | `trialMD_calculator` | +36 lines: cleanup after WBO calc, after each trial MD, and after calcstart copy |
| `src/algos/setuptest.f90` | `trialOPT_calculator` | +13 lines: cleanup after optimization |

Pattern (repeated in each):
```fortran
do j = 1, calc%ncalculations
  if (calc%calcs(j)%id == jobtype%libtorch) call libtorch_cleanup(calc%calcs(j))
  if (calc%calcs(j)%id == jobtype%pymlip)   call pymlip_cleanup(calc%calcs(j))
  if (calc%calcs(j)%id == jobtype%ase_socket) call ase_socket_cleanup(calc%calcs(j))
end do
```

### Global finalization (`src/cleanup.f90`)
Added `call pymlip_finalize()` in `custom_cleanup()` to shut down the embedded Python interpreter at program exit.

---

## 8. SHAKE Fallback for MLIP Calculators

**File**: `src/dynamics/shake_module.f90`

MLIP calculators (UMA, MACE, libtorch) do not provide Wiberg Bond Orders (WBO). The original code called `error stop` when WBO was unavailable with SHAKE mode 2 ("all bonds").

**Change**: Instead of aborting, gracefully fall back to SHAKE mode 1 (X-H bonds only):
```fortran
! Before:
write (*,*) 'No bonding information provided!'
write (*,*) 'Automatic SHAKE setup failed.'
error stop

! After:
write (*,*) 'No bonding information (WBO) available.'
write (*,*) 'Falling back to SHAKE mode 1 (X-H bonds only).'
shk%shake_mode = 1
goto 100   ! jump to X-H SHAKE setup block
```

Note: SHAKE mode 2 already skips metal bonds (existing code at lines 206-220), so this fallback primarily affects non-metal heavy-atom bonds in the ligand framework.

---

## 9. GFN0/GFN2 WBO Fallback for MD Length Setup

**File**: `src/choose_settings.f90`

The MD length setup (`md_length_setup`) requires WBOs to calculate molecular flexibility. Some MLIP environments may not have GFN0 available.

**Change**: Added cascading fallback:
1. Try GFN0-xTB singlepoint for WBOs (original behavior)
2. If GFN0 fails → try GFN2-xTB via tblite
3. If both fail → use size-based default flexibility (`flex = 0.5`)

This required extending `crest_xtbsp()` and `xtbsp()` wrapper with an optional `iostat` output argument (changes in `src/algos/singlepoint.f90` and `src/legacy_wrappers.f90`) to allow non-fatal failure propagation.

---

## 10. Bug Fixes (Pre-existing)

### 10.1 `gfnff_api.F90`: SASA access compile error

**Problem**: `gfnff_dump_sasa` accessed `ff_dat%solvation%sasa` which requires the `WITH_GBSA` preprocessor definition. However, gfnff's CMakeLists.txt defines `WITH_GBSA` as an option but never calls `add_compile_definitions(WITH_GBSA)`, so the placeholder `TBorn` type (with only `integer :: dummy`) is always used, and `sasa` doesn't exist.

**Fix**: Stubbed out the subroutine body since it's a debug utility (writes to `fort.5454`):
```fortran
!> NOTE: sasa access disabled — requires WITH_GBSA compile definition
!> which is not propagated by gfnff's CMake build system.
!> The subroutine is kept as a stub for interface compatibility.
```

### 10.2 `parse_calcdata.f90`: Segfault with `--input` TOML

**Problem**: When using `crest --input file.toml`, `parse_calculation_data()` calls `axis(moltmp%nat, moltmp%at, moltmp%xyz)` before the structure file is loaded (called from `parseinputfile()` at `confparse.f90:336`, but `inputcoords()` isn't called until line 779).

**Fix**: Added null guard:
```fortran
if (moltmp%nat > 0 .and. allocated(moltmp%at)) then
  call axis(moltmp%nat,moltmp%at,moltmp%xyz)
end if
```

### 10.3 `tblite_api.F90`: Missing `error` argument

**Problem**: `eeq_guess()` call was missing the required `error` argument (tblite API change).

**Fix**: Added `error` argument: `call eeq_guess(mctcmol,tblite%calc,tblite%wfn,error)`

### 10.4 `ancopt.f90`: Fortran format string warnings

**Problem**: Format strings like `'("("f7.2"%)")'` triggered compiler warnings due to non-standard quoting.

**Fix**: Changed to standard format: `'("(",f7.2,"%)")'` (3 occurrences).

---

## 11. Minor / Cosmetic Changes

| File | Change |
|------|--------|
| `src/confparse.f90` | Added blank line (no functional change) |
| `src/printouts.f90` | Added `WITH_LIBTORCH` and `WITH_PYMLIP` to metadata printout |

---

## 12. TOML Configuration Reference

### UMA (embedded Python)
```toml
[[calculation.level]]
method = "uma"
model_path = "/path/to/uma-s-1.pt"
device = "cuda"
task = "omol"
atom_refs = "/path/to/iso_atom_elem_refs.yaml"
turbo = true                    # optional: tf32 + compile
compile_mode = "reduce-overhead" # optional: torch.compile
```

### MACE (embedded Python)
```toml
[[calculation.level]]
method = "mace"
model_path = "/path/to/mace-model.model"
device = "cuda"
task = "mp_pbe_refit_add"        # MACE task name
compile_mode = "reduce-overhead"
dtype = "float32"                # optional: mixed precision
```

### MACE (libtorch direct)
```toml
[[calculation.level]]
method = "libtorch"
model_path = "/path/to/mace-model-lammps.pt"
model_format = "mace-lammps"
device = "cuda"
cutoff = 6.0
```

### ASE Socket
```toml
[[calculation.level]]
method = "ase-socket"
host = "127.0.0.1"
port = 6789
```

Server: `python crest_ase_server.py -c ase.calculators.emt.EMT -p 6789`
