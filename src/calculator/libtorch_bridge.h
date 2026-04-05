/*
 * This file is part of crest.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Modifications for MLIP support:
 * Copyright (C) 2024-2026 Alexander Kolganov
 *
 * crest is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * crest is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with crest.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * libtorch_bridge.h — C-linkage interface for TorchScript MLIP inference.
 *
 * Supports two model formats:
 *   format=0 (generic): forward(positions_bohr, atomic_numbers) -> (energy, gradient)
 *   format=1 (MACE LAMMPS): forward(Dict, local_or_ghost) -> Dict with forces
 *
 * Thread safety:
 *   - Per-thread mode: Each handle owns its own model instance.
 *   - Shared mode: Multiple threads share one model via libtorch_load_shared().
 *   - Batched mode: libtorch_engrad_batch() for GPU throughput.
 */

#ifndef LIBTORCH_BRIDGE_H
#define LIBTORCH_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a loaded TorchScript model context */
typedef void* libtorch_handle_t;

/* Model format constants */
#define LIBTORCH_FMT_GENERIC  0
#define LIBTORCH_FMT_MACE     1

/*
 * Set ATen internal thread count. Call once before any model loading.
 * For GPU: set to 1 (GPU handles parallelism).
 * For CPU: set to total_cpus / crest_threads.
 * aten_threads: number of ATen intra-op threads (0 = leave default)
 */
void libtorch_set_num_threads(int aten_threads);

/*
 * Load a TorchScript model from file (per-thread, independent copy).
 *
 * model_path:   null-terminated path to .pt file
 * device:       0 = CPU, 1 = CUDA:0, 2 = MPS
 * model_format: 0 = generic (positions,Z)->tuple, 1 = MACE LAMMPS dict format
 * cutoff:       neighbor list cutoff in Angstrom (used for MACE format)
 * err_msg:      buffer for error message (at least err_len bytes)
 * err_len:      size of err_msg buffer
 *
 * Returns: handle on success, NULL on error (message in err_msg)
 */
libtorch_handle_t libtorch_load(const char* model_path, int device,
                                 int model_format, double cutoff,
                                 char* err_msg, int err_len);

/*
 * Load or retrieve a shared model (thread-safe, idempotent).
 * All calls with the same model_path return the same handle.
 * The returned handle must NOT be passed to libtorch_free();
 * call libtorch_shared_free_all() at program exit instead.
 */
libtorch_handle_t libtorch_load_shared(const char* model_path, int device,
                                        int model_format, double cutoff,
                                        char* err_msg, int err_len);

/*
 * Compute energy and gradient for a single molecular geometry.
 */
int libtorch_engrad(libtorch_handle_t handle,
                    int nat,
                    const double* positions_bohr,
                    const int* atomic_numbers,
                    double* energy_out,
                    double* gradient_out,
                    char* err_msg, int err_len);

/*
 * Compute energy and gradient for a batch of structures (same nat and Z).
 * Designed for GPU throughput: single forward pass for all structures.
 *
 * handle:          model handle (shared or per-thread)
 * batch_size:      number of structures in the batch
 * nat:             atoms per structure (same for all)
 * positions_bohr:  flat [batch_size * nat * 3] all positions concatenated
 * atomic_numbers:  flat [nat] atomic numbers (same for all structures)
 * energies_out:    flat [batch_size] output energies in Hartree
 * gradients_out:   flat [batch_size * nat * 3] output gradients in Hartree/Bohr
 *
 * Returns: 0 on success, non-zero on error
 */
int libtorch_engrad_batch(libtorch_handle_t handle,
                          int batch_size,
                          int nat,
                          const double* positions_bohr,
                          const int* atomic_numbers,
                          double* energies_out,
                          double* gradients_out,
                          char* err_msg, int err_len);

/*
 * Pipelined batch inference (single GPU, double-buffered).
 * Overlaps CPU neighbor-list construction with GPU forward pass.
 * Processes ALL structures in one call, internally batching with pipeline.
 *
 * handle:              shared model handle on GPU
 * total_structures:    total number of structures to process
 * nat:                 atoms per structure (same for all)
 * all_positions_bohr:  flat [total_structures * nat * 3]
 * atomic_numbers:      flat [nat] (same Z for all structures)
 * batch_size:          structures per GPU batch
 * energies_out:        flat [total_structures]
 * gradients_out:       flat [total_structures * nat * 3]
 */
int libtorch_engrad_batch_pipeline(
    libtorch_handle_t handle,
    int total_structures, int nat,
    const double* all_positions_bohr,
    const int* atomic_numbers,
    int batch_size,
    double* energies_out,
    double* gradients_out,
    char* err_msg, int err_len);

/*
 * Multi-GPU pipelined batch inference.
 * Interleaves batches across GPUs with double-buffering.
 *
 * handles:  array of [ngpus] shared model handles (one per GPU)
 * ngpus:    number of GPUs to use
 * (other parameters same as pipeline version)
 */
int libtorch_engrad_batch_multigpu(
    libtorch_handle_t* handles, int ngpus,
    int total_structures, int nat,
    const double* all_positions_bohr,
    const int* atomic_numbers,
    int batch_size,
    double* energies_out,
    double* gradients_out,
    char* err_msg, int err_len);

/*
 * Load shared model on a specific CUDA device index.
 * cuda_device_index: 0 for CUDA:0, 1 for CUDA:1, etc.
 */
libtorch_handle_t libtorch_load_shared_on_device(
    const char* model_path, int cuda_device_index,
    int model_format, double cutoff,
    char* err_msg, int err_len);

/*
 * Query number of available CUDA devices. Returns 0 if CUDA unavailable.
 */
int libtorch_get_cuda_device_count(void);

/*
 * Release a per-thread model and free all associated memory.
 * Safe to call with NULL handle. Do NOT call on shared handles.
 */
void libtorch_free(libtorch_handle_t handle);

/*
 * Release all shared models. Call once at program exit.
 */
void libtorch_shared_free_all(void);

/*
 * Query device support (1 = available, 0 = not available).
 */
int libtorch_has_cuda(void);
int libtorch_has_mps(void);

/*
 * Query GPU memory via CUDA runtime API.
 * Returns 0 on success, non-zero if CUDA unavailable.
 * total_bytes: total GPU memory in bytes
 * free_bytes:  available GPU memory in bytes
 */
int libtorch_get_gpu_memory(long long* total_bytes, long long* free_bytes);

#ifdef __cplusplus
}
#endif

#endif /* LIBTORCH_BRIDGE_H */
