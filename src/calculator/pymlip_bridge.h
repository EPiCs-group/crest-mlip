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
 * pymlip_bridge.h — C-linkage interface for embedded Python MLIP inference.
 *
 * Embeds CPython to call UMA/MACE calculators directly, eliminating
 * TCP socket overhead while reusing all Python model infrastructure.
 *
 * Thread safety: Handles can be shared across threads (GIL serializes).
 * The GIL is acquired/released per call via PyGILState_Ensure/Release.
 */

#ifndef PYMLIP_BRIDGE_H
#define PYMLIP_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a Python MLIP calculator context */
typedef void* pymlip_handle_t;

/*
 * Initialize the embedded Python interpreter (call once from main thread).
 * Returns 0 on success, non-zero on error.
 * err_msg: buffer for error message (at least err_len bytes)
 */
int pymlip_init_python(char* err_msg, int err_len);

/*
 * Create a new MLIP calculator instance.
 *
 * model_type:  "uma" or "mace"
 * model_path:  null-terminated path to model checkpoint
 * device:      "cpu", "cuda", or "mps"
 * task:        UMA task name (e.g., "omol"); ignored for MACE
 * atom_refs:   path to atom reference YAML (empty string = none)
 * err_msg:     buffer for error message
 * err_len:     size of err_msg buffer
 *
 * Returns: handle on success, NULL on error (message in err_msg)
 */
pymlip_handle_t pymlip_create(const char* model_type,
                               const char* model_path,
                               const char* device,
                               const char* task,
                               const char* atom_refs,
                               int charge,
                               int spin,
                               const char* compile_mode,
                               const char* dtype,
                               int turbo,
                               char* err_msg, int err_len);

/*
 * Compute energy and gradient for a molecular geometry.
 *
 * handle:          calculator handle from pymlip_create
 * nat:             number of atoms
 * positions_bohr:  flat array [3*nat] of coordinates in Bohr
 *                  Layout: x1,y1,z1,x2,y2,z2,...
 * atomic_numbers:  int array [nat] of atomic numbers
 * energy_out:      scalar output, energy in Hartree
 * gradient_out:    flat array [3*nat] output, gradient in Hartree/Bohr
 * err_msg:         buffer for error message
 * err_len:         size of err_msg buffer
 *
 * Returns: 0 on success, non-zero on error
 */
int pymlip_engrad(pymlip_handle_t handle,
                  int nat,
                  const double* positions_bohr,
                  const int* atomic_numbers,
                  double* energy_out,
                  double* gradient_out,
                  char* err_msg, int err_len);

/*
 * Batched energy+gradient: process multiple structures in one GIL acquisition.
 *
 * handle:           shared calculator handle
 * batch_size:       number of structures in the batch
 * nat:              number of atoms (same for all structures)
 * positions_batch:  flat array [batch_size * 3 * nat] in Bohr
 * atomic_numbers:   int array [nat] (shared across batch)
 * energies_out:     array [batch_size] of energies in Hartree
 * gradients_out:    flat array [batch_size * 3 * nat] of gradients
 * err_msg/err_len:  error message buffer
 *
 * Returns: 0 on success, non-zero on error (partial results may be valid)
 */
int pymlip_engrad_batch(pymlip_handle_t handle,
                        int batch_size, int nat,
                        const double* positions_batch,
                        const int* atomic_numbers,
                        double* energies_out,
                        double* gradients_out,
                        char* err_msg, int err_len);

/*
 * Release calculator and free all associated memory.
 * Safe to call with NULL handle.
 */
void pymlip_free(pymlip_handle_t handle);

/*
 * Finalize the embedded Python interpreter (call once at program exit).
 */
void pymlip_finalize_python(void);

/*
 * Query GPU memory via PyTorch (torch.cuda).
 * Returns 0 on success, non-zero if CUDA unavailable.
 * total_bytes: total GPU memory in bytes
 * free_bytes:  available GPU memory in bytes (total - allocated)
 */
int pymlip_get_gpu_memory(long long* total_bytes, long long* free_bytes);

#ifdef __cplusplus
}
#endif

#endif /* PYMLIP_BRIDGE_H */
