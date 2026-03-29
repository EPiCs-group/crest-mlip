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
 * ase_socket_bridge.h — C-linkage interface for ASE socket calculator.
 *
 * Connects to an external Python ASE calculator via TCP socket,
 * using a length-prefixed JSON protocol for energy+gradient requests.
 *
 * Thread safety: Shared handle with pthread mutex serialization.
 */

#ifndef ASE_SOCKET_BRIDGE_H
#define ASE_SOCKET_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a socket connection context */
typedef void* ase_socket_handle_t;

/*
 * Connect to an ASE socket server.
 *
 * host:     hostname or IP address (null-terminated)
 * port:     TCP port number
 * err_msg:  buffer for error message (at least err_len bytes)
 * err_len:  size of err_msg buffer
 *
 * Returns: handle on success, NULL on error (message in err_msg)
 */
ase_socket_handle_t ase_socket_connect(const char* host, int port,
                                        char* err_msg, int err_len);

/*
 * Compute energy and gradient via the socket server.
 *
 * handle:          connection handle from ase_socket_connect
 * nat:             number of atoms
 * positions_bohr:  flat array [3*nat] of coordinates in Bohr
 *                  Layout: x1,y1,z1,x2,y2,z2,...
 * atomic_numbers:  int array [nat] of atomic numbers
 * charge:          molecular charge
 * uhf:             number of unpaired electrons
 * energy_out:      scalar output, energy in Hartree
 * gradient_out:    flat array [3*nat] output, gradient in Hartree/Bohr
 * err_msg:         buffer for error message
 * err_len:         size of err_msg buffer
 *
 * Returns: 0 on success, non-zero on error
 */
int ase_socket_engrad(ase_socket_handle_t handle,
                      int nat,
                      const double* positions_bohr,
                      const int* atomic_numbers,
                      int charge, int uhf,
                      double* energy_out,
                      double* gradient_out,
                      char* err_msg, int err_len);

/*
 * Disconnect from the server and free all associated memory.
 * Safe to call with NULL handle.
 */
void ase_socket_disconnect(ase_socket_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif /* ASE_SOCKET_BRIDGE_H */
