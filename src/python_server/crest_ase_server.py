#!/usr/bin/env python3
# This file is part of crest.
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Modifications for MLIP support:
# Copyright (C) 2024-2026 Alexander Kolganov
#
# crest is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# crest is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with crest.  If not, see <https://www.gnu.org/licenses/>.

"""
CREST ASE Socket Server — wrap any ASE calculator for use with CREST.

Usage:
    # Option 1: Command line with built-in calculator
    python crest_ase_server.py --calculator ase.calculators.emt.EMT --port 6789

    # Option 2: Programmatic usage
    from crest_ase_server import CrestASEServer
    from ase.calculators.emt import EMT
    server = CrestASEServer(calculator=EMT(), port=6789)
    server.run()

Protocol:
    Length-prefixed JSON over TCP.
    Each message: 4-byte uint32 (network byte order) payload length, then JSON.
    Coordinates in Bohr, energies in Hartree (CREST native units).
    This server handles ASE unit conversion (Angstrom, eV) internally.
"""

import json
import signal
import socket
import struct
import sys
import threading

import numpy as np
from ase import Atoms

# Unit conversion constants
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV


class CrestASEServer:
    """TCP socket server that wraps an ASE calculator for CREST."""

    PROTOCOL_VERSION = 1

    def __init__(self, calculator, host="127.0.0.1", port=6789):
        """
        Parameters
        ----------
        calculator : ase.calculators.calculator.Calculator
            Any ASE-compatible calculator instance.
        host : str
            Bind address. Default '127.0.0.1' (localhost only).
            Use '0.0.0.0' to accept remote connections.
        port : int
            TCP port to listen on.
        """
        self.calculator = calculator
        self.host = host
        self.port = port
        self._running = False
        self._server_socket = None
        self._call_count = 0
        self._lock = threading.Lock()

    def run(self):
        """Start the server and block until CREST sends 'exit' or SIGINT."""
        self._running = True

        # Graceful shutdown on Ctrl+C
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            print("\n[crest-ase-server] Shutting down...")
            self._running = False
            if self._server_socket:
                try:
                    self._server_socket.close()
                except OSError:
                    pass

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(1)
            self._server_socket.settimeout(1.0)  # Allow checking _running flag

            print(f"[crest-ase-server] Listening on {self.host}:{self.port}")
            print(f"[crest-ase-server] Calculator: {self.calculator.__class__.__name__}")
            print("[crest-ase-server] Waiting for CREST to connect...")

            while self._running:
                try:
                    conn, addr = self._server_socket.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                print(f"[crest-ase-server] Connection from {addr[0]}:{addr[1]}")
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._handle_connection(conn)
                conn.close()
                print(f"[crest-ase-server] Connection closed ({self._call_count} calls served)")

        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            if self._server_socket:
                try:
                    self._server_socket.close()
                except OSError:
                    pass
            print("[crest-ase-server] Server stopped.")

    def _handle_connection(self, conn):
        """Handle a single CREST connection (init → engrad* → exit)."""
        try:
            # Expect init message
            msg = self._recv_msg(conn)
            if msg is None:
                return
            if msg.get("type") != "init":
                self._send_error(conn, f"Expected 'init', got '{msg.get('type')}'")
                return

            version = msg.get("version", 0)
            if version != self.PROTOCOL_VERSION:
                print(f"[crest-ase-server] WARNING: client version {version}, "
                      f"server version {self.PROTOCOL_VERSION}")

            # Send ready
            self._send_msg(conn, {"type": "ready", "version": self.PROTOCOL_VERSION})

            # Serve engrad requests until exit
            while self._running:
                msg = self._recv_msg(conn)
                if msg is None:
                    break

                msg_type = msg.get("type", "")

                if msg_type == "exit":
                    self._send_msg(conn, {"type": "bye"})
                    break

                elif msg_type == "engrad":
                    self._handle_engrad(conn, msg)

                else:
                    self._send_error(conn, f"Unknown message type: {msg_type}")

        except (ConnectionResetError, BrokenPipeError):
            print("[crest-ase-server] Connection lost")
        except Exception as e:
            print(f"[crest-ase-server] Error: {e}")

    def _handle_engrad(self, conn, msg):
        """Process an engrad request: compute energy and gradient."""
        try:
            nat = msg["nat"]
            atomic_numbers = np.array(msg["atomic_numbers"], dtype=int)
            positions_bohr = np.array(msg["positions_bohr"], dtype=float)
            charge = msg.get("charge", 0)
            uhf = msg.get("uhf", 0)

            if len(atomic_numbers) != nat:
                raise ValueError(
                    f"atomic_numbers length {len(atomic_numbers)} != nat {nat}"
                )
            if len(positions_bohr) != 3 * nat:
                raise ValueError(
                    f"positions_bohr length {len(positions_bohr)} != 3*nat {3*nat}"
                )

            # Convert positions from Bohr to Angstrom, reshape to (nat, 3)
            positions_ang = positions_bohr.reshape(nat, 3) * BOHR_TO_ANG

            # Create or update ASE Atoms object
            with self._lock:
                atoms = Atoms(numbers=atomic_numbers, positions=positions_ang, pbc=False)
                atoms.info["charge"] = charge
                atoms.info["spin"] = uhf
                atoms.calc = self.calculator

                # Compute energy (eV) and forces (eV/Angstrom)
                energy_ev = atoms.get_potential_energy()
                forces_ev_ang = atoms.get_forces()

            # Convert to CREST units
            energy_hartree = energy_ev * EV_TO_HARTREE

            # gradient = -forces, convert from eV/Ang to Hartree/Bohr
            # gradient(Hartree/Bohr) = -forces(eV/Ang) * (eV_to_Hartree) / (Ang_to_Bohr)
            #                        = -forces(eV/Ang) * EV_TO_HARTREE * BOHR_TO_ANG
            gradient_hartree_bohr = -forces_ev_ang * EV_TO_HARTREE * BOHR_TO_ANG

            # Flatten to [3*nat] array (row-major: x1,y1,z1,x2,y2,z2,...)
            gradient_flat = gradient_hartree_bohr.reshape(-1).tolist()

            self._call_count += 1
            self._send_msg(conn, {
                "type": "result",
                "energy_hartree": energy_hartree,
                "gradient_hartree_bohr": gradient_flat,
            })

        except Exception as e:
            self._send_error(conn, f"engrad failed: {e}")

    # ----------------------------------------------------------------
    # Protocol helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _send_msg(conn, obj):
        """Send a length-prefixed JSON message."""
        payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        header = struct.pack("!I", len(payload))
        conn.sendall(header + payload)

    @staticmethod
    def _recv_msg(conn):
        """Receive a length-prefixed JSON message. Returns None on disconnect."""
        header = CrestASEServer._recv_exact(conn, 4)
        if header is None:
            return None
        (payload_len,) = struct.unpack("!I", header)
        if payload_len == 0 or payload_len > 256 * 1024 * 1024:
            return None
        payload = CrestASEServer._recv_exact(conn, payload_len)
        if payload is None:
            return None
        return json.loads(payload.decode("utf-8"))

    @staticmethod
    def _recv_exact(conn, n):
        """Receive exactly n bytes. Returns None on disconnect."""
        data = bytearray()
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def _send_error(self, conn, message):
        """Send an error response."""
        print(f"[crest-ase-server] ERROR: {message}")
        self._send_msg(conn, {"type": "error", "message": message})


# ----------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CREST ASE Socket Server — wrap any ASE calculator for use with CREST.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Use ASE's EMT calculator (for testing)
  python crest_ase_server.py --calculator ase.calculators.emt.EMT

  # Use a MACE model
  python crest_ase_server.py --calculator mace.calculators.MACECalculator \\
      --kwargs model_paths=/path/to/model.pt device=cuda

  # Listen on all interfaces (for remote access)
  python crest_ase_server.py --calculator ase.calculators.emt.EMT --host 0.0.0.0
""",
    )
    parser.add_argument(
        "--calculator", "-c", required=True,
        help="Dotted path to ASE calculator class (e.g., 'ase.calculators.emt.EMT')",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=6789, help="TCP port (default: 6789)")
    parser.add_argument(
        "--kwargs", nargs="*", default=[],
        help="Key=value arguments to pass to the calculator constructor",
    )

    args = parser.parse_args()

    # Parse constructor kwargs
    kwargs = {}
    for kv in args.kwargs:
        if "=" not in kv:
            print(f"Error: --kwargs expects key=value pairs, got '{kv}'", file=sys.stderr)
            sys.exit(1)
        k, v = kv.split("=", 1)
        # Try to interpret as int, float, or bool
        if v.lower() in ("true", "false"):
            kwargs[k] = v.lower() == "true"
        else:
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    kwargs[k] = v

    # Import and instantiate calculator
    parts = args.calculator.rsplit(".", 1)
    if len(parts) != 2:
        print(f"Error: Expected 'module.ClassName', got '{args.calculator}'", file=sys.stderr)
        sys.exit(1)
    module_path, class_name = parts

    import importlib
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing calculator: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        calc = cls(**kwargs)
    except Exception as e:
        print(f"Error creating calculator: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[crest-ase-server] Calculator: {class_name}({kwargs if kwargs else ''})")

    server = CrestASEServer(calculator=calc, host=args.host, port=args.port)
    server.run()
