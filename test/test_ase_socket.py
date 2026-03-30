"""
Tests for the CREST ASE Socket Server protocol.

Uses ASE's built-in EMT calculator (no MLIP model needed).
Run with: python -m pytest test/test_ase_socket.py -v
"""

import json
import socket
import struct
import subprocess
import sys
import time

import pytest


# Unit conversion constants (must match crest_ase_server.py)
BOHR_TO_ANG = 0.529177210903
EV_TO_HARTREE = 1.0 / 27.211386245988


def _send_msg(sock, obj):
    """Send a length-prefixed JSON message."""
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def _recv_msg(sock):
    """Receive a length-prefixed JSON message."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (payload_len,) = struct.unpack("!I", header)
    payload = _recv_exact(sock, payload_len)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock, n):
    """Receive exactly n bytes."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _find_free_port():
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Water molecule in Bohr (3 atoms)
WATER_NAT = 3
WATER_ATOMIC_NUMBERS = [8, 1, 1]
WATER_POSITIONS_BOHR = [
    0.0, 0.0, 0.0,
    1.8, 0.0, 0.0,
    -0.6, 1.7, 0.0,
]


@pytest.fixture
def server_port():
    """Start a CrestASEServer as a subprocess and return its port."""
    port = _find_free_port()
    proc = subprocess.Popen(
        [sys.executable, "src/python_server/crest_ase_server.py",
         "--calculator", "ase.calculators.emt.EMT",
         "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    for _ in range(50):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.1)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("Server did not start in time")

    yield port

    proc.terminate()
    proc.wait(timeout=5)


def test_init_handshake(server_port):
    """Test that the server responds to an init message with 'ready'."""
    sock = socket.create_connection(("127.0.0.1", server_port), timeout=5)
    try:
        _send_msg(sock, {"type": "init", "version": 1})
        reply = _recv_msg(sock)
        assert reply is not None
        assert reply["type"] == "ready"
        assert reply["version"] == 1

        _send_msg(sock, {"type": "exit"})
        bye = _recv_msg(sock)
        assert bye["type"] == "bye"
    finally:
        sock.close()


def test_engrad_water(server_port):
    """Test energy+gradient calculation for a water molecule via EMT."""
    sock = socket.create_connection(("127.0.0.1", server_port), timeout=5)
    try:
        # Init
        _send_msg(sock, {"type": "init", "version": 1})
        reply = _recv_msg(sock)
        assert reply["type"] == "ready"

        # Engrad
        _send_msg(sock, {
            "type": "engrad",
            "nat": WATER_NAT,
            "atomic_numbers": WATER_ATOMIC_NUMBERS,
            "positions_bohr": WATER_POSITIONS_BOHR,
            "charge": 0,
            "uhf": 0,
        })
        result = _recv_msg(sock)
        assert result is not None
        assert result["type"] == "result"

        # Energy should be a finite float (in Hartree)
        energy = result["energy_hartree"]
        assert isinstance(energy, float)
        assert abs(energy) < 1e6  # sanity: not NaN/Inf

        # Gradient should be 3*nat floats
        gradient = result["gradient_hartree_bohr"]
        assert len(gradient) == 3 * WATER_NAT
        assert all(isinstance(g, float) for g in gradient)

        # Clean exit
        _send_msg(sock, {"type": "exit"})
        bye = _recv_msg(sock)
        assert bye["type"] == "bye"
    finally:
        sock.close()


def test_multiple_engrad_calls(server_port):
    """Test that the server handles multiple sequential engrad calls."""
    sock = socket.create_connection(("127.0.0.1", server_port), timeout=5)
    try:
        _send_msg(sock, {"type": "init", "version": 1})
        reply = _recv_msg(sock)
        assert reply["type"] == "ready"

        energies = []
        for _ in range(3):
            _send_msg(sock, {
                "type": "engrad",
                "nat": WATER_NAT,
                "atomic_numbers": WATER_ATOMIC_NUMBERS,
                "positions_bohr": WATER_POSITIONS_BOHR,
            })
            result = _recv_msg(sock)
            assert result["type"] == "result"
            energies.append(result["energy_hartree"])

        # Same geometry should give same energy each time
        assert abs(energies[0] - energies[1]) < 1e-12
        assert abs(energies[0] - energies[2]) < 1e-12

        _send_msg(sock, {"type": "exit"})
        _recv_msg(sock)
    finally:
        sock.close()
