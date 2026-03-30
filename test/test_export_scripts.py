"""
Sanity tests for model export scripts.

These verify the scripts can be invoked without crashing.
Full model export requires PyTorch and model checkpoints (GPU tests only).
Run with: python -m pytest test/test_export_scripts.py -v
"""

import subprocess
import sys

import pytest


def _has_torch():
    """Check if PyTorch is available."""
    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True, timeout=10,
    )
    return result.returncode == 0


@pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
def test_export_model_help():
    """export_model.py --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "scripts/export_model.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "model" in result.stdout.lower()


@pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
def test_export_mace_help():
    """export_mace.py --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "scripts/export_mace.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "model" in result.stdout.lower()


def test_ase_server_help():
    """crest_ase_server.py --help should exit 0."""
    result = subprocess.run(
        [sys.executable, "src/python_server/crest_ase_server.py", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "calculator" in result.stdout.lower()


def test_export_model_missing_args():
    """export_model.py with no args should exit with error (not crash)."""
    result = subprocess.run(
        [sys.executable, "scripts/export_model.py"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0  # Should fail gracefully
