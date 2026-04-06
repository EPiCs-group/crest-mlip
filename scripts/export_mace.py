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
Export a MACE model to TorchScript for direct libtorch inference in CREST.

MACE models are TorchScript-compatible (used in LAMMPS integration).
This script wraps the model to handle:
  - Input: positions in Bohr, atomic numbers
  - Graph construction (neighbor list via torch.cdist)
  - Model forward pass
  - Output: energy in Hartree, gradient in Hartree/Bohr

Usage:
    python export_mace.py \
        --model /path/to/mace-mh-0.model \
        --output mace_exported.pt \
        --test-xyz water.xyz \
        --device cpu \
        --cutoff 6.0
"""

import argparse
import sys
import os

import torch
import numpy as np


# Unit conversion constants
BOHR_TO_ANG = 0.529177210903
EV_TO_HARTREE = 1.0 / 27.211386245988
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


class MACETorchScriptWrapper(torch.nn.Module):
    """Wraps a MACE model for TorchScript export.

    Input:  positions_bohr [nat, 3], atomic_numbers [nat]
    Output: (energy_hartree [1], gradient_hartree_bohr [nat, 3])
    """

    def __init__(self, model, cutoff: float, z_table: list):
        super().__init__()
        self.model = model
        self.cutoff = cutoff
        self.z_table = z_table
        self.num_elements = len(z_table)

        # Register z_table as buffer for TorchScript
        self.register_buffer('z_table_tensor',
                             torch.tensor(z_table, dtype=torch.long))

    def _build_graph(self, positions: torch.Tensor) -> torch.Tensor:
        """Build neighbor list via distance matrix (O(N^2), fine for <500 atoms)."""
        dists = torch.cdist(positions.unsqueeze(0),
                           positions.unsqueeze(0)).squeeze(0)
        mask = (dists < self.cutoff) & (dists > 0.0)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        return edge_index

    def _one_hot(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """Create one-hot encoding based on z_table."""
        node_attrs = torch.zeros(atomic_numbers.shape[0], self.num_elements,
                                dtype=positions.dtype, device=positions.device)
        for i in range(self.num_elements):
            node_attrs[:, i] = (atomic_numbers == self.z_table_tensor[i]).float()
        return node_attrs

    def forward(self, positions_bohr: torch.Tensor,
                atomic_numbers: torch.Tensor):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """
        Args:
            positions_bohr: [nat, 3] atomic positions in Bohr
            atomic_numbers: [nat] atomic numbers (int64)

        Returns:
            Tuple of (energy in Hartree [1], gradient in Hartree/Bohr [nat, 3])
        """
        nat = positions_bohr.shape[0]

        # Convert Bohr -> Angstrom
        positions = positions_bohr * BOHR_TO_ANG
        positions = positions.to(torch.float32)
        positions.requires_grad_(True)

        # Build neighbor list
        edge_index = self._build_graph(positions)

        # One-hot node features
        node_attrs = torch.zeros(nat, self.num_elements,
                                dtype=torch.float32,
                                device=positions.device)
        for i in range(self.num_elements):
            z_val = self.z_table_tensor[i]
            node_attrs[:, i] = (atomic_numbers == z_val).float()

        # Edge vectors and shifts (non-periodic)
        num_edges = edge_index.shape[1]
        shifts = torch.zeros(num_edges, 3, dtype=torch.float32,
                           device=positions.device)

        # Batch info (single molecule)
        batch = torch.zeros(nat, dtype=torch.long, device=positions.device)
        ptr = torch.tensor([0, nat], dtype=torch.long,
                          device=positions.device)

        # Cell (non-periodic)
        cell = torch.zeros(3, 3, dtype=torch.float32,
                          device=positions.device)

        # Build input dict for MACE
        data = {
            'positions': positions,
            'node_attrs': node_attrs,
            'edge_index': edge_index,
            'shifts': shifts,
            'batch': batch,
            'ptr': ptr,
            'cell': cell,
            'unit_shifts': shifts,  # same as shifts for non-periodic
        }

        # Forward pass
        output = self.model(data, training=False)

        # Extract energy
        if 'energy' in output:
            energy_ev = output['energy']
        elif 'E' in output:
            energy_ev = output['E']
        else:
            energy_ev = output[list(output.keys())[0]]

        energy_ev = energy_ev.squeeze()

        # Get forces via autograd
        forces_ev_ang = torch.autograd.grad(
            energy_ev, positions,
            grad_outputs=torch.ones_like(energy_ev),
            create_graph=False,
            retain_graph=False
        )[0]

        # Convert units
        energy_hartree = energy_ev * EV_TO_HARTREE
        # gradient = -forces, eV/Ang -> Hartree/Bohr
        gradient_hartree_bohr = forces_ev_ang * EV_TO_HARTREE / BOHR_TO_ANG

        return (energy_hartree.unsqueeze(0).to(torch.float64),
                gradient_hartree_bohr.to(torch.float64))


def load_mace_model(model_path, device='cpu'):
    """Load a MACE model and extract z_table and cutoff."""
    model = torch.load(model_path, map_location=device, weights_only=False)

    # Extract cutoff
    cutoff = None
    if hasattr(model, 'r_max'):
        cutoff = float(model.r_max)
    elif hasattr(model, 'cutoff'):
        cutoff = float(model.cutoff)

    # Extract z_table (element list)
    z_table = None
    if hasattr(model, 'atomic_numbers'):
        z_table = model.atomic_numbers.tolist()
    elif hasattr(model, 'z_table'):
        if hasattr(model.z_table, 'zs'):
            z_table = list(model.z_table.zs)
        else:
            z_table = list(model.z_table)

    return model, cutoff, z_table


def test_wrapper(wrapper, test_xyz, device='cpu'):
    """Test the wrapper against reference ASE calculation."""
    from ase.io import read

    atoms = read(test_xyz)
    positions_bohr = torch.tensor(
        atoms.get_positions() / BOHR_TO_ANG,
        dtype=torch.float64, device=device)
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(),
        dtype=torch.long, device=device)

    # Wrapper result
    wrapper.eval()
    with torch.no_grad():
        energy, gradient = wrapper(positions_bohr, atomic_numbers)

    energy_val = energy.item()
    grad_np = gradient.detach().cpu().numpy()

    print(f"\nWrapper results:")
    print(f"  Energy: {energy_val:.10f} Hartree ({energy_val/EV_TO_HARTREE:.6f} eV)")
    print(f"  Max |gradient|: {np.max(np.abs(grad_np)):.8f} Hartree/Bohr")

    # ASE reference
    try:
        from mace.calculators import MACECalculator
        calc = MACECalculator(
            model_paths=[args.model],
            device=device,
            default_dtype='float64',
        )
        atoms.calc = calc
        ref_energy_ev = atoms.get_potential_energy()
        ref_forces = atoms.get_forces()
        ref_energy = ref_energy_ev * EV_TO_HARTREE
        ref_gradient = -ref_forces * EV_TO_HARTREE / BOHR_TO_ANG

        energy_diff = abs(energy_val - ref_energy)
        grad_diff = np.max(np.abs(grad_np - ref_gradient))

        print(f"\nASE reference:")
        print(f"  Energy: {ref_energy:.10f} Hartree")
        print(f"  Energy diff: {energy_diff:.2e} Hartree")
        print(f"  Max gradient diff: {grad_diff:.2e} Hartree/Bohr")

        if energy_diff < 1e-6 and grad_diff < 1e-5:
            print("  PASS: within tolerance")
        else:
            print("  WARNING: differences exceed tolerance")

    except ImportError:
        print("\n  (MACE not available for ASE reference comparison)")


def main():
    parser = argparse.ArgumentParser(
        description='Export MACE model to TorchScript for CREST libtorch inference')
    parser.add_argument('--model', required=True,
                       help='Path to MACE .model file')
    parser.add_argument('--output', default='mace_exported.pt',
                       help='Output TorchScript file (default: mace_exported.pt)')
    parser.add_argument('--device', default='cpu',
                       help='Device for export (default: cpu)')
    parser.add_argument('--cutoff', type=float, default=None,
                       help='Cutoff radius in Angstrom (auto-detected if not set)')
    parser.add_argument('--test-xyz', default=None,
                       help='XYZ file for numerical validation')
    parser.add_argument('--z-table', type=int, nargs='+', default=None,
                       help='Atomic numbers in z_table (auto-detected if not set)')

    global args
    args = parser.parse_args()

    print(f"Loading MACE model from: {args.model}")
    model, detected_cutoff, detected_z_table = load_mace_model(
        args.model, args.device)

    cutoff = args.cutoff or detected_cutoff
    z_table = args.z_table or detected_z_table

    if cutoff is None:
        print("ERROR: Could not detect cutoff. Please specify with --cutoff")
        sys.exit(1)
    if z_table is None:
        print("ERROR: Could not detect z_table. Please specify with --z-table")
        sys.exit(1)

    print(f"  Cutoff: {cutoff} Angstrom")
    print(f"  Z-table: {z_table}")

    # Create wrapper
    wrapper = MACETorchScriptWrapper(model, cutoff, z_table)
    wrapper = wrapper.to(args.device)
    wrapper.eval()

    # Test if provided
    if args.test_xyz:
        test_wrapper(wrapper, args.test_xyz, args.device)

    # Try TorchScript export via tracing
    print(f"\nAttempting TorchScript trace export...")
    try:
        # Create example inputs for tracing
        example_pos = torch.randn(3, 3, dtype=torch.float64,
                                  device=args.device) * 2.0
        example_z = torch.tensor([1, 8, 1], dtype=torch.long,
                                device=args.device)

        # Since MACE uses autograd.grad for forces, we try torch.jit.script
        # instead of trace (script handles autograd better)
        scripted = torch.jit.script(wrapper)
        scripted.save(args.output)
        print(f"  Exported via torch.jit.script to: {args.output}")

    except Exception as e:
        print(f"  torch.jit.script failed: {e}")
        print(f"\n  Trying torch.jit.trace...")
        try:
            traced = torch.jit.trace(wrapper, (example_pos, example_z))
            traced.save(args.output)
            print(f"  Exported via torch.jit.trace to: {args.output}")
        except Exception as e2:
            print(f"  torch.jit.trace also failed: {e2}")
            print(f"\n  Trying torch.save (pickle) as fallback...")
            torch.save(wrapper, args.output)
            print(f"  Saved via torch.save to: {args.output}")
            print(f"  NOTE: This requires Python+MACE to load (not pure libtorch)")
            return

    # Verify reload
    print(f"\nVerifying exported model...")
    try:
        loaded = torch.jit.load(args.output, map_location=args.device)
        print(f"  Model loaded successfully from {args.output}")

        if args.test_xyz:
            from ase.io import read
            atoms = read(args.test_xyz)
            pos = torch.tensor(atoms.get_positions() / BOHR_TO_ANG,
                             dtype=torch.float64, device=args.device)
            z = torch.tensor(atoms.get_atomic_numbers(),
                           dtype=torch.long, device=args.device)

            with torch.no_grad():
                e, g = loaded(pos, z)
            print(f"  Reload test energy: {e.item():.10f} Hartree")
            print(f"  Export successful!")

    except Exception as e:
        print(f"  Verification failed: {e}")


if __name__ == '__main__':
    main()
