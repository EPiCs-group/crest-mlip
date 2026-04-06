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
Export UMA or MACE models to TorchScript for direct libtorch inference.

Creates a self-contained TorchScript module that embeds:
  - Graph construction (neighbor list via torch.cdist, non-periodic)
  - One-hot node features and edge vectors
  - Model forward pass
  - Unit conversions (Bohr<->Angstrom, eV<->Hartree)

The exported model takes (positions_bohr[nat,3], atomic_numbers[nat])
and returns (energy_hartree[1], gradient_hartree_bohr[nat,3]).

Usage:
  # UMA
  python scripts/export_model.py --model-type uma --model uma-s-1p1 \\
      --output uma_exported.pt --device cuda

  # MACE
  python scripts/export_model.py --model-type mace --model medium \\
      --output mace_exported.pt --device cuda

  # Verify against Python reference
  python scripts/export_model.py --model-type uma --model uma-s-1p1 \\
      --output uma_exported.pt --verify

Requirements:
  UMA:  pip install fairchem-core
  MACE: pip install mace-torch
"""
import argparse
import sys
import os
import time
import json

import torch
import numpy as np

# Unit conversion constants
try:
    from ase.units import Hartree, Bohr
    EV_TO_HARTREE = 1.0 / Hartree
    BOHR_TO_ANGSTROM = Bohr
except ImportError:
    EV_TO_HARTREE = 0.036749322176
    BOHR_TO_ANGSTROM = 0.529177210903

# Force -> gradient: gradient = -force, with unit conversion
# force is eV/Angstrom, gradient is Hartree/Bohr
# gradient[Hartree/Bohr] = -force[eV/Ang] * EV_TO_HARTREE / (1/BOHR_TO_ANGSTROM)
#                        = -force[eV/Ang] * EV_TO_HARTREE * BOHR_TO_ANGSTROM
FORCE_TO_GRADIENT = -EV_TO_HARTREE * BOHR_TO_ANGSTROM


# ---------------------------------------------------------------------------
# Generic neighbor list in pure PyTorch (TorchScript-compatible)
# ---------------------------------------------------------------------------

def build_neighbor_list_torch(positions: torch.Tensor, cutoff: float):
    """Build neighbor list for non-periodic system using O(N^2) distance matrix.

    Args:
        positions: [nat, 3] float tensor (Angstrom)
        cutoff: neighbor cutoff radius (Angstrom)

    Returns:
        edge_index: [2, num_edges] long tensor (sender, receiver)
        edge_vectors: [num_edges, 3] float tensor (displacement vectors)
    """
    # Pairwise distance matrix [nat, nat]
    diffs = positions.unsqueeze(0) - positions.unsqueeze(1)  # [nat, nat, 3]
    dists = torch.norm(diffs, dim=-1)  # [nat, nat]

    # Mask: within cutoff and not self-loops
    mask = (dists < cutoff) & (dists > 1e-8)

    # Edge index from mask
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # [2, E]

    # Edge vectors (receiver - sender)
    senders = edge_index[0]
    receivers = edge_index[1]
    edge_vectors = positions[receivers] - positions[senders]  # [E, 3]

    return edge_index, edge_vectors


# ---------------------------------------------------------------------------
# UMA (FAIRChem) TorchScript Wrapper
# ---------------------------------------------------------------------------

class UMAWrapper(torch.nn.Module):
    """TorchScript-exportable wrapper for UMA/FAIRChem models.

    Embeds graph construction + model forward pass + unit conversions.
    Input:  positions_bohr[nat,3], atomic_numbers[nat] (int32 or int64)
    Output: (energy_hartree[1], gradient_hartree_bohr[nat,3])
    """

    def __init__(self, model, cutoff: float, max_neighbors: int = 50):
        super().__init__()
        self.model = model
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        # Store conversion constants as buffers (included in TorchScript)
        self.register_buffer('_bohr_to_ang',
                             torch.tensor(BOHR_TO_ANGSTROM, dtype=torch.float64))
        self.register_buffer('_ev_to_hartree',
                             torch.tensor(EV_TO_HARTREE, dtype=torch.float64))
        self.register_buffer('_force_to_grad',
                             torch.tensor(FORCE_TO_GRADIENT, dtype=torch.float64))

    def forward(self, positions_bohr: torch.Tensor,
                atomic_numbers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions_bohr: [nat, 3] float64, coordinates in Bohr
            atomic_numbers: [nat] int32 or int64

        Returns:
            energy_hartree: [1] float64
            gradient_hartree_bohr: [nat, 3] float64
        """
        nat = positions_bohr.shape[0]
        device = positions_bohr.device

        # Convert to model precision and units
        positions_ang = (positions_bohr * self._bohr_to_ang).to(torch.float32)
        z = atomic_numbers.to(torch.long)

        # Build neighbor list
        edge_index, _ = build_neighbor_list_torch(positions_ang, self.cutoff)

        # Compute cell offsets (zeros for non-periodic)
        num_edges = edge_index.shape[1]
        cell_offsets = torch.zeros((num_edges, 3), dtype=torch.float32, device=device)

        # Build input dict matching FAIRChem's expected format
        data = {
            "pos": positions_ang,
            "atomic_numbers": z,
            "edge_index": edge_index,
            "cell": torch.zeros((1, 3, 3), dtype=torch.float32, device=device),
            "cell_offsets": cell_offsets,
            "natoms": torch.tensor([nat], dtype=torch.long, device=device),
            "batch": torch.zeros(nat, dtype=torch.long, device=device),
            "tags": torch.zeros(nat, dtype=torch.long, device=device),
            "fixed": torch.zeros(nat, dtype=torch.long, device=device),
        }

        # Forward pass
        with torch.no_grad():
            output = self.model(data)

        # Extract energy (eV) and forces (eV/Angstrom)
        energy_ev = output["energy"].to(torch.float64)  # [1] or scalar
        forces_ev_ang = output["forces"].to(torch.float64)  # [nat, 3]

        # Convert units
        energy_hartree = energy_ev * self._ev_to_hartree
        gradient_hartree_bohr = forces_ev_ang * self._force_to_grad

        return energy_hartree.reshape(1), gradient_hartree_bohr.reshape(nat, 3)


# ---------------------------------------------------------------------------
# MACE TorchScript Wrapper
# ---------------------------------------------------------------------------

class MACEWrapper(torch.nn.Module):
    """TorchScript-exportable wrapper for MACE models.

    Embeds graph construction + model forward pass + unit conversions.
    Input:  positions_bohr[nat,3], atomic_numbers[nat]
    Output: (energy_hartree[1], gradient_hartree_bohr[nat,3])
    """

    def __init__(self, model, cutoff: float, z_table_values: torch.Tensor,
                 num_elements: int, head_index: int = 0, num_heads: int = 1,
                 use_float32: bool = False):
        super().__init__()
        self.model = model
        self.cutoff = cutoff
        self.num_elements = num_elements
        self.head_index = head_index
        self.num_heads = num_heads
        self.use_float32 = use_float32

        # z_table maps atomic number -> index in node_attrs one-hot
        # Store as buffer for TorchScript
        self.register_buffer('z_table', z_table_values)

        self.register_buffer('_bohr_to_ang',
                             torch.tensor(BOHR_TO_ANGSTROM, dtype=torch.float64))
        self.register_buffer('_ev_to_hartree',
                             torch.tensor(EV_TO_HARTREE, dtype=torch.float64))
        self.register_buffer('_force_to_grad',
                             torch.tensor(FORCE_TO_GRADIENT, dtype=torch.float64))

    def forward(self, positions_bohr: torch.Tensor,
                atomic_numbers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nat = positions_bohr.shape[0]
        device = positions_bohr.device

        compute_dtype = torch.float32 if self.use_float32 else torch.float64

        # Convert to model units (requires_grad for MACE autograd forces)
        positions_ang = (positions_bohr * self._bohr_to_ang).to(compute_dtype)
        positions_ang.requires_grad_(True)
        z = atomic_numbers.to(torch.long)

        # Build neighbor list
        edge_index, edge_vectors = build_neighbor_list_torch(
            positions_ang, self.cutoff)

        num_edges = edge_index.shape[1]

        # Build one-hot node attributes via z_table lookup
        # z_table[atomic_number] = index in one-hot encoding
        z_indices = torch.zeros(nat, dtype=torch.long, device=device)
        for i in range(nat):
            z_val = z[i].item()
            # Find index in z_table
            for j in range(self.z_table.shape[0]):
                if self.z_table[j].item() == z_val:
                    z_indices[i] = j
                    break

        node_attrs = torch.zeros(nat, self.num_elements,
                                 dtype=compute_dtype, device=device)
        node_attrs.scatter_(1, z_indices.unsqueeze(1), 1.0)

        # Build input dict for MACE
        shifts = torch.zeros((num_edges, 3), dtype=compute_dtype, device=device)
        unit_shifts = torch.zeros((num_edges, 3), dtype=compute_dtype, device=device)
        batch = torch.zeros(nat, dtype=torch.long, device=device)
        ptr = torch.tensor([0, nat], dtype=torch.long, device=device)
        cell = torch.zeros((3, 3), dtype=compute_dtype, device=device)

        # Head handling
        head = torch.zeros(nat, dtype=torch.long, device=device)
        if self.head_index > 0:
            head[:] = self.head_index

        data = {
            "positions": positions_ang,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "batch": batch,
            "ptr": ptr,
            "cell": cell,
            "head": head,
        }

        output = self.model(data, training=False)

        energy_ev = output["energy"].detach().to(torch.float64)
        forces_ev_ang = output["forces"].detach().to(torch.float64)

        energy_hartree = energy_ev * self._ev_to_hartree
        gradient_hartree_bohr = forces_ev_ang * self._force_to_grad

        return energy_hartree.reshape(1), gradient_hartree_bohr.reshape(nat, 3)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def load_uma_model(model_spec, task, device, atom_refs=None):
    """Load UMA model and extract the backbone + cutoff."""
    from fairchem.core import FAIRChemCalculator

    is_local = os.path.isfile(model_spec)

    if is_local:
        from fairchem.core.calculate.pretrained_mlip import load_predict_unit
        lpu_kwargs = {"path": model_spec, "device": device}
        if atom_refs:
            from omegaconf import OmegaConf
            refs = OmegaConf.load(atom_refs)
            lpu_kwargs["atom_refs"] = refs
        pred_unit = load_predict_unit(**lpu_kwargs)
        calc = FAIRChemCalculator(predict_unit=pred_unit, task_name=task)
    else:
        calc = FAIRChemCalculator.from_model_checkpoint(
            name_or_path=model_spec, task_name=task, device=device)

    # Extract the inner model and cutoff
    model = calc.predictor.model
    model.eval()

    # Get cutoff from the model config
    cutoff = float(calc.predictor.model_config.get(
        'cutoff', calc.predictor.model_config.get('max_radius', 6.0)))

    print(f"  Model type: UMA/FAIRChem")
    print(f"  Cutoff: {cutoff} Angstrom")
    print(f"  Device: {device}")

    return model, cutoff, calc


def load_mace_model(model_spec, device, head=None, dtype="float64"):
    """Load MACE model and extract the backbone + metadata."""
    from mace.calculators import MACECalculator
    try:
        from mace.calculators import mace_mp, mace_off, mace_omol
    except ImportError:
        from mace.calculators import mace_mp, mace_off
        mace_omol = None

    is_local = os.path.isfile(model_spec)

    if is_local:
        mace_kwargs = dict(model_paths=model_spec, device=device, default_dtype=dtype)
        if head:
            mace_kwargs['head'] = head
        calc = MACECalculator(**mace_kwargs)
    elif model_spec.startswith("off/"):
        calc = mace_off(model=model_spec[4:], device=device, default_dtype=dtype)
    elif model_spec.startswith("omol/"):
        if mace_omol is None:
            raise ImportError("mace_omol not available")
        calc = mace_omol(model=model_spec[5:], device=device, default_dtype=dtype)
    else:
        calc = mace_mp(model=model_spec, device=device, default_dtype=dtype)

    model = calc.models[0]
    model.eval()
    cutoff = float(calc.r_max)
    z_table = calc.z_table
    z_table_values = torch.tensor(z_table.zs, dtype=torch.long)
    num_elements = len(z_table.zs)

    # Determine head index
    head_index = 0
    num_heads = 1
    available_heads = getattr(calc, 'available_heads', ['Default'])
    if head and len(available_heads) > 1:
        try:
            head_index = available_heads.index(head)
        except ValueError:
            print(f"  Warning: head '{head}' not found in {available_heads}")
    num_heads = len(available_heads)

    print(f"  Model type: MACE")
    print(f"  Cutoff: {cutoff} Angstrom")
    print(f"  Z-table: {z_table.zs}")
    print(f"  Num elements: {num_elements}")
    print(f"  Heads: {available_heads} (using index {head_index})")
    print(f"  Device: {device}")

    return model, cutoff, calc, z_table_values, num_elements, head_index, num_heads


def get_test_molecule():
    """Return a water molecule for testing."""
    # H2O in Angstrom
    positions_ang = np.array([
        [0.000, 0.000, 0.119],
        [0.000, 0.763, -0.476],
        [0.000, -0.763, -0.476],
    ])
    atomic_numbers = np.array([8, 1, 1])
    positions_bohr = positions_ang / BOHR_TO_ANGSTROM
    return atomic_numbers, positions_bohr, positions_ang


def compute_reference(calc, atomic_numbers, positions_ang):
    """Compute reference energy/forces via the ASE calculator."""
    from ase import Atoms

    atoms = Atoms(numbers=atomic_numbers, positions=positions_ang)
    atoms.calc = calc

    energy_ev = atoms.get_potential_energy()
    forces_ev_ang = atoms.get_forces()

    energy_hartree = energy_ev * EV_TO_HARTREE
    gradient_hartree_bohr = forces_ev_ang * FORCE_TO_GRADIENT

    return energy_hartree, gradient_hartree_bohr


def export_uma(args):
    """Export UMA model to TorchScript."""
    print("\n=== Loading UMA model ===")
    model, cutoff, calc = load_uma_model(
        args.model, args.task, args.device, args.atom_refs)

    print("\n=== Creating TorchScript wrapper ===")
    wrapper = UMAWrapper(model, cutoff)
    wrapper.eval()
    wrapper.to(args.device)

    # Test inputs
    z_np, pos_bohr_np, pos_ang_np = get_test_molecule()
    pos_bohr = torch.tensor(pos_bohr_np, dtype=torch.float64, device=args.device)
    z = torch.tensor(z_np, dtype=torch.int64, device=args.device)

    # Test wrapper in eager mode first
    print("\n=== Testing wrapper (eager mode) ===")
    energy, gradient = wrapper(pos_bohr, z)
    print(f"  Energy:  {energy.item():.10f} Hartree")
    print(f"  |grad|:  {torch.norm(gradient).item():.6e} Hartree/Bohr")

    # Export via tracing
    print("\n=== Exporting to TorchScript (tracing) ===")
    try:
        traced = torch.jit.trace(wrapper, (pos_bohr, z))
        traced.save(args.output)
        print(f"  Saved to: {args.output}")
    except Exception as e:
        print(f"  Tracing failed: {e}")
        print("  Trying torch.jit.script instead...")
        try:
            scripted = torch.jit.script(wrapper)
            scripted.save(args.output)
            print(f"  Saved to: {args.output}")
        except Exception as e2:
            print(f"  Scripting also failed: {e2}")
            print("\n  FALLBACK: Exporting backbone model only (without wrapper).")
            print("  You will need C++ graph construction.")
            try:
                traced_model = torch.jit.trace(model, (get_uma_example_input(
                    z_np, pos_ang_np, cutoff, args.device),))
                traced_model.save(args.output)
                print(f"  Saved backbone to: {args.output}")
            except Exception as e3:
                print(f"  Backbone export also failed: {e3}")
                sys.exit(1)

    # Save metadata
    metadata = {
        "model_type": "uma",
        "cutoff": cutoff,
        "model_spec": args.model,
        "task": args.task,
        "device_exported": args.device,
        "bohr_to_angstrom": BOHR_TO_ANGSTROM,
        "ev_to_hartree": EV_TO_HARTREE,
    }
    meta_path = args.output.replace('.pt', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    # Verify
    if args.verify:
        verify_export(args.output, calc, z_np, pos_bohr_np, pos_ang_np, args.device)

    return energy, gradient


def export_mace(args):
    """Export MACE model to TorchScript."""
    dtype = "float32" if args.device == "mps" else "float64"

    print("\n=== Loading MACE model ===")
    model, cutoff, calc, z_table_values, num_elements, head_index, num_heads = \
        load_mace_model(args.model, args.device, args.head, dtype)

    print("\n=== Creating TorchScript wrapper ===")
    use_float32 = (dtype == "float32")
    wrapper = MACEWrapper(model, cutoff, z_table_values, num_elements,
                          head_index, num_heads, use_float32)
    wrapper.eval()
    wrapper.to(args.device)

    # Test inputs
    z_np, pos_bohr_np, pos_ang_np = get_test_molecule()
    pos_bohr = torch.tensor(pos_bohr_np, dtype=torch.float64, device=args.device)
    z = torch.tensor(z_np, dtype=torch.int64, device=args.device)

    print("\n=== Testing wrapper (eager mode) ===")
    energy, gradient = wrapper(pos_bohr, z)
    print(f"  Energy:  {energy.item():.10f} Hartree")
    print(f"  |grad|:  {torch.norm(gradient).item():.6e} Hartree/Bohr")

    # Export
    print("\n=== Exporting to TorchScript (tracing) ===")
    try:
        traced = torch.jit.trace(wrapper, (pos_bohr, z))
        traced.save(args.output)
        print(f"  Saved to: {args.output}")
    except Exception as e:
        print(f"  Tracing failed: {e}")
        print("  Trying torch.jit.script instead...")
        try:
            scripted = torch.jit.script(wrapper)
            scripted.save(args.output)
            print(f"  Saved to: {args.output}")
        except Exception as e2:
            print(f"  Scripting also failed: {e2}")
            sys.exit(1)

    # Save metadata
    metadata = {
        "model_type": "mace",
        "cutoff": cutoff,
        "model_spec": args.model,
        "z_table": z_table_values.tolist(),
        "num_elements": num_elements,
        "head_index": head_index,
        "num_heads": num_heads,
        "use_float32": use_float32,
        "device_exported": args.device,
        "bohr_to_angstrom": BOHR_TO_ANGSTROM,
        "ev_to_hartree": EV_TO_HARTREE,
    }
    meta_path = args.output.replace('.pt', '_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    if args.verify:
        verify_export(args.output, calc, z_np, pos_bohr_np, pos_ang_np, args.device)

    return energy, gradient


def verify_export(model_path, ase_calc, z_np, pos_bohr_np, pos_ang_np, device):
    """Verify exported TorchScript model against ASE calculator reference."""
    print("\n=== Verification ===")

    # Reference from ASE calculator
    print("  Computing ASE reference...")
    ref_energy, ref_gradient = compute_reference(ase_calc, z_np, pos_ang_np)
    print(f"  Ref energy:  {ref_energy:.10f} Hartree")
    print(f"  Ref |grad|:  {np.linalg.norm(ref_gradient):.6e} Hartree/Bohr")

    # TorchScript model
    print("  Loading exported TorchScript model...")
    loaded = torch.jit.load(model_path, map_location=device)
    loaded.eval()

    pos_bohr = torch.tensor(pos_bohr_np, dtype=torch.float64, device=device)
    z = torch.tensor(z_np, dtype=torch.int64, device=device)

    ts_energy, ts_gradient = loaded(pos_bohr, z)
    ts_energy = ts_energy.cpu().detach().numpy().item()
    ts_gradient = ts_gradient.cpu().detach().numpy()

    # Compare
    energy_diff = abs(ts_energy - ref_energy)
    grad_max_diff = np.max(np.abs(ts_gradient - ref_gradient))

    print(f"\n  TorchScript energy:  {ts_energy:.10f} Hartree")
    print(f"  Energy difference:   {energy_diff:.2e} Hartree")
    print(f"  Max gradient diff:   {grad_max_diff:.2e} Hartree/Bohr")

    energy_tol = 1e-6
    grad_tol = 1e-5

    if energy_diff < energy_tol and grad_max_diff < grad_tol:
        print(f"\n  PASS: Within tolerances (energy<{energy_tol}, grad<{grad_tol})")
    else:
        print(f"\n  WARNING: Exceeds tolerances!")
        if energy_diff >= energy_tol:
            print(f"    Energy diff {energy_diff:.2e} >= {energy_tol}")
        if grad_max_diff >= grad_tol:
            print(f"    Grad diff {grad_max_diff:.2e} >= {grad_tol}")
        print("  This may be due to float32/float64 precision differences.")
        print("  Check if results are acceptable for your use case.")

    # Benchmark
    print("\n=== Benchmark (10 calls) ===")
    times = []
    for i in range(10):
        t0 = time.monotonic()
        loaded(pos_bohr, z)
        torch.cuda.synchronize() if device == "cuda" else None
        times.append(time.monotonic() - t0)

    avg_ms = np.mean(times[1:]) * 1000  # skip first (warmup)
    print(f"  Avg time per call: {avg_ms:.2f} ms (excluding warmup)")
    print(f"  Throughput: {1000.0/avg_ms:.1f} calls/sec")


def get_uma_example_input(z_np, pos_ang_np, cutoff, device):
    """Build example FAIRChem input dict for tracing the backbone."""
    pos = torch.tensor(pos_ang_np, dtype=torch.float32, device=device)
    z = torch.tensor(z_np, dtype=torch.long, device=device)
    edge_index, _ = build_neighbor_list_torch(pos, cutoff)
    num_edges = edge_index.shape[1]
    nat = len(z_np)

    return {
        "pos": pos,
        "atomic_numbers": z,
        "edge_index": edge_index,
        "cell": torch.zeros((1, 3, 3), dtype=torch.float32, device=device),
        "cell_offsets": torch.zeros((num_edges, 3), dtype=torch.float32, device=device),
        "natoms": torch.tensor([nat], dtype=torch.long, device=device),
        "batch": torch.zeros(nat, dtype=torch.long, device=device),
        "tags": torch.zeros(nat, dtype=torch.long, device=device),
        "fixed": torch.zeros(nat, dtype=torch.long, device=device),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export UMA/MACE models to TorchScript for libtorch inference',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--model-type', required=True, choices=['uma', 'mace'],
                        help='Model family to export')
    parser.add_argument('--model', default=None,
                        help='Model name or path (default: uma-s-1p1 / medium)')
    parser.add_argument('--output', required=True,
                        help='Output TorchScript file path (.pt)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                        help='Device for export (default: cpu)')
    parser.add_argument('--task', default='omol',
                        help='UMA task name (default: omol)')
    parser.add_argument('--atom-refs', default=None,
                        help='Atom reference YAML file (UMA local models)')
    parser.add_argument('--head', default=None,
                        help='Model head for multi-head MACE models')
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported model against ASE calculator')

    args = parser.parse_args()

    # Set defaults
    if args.model is None:
        args.model = 'uma-s-1p1' if args.model_type == 'uma' else 'medium'

    print("=" * 60)
    print("MLIP Model Export to TorchScript")
    print("=" * 60)
    print(f"  Model type: {args.model_type}")
    print(f"  Model:      {args.model}")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {args.device}")

    if args.model_type == 'uma':
        export_uma(args)
    elif args.model_type == 'mace':
        export_mace(args)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
