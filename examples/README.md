# CREST-UMA Examples

Example TOML input files for MLIP-powered calculations with CREST.

| File | Backend | Description |
|------|---------|-------------|
| `conformer_search_uma.toml` | pymlip (UMA) | GPU conformer search with Meta's UMA |
| `conformer_search_mace_libtorch.toml` | libtorch | GPU conformer search with MACE TorchScript |
| `optimization_mace_pymlip.toml` | pymlip (MACE) | Geometry optimization with MACE via embedded Python |
| `singlepoint_ase_socket.toml` | ASE socket | Single-point energy via TCP socket to ASE server |
| `ethane.xyz` | — | Simple test structure (8 atoms) |

## Usage

```bash
crest examples/ethane.xyz --input examples/conformer_search_uma.toml
```

All paths in the TOML files are relative. Adjust `model_path` to point to your
downloaded model checkpoint. See the main README for build instructions and the
full TOML keyword reference.
