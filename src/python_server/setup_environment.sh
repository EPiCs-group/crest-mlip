#!/bin/bash
#
# Setup self-contained Python virtual environments for the CREST MLIP server.
# Creates separate venvs per backend to avoid dependency conflicts.
#
# Usage:
#   ./setup_environment.sh --uma                    # UMA backend only
#   ./setup_environment.sh --mace                   # MACE backend only
#   ./setup_environment.sh --all                    # both backends (separate venvs)
#   ./setup_environment.sh --uma --device cuda      # UMA with CUDA PyTorch
#   ./setup_environment.sh --mace --venv /opt/mace  # MACE in custom location
#
# Default venv locations:
#   UMA:  src/python_server/mlip_venv_uma/
#   MACE: src/python_server/mlip_venv_mace/
#
# After setup, the test runner auto-detects the correct venv:
#   cd crest_uma_test && ./run_tests.sh --uma
#   cd crest_uma_test && ./run_tests.sh --mace
#

set -e

# ---------- Defaults ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_UMA=false
INSTALL_MACE=false
DEVICE="cpu"
CUSTOM_VENV=""

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)     CUSTOM_VENV="$2"; shift 2;;
    --uma)      INSTALL_UMA=true; shift;;
    --mace)     INSTALL_MACE=true; shift;;
    --all)      INSTALL_UMA=true; INSTALL_MACE=true; shift;;
    --device)   DEVICE="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Backend selection (at least one required):"
      echo "  --uma          Install FAIRChem/UMA backend (-> mlip_venv_uma/)"
      echo "  --mace         Install MACE backend (-> mlip_venv_mace/)"
      echo "  --all          Install both backends (separate venvs)"
      echo ""
      echo "Options:"
      echo "  --venv DIR     Custom venv location (only with single backend)"
      echo "  --device STR   PyTorch device: cpu (default) or cuda"
      exit 0;;
    *)
      echo "Unknown option: $1"
      echo "Run $0 --help for usage."
      exit 1;;
  esac
done

# ---------- Validate arguments ----------
if ! $INSTALL_UMA && ! $INSTALL_MACE; then
  echo "ERROR: Specify at least one backend: --uma, --mace, or --all"
  echo "Run $0 --help for usage."
  exit 1
fi

if [ -n "$CUSTOM_VENV" ] && $INSTALL_UMA && $INSTALL_MACE; then
  echo "ERROR: --venv cannot be used with --all (each backend needs its own venv)."
  echo "  Use --venv with --uma or --mace individually."
  exit 1
fi

echo "============================================"
echo " CREST MLIP Server - Environment Setup"
echo "============================================"
echo ""

# ---------- Check Python ----------
PYTHON_BIN=""
for py in python3 python; do
  if command -v "$py" &>/dev/null; then
    PY_VERSION=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
    PY_MAJOR=$("$py" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
    PY_MINOR=$("$py" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 9 ]; then
      PYTHON_BIN="$py"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: Python >= 3.9 not found."
  echo "  Install Python 3.9+ and ensure it is on your PATH."
  exit 1
fi
echo "[OK] Found Python: $PYTHON_BIN ($PY_VERSION)"

# ---------- Check venv module ----------
if ! "$PYTHON_BIN" -c "import venv" 2>/dev/null; then
  echo "ERROR: Python venv module not available."
  echo "  On Debian/Ubuntu: sudo apt install python3-venv"
  echo "  On RHEL/Fedora:   sudo dnf install python3-venv"
  exit 1
fi
echo "[OK] venv module available"
echo ""

# ---------- Helper: create and populate a venv ----------
create_venv() {
  local VENV_DIR="$1"
  local BACKEND="$2"   # "uma" or "mace"

  echo "--------------------------------------------"
  echo " Setting up $BACKEND venv: $VENV_DIR"
  echo "--------------------------------------------"

  # Create venv
  if [ -d "$VENV_DIR" ]; then
    echo "[OK] Using existing venv: $VENV_DIR"
  else
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "[OK] Virtual environment created"
  fi

  # Activate
  source "$VENV_DIR/bin/activate"

  # Upgrade pip
  echo "--- Upgrading pip ---"
  pip install --upgrade pip setuptools wheel 2>&1 | tail -1

  # Base dependencies
  echo "--- Installing base dependencies (ase, numpy) ---"
  pip install numpy ase 2>&1 | tail -1

  # PyTorch
  echo "--- Installing PyTorch (device=$DEVICE) ---"
  if [ "$DEVICE" = "cuda" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
  else
    pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
  fi

  # Backend-specific packages
  case "$BACKEND" in
    uma)
      echo "--- Installing FAIRChem (UMA) ---"
      pip install fairchem-core huggingface_hub 2>&1 | tail -1
      ;;
    mace)
      echo "--- Installing MACE ---"
      pip install mace-torch 2>&1 | tail -1
      ;;
  esac

  # Verify
  echo "--- Verifying ---"
  python -c "import numpy; print(f'  numpy    {numpy.__version__}')"
  python -c "import ase; print(f'  ase      {ase.__version__}')"
  python -c "import torch; print(f'  torch    {torch.__version__}')" 2>/dev/null || echo "  torch    (FAILED)"
  case "$BACKEND" in
    uma)  python -c "import fairchem; print('  fairchem OK')" 2>/dev/null || echo "  fairchem (FAILED)" ;;
    mace) python -c "import mace; print('  mace     OK')" 2>/dev/null || echo "  mace     (FAILED)" ;;
  esac

  deactivate
  echo "[OK] $BACKEND venv ready: $VENV_DIR"
  echo ""
}

# ---------- Create backend venvs ----------
CREATED_VENVS=()

if $INSTALL_UMA; then
  UMA_VENV="${CUSTOM_VENV:-${SCRIPT_DIR}/mlip_venv_uma}"
  create_venv "$UMA_VENV" "uma"
  CREATED_VENVS+=("UMA:  $UMA_VENV")
fi

if $INSTALL_MACE; then
  MACE_VENV="${CUSTOM_VENV:-${SCRIPT_DIR}/mlip_venv_mace}"
  create_venv "$MACE_VENV" "mace"
  CREATED_VENVS+=("MACE: $MACE_VENV")
fi

# ---------- Summary ----------
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
for v in "${CREATED_VENVS[@]}"; do
  echo "  $v"
done
echo ""
echo "To use:"
if $INSTALL_UMA; then
  echo "  ${UMA_VENV}/bin/python mlip_server.py --calc ext_calculator_uma.py"
fi
if $INSTALL_MACE; then
  echo "  ${MACE_VENV}/bin/python mlip_server.py --calc ext_calculator_mace.py"
fi
echo ""
echo "The test runner auto-detects the correct venv:"
echo "  cd crest_uma_test && ./run_tests.sh --uma"
echo "  cd crest_uma_test && ./run_tests.sh --mace"
echo ""
