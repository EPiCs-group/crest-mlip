#!/bin/bash
# MLIP integration tests — requires GPU + conda environments.
# Usage: bash test/test_mlip_integration.sh [CREST_BINARY]
#
# Runs a quick singlepoint on ethane with each MLIP backend to verify
# the full stack works: TOML parsing → calculator init → engrad → cleanup.
# Also tests WBO cascade (GFN-FF topology) and ASE socket round-trip.

set -euo pipefail

CREST="${1:-crest}"
PASS=0
FAIL=0
TESTDIR=$(mktemp -d)
XYZ="$(cd "$(dirname "$0")/.." && pwd)/examples/ethane.xyz"
INPUTS="$(cd "$(dirname "$0")" && pwd)/inputs"

cleanup() { rm -rf "$TESTDIR"; }
trap cleanup EXIT

run_test() {
    local name="$1"; shift
    echo -n "  [$name] ... "
    if "$@" > "$TESTDIR/$name.log" 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (exit=$?)"
        tail -5 "$TESTDIR/$name.log" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}

echo "=== MLIP Integration Tests ==="
echo "Binary: $CREST"
echo "Test molecule: $XYZ"
echo

# Test 1: UMA singlepoint
echo "[1/5] UMA singlepoint"
run_test "sp_uma" bash -c "cd $TESTDIR && $CREST $XYZ --input $INPUTS/sp_uma.toml"

# Test 2: MACE-pymlip singlepoint (auto-downloads mace-mp-0-medium)
echo "[2/5] MACE-pymlip singlepoint"
run_test "sp_mace" bash -c "cd $TESTDIR && $CREST $XYZ --input $INPUTS/sp_mace_pymlip.toml"

# Test 3: ASE socket singlepoint
echo "[3/5] ASE socket singlepoint (EMT)"
run_test "sp_ase" bash -c "
    cd $TESTDIR
    python $(cd $(dirname $0)/.. && pwd)/src/python_server/crest_ase_server.py \
        --calculator ase.calculators.emt.EMT --port 6790 &
    SERVER_PID=\$!
    sleep 2
    $CREST $XYZ --input $INPUTS/sp_ase_socket.toml
    RC=\$?
    kill \$SERVER_PID 2>/dev/null || true
    wait \$SERVER_PID 2>/dev/null || true
    exit \$RC
"

# Test 4: UMA optimization (tests WBO cascade for SHAKE)
echo "[4/5] UMA optimization (WBO cascade test)"
run_test "opt_uma" bash -c "cd $TESTDIR && $CREST $XYZ --input $INPUTS/opt_uma.toml"

# Test 5: Check WBO cascade output
echo "[5/5] Verify WBO cascade messages"
if grep -qE 'GFN2-xTB WBOs|GFN-FF topology WBOs|flexibility' "$TESTDIR/opt_uma.log" 2>/dev/null; then
    echo "  [wbo_cascade] ... PASS"
    PASS=$((PASS + 1))
else
    echo "  [wbo_cascade] ... FAIL (no WBO messages found)"
    FAIL=$((FAIL + 1))
fi

echo
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
