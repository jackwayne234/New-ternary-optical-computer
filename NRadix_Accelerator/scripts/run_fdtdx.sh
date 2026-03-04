#!/bin/bash
# ============================================================================
# run_fdtdx.sh — NRadix FDTD Validation Runner
#
# Wraps fdtdx_validation.py with:
#   - tmux session (survives SSH disconnects)
#   - GitHub auto-push on exit / crash / disconnect
#   - JAX GPU memory pre-allocation
#
# Usage:
#   GITHUB_TOKEN=ghp_xxx bash run_fdtdx.sh
#
# Prerequisites: results/ dir with .npy and .json files from mac_inverse_design.py
# ============================================================================
set -euo pipefail

# ---- Auto-launch tmux if not already inside one ----
if [ -z "${TMUX:-}" ]; then
    command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }
    echo "[run_fdtdx] Launching in tmux session 'fdtdx'..."
    exec tmux new-session -s fdtdx "GITHUB_TOKEN=${GITHUB_TOKEN:-} bash $(realpath "$0")"
fi

echo "[run_fdtdx] Running inside tmux session: $(tmux display-message -p '#S')"

# ---- Git configuration ----
REPO_URL="https://github.com/jackwayne234/New-ternary-optical-computer.git"
RESULTS_DIR="NRadix_Accelerator/simulation/results"

_configure_git() {
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        git config user.email "nradix-runner@runpod.io"
        git config user.name  "NRadix Runner"
        git remote set-url origin \
            "https://x-token-auth:${GITHUB_TOKEN}@github.com/jackwayne234/New-ternary-optical-computer.git"
        echo "[run_fdtdx] Git configured with token."
    else
        echo "[run_fdtdx] WARNING: GITHUB_TOKEN not set. Results won't be pushed."
    fi
}

_push_results() {
    echo ""
    echo "[run_fdtdx] ============================================"
    echo "[run_fdtdx] Saving and pushing FDTD results to GitHub..."
    echo "[run_fdtdx] ============================================"

    cd /root/New-ternary-optical-computer 2>/dev/null || \
    cd /workspace/New-ternary-optical-computer 2>/dev/null || {
        echo "[run_fdtdx] ERROR: can't find repo directory"; return 1
    }

    git fetch origin --quiet
    git add -f "${RESULTS_DIR}/fdtdx_validation.json" \
               "${RESULTS_DIR}/fdtdx_validation_mul.json" 2>/dev/null || true

    if git diff --cached --quiet; then
        echo "[run_fdtdx] No new FDTD results to commit."
    else
        git commit -m "FDTD validation results — $(date '+%Y-%m-%d %H:%M') UTC"
        git push origin HEAD
        echo "[run_fdtdx] Results pushed to GitHub."
    fi
}

# Push on any exit: clean finish, Ctrl-C, crash, SSH disconnect
trap '_push_results' EXIT SIGINT SIGTERM SIGHUP

# ---- Find repo directory ----
REPO_DIR=""
for candidate in \
    /root/New-ternary-optical-computer \
    /workspace/New-ternary-optical-computer \
    ~/New-ternary-optical-computer; do
    if [ -d "$candidate" ]; then
        REPO_DIR="$candidate"
        break
    fi
done

if [ -z "$REPO_DIR" ]; then
    echo "[run_fdtdx] Repo not found. Cloning..."
    cd /workspace 2>/dev/null || cd /root
    git clone "$REPO_URL"
    REPO_DIR="$(pwd)/New-ternary-optical-computer"
fi

cd "$REPO_DIR"
_configure_git

# Sync latest code from GitHub
echo "[run_fdtdx] Syncing repo..."
git fetch origin --quiet
git reset --hard origin/master --quiet

# ---- Check for density files ----
SIM_DIR="$REPO_DIR/NRadix_Accelerator/simulation"
if [ ! -f "$SIM_DIR/results/multiply_unit_density.npy" ] || \
   [ ! -f "$SIM_DIR/results/demux_density.npy" ]; then
    echo "[run_fdtdx] ERROR: density .npy files not found in results/."
    echo "[run_fdtdx] Run mac_inverse_design.py first to generate them."
    exit 1
fi

echo "[run_fdtdx] Found density files:"
ls -lh "$SIM_DIR/results/"*.npy

# ---- Install dependencies ----
echo ""
echo "[run_fdtdx] Installing Python dependencies..."
pip install --quiet "jax[cuda12]" numpy matplotlib scipy 2>&1 | tail -5

# ---- GPU check ----
echo ""
echo "[run_fdtdx] GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

python3 -c "
import jax
print(f'  JAX version: {jax.__version__}')
print(f'  Devices: {jax.devices()}')
" 2>/dev/null || echo "  (JAX not yet importable)"

# ---- Run validation ----
cd "$SIM_DIR"
echo ""
echo "[run_fdtdx] Starting FDTD validation..."
echo "[run_fdtdx] Logs will appear below."
echo "[run_fdtdx] Detach safely: Ctrl-B, D"
echo "[run_fdtdx] Reattach later: tmux attach -t fdtdx"
echo ""

# XLA flags for A100 performance
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

python3 fdtdx_validation.py 2>&1 | tee results/fdtdx_run.log

echo ""
echo "[run_fdtdx] Validation complete."
