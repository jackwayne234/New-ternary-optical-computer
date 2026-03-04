#!/bin/bash
# =============================================================================
# run_and_save.sh — NRadix MAC Inverse Design Runner
# Runs mac_inverse_design.py and pushes all results to GitHub on exit.
# Works on RunPod (A100 80GB recommended).
# =============================================================================
#
# ONE-TIME RUNPOD SETUP:
# ----------------------
#   # 1. Install Python deps
#   pip install "jax[cuda12]" numpy scipy matplotlib gdstk
#
#   # 2. Clone the repo
#   git clone https://github.com/jackwayne234/New-ternary-optical-computer.git
#   cd New-ternary-optical-computer
#
#   # 3. Set your GitHub token (create one at github.com/settings/tokens)
#   export GITHUB_TOKEN=ghp_your_token_here
#
#   # 4. Run
#   bash NRadix_Accelerator/simulation/run_and_save.sh
#
# If your SSH connection drops: reconnect and run:
#   tmux attach -t nradix
#
# RESUME FROM CHECKPOINT:
#   If the run was interrupted, just re-run the same command.
#   Stages 2 and 3 will detect their checkpoints and skip re-optimization.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Auto-launch inside tmux so SSH disconnects can't kill the process
# ---------------------------------------------------------------------------
if [ -z "${TMUX:-}" ]; then
    echo ""
    echo "  Not inside tmux. Launching tmux session 'nradix'..."
    echo "  If you get disconnected, reconnect and run: tmux attach -t nradix"
    echo ""
    # Install tmux if missing (some RunPod images don't have it)
    command -v tmux >/dev/null 2>&1 || apt-get install -y tmux -q
    exec tmux new-session -s nradix "GITHUB_TOKEN=${GITHUB_TOKEN:-} bash $(realpath "$0")"
fi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GIT_EMAIL="chrisriner45@gmail.com"
GIT_NAME="Christopher Riner"
REPO_OWNER="jackwayne234"
REPO_NAME="New-ternary-optical-computer"
BRANCH="master"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# ---------------------------------------------------------------------------
# Validate GitHub token
# ---------------------------------------------------------------------------
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo ""
    echo "  ERROR: GITHUB_TOKEN is not set."
    echo ""
    echo "  Fix:"
    echo "    export GITHUB_TOKEN=ghp_your_token_here"
    echo "    bash NRadix_Accelerator/simulation/run_and_save.sh"
    echo ""
    exit 1
fi

# ---------------------------------------------------------------------------
# Configure git
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"
git config user.email "$GIT_EMAIL"
git config user.name "$GIT_NAME"
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REPO_OWNER}/${REPO_NAME}.git"

# ---------------------------------------------------------------------------
# Push-on-exit — fires on normal exit, Ctrl+C, kill, SSH drop (SIGHUP)
# ---------------------------------------------------------------------------
_push_results() {
    local exit_code=$?
    echo ""
    echo "============================================================"
    if [ $exit_code -eq 0 ]; then
        echo "  Simulation complete. Pushing results to GitHub..."
    else
        echo "  Run interrupted (exit $exit_code). Pushing checkpoints..."
    fi
    echo "============================================================"

    cd "$REPO_ROOT"

    if git add NRadix_Accelerator/simulation/results/ 2>/dev/null; then
        if git diff --cached --quiet; then
            echo "  Nothing new to push (results unchanged)."
        else
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M UTC')
            if [ $exit_code -eq 0 ]; then
                MSG="Add MAC inverse design results — ${TIMESTAMP}"
            else
                MSG="Save MAC inverse design checkpoints (interrupted) — ${TIMESTAMP}"
            fi
            git commit -m "$MSG"
            git push origin "$BRANCH"
            echo ""
            echo "  Pushed to github.com/${REPO_OWNER}/${REPO_NAME}"
        fi
    else
        echo "  WARNING: git add failed. Results may not have been pushed."
    fi
}

# Catch everything: normal exit, Ctrl+C, kill, and SSH disconnect (SIGHUP)
trap '_push_results' EXIT SIGINT SIGTERM SIGHUP

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  NRadix MAC Inverse Design"
echo "  Host:  $(hostname)"
echo "  Start: $(date)"
echo "  GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "  tmux:  session 'nradix' (reconnect: tmux attach -t nradix)"
echo "============================================================"
echo ""

mkdir -p "$RESULTS_DIR"
cd "$SCRIPT_DIR"
python mac_inverse_design.py

# EXIT trap fires here automatically
