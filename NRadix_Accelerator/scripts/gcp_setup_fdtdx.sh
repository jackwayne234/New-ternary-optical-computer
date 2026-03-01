#!/bin/bash
# ============================================================================
# GCP GPU Instance Setup for FDTDX Inverse Design
#
# Provisions a g2-standard-4 spot VM with:
#   - 1x NVIDIA L4 GPU (24GB VRAM)
#   - CUDA 12 (pre-installed via Deep Learning VM image)
#   - Python 3.10+
#   - fdtdx[cuda12] + jax[cuda12]
#   - 50GB boot disk
#
# Usage:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install
#   2. Authenticate: gcloud auth login
#   3. Run: bash gcp_setup_fdtdx.sh
#   4. SSH into the instance and run the simulation
#
# Cost: ~$0.35/hr with spot pricing (vs ~$0.98/hr on-demand)
# ============================================================================

set -euo pipefail

# --- Configuration ---
PROJECT_ID=""  # Will be detected from gcloud config
ZONE="us-central1-a"
MACHINE_TYPE="g2-standard-4"    # 4 vCPUs, 16GB RAM, 1x NVIDIA L4 24GB
INSTANCE_NAME="fdtdx-simulation"
DISK_SIZE_GB=50
IMAGE_FAMILY="common-cu124-debian-12-py310"  # Deep Learning VM with CUDA 12.4
IMAGE_PROJECT="deeplearning-platform-release"

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# --- Pre-flight checks ---
log "Checking gcloud CLI..."
if ! command -v gcloud &> /dev/null; then
    err "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1 | grep -q '@'; then
    err "Not authenticated. Run: gcloud auth login"
    exit 1
fi

# Detect project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    err "No project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
log "GCP Project: $PROJECT_ID"
log "Account: $ACCOUNT"
log "Zone: $ZONE"

# --- Enable required APIs ---
log "Enabling Compute Engine API (if not already)..."
gcloud services enable compute.googleapis.com --project="$PROJECT_ID" 2>/dev/null || true

# --- Check GPU quota ---
log "Checking GPU quota in $ZONE..."
GPU_QUOTA=$(gcloud compute regions describe "$(echo $ZONE | sed 's/-[a-z]$//')" \
    --project="$PROJECT_ID" \
    --format="json(quotas)" 2>/dev/null | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if 'NVIDIA_L4' in q.get('metric', ''):
        print(f\"L4 quota: {q['limit']} (used: {q['usage']})\")
        if q['limit'] - q['usage'] < 1:
            print('INSUFFICIENT')
        else:
            print('OK')
        sys.exit(0)
print('L4 quota not found - you may need to request GPU quota')
print('Request at: https://console.cloud.google.com/iam-admin/quotas')
" 2>/dev/null) || true

if echo "$GPU_QUOTA" | grep -q "INSUFFICIENT"; then
    err "Insufficient GPU quota. Request more at:"
    err "https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
    err "Filter for 'NVIDIA L4' in region $(echo $ZONE | sed 's/-[a-z]$//')"
    exit 1
fi

if [ -n "$GPU_QUOTA" ]; then
    echo "$GPU_QUOTA" | while read -r line; do log "$line"; done
fi

# --- Create firewall rule for SSH (if not exists) ---
if ! gcloud compute firewall-rules describe fdtdx-allow-ssh --project="$PROJECT_ID" &>/dev/null; then
    log "Creating firewall rule for SSH..."
    gcloud compute firewall-rules create fdtdx-allow-ssh \
        --project="$PROJECT_ID" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp:22 \
        --source-ranges=0.0.0.0/0 \
        --target-tags=fdtdx \
        --description="Allow SSH for FDTDX simulation instances"
else
    log "Firewall rule fdtdx-allow-ssh already exists"
fi

# --- Startup script (runs on first boot) ---
STARTUP_SCRIPT='#!/bin/bash
set -euo pipefail

exec > /var/log/fdtdx-setup.log 2>&1
echo "=== FDTDX Setup Started: $(date) ==="

# The Deep Learning VM image has CUDA pre-installed.
# Verify CUDA is available.
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version
    nvidia-smi
else
    echo "WARNING: nvcc not found, CUDA may still be loading..."
    # Deep Learning VMs sometimes need a moment after first boot
    sleep 30
    nvidia-smi || echo "GPU driver not ready yet"
fi

# Set up Python virtual environment
echo "Setting up Python environment..."
python3 -m venv /home/fdtdx-env
source /home/fdtdx-env/bin/activate

# Install JAX with CUDA 12 support
echo "Installing JAX with CUDA 12..."
pip install --upgrade pip
pip install "jax[cuda12]"

# Install FDTDX
echo "Installing FDTDX..."
pip install "fdtdx[cuda12]"

# Install additional dependencies
pip install numpy matplotlib gdstk

# Create working directory
mkdir -p /home/simulation
chmod 777 /home/simulation /home/fdtdx-env

echo "=== FDTDX Setup Complete: $(date) ==="
echo "Activate with: source /home/fdtdx-env/bin/activate"
'

# --- Create the VM ---
log "Creating spot VM: $INSTANCE_NAME ($MACHINE_TYPE + 1x L4)..."

# Check if instance already exists
if gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    warn "Instance $INSTANCE_NAME already exists."
    warn "To delete it: gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
    warn "Continuing with existing instance..."
else
    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --accelerator=type=nvidia-l4,count=1 \
        --image-family="$IMAGE_FAMILY" \
        --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${DISK_SIZE_GB}GB" \
        --boot-disk-type=pd-balanced \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --tags=fdtdx \
        --labels=project=nradix-accelerator \
        --metadata=install-nvidia-driver=True \
        --metadata-from-file=startup-script=<(echo "$STARTUP_SCRIPT") \
        --scopes=default

    log "Instance created successfully"
fi

# --- Wait for instance to be running ---
log "Waiting for instance to be running..."
for i in {1..30}; do
    STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" \
        --format="value(status)" 2>/dev/null)

    if [ "$STATUS" = "RUNNING" ]; then
        break
    elif [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "SUSPENDED" ]; then
        err "Instance is $STATUS. It may have been preempted."
        err "Restart with: gcloud compute instances start $INSTANCE_NAME --zone=$ZONE"
        exit 1
    fi

    echo -n "."
    sleep 10
done
echo

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)")

log "Instance ready!"
echo ""
echo "============================================"
echo "  FDTDX GPU Instance (Google Cloud)"
echo "============================================"
echo "  Instance:     $INSTANCE_NAME"
echo "  Zone:         $ZONE"
echo "  Machine:      $MACHINE_TYPE (1x NVIDIA L4, 24GB VRAM)"
echo "  External IP:  $EXTERNAL_IP"
echo "  Pricing:      Spot (~\$0.35/hr)"
echo ""
echo "  SSH (gcloud handles keys automatically):"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "  Setup log (check after ~5 min):"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tail -f /var/log/fdtdx-setup.log'"
echo ""
echo "  Copy simulation scripts:"
echo "    gcloud compute scp ../simulation/*.py $INSTANCE_NAME:/home/simulation/ --zone=$ZONE"
echo ""
echo "  Run simulation:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "    source /home/fdtdx-env/bin/activate"
echo "    cd /home/simulation"
echo "    python trit_multiplier_inverse_design.py"
echo ""
echo "  Copy results back:"
echo "    gcloud compute scp $INSTANCE_NAME:/home/simulation/results/* ./results/ --zone=$ZONE"
echo ""
echo "  IMPORTANT: Stop/delete when done to avoid charges:"
echo "    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo "    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo "============================================"
