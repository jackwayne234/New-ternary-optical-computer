#!/bin/bash
# ============================================================================
# AWS GPU Instance Setup for FDTDX Inverse Design
#
# Provisions a g5.xlarge spot instance with:
#   - 1x NVIDIA A10G GPU (24GB VRAM)
#   - CUDA 12
#   - Python 3.10+
#   - fdtdx[cuda12] + jax[cuda12]
#   - 50GB EBS volume
#
# Usage:
#   1. Configure AWS CLI: aws configure
#   2. Run: bash aws_setup_fdtdx.sh
#   3. SSH into the instance and run the simulation
#
# Cost: ~$0.40/hr with spot pricing (vs ~$1.00/hr on-demand)
# ============================================================================

set -euo pipefail

# --- Configuration ---
INSTANCE_TYPE="g5.xlarge"
AMI_ID=""  # Will be looked up dynamically
REGION="us-east-1"
KEY_NAME="fdtdx-keypair"
SECURITY_GROUP_NAME="fdtdx-sg"
VOLUME_SIZE_GB=50
SPOT_MAX_PRICE="0.50"  # Max spot price (on-demand is ~$1.00)

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

# --- Pre-flight checks ---
log "Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    err "AWS CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    err "AWS CLI not configured. Run: aws configure"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log "AWS Account: $ACCOUNT_ID, Region: $REGION"

# --- Find latest Ubuntu 22.04 AMI with NVIDIA drivers ---
log "Looking up latest Deep Learning AMI (Ubuntu 22.04)..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning AMI GPU PyTorch * (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ "$AMI_ID" = "None" ] || [ -z "$AMI_ID" ]; then
    warn "Deep Learning AMI not found, falling back to base Ubuntu 22.04..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$REGION" \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

log "AMI: $AMI_ID"

# --- Create key pair if it doesn't exist ---
if ! aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" &> /dev/null; then
    log "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair \
        --region "$REGION" \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    log "Key saved to ${KEY_NAME}.pem"
else
    log "Key pair $KEY_NAME already exists"
fi

# --- Create security group if it doesn't exist ---
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    log "Creating security group: $SECURITY_GROUP_NAME"
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "FDTDX simulation - SSH access only" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' \
        --output text)
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0
    log "Security group created: $SG_ID (SSH only)"
else
    log "Security group exists: $SG_ID"
fi

# --- User data script (runs on first boot) ---
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

# Log setup progress
exec > /var/log/fdtdx-setup.log 2>&1
echo "=== FDTDX Setup Started: $(date) ==="

# Install CUDA 12 if not already present (Deep Learning AMI has it)
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA 12..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
fi

# Set up Python virtual environment
echo "Setting up Python environment..."
apt-get install -y -qq python3.10-venv python3-pip
sudo -u ubuntu python3 -m venv /home/ubuntu/fdtdx-env
source /home/ubuntu/fdtdx-env/bin/activate

# Install JAX with CUDA 12 support
echo "Installing JAX with CUDA 12..."
pip install --upgrade pip
pip install "jax[cuda12]"

# Install FDTDX with CUDA 12 support
echo "Installing FDTDX..."
pip install "fdtdx[cuda12]"

# Install additional dependencies
pip install numpy matplotlib gdstk

# Create working directory
mkdir -p /home/ubuntu/simulation
chown -R ubuntu:ubuntu /home/ubuntu/simulation /home/ubuntu/fdtdx-env

echo "=== FDTDX Setup Complete: $(date) ==="
echo "Activate with: source /home/ubuntu/fdtdx-env/bin/activate"
USERDATA
)

USER_DATA_B64=$(echo "$USER_DATA" | base64 -w 0)

# --- Launch spot instance ---
log "Requesting spot instance: $INSTANCE_TYPE"
SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --region "$REGION" \
    --spot-price "$SPOT_MAX_PRICE" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$SG_ID\"],
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {
                \"VolumeSize\": $VOLUME_SIZE_GB,
                \"VolumeType\": \"gp3\",
                \"DeleteOnTermination\": true
            }
        }],
        \"UserData\": \"$USER_DATA_B64\"
    }" \
    --query 'SpotInstanceRequests[0].SpotInstanceRequestId' \
    --output text)

log "Spot request: $SPOT_REQUEST"
log "Waiting for instance to launch..."

# Wait for spot request to be fulfilled
for i in {1..30}; do
    STATE=$(aws ec2 describe-spot-instance-requests \
        --region "$REGION" \
        --spot-instance-request-ids "$SPOT_REQUEST" \
        --query 'SpotInstanceRequests[0].State' \
        --output text)

    if [ "$STATE" = "active" ]; then
        break
    elif [ "$STATE" = "failed" ] || [ "$STATE" = "cancelled" ]; then
        err "Spot request $STATE. Try on-demand or a different AZ."
        exit 1
    fi

    echo -n "."
    sleep 10
done
echo

INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
    --region "$REGION" \
    --spot-instance-request-ids "$SPOT_REQUEST" \
    --query 'SpotInstanceRequests[0].InstanceId' \
    --output text)

log "Instance launched: $INSTANCE_ID"

# Tag the instance
aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" \
    --tags Key=Name,Value=fdtdx-simulation Key=Project,Value=NRadix-Accelerator

# Wait for instance to be running
log "Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

log "Instance ready!"
echo ""
echo "============================================"
echo "  FDTDX GPU Instance"
echo "============================================"
echo "  Instance ID:  $INSTANCE_ID"
echo "  Public IP:    $PUBLIC_IP"
echo "  Instance:     $INSTANCE_TYPE (1x A10G, 24GB VRAM)"
echo "  Spot price:   ~\$${SPOT_MAX_PRICE}/hr max"
echo ""
echo "  SSH:"
echo "    ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "  Setup log (check after ~5 min):"
echo "    ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/fdtdx-setup.log'"
echo ""
echo "  Copy simulation scripts:"
echo "    scp -i ${KEY_NAME}.pem ../simulation/*.py ubuntu@${PUBLIC_IP}:/home/ubuntu/simulation/"
echo ""
echo "  IMPORTANT: Terminate when done to avoid charges:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo "============================================"
