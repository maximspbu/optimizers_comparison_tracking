#!/usr/bin/env bash
# Rent a vast.ai GPU instance and launch the optimizer comparison run.
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key <your_api_key>
#   export DOCKERHUB_USER=yourusername
#
# Usage:
#   ./deploy/create_instance.sh [gpu_type] [num_gpus] [disk_gb]
#
# Examples:
#   ./deploy/create_instance.sh RTX_4080 4 100
#   ./deploy/create_instance.sh A100_SXM4_80GB 4 200
set -euo pipefail

GPU_TYPE="${1:-RTX_4080}"
NUM_GPUS="${2:-4}"
DISK_GB="${3:-100}"
DOCKERHUB_USER="${DOCKERHUB_USER:?Set DOCKERHUB_USER}"
TAG="${IMAGE_TAG:-latest}"
IMAGE="${DOCKERHUB_USER}/optimizer-runner:${TAG}"

echo "==> Searching for cheapest ${GPU_TYPE} x${NUM_GPUS} offer ..."
OFFER_ID=$(vastai search offers \
    "gpu_name=${GPU_TYPE} num_gpus=${NUM_GPUS} disk_space>=${DISK_GB} cuda_vers>=12.4 rentable=True" \
    --order dph_total --limit 5 -q | awk 'NR==1 {print $1}')

if [[ -z "$OFFER_ID" ]]; then
    echo "ERROR: No matching offers found. Try a different gpu_type or num_gpus." >&2
    exit 1
fi

echo "==> Best offer ID: $OFFER_ID"
echo "==> Creating instance with image: $IMAGE"

INSTANCE_JSON=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --onstart "bash /workspace/deploy/onstart.sh" \
    --env "NUM_GPUS=${NUM_GPUS}" \
    --env "VAST_AI_RUN=1" \
    --raw)

INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('new_contract','unknown'))")

echo "==> Instance created: $INSTANCE_ID"
echo ""
echo "Next steps:"
echo "  Watch logs:     ./deploy/watch_instance.sh $INSTANCE_ID"
echo "  Sync results:   ./deploy/sync_results.sh   $INSTANCE_ID"
echo "  Destroy when done: ./deploy/destroy_instance.sh $INSTANCE_ID"
echo ""
echo "Instance ID saved to: .vastai_instance_id"
echo "$INSTANCE_ID" > .vastai_instance_id
