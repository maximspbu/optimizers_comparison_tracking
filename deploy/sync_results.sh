#!/usr/bin/env bash
# Rsync outputs, mlruns, and data from a vast.ai instance to local machine.
# Usage: ./deploy/sync_results.sh <instance_id> [local_base_dir]
set -euo pipefail

INSTANCE_ID="${1:-$(cat .vastai_instance_id 2>/dev/null || echo '')}"
LOCAL_BASE="${2:-.}"

if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: $0 <instance_id> [local_base_dir]" >&2
    exit 1
fi

SSH_INFO=$(bash "$(dirname "$0")/ssh_info.sh" "$INSTANCE_ID")

# Build SSH options
SSH_CMD="ssh -o StrictHostKeyChecking=no"
# Extract host and port separately for rsync
HOST=$(echo "$SSH_INFO" | awk '{print $1}')  # root@hostname
PORT=$(echo "$SSH_INFO" | awk '{print $3}')  # port number

echo "==> Syncing outputs from instance $INSTANCE_ID ..."

rsync -avz --progress \
    -e "$SSH_CMD -p $PORT" \
    "${HOST}:/workspace/outputs/" "${LOCAL_BASE}/outputs/"

echo "==> Syncing mlruns ..."
rsync -avz --progress \
    -e "$SSH_CMD -p $PORT" \
    "${HOST}:/workspace/mlruns/" "${LOCAL_BASE}/mlruns/"

echo "==> Sync complete. Results in ${LOCAL_BASE}/outputs/"
