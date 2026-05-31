#!/usr/bin/env bash
# Stream live run.log from a vast.ai instance.
# Usage: ./deploy/watch_instance.sh <instance_id>
set -euo pipefail

INSTANCE_ID="${1:-$(cat .vastai_instance_id 2>/dev/null || echo '')}"
if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: $0 <instance_id>" >&2
    exit 1
fi

SSH_INFO=$(bash "$(dirname "$0")/ssh_info.sh" "$INSTANCE_ID")
echo "==> Connecting to instance $INSTANCE_ID ..."
# shellcheck disable=SC2086
ssh -o StrictHostKeyChecking=no $SSH_INFO \
    "tail -f /workspace/outputs/run.log"
