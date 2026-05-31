#!/usr/bin/env bash
# Sync results then destroy a vast.ai instance to stop billing.
# Usage: ./deploy/destroy_instance.sh <instance_id>
set -euo pipefail

INSTANCE_ID="${1:-$(cat .vastai_instance_id 2>/dev/null || echo '')}"
if [[ -z "$INSTANCE_ID" ]]; then
    echo "Usage: $0 <instance_id>" >&2
    exit 1
fi

SCRIPT_DIR="$(dirname "$0")"

echo "==> Syncing results before destroy ..."
bash "${SCRIPT_DIR}/sync_results.sh" "$INSTANCE_ID"

echo "==> Destroying instance $INSTANCE_ID ..."
vastai destroy instance "$INSTANCE_ID"

echo "==> Instance $INSTANCE_ID destroyed. Billing stopped."
echo "    Results saved to: ./outputs/  ./mlruns/"
