#!/usr/bin/env bash
# Helper: print SSH connection string for a vast.ai instance.
# Used internally by other deploy scripts.
set -euo pipefail

INSTANCE_ID="${1:?Usage: ssh_info.sh <instance_id>}"

RAW=$(vastai show instance "$INSTANCE_ID" --raw)
SSH_HOST=$(echo "$RAW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ssh_host'])")
SSH_PORT=$(echo "$RAW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['ssh_port'])")

# Output: "user@host -p port"
echo "root@${SSH_HOST} -p ${SSH_PORT}"
