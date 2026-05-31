#!/usr/bin/env bash
# Build Docker image and push to Docker Hub.
# Usage: ./deploy/build_and_push.sh [tag]
#        DOCKERHUB_USER=myusername ./deploy/build_and_push.sh v1.0
set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-}"
TAG="${1:-latest}"

if [[ -z "$DOCKERHUB_USER" ]]; then
    echo "ERROR: set DOCKERHUB_USER environment variable or edit this script." >&2
    echo "  export DOCKERHUB_USER=yourusername" >&2
    exit 1
fi

IMAGE="${DOCKERHUB_USER}/optimizer-runner:${TAG}"

echo "==> Building $IMAGE ..."
docker build -t "$IMAGE" .

echo "==> Pushing $IMAGE ..."
docker push "$IMAGE"

echo "==> Done.  Image: $IMAGE"
