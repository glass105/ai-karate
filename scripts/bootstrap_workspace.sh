#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/workspace/ai-karate}"
mkdir -p \
  "${workspace}/models" \
  "${workspace}/runs" \
  "${workspace}/videos/input" \
  "${workspace}/videos/output"

echo "Workspace ready: ${workspace}"
echo "Upload a clip to ${workspace}/videos/input/fight1.mp4"
