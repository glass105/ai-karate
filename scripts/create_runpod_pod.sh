#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_IMAGE:?Set RUNPOD_IMAGE to your published container image}"
: "${RUNPOD_GPU_ID:=NVIDIA GeForce RTX 4090}"
: "${RUNPOD_POD_NAME:=ai-karate-dev}"
: "${RUNPOD_TERMINATE_AFTER:=8h}"

args=(
  pod create
  --name "${RUNPOD_POD_NAME}"
  --image "${RUNPOD_IMAGE}"
  --gpu-id "${RUNPOD_GPU_ID}"
  --gpu-count 1
  --cloud-type SECURE
  --container-disk-in-gb 40
  --volume-mount-path /workspace
  --ports "8888/http,22/tcp"
  --terminate-after "${RUNPOD_TERMINATE_AFTER}"
)

if [[ -n "${RUNPOD_NETWORK_VOLUME_ID:-}" ]]; then
  args+=(--network-volume-id "${RUNPOD_NETWORK_VOLUME_ID}")
else
  args+=(--volume-in-gb 50)
fi

runpodctl "${args[@]}"
