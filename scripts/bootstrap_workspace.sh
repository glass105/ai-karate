#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/workspace/ai-karate}"
venv="${AI_KARATE_VENV:-/root/.venvs/ai-karate}"
mkdir -p \
  "$(dirname "${venv}")" \
  "${workspace}/models" \
  "${workspace}/runs" \
  "${workspace}/videos/input" \
  "${workspace}/videos/output"

cd "${workspace}"
if [[ ! -x "${venv}/bin/python" ]]; then
  python -m venv --system-site-packages "${venv}"
fi

"${venv}/bin/python" -m pip install --upgrade pip
"${venv}/bin/python" -m pip install \
  --upgrade \
  torch==2.9.1 \
  torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu128
"${venv}/bin/python" -m pip install --no-cache-dir -r requirements.txt
"${venv}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")

print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
PY

echo "Workspace ready: ${workspace}"
echo "Virtual environment ready: ${venv}"
echo "Upload a clip to ${workspace}/videos/input/fight1.mp4"
