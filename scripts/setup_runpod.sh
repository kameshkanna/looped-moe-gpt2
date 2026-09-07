#!/bin/bash
# Environment setup for a RunPod GPU pod (tested against the "RunPod PyTorch 2.8.0" template).
#
# Usage (on the pod, after cloning this repo):
#   bash scripts/setup_runpod.sh
#
# This creates a fresh venv (kept separate from the pod's base image site-packages, so this
# project's exact dependency versions are reproducible and don't drift with the base image),
# installs this package in editable mode, and verifies CUDA + torch.compile (Triton) both work
# before you spend any time on data prep or training.

set -euo pipefail

echo "=== looped-moe-gpt2 RunPod setup ==="

# --- Sanity checks before doing anything expensive ---
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. This script expects the RunPod PyTorch base image." >&2
    exit 1
fi

echo "--- Python version ---"
python3 --version

echo "--- GPU check ---"
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi failed -- no GPU visible to this pod. Check your RunPod GPU allocation." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- Virtual environment ---
# NOTE: RunPod's base image already has a working torch+CUDA install system-wide. Creating a
# venv WITHOUT --system-site-packages would force a from-scratch torch download (large, slow,
# and risks mismatching the pod's pre-validated CUDA driver/toolkit pairing). Use
# --system-site-packages so the venv inherits the base image's torch/CUDA stack and only adds
# this project's own dependencies on top.
if [ ! -d ".venv" ]; then
    echo "--- Creating venv (inheriting base image's torch/CUDA install) ---"
    python3 -m venv --system-site-packages .venv
else
    echo "--- .venv already exists, reusing it ---"
fi

source .venv/bin/activate

echo "--- Installing looped-moe-gpt2 in editable mode ---"
pip install --quiet --upgrade pip
pip install --quiet -e .

echo "--- Verifying torch sees the GPU ---"
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available inside the venv -- system-site-packages inheritance may have failed.'
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
"

echo "--- Verifying torch.compile works (Triton check) ---"
python3 -c "
import torch
@torch.compile
def f(x):
    return x.sin() + x.cos()
x = torch.randn(1024, device='cuda')
y = f(x)
torch.cuda.synchronize()
print('torch.compile smoke test passed.')
"

echo "--- Running test suite ---"
pip install --quiet pytest
pytest tests/ -q

echo ""
echo "=== Setup complete ==="
echo "Next: run scripts/prepare_curriculum_data.py to generate training data, then"
echo "      scripts/profile_h100_batch_size.py to find the right batch size for this GPU,"
echo "      then scripts/train.py to launch training."
