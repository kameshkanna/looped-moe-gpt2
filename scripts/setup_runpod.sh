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

echo "--- Installing Mamba-2 dependencies (mamba_ssm, causal-conv1d) ---"
echo "This compiles CUDA kernels from source -- expect several minutes. See"
echo "docs/mamba_investigation.md for the version pins and why they matter:"
echo "mamba-ssm 2.3.x pulls in a heavy TileLang/TVM/CUDA-13 stack that would"
echo "upgrade torch itself; BOTH mamba-ssm==2.2.4 and causal-conv1d==1.4.0's"
echo "PyPI sdists are missing their own csrc/ CUDA sources (confirmed on two"
echo "separate machines) -- both must come from their git tags instead, with"
echo "--no-build-isolation so they build against the venv's own torch rather"
echo "than resolving a fresh (and possibly CUDA-version-mismatched) one."
pip install --quiet 'transformers==4.44.2'
pip install --quiet packaging ninja setuptools wheel
pip install --quiet --no-build-isolation \
    'mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.4'
MAX_JOBS=4 pip install --quiet \
    'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0' \
    --no-build-isolation
python3 -c "
from mamba_ssm import Mamba2
import torch
x = torch.randn(2, 64, 256, device='cuda')
m = Mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=64).cuda()
y = m(x)
print(f'mamba_ssm smoke test passed: output shape {tuple(y.shape)}')
"

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
echo "Next: run scripts/prepare_data.py (or prepare_curriculum_data.py) to generate training"
echo "      data, then scripts/profile_batch_size.py --config <your config>.yaml to find the"
echo "      right batch size for THIS GPU, then scripts/train.py to launch training."
