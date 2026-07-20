#!/usr/bin/env bash
# Bootstrap the fine-tuning environment on an ASUS Ascent GX10 (NVIDIA GB10
# Grace-Blackwell, aarch64, 128 GB unified memory, DGX OS / Ubuntu).
#
# Design choices for this hardware:
#   * bf16 LoRA, NOT 4-bit QLoRA -- 128 GB unified memory makes quantization
#     unnecessary, and it avoids bitsandbytes, whose aarch64 support is shaky.
#   * PyTorch built for Blackwell (sm_121) -> CUDA 12.8 wheels (cu128).
#   * transformers >= 4.56 (Apertus modelling code).
#
# Usage:  bash train/setup_gb10.sh   (from the repo root on the GX10)
set -euo pipefail

echo "== hardware =="
uname -m
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || {
  echo "nvidia-smi not found -- is this the GB10 box with drivers installed?"; exit 1; }

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "WARNING: expected aarch64 (GB10); got $(uname -m). Continuing anyway."
fi

python3 -m venv .venv-train
# shellcheck disable=SC1091
source .venv-train/bin/activate
python -m pip install -U pip wheel

# PyTorch for Blackwell. If cu128 stable isn't resolvable yet, fall back to the
# nightly cu128 index (Blackwell support landed there first).
pip install torch --index-url https://download.pytorch.org/whl/cu128 || \
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

pip install "transformers>=4.56" "trl>=0.12" "peft>=0.13" \
            "datasets>=3.0" "accelerate>=1.0" "huggingface_hub>=0.25"

echo "== sanity =="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0),
          "cc", torch.cuda.get_device_capability(0))
PY

echo
echo "Setup done. To train (bf16 LoRA, sized for GB10's 128 GB):"
echo "  source .venv-train/bin/activate"
echo "  export HF_TOKEN=...   # only if the Apertus repo needs auth"
echo "  python train/sft_lora.py --quant none --batch 8 --grad-accum 2 \\"
echo "         --max-seq-len 1024 --epochs 3 --merge"
