#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Activate conda n2n and check GPU access.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

run_output="reproduction/runs/mnist_fgsm_$(date +%Y%m%d_%H%M%S)_$$"
python -u reproduction/run.py \
  --device cuda \
  --output "$run_output" \
  --attack fgsm --attack-epsilon 0.15 \
  --train-size 4500 --validation-size 500 --eval-size 5000 \
  --epochs 30 --batch-size 32 \
  "$@"
