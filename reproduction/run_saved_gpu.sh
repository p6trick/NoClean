#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Activate conda n2n and check GPU access.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

saved_root="${N2N_PM_ROOT:?Set N2N_PM_ROOT}/Self2Self/noise2noise/temporary/data/MNIST"
run_output="reproduction/runs/mnist_saved_fgsm_$(date +%Y%m%d_%H%M%S)_$$"
python -u reproduction/run.py \
  --device cuda --output "$run_output" \
  --input-mode saved --fit-half last \
  --saved-train-attacks "$saved_root/FGSM_train(attack)" \
  --saved-eval-attacks "$saved_root/real_FGSM_valid" \
  --saved-targets "${N2N_PM_ROOT}/PGD/MNIST/FGSM/FGSM_DEP_PGD" \
  --train-size 4500 --validation-size 500 --eval-size 5000 \
  --epochs 30 --batch-size 32 \
  "$@"
