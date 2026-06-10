#!/usr/bin/env bash
#SBATCH --job-name=prov_gpu80_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/prov_gpu80_eval_%j.out
#SBATCH --error=logs/prov_gpu80_eval_%j.err

set -Eeuo pipefail

cd "${PRO_V_REPO_ROOT:-/scratch/network/ak7587/Pro-V}"
mkdir -p logs

export PRO_V_PYTHON="${PRO_V_PYTHON:-/scratch/network/ak7587/envs/pro-v/bin/python}"
export PATH="$(dirname "${PRO_V_PYTHON}"):${PATH}"
export PRO_V_MAX_CMB_VECTORS="${PRO_V_MAX_CMB_VECTORS:-100}"
export PRO_V_MAX_SEQ_SCENARIOS="${PRO_V_MAX_SEQ_SCENARIOS:-24}"
export PRO_V_MAX_SEQ_CYCLES="${PRO_V_MAX_SEQ_CYCLES:-32}"
export PRO_V_MAX_SEQ_TOTAL_CYCLES="${PRO_V_MAX_SEQ_TOTAL_CYCLES:-512}"
export MAX_LEN="${MAX_LEN:-16384}"
export MAX_TOKEN="${MAX_TOKEN:-2048}"
export EXTRA_ARGS="${EXTRA_ARGS:---max_concurrency 12 --sampling_size 0}"

echo "[INFO] Started on $(hostname) at $(date)"
nvidia-smi || true
"${PRO_V_PYTHON}" - <<'PY'
import torch
print("[INFO] torch", torch.__version__, "cuda", torch.version.cuda)
print("[INFO] cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY

bash scripts/run_evaluation_think.sh
