#!/usr/bin/env bash
# Launch the local PRO-V-R1-8B weights as an OpenAI-compatible vLLM server.
# Requires a CUDA GPU box with `vllm` installed (pip install vllm). This machine
# (macOS/no-CUDA) cannot run it -- copy the repo + ./models to a GPU host.
#
# Usage:
#   bash scripts/launch_vllm_local.sh            # serve on :8020
#   PORT=8020 TP=1 bash scripts/launch_vllm_local.sh
# Then point the eval at it:
#   VLLM_ENDPOINTS_CSV="http://127.0.0.1:8020" bash scripts/run_evaluation_think_simple.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/PRO-V-R1-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PRO-V-R1-8B}"
PORT="${PORT:-8020}"
TP="${TP:-1}"                 # tensor-parallel = #GPUs
MAX_LEN="${MAX_LEN:-32768}"
LOG_FILE="${REPO_ROOT}/vllm_${PORT}.log"

echo "== Preflight =="
# 1) model files present
for f in config.json tokenizer.json model.safetensors.index.json; do
  [[ -f "${MODEL_PATH}/${f}" ]] || { echo "[ERROR] missing ${MODEL_PATH}/${f}"; exit 1; }
done
shards=$(ls "${MODEL_PATH}"/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
echo "  model dir : ${MODEL_PATH} (${shards} safetensor shards)"
# 2) tooling present
command -v python >/dev/null || { echo "[ERROR] python not found"; exit 1; }
python -c "import vllm" 2>/dev/null || { echo "[ERROR] vllm not installed (pip install vllm on the GPU box)"; exit 1; }
python -c "import torch,sys; assert torch.cuda.is_available(); print('  gpus      :', torch.cuda.device_count())" \
  || { echo "[ERROR] CUDA GPU not available -- this step must run on a GPU host"; exit 1; }

echo "== Launching vLLM on :${PORT} (TP=${TP}) =="
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --port "${PORT}" \
  --trust-remote-code >"${LOG_FILE}" 2>&1 &
VLLM_PID=$!
echo "  pid ${VLLM_PID}, logging to ${LOG_FILE}"

echo "== Waiting for readiness =="
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "  ready after ${i}s"
    curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 400; echo
    echo "Endpoint: http://127.0.0.1:${PORT}   (set VLLM_ENDPOINTS_CSV to this)"
    exit 0
  fi
  sleep 1
done
echo "[ERROR] vLLM did not become ready in 120s -- see ${LOG_FILE}"; exit 1
