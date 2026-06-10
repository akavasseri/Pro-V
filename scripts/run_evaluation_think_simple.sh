#!/usr/bin/env bash
set -Eeo pipefail

########################################
# Config
########################################
GPU_IDS="0"
MODEL_PATH="${MODEL_PATH:-./models/PRO-V-R1-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PRO-V-R1-8B}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_non_think}"
MAX_LEN="${MAX_LEN:-4096}"
PORT="${PORT:-8020}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-50}"
TASK_NUMBERS="${TASK_NUMBERS:-}"
VLLM_EXTRA="${VLLM_EXTRA:-}"

# Sampling settings
TEMPERATURE=0
TOP_P=0.95
TEMPERATURE_SAMPLE=0
TOP_P_SAMPLE=0.95
MAX_TOKEN=20000
ENABLE_THINKING=true
FILTER_INSTANCE=""
FOLDER_PATH="${FOLDER_PATH:-./verilog-eval/HDLBits/test_benchmark_new.json}"
BENCHMARK_FORMAT="${BENCHMARK_FORMAT:-auto}"
RUN_IDENTIFIER="gen_tb"
KEY_CFG_PATH="../key.cfg"
USE_GOLDEN_REF=true
SAMPLING_SIZE=5
STIMULI_SAMPLING_SIZE=3
MAX_TRIALS=5
STAGE=0
DAY="20250408"
DUT=false

export TEMPERATURE TOP_P TEMPERATURE_SAMPLE TOP_P_SAMPLE MAX_TOKEN \
       ENABLE_THINKING FOLDER_PATH RUN_IDENTIFIER KEY_CFG_PATH USE_GOLDEN_REF \
       SAMPLING_SIZE STIMULI_SAMPLING_SIZE MAX_TRIALS STAGE DAY DUT FILTER_INSTANCE \
       BENCHMARK_FORMAT

########################################
# Setup
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRO_V_ENV_PY="/scratch/network/ak7587/envs/pro-v/bin/python"
if [[ -n "${PRO_V_PYTHON:-}" ]]; then
  PY="${PRO_V_PYTHON}"
elif [[ -x "${PRO_V_ENV_PY}" ]]; then
  PY="${PRO_V_ENV_PY}"
else
  PY=$(command -v python3 || command -v python)
fi
[[ -z "$PY" ]] && { echo "[ERROR] Python not found"; exit 1; }

if [[ -z "${CHAT_TEMPLATE}" ]]; then
  CHAT_TEMPLATE="${REPO_ROOT}/qwen3_nonthinking.jinja"
fi
if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "[ERROR] Chat template not found: ${CHAT_TEMPLATE}" >&2
  echo "[ERROR] Set CHAT_TEMPLATE=/path/to/qwen3_nonthinking.jinja or add it to the repo root." >&2
  exit 1
fi
LOG_FILE="${REPO_ROOT}/vllm_${PORT}.log"

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
TOTAL_GPUS_NEEDED=$((TP_SIZE * DP_SIZE))
[[ ${#GPU_ARR[@]} -lt $TOTAL_GPUS_NEEDED ]] && { echo "[ERROR] Not enough GPUs"; exit 1; }
SELECTED_GPUS=("${GPU_ARR[@]:0:TOTAL_GPUS_NEEDED}")
CUDA_DEVICES=$(IFS=','; echo "${SELECTED_GPUS[*]}")

${PY} - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(0)

if not torch.cuda.is_available():
    sys.exit(0)

bad = []
for idx in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(idx)
    if (major, minor) < (7, 5):
        bad.append(f"{idx}:{torch.cuda.get_device_name(idx)} cc{major}.{minor}")

if bad:
    print("[ERROR] This Python env has torch built for CUDA 13 and does not support V100/compute capability 7.0 GPUs.", file=sys.stderr)
    print("[ERROR] Unsupported visible GPU(s): " + ", ".join(bad), file=sys.stderr)
    print("[ERROR] Request an A100 job, for example: salloc --partition=gpu --gres=gpu:nvidia_a100:1 --time=02:00:00 --mem=32G", file=sys.stderr)
    sys.exit(1)
PY

########################################
# Global variables
########################################
VLLM_PID=""
PYTHON_PID=""
CLEANUP_DONE=false

########################################
# Cleanup handler
########################################
cleanup() {
  $CLEANUP_DONE && return
  CLEANUP_DONE=true
  
  echo ""
  echo "[INFO] Cleaning up..."
  
  [[ -n "$PYTHON_PID" ]] && kill -9 $PYTHON_PID 2>/dev/null || true
  [[ -n "$VLLM_PID" ]] && {
    echo "[INFO] Stopping vLLM (PID=$VLLM_PID)"
    pkill -9 -P $VLLM_PID 2>/dev/null || true
    kill -9 $VLLM_PID 2>/dev/null || true
  }
  
  echo "[INFO] Cleanup done"
}

trap 'echo "[INFO] Interrupted"; cleanup; exit 130' INT
trap 'cleanup' EXIT

########################################
# Check for existing vLLM (3s timeout)
########################################
echo "[INFO] Checking for vLLM on port $PORT (3s)..."
VLLM_EXISTS=false

for i in {1..3}; do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] Found existing vLLM on port $PORT"
    VLLM_EXISTS=true
    break
  fi
  sleep 1
done

########################################
# Launch vLLM if not found
########################################
if ! $VLLM_EXISTS; then
  echo "[INFO] No vLLM found. Starting vLLM..."
  echo "[INFO] Model: $MODEL_PATH"
  echo "[INFO] GPUs: $GPU_IDS (TP=$TP_SIZE)"
  echo "[INFO] Max length: $MAX_LEN"
  echo "[INFO] Port: $PORT"
  echo "-----------------------------------"
  
  (
    set -x
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    ${PY} -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_LEN}" \
      --port "${PORT}" \
      --chat-template "${CHAT_TEMPLATE}" \
      ${VLLM_EXTRA} \
      >"${LOG_FILE}" 2>&1
  ) &
  VLLM_PID=$!
  echo ""
  echo "[INFO] vLLM started (PID=$VLLM_PID)"
  
  # Wait for vLLM to be ready (120s timeout)
  echo "[INFO] Waiting for vLLM to be ready..."
  for i in {1..120}; do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] vLLM is ready!"
      break
    fi
    [[ $i -eq 120 ]] && { echo "[ERROR] vLLM timeout. Check vllm_$PORT.log"; exit 1; }
    sleep 1
  done
fi

########################################
# Run evaluation
########################################
echo "-----------------------------------"
echo "[INFO] Starting evaluation..."
echo "[INFO] Experiment: $EXPERIMENT_NAME"
echo "[INFO] Concurrency: $MAX_CONCURRENCY"
[[ -n "$TASK_NUMBERS" ]] && echo "[INFO] Tasks: $TASK_NUMBERS" || echo "[INFO] Tasks: ALL"
echo "-----------------------------------"

VLLM_ENDPOINTS_CSV="http://127.0.0.1:${PORT}"
export VLLM_ENDPOINTS_CSV
$PY pro_v/prompting_top_agent_simple.py \
  --model "$SERVED_MODEL_NAME" \
  --vllm_endpoints "$VLLM_ENDPOINTS_CSV" \
  --experiment_name "$EXPERIMENT_NAME" \
  --benchmark_path "$FOLDER_PATH" \
  --benchmark_format "$BENCHMARK_FORMAT" \
  --max_concurrency $MAX_CONCURRENCY \
  ${TASK_NUMBERS:+--task_numbers $TASK_NUMBERS} &

PYTHON_PID=$!
wait $PYTHON_PID 2>/dev/null || true


echo ""
echo "[INFO] Main evaluation finished"

########################################
# Run evaluate2 - Mutant Detection
########################################


echo ""
echo "[INFO] Mutant detection evaluation completed. Report saved to: ${EVAL2_REPORT_FILE}"
echo "[INFO] All evaluations finished"
