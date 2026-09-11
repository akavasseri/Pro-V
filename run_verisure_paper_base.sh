#!/usr/bin/env bash
set -Eeuo pipefail

GPU_IDS="${GPU_IDS:-0,1}"
MODEL_PATH="${MODEL_PATH:-/home/abhijna/Pro-V-paper-base/models/PRO-V-R1-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-PRO-V-R1-8B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-verisure_paper_base_smoke}"
TASK_NUMBERS="${TASK_NUMBERS-1,2,3}"
MAX_LEN="${MAX_LEN:-36000}"
PORT="${PORT:-8621}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-2}"
EXTRA_ARGS="${EXTRA_ARGS:---max_concurrency 2 --sampling_size 5}"
PRO_V_ENV_DIR="${PRO_V_ENV_DIR:-/home/abhijna/miniforge3/envs/pro-v}"
PY="${PRO_V_PYTHON:-${PRO_V_ENV_DIR}/bin/python}"

export PATH="${PRO_V_ENV_DIR}/bin:${PATH}"
export FOLDER_PATH="${FOLDER_PATH:-/home/abhijna/Pro-V-paper-base/generated_benchmarks/verisure_v2_ext_209_mutants10_labeled_strict_20260818.json}"
export RUN_IDENTIFIER="gen_tb"
export KEY_CFG_PATH="../key.cfg"
export USE_GOLDEN_REF=true
export SAMPLING_SIZE="${SAMPLING_SIZE:-5}"
export STIMULI_SAMPLING_SIZE="${STIMULI_SAMPLING_SIZE:-3}"
export MAX_TRIALS="${MAX_TRIALS:-5}"
export STAGE=0
export DAY="20250408"
export DUT=false
export TEMPERATURE=0
export TOP_P=0.1
export TEMPERATURE_SAMPLE=0.6
export TOP_P_SAMPLE=0.95
export MAX_TOKEN="${MAX_TOKEN:-20000}"
export ENABLE_THINKING=true

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if (( ${#GPU_ARR[@]} < DP_SIZE * TP_SIZE )); then
  echo "Not enough GPUs: GPU_IDS=${GPU_IDS}, need $((DP_SIZE * TP_SIZE))" >&2
  exit 1
fi

STARTED_PIDS=()
cleanup() {
  for pid in "${STARTED_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 1
      pkill -P "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup EXIT INT TERM ERR

wait_ready() {
  local port="$1"
  for _ in $(seq 1 600); do
    curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

ENDPOINTS=()
for ((i=0; i<DP_SIZE; i++)); do
  port=$((PORT+i))
  if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] Reusing existing server on ${port}"
  else
    if lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "Port ${port} occupied by non-ready service" >&2
      exit 1
    fi
    start=$((i*TP_SIZE))
    cuda=""
    for ((j=start; j<start+TP_SIZE; j++)); do
      cuda+="${GPU_ARR[j]},"
    done
    cuda="${cuda%,}"
    echo "[INFO] Launching vLLM replica ${i} on GPU(s) ${cuda}, port ${port}"
    CUDA_VISIBLE_DEVICES="${cuda}" "${PY}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_LEN}" \
      --port "${port}" \
      >"vllm_${port}.log" 2>&1 &
    STARTED_PIDS+=("$!")
    wait_ready "${port}"
  fi
  ENDPOINTS+=("http://127.0.0.1:${port}")
done

VLLM_ENDPOINTS_CSV="$(IFS=','; echo "${ENDPOINTS[*]}")"
TASK_FLAG=""
if [[ -n "${TASK_NUMBERS}" ]]; then
  TASK_FLAG="--task_numbers ${TASK_NUMBERS}"
fi

mkdir -p logs
set -x
"${PY}" pro_v/prompting_top_agent_ray.py \
  --model "${SERVED_MODEL_NAME}" \
  --vllm_endpoints "${VLLM_ENDPOINTS_CSV}" \
  --provider vllm \
  --experiment_name "${EXPERIMENT_NAME}" \
  --enable_thinking \
  ${TASK_FLAG} \
  ${EXTRA_ARGS}
set +x
