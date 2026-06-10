#!/usr/bin/env bash
set -Eeuo pipefail

########################################
# >>> Config (edit these) <<<
########################################

GPU_IDS="0"

MODEL_PATH="./models/PRO-V-R1-8B"
SERVED_MODEL_NAME="PRO-V-R1-8B"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

EXPERIMENT_NAME="verilog_eval_$(date +%Y%m%d_%H%M%S)"

LOG_FILE="logs/${EXPERIMENT_NAME}.log"

# vLLM max context length
MAX_LEN="${MAX_LEN:-16384}"

# Base server port
PORT=8020

TP_SIZE=1
DP_SIZE=1

VLLM_EXTRA=""

EXTRA_ARGS="${EXTRA_ARGS:---max_concurrency 12 --sampling_size 4}"

# Leave empty to process all tasks.
# Resume mode will override this automatically.
TASK_NUMBERS="${TASK_NUMBERS-80,140,150}"

TEMPERATURE=0
TOP_P=0.1
TEMPERATURE_SAMPLE=0.6
TOP_P_SAMPLE=0.95
MAX_TOKEN="${MAX_TOKEN:-2048}"
ENABLE_THINKING=true

FILTER_INSTANCE=""

FOLDER_PATH="${FOLDER_PATH:-./verilog-eval/HDLBits/test_benchmark_new.json}"
BENCHMARK_FORMAT="${BENCHMARK_FORMAT:-auto}"
RUN_IDENTIFIER="gen_tb"
KEY_CFG_PATH="../key.cfg"
USE_GOLDEN_REF=true
SAMPLING_SIZE=5
STIMULI_SAMPLING_SIZE=5
MAX_TRIALS=5
STAGE=0
DAY="$(date +%Y%m%d)"
DUT=false

########################################
# Repository / Python setup
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRO_V_ENV_PY="/scratch/network/ak7587/envs/pro-v/bin/python"
if [[ -n "${PRO_V_PYTHON:-}" ]]; then
  PY="${PRO_V_PYTHON}"
elif [[ -x "${PRO_V_ENV_PY}" ]]; then
  PY="${PRO_V_ENV_PY}"
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[ERROR] Python not found. Please install python3 or adjust PATH." >&2
  exit 1
fi

if [[ -z "${CHAT_TEMPLATE}" ]]; then
  CHAT_TEMPLATE="${REPO_ROOT}/qwen3_nonthinking.jinja"
fi
if [[ ! -f "${CHAT_TEMPLATE}" ]]; then
  echo "[ERROR] Chat template not found: ${CHAT_TEMPLATE}" >&2
  echo "[ERROR] Set CHAT_TEMPLATE=/path/to/qwen3_nonthinking.jinja or add it to the repo root." >&2
  exit 1
fi

########################################
# Resume support
########################################

RESUME_EXPERIMENT="${RESUME_EXPERIMENT:-}"

if [[ -n "${RESUME_EXPERIMENT}" ]]; then
  echo "[INFO] Resume mode enabled from outputs/${RESUME_EXPERIMENT}"
  EXPERIMENT_NAME="${RESUME_EXPERIMENT}"
  LOG_FILE="logs/${EXPERIMENT_NAME}.log"

  RESUME_OUT_DIR="outputs/${EXPERIMENT_NAME}"
  BENCHMARK_PATH="${FOLDER_PATH}"

  TASK_NUMBERS=$(${PY} - <<PY
import json, os

out_dir = "${RESUME_OUT_DIR}"
benchmark_path = "${BENCHMARK_PATH}"

data = json.load(open(benchmark_path))
if isinstance(data, dict):
    tasks = [data.get("task_number")]
else:
    tasks = [t.get("task_number") for t in data]

missing = []
for t in tasks:
    if t is None:
        continue
    result_path = os.path.join(out_dir, f"task_{t}", "task_result.json")
    if not os.path.exists(result_path):
        missing.append(str(t))

print(",".join(missing))
PY
)

  if [[ -z "${TASK_NUMBERS}" ]]; then
    echo "[INFO] Resume found no missing tasks. Everything with task_result.json is already complete."
    exit 0
  fi

  echo "[INFO] Resume will run missing tasks:"
  echo "${TASK_NUMBERS}"
fi

########################################
# Exports
########################################

export TEMPERATURE TOP_P TEMPERATURE_SAMPLE TOP_P_SAMPLE MAX_TOKEN \
       ENABLE_THINKING FOLDER_PATH RUN_IDENTIFIER KEY_CFG_PATH USE_GOLDEN_REF \
       SAMPLING_SIZE STIMULI_SAMPLING_SIZE MAX_TRIALS STAGE DAY DUT BENCHMARK_FORMAT \
       FILTER_INSTANCE

export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ -z "${SERVED_MODEL_NAME}" ]]; then
  SERVED_MODEL_NAME="$(basename "${MODEL_PATH}")"
fi

command -v curl >/dev/null 2>&1 || { echo "[ERROR] 'curl' is required." >&2; exit 1; }
command -v lsof >/dev/null 2>&1 || { echo "[ERROR] 'lsof' is required for port checks." >&2; exit 1; }

########################################
# GPU parsing
########################################

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
TOTAL_GPUS_AVAILABLE="${#GPU_ARR[@]}"
TOTAL_GPUS_NEEDED=$(( TP_SIZE * DP_SIZE ))

if (( TOTAL_GPUS_NEEDED < 1 )); then
  echo "[ERROR] Invalid TOTAL_GPUS_NEEDED=${TOTAL_GPUS_NEEDED}. Check TP_SIZE and DP_SIZE." >&2
  exit 1
fi

if (( TOTAL_GPUS_AVAILABLE < TOTAL_GPUS_NEEDED )); then
  echo "[ERROR] Not enough GPUs. Available=${TOTAL_GPUS_AVAILABLE} from GPU_IDS='${GPU_IDS}', needed=${TOTAL_GPUS_NEEDED}." >&2
  exit 1
fi

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

replica_cuda_devices=()
for (( i=0; i<DP_SIZE; i++ )); do
  start=$(( i * TP_SIZE ))
  end=$(( start + TP_SIZE - 1 ))
  slice=""
  for (( j=start; j<=end; j++ )); do
    slice+="${GPU_ARR[j]},"
  done
  slice="${slice%,}"
  replica_cuda_devices+=("${slice}")
done

########################################
# Helpers
########################################

wait_for_ready() {
  local port="$1"
  local seconds="${2:-120}"
  for _ in $(seq 1 "${seconds}"); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

get_remote_model_name() {
  curl -fsS "http://127.0.0.1:${1}/v1/models" \
    | ${PY} - <<'PY' || true
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get("data", [])
    if models:
        print(models[0].get("id",""))
except Exception:
    pass
PY
}

launch_replica() {
  local idx="$1"
  local port="$2"
  local cuda="$3"
  local log_file="vllm_${port}.log"

  echo "[INFO] Launching vLLM replica #${idx}: CUDA=${cuda}, port=${port}, model='${MODEL_PATH}', served-name='${SERVED_MODEL_NAME}', TP=${TP_SIZE}, MAX_LEN=${MAX_LEN}"

  (
    set -x
    export CUDA_VISIBLE_DEVICES="${cuda}"
    export VLLM_USE_FLASHINFER_SAMPLER=0

    ${PY} -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_LEN}" \
      --port "${port}" \
      --chat-template "${CHAT_TEMPLATE}" \
      ${VLLM_EXTRA} \
      >"${log_file}" 2>&1
  ) &

  local pid=$!
  echo "[INFO] vLLM replica #${idx} started (PID=${pid}); logs: ${log_file}"
  STARTED_PIDS+=("${pid}")
  STARTED_LOGS+=("${log_file}")
  STARTED_PORTS+=("${port}")
  STARTED_FLAGS+=("launched")
}

try_link_or_launch() {
  local idx="$1"
  local port="$2"
  local cuda="$3"

  echo "[INFO] Probing port :${port} for an existing vLLM/OpenAI server (3s retry)..."
  if wait_for_ready "${port}" 3; then
    local name
    name="$(get_remote_model_name "${port}" || true)"
    if [[ -n "${name}" ]]; then
      echo "[INFO] Reusing running server on :${port} serving '${name}'."
      SERVED_MODEL_NAME="${name}"
    else
      echo "[WARN] Server on :${port} responded but model list was empty/unparseable; reusing anyway."
    fi
    STARTED_PIDS+=("")
    STARTED_LOGS+=("<external>")
    STARTED_PORTS+=("${port}")
    STARTED_FLAGS+=("existing")
    return 0
  fi

  if lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[ERROR] Port ${port} is occupied by a non-responsive service. Stop it or change PORT." >&2
    exit 1
  fi

  echo "[INFO] No server detected on :${port} within 3s. Launching our vLLM and waiting up to 120s..."
  launch_replica "${idx}" "${port}" "${cuda}"

  if ! wait_for_ready "${port}" 120; then
    echo "[ERROR] vLLM on port ${port} did not become ready in 120s. Check logs: ${STARTED_LOGS[-1]}." >&2
    exit 1
  fi

  echo "[INFO] vLLM is ready on :${port}."
}

########################################
# Launch orchestration
########################################

STARTED_PIDS=()
STARTED_LOGS=()
STARTED_PORTS=()
STARTED_FLAGS=()

cleanup() {
  local code=$?
  echo "[INFO] Cleaning up (exit code ${code})."
  for i in "${!STARTED_PIDS[@]}"; do
    local pid="${STARTED_PIDS[$i]}"
    local flag="${STARTED_FLAGS[$i]}"
    if [[ -n "${pid}" && "${flag}" == "launched" ]]; then
      if ps -p "${pid}" >/dev/null 2>&1; then
        echo "[INFO] Terminating vLLM process (PID=${pid})"
        kill "${pid}" >/dev/null 2>&1 || true
        sleep 1
        pkill -P "${pid}" >/dev/null 2>&1 || true
        if ps -p "${pid}" >/dev/null 2>&1; then
          kill -9 "${pid}" >/dev/null 2>&1 || true
        fi
      fi
    fi
  done
}
trap cleanup EXIT INT TERM ERR

if (( DP_SIZE == 1 )); then
  try_link_or_launch 0 "${PORT}" "${replica_cuda_devices[0]}"
else
  echo "[INFO] Launching ${DP_SIZE} vLLM replicas in parallel..."
  for (( i=0; i<DP_SIZE; i++ )); do
    rport=$(( PORT + i ))
    cuda="${replica_cuda_devices[$i]}"

    if wait_for_ready "${rport}" 3; then
      name="$(get_remote_model_name "${rport}" || true)"
      if [[ -n "${name}" ]]; then
        echo "[INFO] Reusing running server on :${rport} serving '${name}'."
        SERVED_MODEL_NAME="${name}"
      fi
      STARTED_PIDS+=("")
      STARTED_LOGS+=("<external>")
      STARTED_PORTS+=("${rport}")
      STARTED_FLAGS+=("existing")
    else
      if lsof -iTCP:"${rport}" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "[ERROR] Port ${rport} is occupied by a non-responsive service. Stop it or change PORT." >&2
        exit 1
      fi
      launch_replica "${i}" "${rport}" "${cuda}"
    fi
  done

  echo "[INFO] Waiting for all newly launched vLLM replicas to become ready..."
  for i in "${!STARTED_PORTS[@]}"; do
    prt="${STARTED_PORTS[$i]}"
    flag="${STARTED_FLAGS[$i]}"
    if [[ "${flag}" == "launched" ]]; then
      if ! wait_for_ready "${prt}" 120; then
        echo "[ERROR] vLLM on port ${prt} did not become ready in 120s. Check logs: ${STARTED_LOGS[$i]}." >&2
        exit 1
      fi
      echo "[INFO] vLLM is ready on :${prt}."
    fi
  done
fi

########################################
# Build endpoint list
########################################

ENDPOINTS=()
for prt in "${STARTED_PORTS[@]}"; do
  ENDPOINTS+=("http://127.0.0.1:${prt}")
done

export VLLM_ENDPOINTS_CSV
VLLM_ENDPOINTS_CSV="$(IFS=','; echo "${ENDPOINTS[*]}")"

echo "[INFO] Active endpoints: ${VLLM_ENDPOINTS_CSV}"
echo "[INFO] Model served as: ${SERVED_MODEL_NAME}"
echo "[INFO] TP_SIZE=${TP_SIZE}, DP_SIZE=${DP_SIZE}, Total GPUs used=${TOTAL_GPUS_NEEDED}"

########################################
# Run main evaluation
########################################

echo "[INFO] Running prompting_top_agent_ray.py"
echo "[INFO] Load balancing across ${DP_SIZE} vLLM replica(s) will be handled by Ray"

THINKING_FLAG=""

TASK_NUMBERS_FLAG=""
if [[ -n "${TASK_NUMBERS}" ]]; then
  TASK_NUMBERS_FLAG="--task_numbers ${TASK_NUMBERS}"
  echo "[INFO] Processing specified tasks: ${TASK_NUMBERS}"
else
  echo "[INFO] Processing all tasks (1-156)"
fi

set -x
${PY} pro_v/prompting_top_agent_ray.py \
  --model "${SERVED_MODEL_NAME}" \
  --vllm_endpoints "${VLLM_ENDPOINTS_CSV}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --benchmark_path "${FOLDER_PATH}" \
  --benchmark_format "${BENCHMARK_FORMAT}" \
  ${THINKING_FLAG} \
  ${TASK_NUMBERS_FLAG} \
  ${EXTRA_ARGS}
set +x

echo "[INFO] Main evaluation finished."
echo "[INFO] Starting evaluate2 - Mutant detection analysis..."

EVAL2_REPORT_FILE="evaluation_report_${EXPERIMENT_NAME}_${DAY}.json"
EXPERIMENT_OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

echo "[INFO] All evaluations finished."
