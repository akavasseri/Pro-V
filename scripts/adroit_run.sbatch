#!/bin/bash
#SBATCH --job-name=prov-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=prov_eval_%j.log
# NOTE: verify the right partition/constraint first with:  sinfo -s   and   snodes
# On Adroit you may need e.g.  #SBATCH --partition=gpu   or  #SBATCH --constraint=a100
#
# Run from the repo root on Adroit:   sbatch scripts/adroit_run.sbatch
# Watch it with:                      tail -f prov_eval_<jobid>.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- environment (adjust to how Pro-V is installed on Adroit) --------------
module purge 2>/dev/null || true
module load anaconda3/2024.6 2>/dev/null || module load anaconda3 2>/dev/null || true
# Use the pro-v conda env if it exists, else create/point PY at your python
conda activate pro-v 2>/dev/null || true
PY="${PY:-python}"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/PRO-V-R1-8B}"
PORT="${PORT:-8020}"
BENCH="${BENCH:-verilog-eval/HDLBits/merged_benchmark.json}"   # the REAL benchmark with mutants
TASKS="${TASKS:-1,2,3,4,5}"                                     # small subset

echo "== node: $(hostname) =="; nvidia-smi -L || true

# ---- 1) launch vLLM in the background --------------------------------------
"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name PRO-V-R1-8B \
  --tensor-parallel-size 1 --max-model-len 32768 --port "$PORT" \
  --trust-remote-code > "vllm_${PORT}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "== waiting for vLLM on :$PORT =="
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready ($i s)"; break; }
  sleep 1; [ "$i" -eq 180 ] && { echo "vLLM timeout; see vllm_${PORT}.log"; exit 1; }
done

# ---- 2) sanity: our GPU-free connect test against the LIVE model -----------
VLLM_URL="http://127.0.0.1:$PORT" "$PY" pro_v/connect_test.py || true

# ---- 3) the real Pro-V eval (generates FRMs/stimulus via the model) --------
if [ -f "$BENCH" ]; then
  VLLM_ENDPOINTS_CSV="http://127.0.0.1:$PORT" \
  FOLDER_PATH="$BENCH" TASK_NUMBERS="$TASKS" \
    bash scripts/run_evaluation_think_simple.sh
  # ---- 4) score testbench strength on the produced outputs ----------------
  "$PY" pro_v/simulate_and_evaluate_mutants.py \
     -e outputs/pro_v_eval --strength --exclude-noncompiling -o strength_report.json || true
else
  echo "WARNING: real benchmark $BENCH not found; running only the mini fixture"
  "$PY" - <<'PYEOF'
import json,sys; sys.path.insert(0,".")
from pro_v.mutation_strength import classify_mutant
b=json.load(open("benchmark/prov_mini_benchmark.json"))
for t in b:
  for i,m in enumerate(t["mutants"]):
    v=classify_mutant(t["module_code"],m,t["circuit_type"],max_exhaustive_bits=17)
    print(t["task_id"],i,"equiv" if v.equivalent else "nonequiv",len(v.witnesses),"witnesses")
PYEOF
fi
echo "== done =="
