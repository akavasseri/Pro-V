#!/bin/bash
#SBATCH --job-name=fca-h2h
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=fca_h2h_%j.log
#
# Fresh real run of the Pro-V paper pipeline on 15 random HDLBits tasks, on an
# 80GB A100, so we can measure the Functional Coverage Agent's testbench
# strength head-to-head against the paper's method. GENERATION ONLY here;
# CPU scoring (mutant detection + FCA head-to-head) is run afterwards from the
# login node so the analysis can be iterated without holding the GPU.
#
#   sbatch scripts/adroit_fca_run.sbatch
#   tail -f fca_h2h_<jobid>.log

set -x
REPO_ROOT="/scratch/network/ak7587/Pro-V"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd $REPO_ROOT"; exit 2; }

source /etc/profile 2>/dev/null || true
module load anaconda3/2024.6 2>/dev/null || module load anaconda3 2>/dev/null || true
module load cudatoolkit/13.0 2>/dev/null || true   # CUDA dev headers matching torch cu130

# vLLM: use the Torch-native top-k/top-p sampler (prior working runs did this).
# Avoids flashinfer JIT-compiling a CUDA kernel that needs curand.h at runtime.
export VLLM_USE_FLASHINFER_SAMPLER=0

export PRO_V_PYTHON=/scratch/network/ak7587/envs/pro-v/bin/python
# env bin on PATH so torch's cpp_extension finds `ninja`, and iverilog/verilator resolve
export PATH=/scratch/network/ak7587/envs/pro-v/bin:$PATH

# ---- experiment knobs ------------------------------------------------------
export EXPERIMENT_NAME="fca_headtohead_15"
export TASK_NUMBERS="2,5,8,9,22,24,26,30,61,89,91,113,119,124,150"   # seed=1234, 11 cmb / 4 seq
export FOLDER_PATH="verilog-eval/HDLBits/test_benchmark_new.json"
export MODEL_PATH="$REPO_ROOT/models/PRO-V-R1-8B"
export MAX_LEN=32768
export PORT=8321
export GPU_IDS=0
export MAX_CONCURRENCY=15

echo "== node: $(hostname) =="
nvidia-smi -L || true
echo "== experiment: $EXPERIMENT_NAME  tasks: $TASK_NUMBERS =="

# clean any stale output dir so this is a genuinely fresh generation
rm -rf "outputs/${EXPERIMENT_NAME}"

# ---- the paper pipeline: gen_tb (stimulus) + pychecker (FRM) + judge --------
bash scripts/run_evaluation_think_simple.sh

echo "== GENERATION DONE =="
echo "== output dir: outputs/${EXPERIMENT_NAME} =="
ls -1 "outputs/${EXPERIMENT_NAME}" 2>/dev/null | head
echo "== next: score from login node (mutant detection + FCA head-to-head) =="
