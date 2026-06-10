# Reproducing The Evaluation

This branch keeps the original Pro-V evaluation flow, but adds safer testbench
generation, golden-DUT sample selection, benchmark adapters, and run scripts.

## Required Local Assets

These are intentionally not committed:

- `models/PRO-V-R1-8B`: download from Hugging Face.
- `verilog-eval/`: clone or restore the HDLBits benchmark data.
- `RTLLM/`: optional external benchmark clone.
- `key.cfg`: API keys if using a remote API instead of local vLLM.

The Qwen3 non-thinking chat template is committed as
`qwen3_nonthinking.jinja` and used by default.

## HDLBits Run

```bash
cd /path/to/Pro-V
export PRO_V_PYTHON=/path/to/env/bin/python
export MODEL_PATH=./models/PRO-V-R1-8B
export FOLDER_PATH=./verilog-eval/HDLBits/test_benchmark_new.json
export BENCHMARK_FORMAT=hdlbits_json
export TASK_NUMBERS=""
bash scripts/run_evaluation_think.sh
```

For a smoke test, set `TASK_NUMBERS=1,2,3`.

## RTLLM Adapter Smoke Test

```bash
cd /path/to
git clone https://github.com/hkust-zhiyao/RTLLM.git

cd /path/to/Pro-V
python scripts/inspect_benchmark_adapter.py /path/to/RTLLM \
  --benchmark_format rtllm_folder \
  --limit 20 \
  --write_normalized outputs/rtllm_normalized_preview.json
```

Expected adapter behavior:

- RTLLM modules are renamed to `top_module`.
- Description, header, RTL, and optional testbench are normalized.
- RTLLM has no mutant labels by default, so eval2 is skipped unless mutants are
  supplied.

Run a smoke test:

```bash
export FOLDER_PATH=/path/to/RTLLM
export BENCHMARK_FORMAT=rtllm_folder
export TASK_NUMBERS=1,2,3
export EXTRA_ARGS="--max_concurrency 1 --sampling_size 0"
bash scripts/run_evaluation_think.sh
```

## Generation Caps

Defaults can be overridden with environment variables:

```bash
export PRO_V_MAX_CMB_VECTORS=100
export PRO_V_EXHAUSTIVE_CMB_INPUT_BITS=10
export PRO_V_MAX_SEQ_SCENARIOS=24
export PRO_V_MAX_SEQ_CYCLES=32
export PRO_V_MAX_SEQ_TOTAL_CYCLES=512
```

Combinational tasks with at most 10 input bits may use exhaustive vectors.
Larger combinational tasks and sequential tasks are capped to prevent wasteful
or timeout-prone generation.

## Slurm GPU80 Run

```bash
cd /path/to/Pro-V
mkdir -p logs
PRO_V_REPO_ROOT=$PWD \
PRO_V_PYTHON=/path/to/env/bin/python \
sbatch scripts/run_prov_gpu80_eval.sbatch
```

The Slurm script requests one `gpu80` GPU for six hours.

## Merging Partial Runs

Use this when several smaller runs cover different tasks:

```bash
python scripts/merge_task_results.py \
  --output_dir outputs/merged_eval \
  outputs/run_a outputs/run_b outputs/run_c
```

The merge keeps the best valid task result and does not alter eval definitions.

