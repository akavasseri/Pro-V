#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/abhijna/Pro-V

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in this shell before running GPT-5.2}"

export LLM_PROVIDER=openai
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-gpt52_probe}" \
TASK_NUMBERS="${TASK_NUMBERS:-16,30,39,110,121,148}" \
EXTRA_ARGS="${EXTRA_ARGS:---max_concurrency 2 --sampling_size 3}" \
bash scripts/run_evaluation_full156.sh
