#!/usr/bin/env bash
set -euo pipefail

# Example batched evaluation runner.
#
# This shows how the full held-out evaluation was split into 10 batches of
# 10 queries. Batching makes long local LLM runs easier to monitor and restart.

cd "$(dirname "$0")/.."

BATCH_SIZE=10
NUM_BATCHES=10

for i in $(seq 0 $((NUM_BATCHES - 1))); do
  start=$((i * BATCH_SIZE))
  batch=$(printf "%02d" $((i + 1)))

  echo "Running SciFact Mistral matched batch ${batch}, eval_start=${start}"

  /usr/bin/caffeinate -dimsu /opt/homebrew/bin/python3 scripts/run_experiment.py \
    --documents data/scifact/documents.jsonl \
    --queries data/scifact/queries_150_seed0_llm_gold_v2.jsonl \
    --dataset-name scifact \
    --model mistral \
    --api-url http://localhost:11434/v1 \
    --no-api-key \
    --max-output-tokens 80 \
    --max-eval-queries "${BATCH_SIZE}" \
    --eval-start-index "${start}" \
    --output-dir "outputs/scifact_mistral_80_evalbatch_${batch}" \
    --request-timeout-seconds 300 \
    --require-provider-tokens \
    --seed 0
done
