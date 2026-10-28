#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
TOTAL="${2:-200}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
API_URL="${API_URL:-https://api.berget.ai/v1}"
API_KEY_ENV="${API_KEY_ENV:-BERGET_API_KEY}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: scripts/run_official_adaptive_k_batches.sh {scifact|bioasq|hotpotqa|asqa|msmarco|all} [total]"
  exit 1
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key. Export it first, for example:"
  echo "  export ${API_KEY_ENV}='YOUR_KEY_HERE'"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_one() {
  local dataset="$1"
  local prompt_style="default"
  local max_output_tokens="80"
  local documents=""
  local queries=""
  local dev_queries=""
  local dataset_name=""

  case "$dataset" in
    scifact)
      documents="data/scifact/documents.jsonl"
      queries="data/scifact_250/queries_eval_200.jsonl"
      dev_queries="data/scifact_250/queries_calibration_50.jsonl"
      dataset_name="SciFact"
      ;;
    bioasq)
      documents="data/bioasq_250/documents.jsonl"
      queries="data/bioasq_250/queries_eval_200.jsonl"
      dev_queries="data/bioasq_250/queries_calibration_50.jsonl"
      dataset_name="BioASQ"
      ;;
    hotpotqa)
      documents="data/hotpotqa_250/documents.jsonl"
      queries="data/hotpotqa_250/queries_eval_200.jsonl"
      dev_queries="data/hotpotqa_250/queries_calibration_50.jsonl"
      dataset_name="HotpotQA"
      prompt_style="concise"
      ;;
    asqa)
      documents="data/asqa_250/documents.jsonl"
      queries="data/asqa_250/queries_eval_200.jsonl"
      dev_queries="data/asqa_250/queries_calibration_50.jsonl"
      dataset_name="ASQA"
      max_output_tokens="180"
      ;;
    msmarco)
      documents="data/msmarco_250/documents.jsonl"
      queries="data/msmarco_250/queries_eval_200.jsonl"
      dev_queries="data/msmarco_250/queries_calibration_50.jsonl"
      dataset_name="MSMARCO"
      ;;
    *)
      echo "Unknown dataset: $dataset"
      exit 1
      ;;
  esac

  local output_root="outputs/official_adaptive_k/${dataset}_llama70b_${TOTAL}"

  python scripts/run_llm_batches.py \
    --documents "$documents" \
    --queries "$queries" \
    --dev-queries "$dev_queries" \
    --dataset-name "$dataset_name" \
    --output-root "$output_root" \
    --model "$MODEL" \
    --api-url "$API_URL" \
    --api-key-env "$API_KEY_ENV" \
    --prompt-style "$prompt_style" \
    --max-output-tokens "$max_output_tokens" \
    --batch-size "$BATCH_SIZE" \
    --total "$TOTAL" \
    --seed 0 \
    --require-provider-tokens \
    --methods fixed_10 adaptive_k_official answer_aware_fallback task_aware_coverage_ultra \
    --compression-modes full

  echo
  echo "Final merged table:"
  echo "$output_root/merged/final_table.md"
}

if [[ "$DATASET" == "all" ]]; then
  for dataset in scifact bioasq hotpotqa asqa msmarco; do
    run_one "$dataset"
  done
else
  run_one "$DATASET"
fi
