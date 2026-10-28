#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-}"
TOTAL="${2:-200}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
API_URL="${API_URL:-https://api.berget.ai/v1}"
API_KEY_ENV="${API_KEY_ENV:-BERGET_API_KEY}"

if [[ -z "$DATASET" ]]; then
  echo "Usage: scripts/run_final_tacer_batches.sh {scifact|bioasq|hotpotqa|asqa|msmarco} [total]"
  exit 1
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key. Export it first, for example:"
  echo "  export ${API_KEY_ENV}='YOUR_KEY_HERE'"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROMPT_STYLE="default"
MAX_OUTPUT_TOKENS="80"

case "$DATASET" in
  scifact)
    DOCUMENTS="data/scifact/documents.jsonl"
    QUERIES="data/scifact_250/queries_eval_200.jsonl"
    DEV_QUERIES="data/scifact_250/queries_calibration_50.jsonl"
    DATASET_NAME="SciFact"
    ;;
  bioasq)
    DOCUMENTS="data/bioasq_250/documents.jsonl"
    QUERIES="data/bioasq_250/queries_eval_200.jsonl"
    DEV_QUERIES="data/bioasq_250/queries_calibration_50.jsonl"
    DATASET_NAME="BioASQ"
    ;;
  hotpotqa)
    DOCUMENTS="data/hotpotqa_250/documents.jsonl"
    QUERIES="data/hotpotqa_250/queries_eval_200.jsonl"
    DEV_QUERIES="data/hotpotqa_250/queries_calibration_50.jsonl"
    DATASET_NAME="HotpotQA"
    PROMPT_STYLE="concise"
    ;;
  asqa)
    DOCUMENTS="data/asqa_250/documents.jsonl"
    QUERIES="data/asqa_250/queries_eval_200.jsonl"
    DEV_QUERIES="data/asqa_250/queries_calibration_50.jsonl"
    DATASET_NAME="ASQA"
    MAX_OUTPUT_TOKENS="180"
    ;;
  msmarco)
    DOCUMENTS="data/msmarco_250/documents.jsonl"
    QUERIES="data/msmarco_250/queries_eval_200.jsonl"
    DEV_QUERIES="data/msmarco_250/queries_calibration_50.jsonl"
    DATASET_NAME="MSMARCO"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Use one of: scifact, bioasq, hotpotqa, asqa, msmarco"
    exit 1
    ;;
esac

OUTPUT_ROOT="outputs/final_tacer_batches/${DATASET}_llama70b_${TOTAL}"

python scripts/run_llm_batches.py \
  --documents "$DOCUMENTS" \
  --queries "$QUERIES" \
  --dev-queries "$DEV_QUERIES" \
  --dataset-name "$DATASET_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --model "$MODEL" \
  --api-url "$API_URL" \
  --api-key-env "$API_KEY_ENV" \
  --prompt-style "$PROMPT_STYLE" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --total "$TOTAL" \
  --seed 0 \
  --require-provider-tokens \
  --methods no_retrieval fixed_3 fixed_5 fixed_7 fixed_10 heuristic_rules adaptive_k answer_aware_fallback coverage_guided_ultra task_aware_coverage_ultra \
  --compression-modes full

echo
echo "Final merged table:"
echo "$OUTPUT_ROOT/merged/final_table.md"
