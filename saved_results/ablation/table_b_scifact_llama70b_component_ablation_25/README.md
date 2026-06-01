# Table B: Safe Adaptive Context Component Ablation

This folder contains the focused Table B ablation for Safe Adaptive Context.

## Run Setup

- Dataset: SciFact
- Documents: `data/scifact/documents.jsonl`
- Queries: `data/scifact/queries_150_seed0_llm_gold_v2.jsonl`
- Split: first 50 queries for calibration, held-out evaluation after that
- Evaluated held-out queries: 25
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Provider: Berget AI OpenAI-compatible API
- Max output tokens: 80
- Token source: provider-reported usage
- Seed: 0

## Component Variants

The table compares:

- `Fixed Top-10`: full-context baseline
- `Adaptive Full Only`: adaptive budget without compression or fallback
- `Adaptive Compact Only`: adaptive budget plus compact evidence, without fallback
- `Fixed Top-5 Compact + Fallback`: fallback control without adaptive budget
- `Full Safe Adaptive Context`: complete system

## Main Reading

Safe Adaptive Context keeps almost the same answer F1 as Fixed Top-10 while
using much fewer tokens.

From `table_b_component_ablation.csv`:

- Fixed Top-10 tokens: 3324.6
- Full Safe Adaptive tokens: 780.4
- Token reduction: 76.5%
- Fixed Top-10 Answer F1: 0.247355
- Full Safe Adaptive Answer F1: 0.247376
- Full Safe Adaptive fallback rate: 4.0%

This suggests that adaptive budgeting provides the main token savings, compact
evidence preserves quality at lower cost, and fallback acts as a small safety
layer.

## Files

- `table_b_component_ablation.csv`: clean Table B in CSV format
- `table_b_component_ablation.md`: clean Table B in Markdown format
- `llm_summary.csv`: aggregate metrics for each component variant
- `llm_answers_by_query.csv`: per-query generated answers and metrics
- `retrieval_summary.csv`: retrieval-side summary for the evaluated subset

## Reproduction Command

```bash
cd /Users/joelvazquezlopez/Desktop/adaptive_rag_github

/usr/bin/caffeinate -dimsu /opt/homebrew/bin/python3 scripts/run_table_b_ablation.py \
  --documents data/scifact/documents.jsonl \
  --queries data/scifact/queries_150_seed0_llm_gold_v2.jsonl \
  --dataset-name scifact \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --api-url https://api.berget.ai/v1 \
  --api-key-env BERGET_API_KEY \
  --max-output-tokens 80 \
  --max-eval-queries 25 \
  --eval-start-index 0 \
  --output-dir outputs/table_b_scifact_llama70b_ablation_25 \
  --request-timeout-seconds 120 \
  --require-provider-tokens \
  --seed 0 \
  --sleep-between-calls 1
```

