# BioASQ Llama-70B Final Evaluation Run

This folder contains the merged final BioASQ evaluation for the hosted Llama model.

## Run setup

- Dataset: BioASQ
- Prepared query file: `data/bioasq_candidate/queries.jsonl`
- Documents: `data/bioasq_candidate/documents.jsonl`
- Total prepared queries: 150
- Development/calibration split: 50 queries
- Held-out evaluation split: 100 queries
- Evaluated model: `meta-llama/Llama-3.3-70B-Instruct`
- Provider: Berget AI OpenAI-compatible API
- Run method: 10 batches of 10 held-out evaluation queries, merged at per-query level
- Prompt style: default
- Max output tokens: 80
- Seed: 0
- Token source: provider-reported usage

## Main result

Safe Adaptive Context preserves essentially all of the Fixed Top-10 answer quality while using substantially fewer tokens.

From `final_table.csv`:

- Fixed Top-10 total tokens: 3496.26
- Safe Adaptive total tokens: 1668.18
- Token reduction: 52.3%
- Fixed Top-10 Answer F1: 0.344165
- Safe Adaptive Answer F1: 0.342478
- Fixed Top-10 semantic similarity: 0.774441
- Safe Adaptive semantic similarity: 0.779144
- Safe Adaptive fallback rate: 2.0%

Compared with the heuristic baseline, Safe Adaptive uses fewer tokens while preserving nearly identical semantic similarity:

- Heuristic total tokens: 2126.33
- Safe Adaptive total tokens: 1668.18
- Heuristic Answer F1: 0.349106
- Safe Adaptive Answer F1: 0.342478
- Heuristic semantic similarity: 0.780216
- Safe Adaptive semantic similarity: 0.779144

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
