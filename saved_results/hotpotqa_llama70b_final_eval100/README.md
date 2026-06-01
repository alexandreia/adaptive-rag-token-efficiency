# HotpotQA Llama-70B Final Evaluation Run

This folder contains the merged final HotpotQA evaluation for the hosted Llama model.

## Run setup

- Dataset: HotpotQA
- Prepared query file: `data/hotpotqa_classmate/queries_150.jsonl`
- Documents: `data/hotpotqa_classmate/documents.jsonl`
- Total prepared queries: 150
- Development/calibration split: 50 queries
- Held-out evaluation split: 100 queries
- Evaluated model: `meta-llama/Llama-3.3-70B-Instruct`
- Provider: Berget AI OpenAI-compatible API
- Run method: 10 batches of 10 held-out evaluation queries, merged at per-query level
- Prompt style: concise
- Max output tokens: 80
- Seed: 0
- Token source: provider-reported usage

## Main result

Safe Adaptive Context preserves most of the Fixed Top-10 answer quality while using substantially fewer tokens.

From `final_table.csv`:

- Fixed Top-10 total tokens: 1536.68
- Safe Adaptive total tokens: 1065.72
- Token reduction: 30.6%
- Fixed Top-10 Answer F1: 0.764539
- Safe Adaptive Answer F1: 0.736539
- Fixed Top-10 semantic similarity: 0.768282
- Safe Adaptive semantic similarity: 0.740117
- Safe Adaptive fallback rate: 0.0%

Compared with the heuristic baseline, Safe Adaptive uses more tokens but improves answer quality:

- Heuristic total tokens: 944.98
- Safe Adaptive total tokens: 1065.72
- Heuristic Answer F1: 0.707539
- Safe Adaptive Answer F1: 0.736539
- Heuristic semantic similarity: 0.710674
- Safe Adaptive semantic similarity: 0.740117

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
