# BioASQ Mistral Final Evaluation Run

This folder contains the merged final BioASQ evaluation for the local Mistral model.

## Run setup

- Dataset: BioASQ
- Prepared query file: `data/bioasq_candidate/queries.jsonl`
- Documents: `data/bioasq_candidate/documents.jsonl`
- Total prepared queries: 150
- Development/calibration split: 50 queries
- Held-out evaluation split: 100 queries
- Evaluated model: `mistral`
- Provider: Ollama local OpenAI-compatible API
- Run method: 10 batches of 10 held-out evaluation queries, merged at per-query level
- Prompt style: default
- Max output tokens: 80
- Seed: 0
- Token source: local/Ollama-compatible usage

## Main result

Safe Adaptive Context improves answer quality over Fixed Top-10 while using substantially fewer tokens.

From `final_table.csv`:

- Fixed Top-10 total tokens: 3888.53
- Safe Adaptive total tokens: 2011.28
- Token reduction: 48.3%
- Fixed Top-10 Answer F1: 0.256987
- Safe Adaptive Answer F1: 0.339011
- Fixed Top-10 semantic similarity: 0.606102
- Safe Adaptive semantic similarity: 0.758291
- Safe Adaptive fallback rate: 0.0%

Compared with the heuristic baseline, Safe Adaptive also improves quality while using fewer tokens:

- Heuristic total tokens: 2559.79
- Safe Adaptive total tokens: 2011.28
- Heuristic Answer F1: 0.332534
- Safe Adaptive Answer F1: 0.339011
- Heuristic semantic similarity: 0.745378
- Safe Adaptive semantic similarity: 0.758291

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
