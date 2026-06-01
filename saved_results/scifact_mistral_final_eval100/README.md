# SciFact Mistral Final Evaluation Run

This folder contains the merged final SciFact evaluation for the local Mistral model through Ollama.

This run uses the same max output token setting as the hosted Llama SciFact evaluation.

## Run setup

- Dataset: SciFact
- Prepared query file: `data/scifact/queries_150_seed0_llm_gold_v2.jsonl`
- Documents: `data/scifact/documents.jsonl`
- Total prepared queries: 150
- Development/calibration split: 50 queries
- Held-out evaluation split: 100 queries
- Evaluated model: `mistral`
- Provider: local Ollama OpenAI-compatible API
- API URL: `http://localhost:11434/v1`
- Run method: 10 batches of 10 held-out evaluation queries, merged at per-query level
- Max output tokens: 80
- Seed: 0
- Token source: provider-reported usage

## Main result

Safe Adaptive Context substantially reduces token usage while retaining stronger answer quality than Fixed Top-10.

From `final_table.csv`:

- Fixed Top-10 total tokens: 3781.14
- Safe Adaptive total tokens: 985.01
- Token reduction: 73.9%
- Fixed Top-10 Answer F1: 0.203530
- Safe Adaptive Answer F1: 0.252913
- Fixed Top-10 semantic similarity: 0.363887
- Safe Adaptive semantic similarity: 0.426009
- Safe Adaptive fallback rate: 5.0%

Compared with the heuristic baseline, Safe Adaptive uses substantially fewer tokens while preserving most answer quality:

- Heuristic total tokens: 2651.75
- Safe Adaptive total tokens: 985.01
- Safe Adaptive uses about 62.9% fewer tokens than Heuristic.
- Heuristic Answer F1: 0.264540
- Safe Adaptive Answer F1: 0.252913
- Heuristic semantic similarity: 0.437337
- Safe Adaptive semantic similarity: 0.426009

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
