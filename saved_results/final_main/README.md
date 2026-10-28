# Final Main Results

This folder stores report-ready outputs from the final TACER/Safe Adaptive main benchmark.

Raw batch execution logs remain under `outputs/`. The files here are the merged, paper-facing artifacts.

## Structure

- `llama70b_tfidf/`: final Llama-3.3-70B-Instruct runs with the TF-IDF retriever.
  - `scifact/`
  - `bioasq/`
  - `hotpotqa/`
  - `msmarco/`
  - `asqa/`
  - `final_table_all_datasets.csv`
  - `llm_summary_all_datasets.csv`
- `mistral_tfidf/`: final Mistral runs with the TF-IDF retriever for model-size comparison.
  - `scifact/`
  - `bioasq/`
  - `hotpotqa/`
  - `msmarco/`
  - `asqa/`
  - `final_table_all_datasets.csv`
  - `llm_summary_all_datasets.csv`

Each dataset folder contains:

- `final_table.csv`: method-level aggregate metrics.
- `final_table.md`: Markdown version of the aggregate table.
- `llm_summary.csv`: method-level summary used to build the final table.
- `llm_answers_by_query.csv`: per-query generated answers and metrics.

## Final Method Ladder

The final TF-IDF benchmarks compare:

- No Retrieval
- Fixed Top-3
- Fixed Top-5
- Fixed Top-7
- Fixed Top-10
- Heuristic Rules
- Adaptive-k
- Safe Adaptive Context
- Coverage-Guided Ultra
- TACER

TACER is implemented as `task_aware_coverage_ultra` in the code.
