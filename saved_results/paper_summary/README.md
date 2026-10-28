# Paper Summary Artifacts

This folder contains report-facing tables and plots generated from the final runs.

## Core Tables

- `main_llama_tfidf_core.md`: compact main TF-IDF table for Llama-70B.
- `model_comparison_tfidf.md`: Llama-70B vs Mistral comparison under TF-IDF.
- `retriever_robustness_core.md`: method-level results across TF-IDF, TF-IDF + cross-encoder, and dense + cross-encoder.
- `method_averages_by_retriever.md`: average F1 retention, semantic retention, token reduction, and fallback by retriever.
- `tacer_vs_adaptive_by_retriever.md`: direct TACER vs Adaptive-k comparison by dataset and retriever.
- `report_summary_notes.md`: short prose notes for the paper discussion.

CSV versions of the tables are provided next to each Markdown file.

## Figures

- `pareto_llama_tfidf.svg`
- `pareto_llama_tfidf_cross_encoder.svg`
- `pareto_llama_dense_cross_encoder.svg`

The plots show F1 retention against token reduction. A good point is high and to the right.

## Regeneration

Run:

```bash
python scripts/build_paper_summary_artifacts.py
```
