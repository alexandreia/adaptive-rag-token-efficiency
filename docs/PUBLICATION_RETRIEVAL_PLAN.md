# Publication Retrieval Plan

This note defines the document-ranking settings for the publication expansion.
The main paper question is context selection, so retrieval settings should be
strong enough to test robustness without turning the project into a retriever
benchmark.

## Evaluation Splits

Each dataset uses the same split structure:

- 50 calibration examples for threshold and budget sanity checks.
- 200 held-out evaluation examples for final reporting.

Datasets:

- `data/scifact_250/queries_eval_200.jsonl`
- `data/hotpotqa_250/queries_eval_200.jsonl`
- `data/bioasq_250/queries_eval_200.jsonl`
- `data/asqa_250/queries_eval_200.jsonl` as exploratory long-form QA.

## Ranking Settings

### 1. TF-IDF

Role: main controlled setting.

All methods receive the same TF-IDF top-10 candidate list. This isolates context
selection from retriever optimization and keeps the core experiment transparent.

Use for:

- Full method comparison.
- Both generators when budget allows.
- Main confidence-interval tables.

### 2. Dense Retrieval + Cross-Encoder Reranking

Role: strong retrieval robustness setting.

Planned stack:

```text
Dense retriever retrieves top-50 or top-100 candidates
→ Cross-encoder reranks candidates
→ Context methods use the reranked top-10
```

Candidate Berget models:

- Dense embeddings: `intfloat/multilingual-e5-large-instruct`
- Reranker: `BAAI/bge-reranker-v2-m3`

Use for:

- Retrieval metrics first: nDCG@10, MRR@10, Recall@10, Recall@50.
- Generation only after retrieval metrics improve or change the ranking enough
  to justify the extra cost.

### 3. Cross-Encoder Reranking of TF-IDF Top-10

Role: lower-cost robustness setting already aligned with the current paper.

This keeps the original TF-IDF candidate pool but changes the ordering. It tests
whether Safe Adaptive still works when relevant documents are moved earlier by a
stronger ranker.

Use for:

- Llama-70B first, to control cost.
- Main datasets only: SciFact, HotpotQA, BioASQ.

## Context Baselines Under Each Ranking

Minimum baseline set:

- Fixed Top-10
- Heuristic Rules
- Adaptive-k
- Safe Adaptive Context

Optional diagnostic baselines:

- Fixed Top-10 + compact evidence
- Heuristic Rules + compact evidence
- Adaptive-k + compact evidence

## Adaptive-k Baseline

Adaptive-k chooses a cutoff from the shape of the retrieval-score distribution.
In this implementation, it selects the documents before the largest adjacent
drop in the top-10 score list, following the default `largest_gap` thresholding
rule from the released Adaptive-k code. It is intentionally lightweight:

- no additional LLM call
- no fine-tuning
- no learned critic
- only retrieval scores

This is the cleanest comparison for testing whether Safe Adaptive is more than
adaptive top-k selection.

## Retrieval-Only Command

Before running paid generation, build retrieval-only rankings and metrics:

```bash
python scripts/build_retrieval_rankings.py \
  --documents data/scifact/documents.jsonl \
  --queries data/scifact_250/queries_eval_200.jsonl \
  --dataset-name SciFact \
  --output-dir saved_results/retrieval_rankings/scifact_200
```

This writes:

```text
retrieval_rankings.csv
retrieval_rankings.jsonl
retrieval_summary.csv
```

The same script reports:

- `tfidf`
- `tfidf_cross_encoder`
- `dense`
- `dense_cross_encoder`

Use `--skip-dense` or `--skip-cross-encoder` for cheap smoke tests.
