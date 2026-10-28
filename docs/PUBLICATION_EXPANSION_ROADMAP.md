# Publication Expansion Roadmap

This roadmap turns the course report into a stronger workshop/conference-style
paper. The core claim stays the same:

> Adaptive context selection is not only smaller Top-k and not only compression.
> It works best when task-aware context structure is combined with answer-aware
> fallback.

## Current Paper Baseline

The current paper already includes:

- Three main datasets: SciFact, HotpotQA, BioASQ.
- Two generators: Mistral 7B and Llama-3.3-70B.
- Fixed Top-k baselines.
- A rule-based Heuristic Rules baseline.
- Safe Adaptive Context.
- Component ablation on SciFact.
- Compression/budget diagnostics.
- Cross-encoder reranking robustness.
- Exploratory ASQA long-form evaluation.
- Human annotation over 120 Safe Adaptive outputs.

This is a strong base. The publication expansion should add scale, stronger
baselines, uncertainty estimates, and deeper error analysis.

## Phase 1: Statistical Confidence From Existing Runs

Goal: make the current result claims more defensible without rerunning expensive
LLM calls.

Tasks:

- Add bootstrap confidence intervals for:
  - Token F1
  - Answer coverage
  - Semantic similarity
  - Total tokens
  - Token reduction
- Add paired Safe Adaptive vs Fixed Top-10 comparisons.
- Add paired Safe Adaptive vs Heuristic Rules comparisons.
- Report whether the quality changes are small relative to token savings.

Why this comes first:

- The existing `llm_answers_by_query.csv` files already contain per-query
  metrics.
- This strengthens the paper immediately.
- It gives us a better sense of which claims survive uncertainty.

Output target:

```text
saved_results/publication_stats/
```

## Phase 2: Larger Evaluation Splits

Goal: move beyond 100 examples per dataset with a clean development/evaluation
split.

Tasks:

- Use 50 calibration examples per dataset for threshold/budget sanity checks.
- Use 200 held-out evaluation examples per dataset for final reporting.
- Run the 200-example evaluation split for:
  - SciFact
  - HotpotQA
  - BioASQ
- Keep ASQA as an exploratory long-form setting with the same 50/200 structure.
- Compare whether the 200-query results preserve the same pattern as the
  100-query paper results.

Why this matters:

- The current result is strong, but 100 examples per dataset is easy for a
  reviewer to question.
- A larger evaluation makes the token-efficiency claim more robust.

## Phase 3: Stronger Retrieval Settings

Goal: show that Safe Adaptive Context is not only a TF-IDF artifact.

Tasks:

- Keep the existing TF-IDF setting as the controlled context-selection setup.
- Promote cross-encoder reranking from robustness check to a fuller experiment.
- Add one strong retrieval setting:
  - Dense retrieval with `multilingual-e5-large-instruct`.
  - Cross-encoder reranking with `bge-reranker-v2-m3`.
  - Safe Adaptive selects/compresses context from the reranked top-10.
- Report retrieval metrics first before spending generation budget:
  - nDCG@10
  - MRR@10
  - Recall@10
  - Recall@50, if dense retrieval retrieves a larger candidate pool.

Key question:

> Does Safe Adaptive still save tokens when the retrieved evidence is ranked by a
> stronger retriever?

## Phase 4: Stronger External Baselines

Goal: compare against recent lightweight context-selection and compression
baselines, not only our internal fixed/heuristic baselines.

Priority order:

1. Adaptive-k style baseline.
   - Select the number of passages from retrieval-score distribution.
   - No extra LLM call.
   - Closest conceptual baseline to our budget-selection component.
2. LLMLingua-style compression baseline, if feasible.
   - Tests whether a known prompt-compression approach can match Safe Adaptive.
   - More expensive/complex, so run on selected settings first.
3. RECOMP-style selective/compressed context baseline, if feasible.
   - Useful for positioning, but likely heavier than Adaptive-k.

Recommended first experiment:

- Implement Adaptive-k locally.
- Evaluate it on the 200-example TF-IDF setting.
- Compare against:
  - Fixed Top-10
  - Heuristic Rules
  - Safe Adaptive Context

Key question:

> Is Safe Adaptive better than choosing k adaptively from retrieval scores alone?

## Phase 5: Stronger Context Baselines

Goal: answer the reviewer question: "Is this better than just compressing every
prompt?"

Tasks:

- Add or formalize these baselines:
  - Fixed Top-10 full context.
  - Fixed Top-10 compressed context.
  - Heuristic Rules full context.
  - Heuristic Rules + compact evidence.
  - Safe Adaptive Context.
- If feasible, add an external/simple compression baseline.

Key claim to test:

> The gain comes from combining adaptive budgeting with controlled compression,
> not from compression alone.

## Phase 6: Human Evaluation Expansion

Goal: make the metric-validation section publication-level.

Tasks:

- Adjudicate the 36 disagreed examples.
- Report final adjudicated labels.
- Add a small qualitative table with examples:
  - Low F1 but human-correct.
  - High semantic similarity but incomplete/wrong.
  - Safe Adaptive success after compact context.
  - Safe Adaptive failure due to missing evidence.
- Add coverage buckets to the metric-alignment table.

Key claim:

> Automatic metrics are useful directional indicators, but they are conservative:
> many low-F1 answers are still judged correct or partially correct by humans.

## Phase 7: Error Analysis

Goal: explain failures, not only successes.

Error categories to code:

- Relevant evidence ranked too low.
- Compression removed needed detail.
- Multi-hop evidence split across documents.
- Fallback did not trigger.
- Fallback triggered but retrieval still lacked evidence.
- Reference answer was too narrow or synthetic-reference-sensitive.

Output target:

```text
saved_results/error_analysis/
```

## Phase 8: Practical Cost and Latency

Goal: make the efficiency argument operational.

Tasks:

- Convert token savings into approximate cost savings for at least one hosted
  model pricing setup.
- Report latency where available.
- Separate prompt tokens and completion tokens.

Key claim:

> Token reduction is not only a metric improvement; it is a practical cost and
> latency improvement.

## First Concrete Step

Start with Phase 1:

1. Create a script that reads existing `saved_results/*/llm_answers_by_query.csv`
   files.
2. Compute bootstrap confidence intervals by method.
3. Compute paired differences for:
   - Safe Adaptive Context vs Fixed Top-10.
   - Safe Adaptive Context vs Heuristic Rules.
4. Save publication-ready CSV and Markdown summaries.

This gives us a stronger paper without spending new API money first.
