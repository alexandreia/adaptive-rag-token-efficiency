# Safe Adaptive Context and TACER

This repository contains the code, data files, saved outputs, annotation
materials, and presentation assets for our project:

**Safe Adaptive Context and TACER: Task-Aware Evidence Routing for
Token-Efficient Retrieval-Augmented Generation**

The project asks a focused question:

> Given the same initially retrieved top-10 candidate documents, can a RAG
> system decide how much evidence to send to the generator, and in what form,
> while keeping answer quality close to a fixed Top-10 baseline?

The repository contains two related context controllers:

- **Safe Adaptive Context**, a lightweight controller that starts with limited
  evidence and expands when the generated answer appears weak or unsupported.
- **TACER**, a task-aware extension that routes between compact evidence and
  safer broader context using retrieval-side and evidence-coverage signals.

## Main Result

Across five datasets, two generators, and stronger retriever settings, Safe
Adaptive Context and TACER substantially reduce provider-reported token use while
preserving most Fixed Top-10 answer quality.

In the five-dataset external-baseline stress test, Safe Adaptive Context retains
**99.8%** of Fixed Top-10 F1 with **55.0%** token reduction, while TACER retains
**99.4%** with **60.4%** token reduction. The contribution is not universal
quality improvement, but practical quality-preserving efficiency through
task-aware context control.

## Contributors

- Andreia Alexa
- Simon Backhaus Brudzewski
- Karlis Martins Auce
- Joel Vazquez
- Sheraz Ahmad

## What The System Does

All methods start from the same retrieved candidate pool. The retriever returns
the top-10 documents with TF-IDF cosine similarity. The systems differ only in
how that retrieved list is converted into prompt context.

Safe Adaptive Context has three main parts:

1. **Adaptive budget prediction**
   - Uses retrieval-side signals such as query length, top scores, score gaps,
     score entropy, and score concentration.
   - Chooses an initial context budget before calling the generator.

2. **Evidence compression**
   - Splits selected documents into sentences.
   - Scores sentences by lexical overlap, rare query-term matches, phrase
     overlap, token density, and position.
   - Keeps high-scoring sentence neighborhoods rather than isolated sentences.

3. **Answer-aware fallback**
   - Generates an answer from the compact context.
   - Checks runtime answer-risk signals such as empty answers, weak grounding,
     low answer-context overlap, and malformed short answers.
   - Expands context only when the answer looks risky.

The goal is not simply to compress every prompt. The goal is to spend context
where it helps and save tokens where evidence is concentrated.

TACER adds explicit task/evidence routing. It asks whether the query appears to
need concentrated evidence, distributed multi-hop evidence, or longer synthesis,
then chooses compact coverage-guided evidence, broader selected context, or
fallback-enabled expansion.

## Datasets

The submitted paper uses five datasets with 50 calibration examples and 200
held-out evaluation examples per dataset:

| Dataset | Task Type | Notes |
|---|---|---|
| SciFact | Biomedical claim verification | Gold qrels are original; QA-style reference answers were synthetically generated for answer-level metrics |
| HotpotQA | Multi-hop question answering | Requires broader evidence coverage |
| BioASQ | Biomedical factoid QA | Uses concise biomedical answers |
| MS MARCO | Web passage QA | Often early-answer evidence |
| ASQA | Ambiguous long-form QA | Requires broader synthesis and careful compression |

SciFact does not provide natural-language QA-style gold answers in the same
format as the QA datasets. For answer-level metrics only, we generated SciFact
reference answers with `gpt-oss-120b`. Retrieval metrics such as nDCG@10 and
MRR@10 still use the original dataset qrels. See:

```text
docs/SCIFACT_LLM_GOLD_REPRODUCIBILITY.md
```

## Methods Compared

| Method | Code Mode | Meaning |
|---|---|---|
| No Retrieval | `no_retrieval` | Closed-book generation without retrieved context |
| Fixed Top-3 | `fixed_3_full` | Always sends the top 3 full documents |
| Fixed Top-5 | `fixed_5_full` | Always sends the top 5 full documents |
| Fixed Top-7 | `fixed_7_full` | Always sends the top 7 full documents |
| Fixed Top-10 | `fixed_10_full` | Always sends the top 10 full documents; main expensive baseline |
| Heuristic Rules | `heuristic_rules_full` | Rule-based context controller using retrieval score gaps and query length |
| Adaptive-k | `adaptive_k_full` | Lightweight score-drop baseline |
| Adaptive-k (paper) | `adaptive_k_official_full` | Paper-faithful largest-gap/buffer baseline |
| LLMLingua-2 | `llmlingua2_top10`, `llmlingua2_adaptive_k_official` | External prompt-compression baselines |
| FLARE-lite | `flare_lite` | Answer-risk expansion baseline |
| Safe Adaptive Context | `answer_aware_fallback` | Starts compact, checks answer risk, and expands only when needed |
| TACER | `task_aware_coverage_ultra` | Routes between compact evidence and safer broader context |

## Submitted Paper Artifacts

The submitted paper artifacts are under:

```text
paper/submitted_groundlm_2026/
```

Paper-facing summary tables are under:

```text
saved_results/paper_summary/
saved_results/publication_stats/
```

## Final Results at a Glance

Safe Adaptive Context compared with Fixed Top-10:

| Dataset | Model | Fixed F1 | Safe F1 | Fixed Tokens | Safe Tokens | Token Reduction |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | Mistral | 0.204 | 0.253 | 3,781 | 985 | 73.9% |
| SciFact | Llama-70B | 0.262 | 0.265 | 3,412 | 933 | 72.6% |
| HotpotQA | Mistral | 0.516 | 0.540 | 1,762 | 1,240 | 29.6% |
| HotpotQA | Llama-70B | 0.765 | 0.737 | 1,537 | 1,066 | 30.6% |
| BioASQ | Mistral | 0.257 | 0.339 | 3,889 | 2,011 | 48.3% |
| BioASQ | Llama-70B | 0.344 | 0.342 | 3,496 | 1,668 | 52.3% |

Safe Adaptive Context compared with the Heuristic Rules baseline:

| Dataset | Model | Heuristic F1 | Safe F1 | Heuristic Tokens | Safe Tokens | Token vs Heuristic |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | Mistral | 0.265 | 0.253 | 2,652 | 985 | -62.9% |
| SciFact | Llama-70B | 0.266 | 0.265 | 2,234 | 933 | -58.2% |
| HotpotQA | Mistral | 0.482 | 0.540 | 1,073 | 1,240 | +15.5% |
| HotpotQA | Llama-70B | 0.708 | 0.737 | 945 | 1,066 | +12.8% |
| BioASQ | Mistral | 0.333 | 0.339 | 2,560 | 2,011 | -21.4% |
| BioASQ | Llama-70B | 0.349 | 0.342 | 2,126 | 1,668 | -21.5% |

The comparison with Heuristic Rules is important because the heuristic is already
adaptive. Safe Adaptive is cheaper on SciFact and BioASQ, while on HotpotQA it
spends more tokens than the heuristic because multi-hop questions often need
broader evidence. In those HotpotQA settings, the extra context improves F1.

## Human Annotation Study

Automatic metrics are useful but incomplete, so we also ran a structured human
annotation study.

- 120 Safe Adaptive outputs
- 40 examples each from SciFact, HotpotQA, and BioASQ
- Two independent annotators
- Labels: `CORRECT`, `PARTIALLY_CORRECT`, `INCORRECT`, `NOT_ENOUGH_INFO`
- Raw agreement: 70.0%
- Cohen's kappa: 0.477

The annotation materials are under:

```text
annotation_workform/
```

The browser workform is self-contained and can be opened locally:

```text
annotation_workform/index.html
```

The agreement calculator is:

```text
annotation_workform/kappa.html
```

## Repository Structure

```text
.
├── annotation_workform/        # Human annotation interface and kappa tool
├── data/                       # Prepared dataset files
│   ├── asqa_candidate/
│   ├── bioasq_candidate/
│   ├── hotpotqa/
│   ├── hotpotqa_final/
│   └── scifact/
├── docs/                       # Plain-language docs and reproducibility notes
├── saved_results/              # Final result tables and saved LLM outputs
├── scripts/                    # Experiment and data-preparation scripts
├── src/adaptive_retrieval/     # Main implementation
├── src_cross_enc/              # Cross-encoder reranking variant
├── rpg_presentation_prototype.html
├── PROJECT_PLAN.md
├── README.md
└── requirements.txt
```

## Important Files

| File | Purpose |
|---|---|
| `scripts/run_experiment.py` | Main experiment runner for fixed, heuristic, and Safe Adaptive methods |
| `scripts/run_cross_encoder_experiment.py` | Cross-encoder reranking experiment runner |
| `scripts/run_table_b_ablation.py` | Component ablation runner |
| `src/adaptive_retrieval/llm_budget.py` | Main LLM pipeline, context policies, and answer-aware fallback |
| `src/adaptive_retrieval/retriever.py` | TF-IDF cosine retrieval |
| `src/adaptive_retrieval/learned_budget.py` | Lightweight adaptive budget predictor |
| `src/adaptive_retrieval/metrics.py` | Token F1, coverage, semantic metrics, MRR, and nDCG@10 |
| `docs/CODE_WALKTHROUGH.md` | Plain-language code walkthrough |
| `docs/FULL_SYSTEM_EXPLANATION.md` | Full system explanation |
| `docs/SCIFACT_LLM_GOLD_REPRODUCIBILITY.md` | How SciFact synthetic references were generated |

The best file to inspect first is:

```text
src/adaptive_retrieval/llm_budget.py
```

The main Safe Adaptive function is:

```text
answer_aware_fallback_run(...)
```

## Installation

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local Mistral runs, install and start Ollama:

```bash
brew install ollama
ollama serve
ollama run mistral "What is the capital of Spain?"
```

The local Ollama API should be available at:

```text
http://localhost:11434/v1
```

## Dry Run

Use a dry run to check that loading, retrieval, and table writing work without
calling the LLM:

```bash
python3 scripts/run_experiment.py \
  --dry-run \
  --max-eval-queries 5 \
  --output-dir outputs/dry_run
```

Dry-run outputs are only a pipeline check. They are not final LLM results.

## Example Mistral Run

Run 50 SciFact examples with local Mistral through Ollama:

```bash
python3 scripts/run_experiment.py \
  --documents data/scifact/documents.jsonl \
  --queries data/scifact/queries_150_seed0_llm_gold_v2.jsonl \
  --dataset-name SciFact \
  --output-dir outputs/scifact_mistral_50 \
  --model mistral \
  --api-url http://localhost:11434/v1 \
  --no-api-key \
  --max-eval-queries 50 \
  --seed 0 \
  --require-provider-tokens
```

For final reported runs, provider-reported token counts were required whenever
available:

```bash
--require-provider-tokens
```

## Saved Results

Final saved outputs are under:

```text
saved_results/
```

Important result folders include:

```text
saved_results/scifact_mistral_final_eval100/
saved_results/scifact_llama70b_final_eval100/
saved_results/hotpotqa_mistral_final_eval100/
saved_results/hotpotqa_llama70b_final_eval100/
saved_results/bioasq_mistral_final_eval100/
saved_results/bioasq_llama70b_final_eval100/
saved_results/asqa_llama70b_final_eval100/
saved_results/ablation/
```

Each run typically contains:

```text
final_table.csv
final_table.md
llm_answers_by_query.csv
llm_summary.csv
retrieval_summary.csv
merged_summary.csv
```

Exact files vary slightly by experiment type.

## Metrics

| Metric | Meaning |
|---|---|
| Token F1 | Lexical overlap between generated answer and reference answer |
| Answer coverage | Recall-style coverage of reference answer tokens |
| Semantic similarity | Embedding-style similarity between generated and reference answer |
| nDCG@10 | Whether relevant documents appear high in the retrieved top-10 list |
| MRR@10 | Reciprocal-rank retrieval quality |
| Total tokens | Provider-reported prompt plus completion tokens |
| Token reduction | Token saving relative to Fixed Top-10 |
| Fallback rate | How often Safe Adaptive expands context after the first answer |

Token F1 is useful but imperfect: it can penalize correct paraphrases and reward
short answer forms. For that reason, the final analysis interprets Token F1
together with semantic similarity, answer coverage, retrieval metrics, and human
annotation.

## Reproducibility Notes

The final reported runs used:

- temperature `0`
- fixed evaluation samples
- provider-reported token counts when available
- the same retrieved candidate pool for fixed, heuristic, and adaptive methods

Local LLM outputs may still vary slightly across Ollama versions, model builds,
hardware backends, or hosted API changes. The saved outputs in `saved_results/`
are therefore the source of the final reported tables.

## Presentation

The HTML presentation prototype is:

```text
rpg_presentation_prototype.html
```

## Project Takeaway

The main lesson is that context size should be treated as a decision, not a
constant. Fixed Top-10 is simple, but it often sends more context than the model
needs. Plain compression is also not enough, because some tasks, especially
multi-hop QA, need broader evidence. Safe Adaptive Context combines adaptive
budgeting, lightweight evidence compression, and answer-aware fallback so the
system can save tokens when evidence is concentrated and spend more when the task
requires it.
