# Adaptive RAG for Token Efficiency

This is the GitHub-ready version of our real Adaptive Context experiment.

The project tests whether an LLM/RAG system can use less context while keeping answer quality close to a fixed top-10 baseline.

The important method is:

> **Safe Adaptive Context**: answer first with compact evidence, check whether the answer looks risky, and only then expand to full top-10 context.

## Contributors

- Andreia Alexa
- Simon Backhaus Brudzewski
- Karlis Martins Auce
- Joel Vazquez
- Sheraz Ahmad 

## What The Pipeline Does

For each SciFact query:

1. Load the SciFact biomedical abstracts.
2. Retrieve top-10 candidate documents with TF-IDF cosine retrieval.
3. Compare fixed and adaptive context methods.
4. Send the selected context to Mistral through Ollama.
5. Record real provider token counts from Ollama when available.
6. Measure answer F1, tokens, latency, and fallback rate.

## Full System Workflow

This is the complete work the system does from start to finish.

### 1. Load The Dataset

The system reads:

```text
data/scifact/documents.jsonl
data/scifact/queries_150_seed0.jsonl
```

Documents contain scientific abstracts. Queries contain the claim/question,
the gold relevant document ids, and reference text for evaluation.

### 2. Build The Retriever

The system builds a TF-IDF representation of the document collection.

TF-IDF means:

- common words get lower importance
- rare/informative words get higher importance
- each document becomes a sparse vector of word weights

This is used to compare a query with every document.

### 3. Retrieve Candidate Documents

For each query, the system retrieves a ranked list of candidate documents using
cosine similarity.

The retrieval stage produces the same top candidate list for every method.
The difference is what each method decides to send to the LLM.

### 4. Run Fixed Baselines

The fixed baselines are:

- `no_retrieval`
- `fixed_3_full`
- `fixed_5_full`
- `fixed_10_full`

`no_retrieval` sends no documents. It is the closed-book baseline.

The fixed top-k methods always send the same number of full documents.

They answer the question:

> What happens if we use a normal fixed top-k RAG system?

`fixed_10_full` is the main expensive baseline.

### 4b. Planned Heuristic Adaptive Model

### TODO MAIN IDEA SO FAR

This method would use simple rules, such as query length and retrieval score,
to choose k before generation.

It is useful because it tests whether a lightweight rule system is enough, or
whether the learned/safe adaptive model gives a better tradeoff.

### 5. Train The Basic Adaptive Budget Model

The system builds a simple adaptive budget model using the evaluation split.

It uses cheap features such as:

- query length
- top retrieval score
- score gaps between retrieved documents
- score entropy
- whether evidence is concentrated or spread across documents

The model predicts whether the query should use a small or large context budget.

This is the Basic Adaptive model.

### 6. Compress Evidence

For compact methods, the system does not send the whole document text.
Instead, it uses:

```text
evidence_ngram_neighbors
```

This means:

1. split each document into sentences
2. score sentences by word/phrase overlap with the query
3. keep the strongest evidence sentences
4. keep neighboring sentences too, so the evidence still has context

This creates a smaller evidence package for the LLM.

### 7. Run Safe Adaptive Context

Safe Adaptive Context is our main model.

It works in two passes:

First pass:

1. choose an adaptive context budget
2. compress the chosen documents into evidence spans
3. ask Mistral to answer

Safety check:

1. inspect the generated answer
2. check whether it is empty, too short, uncertain, or poorly grounded

Fallback:

1. if the answer looks risky, expand to full top-10 documents
2. ask Mistral again
3. count both the first-pass cost and fallback cost

This is the main idea:

> easy queries stay cheap, hard queries get more context.

### 8. Call Mistral Through Ollama

The system sends prompts to:

```text
http://localhost:11434/v1/chat/completions
```

This is Ollama's OpenAI-compatible API.

The model is called with:

```text
temperature = 0.0
```

This makes generation as deterministic as possible.

### 9. Record Real Token Counts

For final runs, use:

```bash
--require-provider-tokens
```

This forces the system to use token counts reported by the model/provider.

The important token fields are:

- prompt tokens
- completion tokens
- total tokens

### 10. Evaluate Results

The system evaluates:

- answer F1
- total tokens
- token reduction vs fixed top-10
- generation time
- time reduction vs fixed top-10
- fallback rate

The final report table is saved as:

```text
outputs/<run_name>/final_table.csv
```

## Methods Compared

| Method | Code mode | Meaning |
|---|---|---|
| No Retrieval | `no_retrieval` | Closed-book baseline; Mistral answers without retrieved documents |
| Fixed Top-3 | `fixed_3_full` | Always send 3 full documents |
| Fixed Top-5 | `fixed_5_full` | Always send 5 full documents |
| Fixed Top-7 | `fixed_7_full` | Always send 7 full documents |
| Fixed Top-10 | `fixed_10_full` | Always send 10 full documents; expensive baseline |
| Heuristic Rules | `heuristic_rules_full` | Rule-based controller using query length and retrieval score gaps |
| Basic Adaptive + Compact Evidence | `learned_budget_evidence_ngram_neighbors` | Predict a budget, then send compact evidence spans |
| Safe Adaptive Context | `answer_aware_fallback` | Try compact adaptive evidence first; expand to full top-10 only if the answer looks weak |


## Folder Structure

```text
adaptive_rag_github/
├── data/scifact/
│   ├── documents.jsonl
│   ├── queries_all.jsonl
│   ├── queries_150_seed0.jsonl
│   └── README.md
├── scripts/
│   └── run_experiment.py
├── docs/
│   └── CODE_WALKTHROUGH.md
├── src/adaptive_retrieval/
│   ├── budget_experiment.py
│   ├── data.py
│   ├── learned_budget.py
│   ├── llm_budget.py
│   ├── metrics.py
│   ├── retriever.py
│   └── text.py
├── PROJECT_PLAN.md
├── README.md
└── requirements.txt
```

## Dataset

Dataset: SciFact from BEIR.

This repo includes:

- `data/scifact/documents.jsonl`: 5,183 scientific abstracts.
- `data/scifact/queries_150_seed0.jsonl`: fixed 150-query sample.

The query file contains:

- query text
- relevant document ids from qrels
- reference answer text

## Possible Extra Datasets

The must-ship version uses SciFact because that is the agreed course dataset.
If we have time, the same code structure can be tested on three more BEIR-style
datasets:

| Dataset | Why It Helps |
|---|---|
| NFCorpus | More diverse biomedical/nutrition queries; useful for testing whether the controller handles less uniform data |
| FiQA | Financial question answering; useful for checking if the method works outside biomedical text |
| TREC-COVID | Scientific/medical COVID retrieval; useful as another high-stakes scientific retrieval setting |

These datasets are not required for the core submission, but they would make the
project stronger because they test generalization beyond one dataset.
We may need to use other datasets to prove if is able to generalize. 

## Install

Install Ollama:

```bash
brew install ollama
```

Download/test Mistral:

```bash
ollama run mistral "What is the capital of Spain?"
```

Install Python requirements:

```bash
pip install -r requirements.txt
```

## Dry Run

Use this to check that the code works without calling the LLM:

```bash
python3 scripts/run_experiment.py \
  --dry-run \
  --max-eval-queries 5 \
  --output-dir outputs/dry_run
```

Dry-run results are only a pipeline check. They are not final LLM results.

## Real Ollama Run

Run 50 SciFact queries:

```bash
python3 scripts/run_experiment.py \
  --documents data/scifact/documents.jsonl \
  --queries data/scifact/queries_150_seed0.jsonl \
  --dataset-name SciFact \
  --output-dir outputs/scifact_mistral_50 \
  --model mistral \
  --api-url http://localhost:11434/v1 \
  --no-api-key \
  --max-eval-queries 50 \
  --seed 0 \
  --require-provider-tokens
```

Run the full 150-query sample:

```bash
python3 scripts/run_experiment.py \
  --documents data/scifact/documents.jsonl \
  --queries data/scifact/queries_150_seed0.jsonl \
  --dataset-name SciFact \
  --output-dir outputs/scifact_mistral_150 \
  --model mistral \
  --api-url http://localhost:11434/v1 \
  --no-api-key \
  --max-eval-queries 150 \
  --seed 0 \
  --require-provider-tokens
```

## Reproducibility

The project uses a fixed query sample:

```text
data/scifact/queries_150_seed0.jsonl
```

The runner also has:

```bash
--seed 0
```

Important caveat:

With the same code, same query file, same seed, same Ollama version, same Mistral
model, and same machine settings, the run should be reproducible.

However, local LLM generation can still vary slightly if the Ollama version,
model build, backend, or hardware settings change. So the method comparison
should be stable, but exact generated text may not be byte-for-byte identical
forever across different machines or future model versions.

## Outputs

Each run writes:

```text
outputs/<run_name>/llm_answers_by_query.csv
outputs/<run_name>/llm_summary.csv
outputs/<run_name>/retrieval_summary.csv
outputs/<run_name>/final_table.csv
outputs/<run_name>/final_table.md
```

The file for the report is:

```text
outputs/<run_name>/final_table.csv
```

## Saved Results And Ablations

Final saved result tables are kept in:

```text
saved_results/
```

The ablation section is:

```text
saved_results/ablation/
```

It contains:

- `table_b_scifact_llama70b_component_ablation_25/`: Table B component ablation for Safe Adaptive Context.
- `context_compression/`: compression-focused ablation across completed runs.

Table B isolates the main parts of Safe Adaptive Context:

- adaptive budget
- compact evidence compression
- fallback expansion

## What Each Important File Does

| File | What It Does |
|---|---|
| `scripts/run_experiment.py` | Main script. Loads data, runs methods, writes tables |
| `scripts/run_table_b_ablation.py` | Focused Table B component ablation script |
| `src/adaptive_retrieval/data.py` | Reads `documents.jsonl` and `queries.jsonl` |
| `src/adaptive_retrieval/text.py` | Tokenization, TF-IDF vectors, token estimates |
| `src/adaptive_retrieval/retriever.py` | Retrieves top documents with TF-IDF cosine similarity |
| `src/adaptive_retrieval/budget_experiment.py` | Small helper for retrieval-side budget metrics |
| `src/adaptive_retrieval/learned_budget.py` | Basic adaptive budget predictor |
| `src/adaptive_retrieval/llm_budget.py` | Main LLM pipeline and Safe Adaptive Context model |
| `src/adaptive_retrieval/metrics.py` | Token F1, coverage, precision, recall, MRR, nDCG@10 |

The file to understand first is:

```text
src/adaptive_retrieval/llm_budget.py
```

The function to understand first is:

```text
answer_aware_fallback_run(...)
```

For a plain-language explanation of the code, read:

```text
docs/CODE_WALKTHROUGH.md
```

## Metrics

| Metric | Meaning |
|---|---|
| `answer_f1` | Token-overlap F1 between generated answer and reference text |
| `total_tokens` | Prompt + completion tokens |
| `token_reduction_vs_top10` | Token saving compared with fixed top-10 |
| `generation_time_ms` | Generation latency |
| `time_reduction_vs_top10` | Time saving compared with fixed top-10 |
| `fallback_rate` | How often Safe Adaptive expanded to full top-10 |


| Category | Metric | What It Shows |
|---|---|---|
| `Retrieval quality` | nDCG@10 | Are relevant documents ranked high? |
| `Answer quality` | Token F1 | Does the generated answer overlap with the reference? |
| `Efficiency` | Total tokens | How expensive is the context and generation? |
| `Efficiency` | Token reduction | How much token usage is reduced compared with fixed top-10? |
| `Latency` | Generation time | Is the method faster or slower? |
| `Safety behavior` | Fallback rate | How often does Safe Adaptive expand to more context? |

## Important Note

Token F1 is useful but imperfect. It measures lexical overlap, so correct paraphrases can still score low. In addition for that we also evaluate nDCG@10 to measure whether relevant documents appear high in the ranked list. 
