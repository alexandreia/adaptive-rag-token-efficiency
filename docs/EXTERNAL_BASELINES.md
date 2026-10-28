# External Baselines

This note tracks public baseline code that can strengthen the TACER paper.
Third-party repositories should be fetched into `external_baselines/`, which is
ignored by git, so licenses and external histories remain separate from this
project.

## Fetch Code

From the project root:

```bash
bash scripts/fetch_external_baselines.sh
```

This fetches:

- LLMLingua / LLMLingua-2: `https://github.com/microsoft/LLMLingua`
- Adaptive-k Retrieval: `https://github.com/megagonlabs/adaptive-k-retrieval`
- FLARE: `https://github.com/jzbjyb/FLARE`

## Baseline Status

### LLMLingua-2

Best candidate for a serious additional comparison. It is directly about prompt
compression and has a clean Python package interface:

```bash
python -m pip install llmlingua
```

The fairest TACER comparison is:

- Fixed Top-10
- Adaptive-k (paper)
- FLARE-lite
- Fixed Top-10 + LLMLingua-2 compression
- Adaptive-k (paper) + LLMLingua-2 compression
- Safe Adaptive Context
- TACER

Important caveat: LLMLingua-2 uses an extra compressor model. Report generator
prompt-token savings separately from compressor runtime/compute.

The project exposes these rows as:

- `llmlingua2_top10`
- `llmlingua2_adaptive_k_official`

The default LLMLingua-2 model is
`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`. To use the smaller model,
set:

```bash
export LLMLINGUA2_MODEL=microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
```

The wrapper chooses `cuda`, `mps`, or `cpu` automatically. To force CPU on a
Mac or laptop run, set:

```bash
export LLMLINGUA2_DEVICE=cpu
```

The implementation compresses retrieved evidence as structured document blocks
and then rebuilds the normal RAG prompt around the compressed evidence. This is
important for LLMLingua-2 models with 512-token encoder limits; passing the
entire RAG prompt as one long string can trigger sequence-length warnings and
produce an unfairly degraded baseline.

### Adaptive-k

Official code exists at `https://github.com/megagonlabs/adaptive-k-retrieval`.
The repository README identifies it as the official implementation for
"Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just
Adaptive-k" and uses a BSD-3-Clause license.

The official QA script configures Adaptive-k as:

```text
--adaptive_retrieval
--retrieval_strategy largest_gap
--ignore_extreme_tail 0.1
--retrieve_more 5
```

The project includes three Adaptive-k rows:

- `adaptive_k`: strict largest-adjacent-drop cutoff.
- `adaptive_k_paper`: paper-faithful largest-gap cutoff with fixed buffer
  `B=5` and top-90% gap search, adapted to our fixed top-10 candidate lists.
- `adaptive_k_official`: legacy mode name kept for compatibility; currently
  used for the same paper-faithful Adaptive-k comparison in final tables.

For the final paper comparison, describe this row as `Adaptive-k (paper)` unless
the official repository is run end-to-end inside the same evaluation harness.

### FLARE

FLARE has official code, but it is not a drop-in context-selection baseline. It
is an active retrieval-and-generation system that can issue additional retrievals
during generation and originally relies on an Elasticsearch-backed Wikipedia
index plus iterative LLM calls.

For this project, FLARE should be treated as one of:

- a related-work comparison in the paper, or
- a small appendix experiment using `flare_lite`, our constrained variant that
  uses the fixed candidate pool and clearly labels itself as a TACER-compatible
  adaptation.

It should not be mixed into the main tables unless we can make retrieval corpus,
LLM calls, token accounting, and stopping policy comparable.

Implemented row:

- `flare_lite`: starts with the top-1 retrieved document, checks whether the
  answer appears unsupported by that evidence, then expands to the official
  Adaptive-k budget and finally Fixed Top-10 if still risky. This captures the
  FLARE intuition of retrieval expansion after a weak first generation, but it
  is not official FLARE because the hosted API path does not expose FLARE's
  token-level uncertainty signal during generation.
