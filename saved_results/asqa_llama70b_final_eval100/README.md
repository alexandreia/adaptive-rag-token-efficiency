# ASQA Llama-70B Final Evaluation

Long-answer / explanatory QA experiment using the final Safe Adaptive Context logic.

## Setup

- Dataset: ASQA
- Model: meta-llama/Llama-3.3-70B-Instruct
- Evaluation size: 100 queries
- Retrieval: TF-IDF
- Prompt style: default
- Max output tokens: 160
- Pipeline: main `src` pipeline with final Safe Adaptive Context logic

## Motivation

ASQA was added as a harder long-answer stress test. Unlike compact evidence datasets such as SciFact or short-answer datasets such as HotpotQA, ASQA contains ambiguous questions that often require broader, explanatory answers. This makes it useful for testing whether adaptive context selection can still reduce tokens when the model needs more information to answer well.

The goal of this experiment is not only to maximize answer quality, but to evaluate the quality-token trade-off in a longer-form QA setting.

## Main Result

| Method | F1 | Answer coverage | Semantic similarity | Tokens | Token reduction vs Top-10 |
|---|---:|---:|---:|---:|---:|
| Fixed Top-10 | 0.410959 | 0.418503 | 0.786414 | 1229.47 | 0.0% |
| Heuristic Rules | 0.392896 | 0.414429 | 0.781811 | 629.01 | 48.8% |
| Safe Adaptive Context | 0.416842 | 0.411050 | 0.782406 | 503.41 | 59.1% |
| Fixed Top-3 | 0.397971 | 0.394838 | 0.771982 | 421.86 | 65.7% |

## Interpretation

ASQA gives a strong result for Safe Adaptive Context in a longer-answer setting. Safe Adaptive achieved the highest answer F1 among the compared methods while reducing token usage by 59.1% relative to Fixed Top-10.

Compared with Fixed Top-10, Safe Adaptive retained 101.4% of answer F1 and 99.5% of semantic similarity while using substantially fewer tokens. Answer coverage was slightly lower than Top-10, but remained close, suggesting that the token reduction did not substantially degrade long-form answer quality.

Compared with Heuristic Rules, Safe Adaptive achieved higher F1, slightly higher semantic similarity, and lower token usage. This is important because it shows that the adaptive policy can outperform a simpler heuristic baseline in a longer-form explanatory QA setting.

## Role in the Overall Study

ASQA complements the main compact-evidence experiments by testing a different task regime:

- SciFact and BioASQ test compact scientific/biomedical evidence QA.
- HotpotQA tests short-answer multi-hop QA.
- Cross-encoder experiments test the effect of stronger reranking.
- ASQA tests longer-form ambiguous/explanatory QA.

The ASQA result supports the broader claim that Safe Adaptive Context is not only useful for compact answers, but can also provide a strong quality-token trade-off when answers require broader explanation and synthesis.
