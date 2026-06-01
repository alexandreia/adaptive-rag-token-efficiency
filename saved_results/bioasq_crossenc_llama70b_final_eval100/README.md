# BioASQ Cross-Encoder Llama-70B Final Evaluation

Side experiment using cross-encoder reranking with the final Safe Adaptive Context logic.

## Setup

- Dataset: BioASQ
- Model: meta-llama/Llama-3.3-70B-Instruct
- Evaluation size: 100 queries
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Prompt style: default
- Max output tokens: 80
- Pipeline: `src_cross_enc` with final Safe Adaptive Context logic
- Retrieval regime: TF-IDF candidate retrieval followed by cross-encoder reranking

## Main Result

| Method | F1 | Semantic similarity | Tokens | Token reduction vs Top-10 |
|---|---:|---:|---:|---:|
| Fixed Top-10 | 0.337888 | 0.772815 | 3496.92 | 0.0% |
| Heuristic Rules | 0.339964 | 0.774564 | 1749.54 | 50.0% |
| Safe Adaptive Context | 0.341810 | 0.776074 | 813.63 | 76.7% |
| Fixed Top-3 | 0.333421 | 0.774012 | 1185.91 | 66.1% |

## Interpretation

The BioASQ cross-encoder result is a strong positive result for Safe Adaptive Context. Safe Adaptive slightly exceeds Fixed Top-10 in answer F1 and semantic similarity while reducing token usage by 76.7%. It also outperforms Heuristic Rules while using less than half as many tokens.

This suggests that, for biomedical evidence-based QA, Safe Adaptive remains highly effective even when retrieval is strengthened with cross-encoder reranking. Unlike the HotpotQA cross-encoder setting, where simpler policies become highly competitive, BioASQ still benefits from answer-aware adaptive context control.

Together with the SciFact cross-encoder result, this supports the interpretation that Safe Adaptive is especially useful for scientific and biomedical evidence tasks, while its advantage is more retrieval-regime dependent on general multi-hop QA.
