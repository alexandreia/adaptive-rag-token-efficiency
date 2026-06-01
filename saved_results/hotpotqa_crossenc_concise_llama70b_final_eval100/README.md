# HotpotQA Cross-Encoder Llama-70B Final Evaluation

Side experiment using cross-encoder reranking with the final Safe Adaptive Context logic.

## Setup

- Dataset: HotpotQA
- Model: meta-llama/Llama-3.3-70B-Instruct
- Evaluation size: 100 queries
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Prompt style: concise
- Max output tokens: 80
- Pipeline: `src_cross_enc` with final Safe Adaptive Context logic
- Retrieval regime: TF-IDF candidate retrieval followed by cross-encoder reranking

The concise prompt is required for HotpotQA because the task uses short exact answers. A non-concise/default prompt produced much lower answer F1 and was not comparable with the main HotpotQA evaluation.

## Main Result

| Method | F1 | Semantic similarity | Tokens | Token reduction vs Top-10 |
|---|---:|---:|---:|---:|
| Fixed Top-10 | 0.764039 | 0.837879 | 1500.35 | 0.0% |
| Heuristic Rules | 0.766539 | 0.834787 | 804.22 | 46.4% |
| Safe Adaptive Context | 0.750896 | 0.819072 | 1109.23 | 26.1% |
| Fixed Top-3 | 0.737373 | 0.809881 | 499.93 | 66.7% |

## Interpretation

This result shows an important limitation and operating-regime effect.

With cross-encoder reranking, HotpotQA becomes easier for small-context baselines because the relevant evidence is often concentrated near the top of the reranked list. In this setting, Heuristic Rules gives the best quality-efficiency trade-off: it slightly exceeds Fixed Top-10 in answer F1 while reducing tokens by 46.4%.

Safe Adaptive Context remains quality-preserving, retaining 98.3% of Fixed Top-10 F1 and 97.8% of Fixed Top-10 semantic similarity, but it is more conservative and uses more tokens than Heuristic Rules. This suggests that cross-encoder reranking can reduce the advantage of answer-aware fallback by making simpler policies more competitive.

The result should therefore be interpreted as retrieval-regime sensitivity rather than a failure of Safe Adaptive. Safe Adaptive performs best when evidence depth is uncertain or retrieval is imperfect. When reranking already places the required evidence in the top few documents, simpler fixed or heuristic policies can be more token-efficient.

## Contrast with SciFact Cross-Encoder Result

The SciFact cross-encoder result showed the opposite pattern: Safe Adaptive Context slightly outperformed Fixed Top-10 while reducing tokens by approximately 72.5%. Together, the two side experiments suggest that the best context policy depends on task structure and retrieval quality:

- SciFact: Safe Adaptive remains highly effective under reranking.
- HotpotQA: reranking makes Heuristic Rules and Fixed Top-3 much more competitive.
- Overall: Safe Adaptive is robust and quality-preserving, but not universally the cheapest policy.
