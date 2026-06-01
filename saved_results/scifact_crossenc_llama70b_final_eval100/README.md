# SciFact Cross-Encoder Llama-70B Final Evaluation

Side experiment using cross-encoder reranking with the final Safe Adaptive Context logic.

Dataset: SciFact
Model: meta-llama/Llama-3.3-70B-Instruct
Evaluation size: 100 queries
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
Output tokens: 80
Pipeline: src_cross_enc with final Safe Adaptive logic

Key merged result:
- Safe Adaptive Context F1: 0.265369
- Safe Adaptive Context semantic similarity: 0.574401
- Safe Adaptive Context total tokens: 938.99
- Safe Adaptive Context token reduction vs Fixed Top-10: 72.48%
- Safe Adaptive Context fallback rate: 4%

Compared with Fixed Top-10:
- Fixed Top-10 F1: 0.261028
- Fixed Top-10 semantic similarity: 0.572803
- Fixed Top-10 total tokens: 3411.56
