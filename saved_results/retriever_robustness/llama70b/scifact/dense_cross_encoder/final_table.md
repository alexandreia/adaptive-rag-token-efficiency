| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact | Fixed Top-3 | fixed_3_full | 0.634527 | 0.635 | 0.252087 | 0.829984 | 0.547019 | 101.8% | 99.6% | 1224.52 | 67.7% | 0.0% |
| SciFact | Fixed Top-5 | fixed_5_full | 0.663181 | 0.64875 | 0.251716 | 0.835277 | 0.544674 | 101.7% | 99.2% | 1950.79 | 48.5% | 0.0% |
| SciFact | Fixed Top-10 | fixed_10_full | 0.686115 | 0.657468 | 0.247615 | 0.828207 | 0.549076 | 100.0% | 100.0% | 3789.59 | 0.0% | 0.0% |
| SciFact | Adaptive-k | adaptive_k_full | 0.617563 | 0.619583 | 0.264054 | 0.832711 | 0.549601 | 106.6% | 100.1% | 885.53 | 76.6% | 0.0% |
| SciFact | Safe Adaptive Context | answer_aware_fallback | 0.636681 | 0.63625 | 0.248906 | 0.821381 | 0.550562 | 100.5% | 100.3% | 931.03 | 75.4% | 4.5% |
| SciFact | TACER | task_aware_coverage_ultra | 0.642152 | 0.639583 | 0.254747 | 0.835102 | 0.554197 | 102.9% | 100.9% | 677.16 | 82.1% | 0.0% |
