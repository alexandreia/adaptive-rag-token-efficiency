| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASQA | Fixed Top-3 | fixed_3_full | 0.794749 | 0.985 | 0.415969 | 0.409439 | 0.781881 | 99.6% | 99.8% | 456.65 | 57.2% | 0.0% |
| ASQA | Fixed Top-5 | fixed_5_full | 0.888774 | 0.98625 | 0.418641 | 0.427076 | 0.784611 | 100.2% | 100.2% | 647.54 | 39.3% | 0.0% |
| ASQA | Fixed Top-10 | fixed_10_full | 0.924695 | 0.98625 | 0.417842 | 0.425609 | 0.783324 | 100.0% | 100.0% | 1066.49 | 0.0% | 0.0% |
| ASQA | Adaptive-k | adaptive_k_full | 0.825496 | 0.98625 | 0.414962 | 0.403715 | 0.780825 | 99.3% | 99.7% | 481.64 | 54.8% | 0.0% |
| ASQA | Safe Adaptive Context | answer_aware_fallback | 0.798813 | 0.985 | 0.414382 | 0.411775 | 0.780628 | 99.2% | 99.7% | 486.76 | 54.4% | 4.5% |
| ASQA | TACER | task_aware_coverage_ultra | 0.796831 | 0.985 | 0.417201 | 0.410691 | 0.780995 | 99.8% | 99.7% | 485.88 | 54.4% | 5.5% |
