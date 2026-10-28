| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SciFact | No Retrieval | no_retrieval_full | 0.0 | 0.0 | 0.274837 | 0.495251 | 0.367923 | 113.0% | 67.2% | 119.4 | 96.5% | 0.0% |
| SciFact | Fixed Top-3 | fixed_3_full | 0.517187 | 0.505833 | 0.245145 | 0.818521 | 0.544218 | 100.8% | 99.4% | 1135.12 | 66.7% | 0.0% |
| SciFact | Fixed Top-5 | fixed_5_full | 0.548163 | 0.523583 | 0.245099 | 0.820034 | 0.540502 | 100.8% | 98.7% | 1785.34 | 47.7% | 0.0% |
| SciFact | Fixed Top-7 | fixed_7_full | 0.559277 | 0.528345 | 0.245033 | 0.827207 | 0.547742 | 100.7% | 100.0% | 2433.97 | 28.7% | 0.0% |
| SciFact | Fixed Top-10 | fixed_10_full | 0.56748 | 0.530595 | 0.243247 | 0.815308 | 0.547716 | 100.0% | 100.0% | 3413.59 | 0.0% | 0.0% |
| SciFact | Heuristic Rules | heuristic_rules_full | 0.548681 | 0.523798 | 0.245189 | 0.819101 | 0.546372 | 100.8% | 99.8% | 2147.67 | 37.1% | 0.0% |
| SciFact | Adaptive-k | adaptive_k_full | 0.491895 | 0.485833 | 0.251003 | 0.818284 | 0.545321 | 103.2% | 99.6% | 858.02 | 74.9% | 0.0% |
| SciFact | Safe Adaptive Context | answer_aware_fallback | 0.525143 | 0.510083 | 0.246903 | 0.82047 | 0.546241 | 101.5% | 99.7% | 1026.84 | 69.9% | 10.0% |
| SciFact | Coverage-Guided Ultra | coverage_guided_ultra | 0.500015 | 0.48975 | 0.25395 | 0.813562 | 0.552338 | 104.4% | 100.8% | 653.15 | 80.9% | 0.0% |
| SciFact | TACER | task_aware_coverage_ultra | 0.500015 | 0.48975 | 0.252813 | 0.81385 | 0.55417 | 103.9% | 101.2% | 653.6 | 80.9% | 0.0% |
