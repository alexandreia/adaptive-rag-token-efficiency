| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MSMARCO | No Retrieval | no_retrieval_full | 0.0 | 0.0 | 0.101411 | 0.136235 | 0.282955 | 51.8% | 59.8% | 110.88 | 92.1% | 0.0% |
| MSMARCO | Fixed Top-3 | fixed_3_full | 0.385599 | 0.36 | 0.196697 | 0.580155 | 0.468497 | 100.6% | 99.0% | 528.35 | 62.2% | 0.0% |
| MSMARCO | Fixed Top-5 | fixed_5_full | 0.474519 | 0.40325 | 0.197695 | 0.624256 | 0.472222 | 101.1% | 99.8% | 785.55 | 43.8% | 0.0% |
| MSMARCO | Fixed Top-7 | fixed_7_full | 0.535107 | 0.428607 | 0.198178 | 0.621408 | 0.474793 | 101.3% | 100.3% | 1035.16 | 26.0% | 0.0% |
| MSMARCO | Fixed Top-10 | fixed_10_full | 0.552481 | 0.434579 | 0.195618 | 0.614194 | 0.473325 | 100.0% | 100.0% | 1397.97 | 0.0% | 0.0% |
| MSMARCO | Heuristic Rules | heuristic_rules_full | 0.487252 | 0.40894 | 0.2006 | 0.616731 | 0.475987 | 102.5% | 100.6% | 813.03 | 41.8% | 0.0% |
| MSMARCO | Adaptive-k | adaptive_k_full | 0.431013 | 0.37028 | 0.199485 | 0.549167 | 0.466066 | 102.0% | 98.5% | 718.35 | 48.6% | 0.0% |
| MSMARCO | Safe Adaptive Context | answer_aware_fallback | 0.515606 | 0.420329 | 0.195315 | 0.621966 | 0.468943 | 99.8% | 99.1% | 859.26 | 38.5% | 1.0% |
| MSMARCO | Coverage-Guided Ultra | coverage_guided_ultra | 0.431013 | 0.37028 | 0.193461 | 0.533332 | 0.451645 | 98.9% | 95.4% | 642.04 | 54.1% | 0.0% |
| MSMARCO | TACER | task_aware_coverage_ultra | 0.431013 | 0.37028 | 0.194744 | 0.52266 | 0.455981 | 99.6% | 96.3% | 642.36 | 54.1% | 0.0% |
