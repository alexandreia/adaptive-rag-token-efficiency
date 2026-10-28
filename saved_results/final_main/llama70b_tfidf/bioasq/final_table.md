| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BioASQ | No Retrieval | no_retrieval_full | 0.0 | 0.0 | 0.201836 | 0.233687 | 0.532329 | 61.5% | 70.7% | 121.39 | 96.5% | 0.0% |
| BioASQ | Fixed Top-3 | fixed_3_full | 0.604519 | 0.906667 | 0.322358 | 0.48322 | 0.750155 | 98.2% | 99.7% | 1140.5 | 66.7% | 0.0% |
| BioASQ | Fixed Top-5 | fixed_5_full | 0.710061 | 0.907667 | 0.325737 | 0.498267 | 0.751332 | 99.2% | 99.8% | 1795.32 | 47.5% | 0.0% |
| BioASQ | Fixed Top-7 | fixed_7_full | 0.770304 | 0.907667 | 0.324833 | 0.502418 | 0.749411 | 98.9% | 99.6% | 2453.7 | 28.3% | 0.0% |
| BioASQ | Fixed Top-10 | fixed_10_full | 0.831084 | 0.908167 | 0.328406 | 0.506597 | 0.752756 | 100.0% | 100.0% | 3420.69 | 0.0% | 0.0% |
| BioASQ | Heuristic Rules | heuristic_rules_full | 0.732266 | 0.907667 | 0.325061 | 0.49935 | 0.748764 | 99.0% | 99.5% | 2079.66 | 39.2% | 0.0% |
| BioASQ | Adaptive-k | adaptive_k_full | 0.539145 | 0.8975 | 0.319308 | 0.458208 | 0.738199 | 97.2% | 98.1% | 917.53 | 73.2% | 0.0% |
| BioASQ | Safe Adaptive Context | answer_aware_fallback | 0.607442 | 0.906667 | 0.322513 | 0.49395 | 0.749048 | 98.2% | 99.5% | 870.18 | 74.6% | 4.0% |
| BioASQ | Coverage-Guided Ultra | coverage_guided_ultra | 0.591526 | 0.91 | 0.313178 | 0.462305 | 0.740013 | 95.4% | 98.3% | 679.25 | 80.1% | 0.0% |
| BioASQ | TACER | task_aware_coverage_ultra | 0.581004 | 0.91 | 0.313699 | 0.464341 | 0.740165 | 95.5% | 98.3% | 656.3 | 80.8% | 0.0% |
