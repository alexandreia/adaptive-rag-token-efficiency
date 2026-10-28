| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HotpotQA | No Retrieval | no_retrieval_full | 0.0 | 0.0 | 0.309072 | 0.306123 | 0.470348 | 48.0% | 63.0% | 147.91 | 91.4% | 0.0% |
| HotpotQA | Fixed Top-3 | fixed_3_full | 0.596062 | 0.7675 | 0.53189 | 0.531027 | 0.644139 | 82.5% | 86.3% | 556.92 | 67.7% | 0.0% |
| HotpotQA | Fixed Top-5 | fixed_5_full | 0.64131 | 0.77975 | 0.555306 | 0.554611 | 0.663382 | 86.2% | 88.8% | 863.4 | 49.9% | 0.0% |
| HotpotQA | Fixed Top-7 | fixed_7_full | 0.679501 | 0.786179 | 0.600881 | 0.605397 | 0.709987 | 93.2% | 95.1% | 1192.27 | 30.8% | 0.0% |
| HotpotQA | Fixed Top-10 | fixed_10_full | 0.717349 | 0.790206 | 0.644556 | 0.65073 | 0.746732 | 100.0% | 100.0% | 1721.8 | 0.0% | 0.0% |
| HotpotQA | Heuristic Rules | heuristic_rules_full | 0.665239 | 0.782798 | 0.57514 | 0.579539 | 0.691306 | 89.2% | 92.6% | 1079.75 | 37.3% | 0.0% |
| HotpotQA | Adaptive-k | adaptive_k_full | 0.539579 | 0.749847 | 0.489965 | 0.492777 | 0.605493 | 76.0% | 81.1% | 510.24 | 70.4% | 0.0% |
| HotpotQA | Safe Adaptive Context | answer_aware_fallback | 0.716507 | 0.790206 | 0.627005 | 0.630123 | 0.730021 | 97.3% | 97.8% | 1164.89 | 32.3% | 0.0% |
| HotpotQA | Coverage-Guided Ultra | coverage_guided_ultra | 0.630262 | 0.780417 | 0.463644 | 0.466944 | 0.583081 | 71.9% | 78.1% | 452.37 | 73.7% | 0.5% |
| HotpotQA | TACER | task_aware_coverage_ultra | 0.715832 | 0.790873 | 0.627005 | 0.630123 | 0.724261 | 97.3% | 97.0% | 1160.92 | 32.6% | 0.0% |
