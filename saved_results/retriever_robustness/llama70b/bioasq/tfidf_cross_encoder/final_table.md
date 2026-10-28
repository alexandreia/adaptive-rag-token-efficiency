| dataset | method | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BioASQ | Fixed Top-3 | fixed_3_full | 0.658399 | 0.94 | 0.330337 | 0.501033 | 0.755773 | 101.3% | 99.9% | 1159.58 | 68.1% | 0.0% |
| BioASQ | Fixed Top-5 | fixed_5_full | 0.771813 | 0.94125 | 0.326849 | 0.509961 | 0.751672 | 100.2% | 99.4% | 1852.78 | 49.0% | 0.0% |
| BioASQ | Fixed Top-10 | fixed_10_full | 0.904823 | 0.942639 | 0.326127 | 0.50925 | 0.756476 | 100.0% | 100.0% | 3634.96 | 0.0% | 0.0% |
| BioASQ | Adaptive-k | adaptive_k_full | 0.696999 | 0.940833 | 0.335717 | 0.498658 | 0.75765 | 102.9% | 100.2% | 1222.87 | 66.4% | 0.0% |
| BioASQ | Safe Adaptive Context | answer_aware_fallback | 0.660054 | 0.94 | 0.328688 | 0.505505 | 0.750119 | 100.8% | 99.2% | 833.26 | 77.1% | 2.5% |
| BioASQ | TACER | task_aware_coverage_ultra | 0.713701 | 0.940833 | 0.320255 | 0.490641 | 0.749232 | 98.2% | 99.0% | 804.62 | 77.9% | 0.0% |
