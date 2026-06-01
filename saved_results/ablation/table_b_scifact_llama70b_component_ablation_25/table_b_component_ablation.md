| dataset | component_variant | code_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | answer_coverage | semantic_similarity | f1_retained_vs_top10 | semantic_similarity_retained_vs_top10 | total_tokens | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | Fixed Top-10 | fixed_top10_full | 0.558136 | 0.510048 | 0.247355 | 0.832051 | 0.488839 | 100.0% | 100.0% | 3324.6 | 0.0% | 0.0% |
| scifact | Adaptive Compact Only | adaptive_compact_only | 0.484008 | 0.473333 | 0.247956 | 0.808774 | 0.480824 | 100.2% | 98.4% | 735.76 | 77.9% | 0.0% |
| scifact | Adaptive Full Only | adaptive_full_only | 0.484008 | 0.473333 | 0.239624 | 0.810804 | 0.485116 | 96.9% | 99.2% | 1096.76 | 67.0% | 0.0% |
| scifact | Fixed Top-5 Compact + Fallback | fixed_top5_compact_fallback | 0.532184 | 0.499333 | 0.258336 | 0.831013 | 0.489237 | 104.4% | 100.1% | 1208.68 | 63.6% | 4.0% |
| scifact | Full Safe Adaptive Context | full_safe_adaptive | 0.484008 | 0.473333 | 0.247376 | 0.808461 | 0.477319 | 100.0% | 97.6% | 780.4 | 76.5% | 4.0% |
