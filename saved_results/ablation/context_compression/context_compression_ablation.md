| run | mode | method_name | compression_mode | ndcg_at_10 | mrr_at_10 | answer_f1 | semantic_similarity | total_tokens | f1_retained_vs_top10 | semantic_retained_vs_top10 | token_reduction_vs_top10 | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scifact_llama70b | fixed_10_full | Fixed Full Context | full | 0.602129 | 0.562762 | 0.261904 | 0.56934 | 3411.76 | 100.0% | 100.0% | 0.0% | 0.0% |
| scifact_llama70b | fixed_10_evidence_ngram_neighbors | Compressed Fixed Full Context | evidence_ngram_neighbors | 0.602129 | 0.562762 | 0.257487 | 0.570925 | 2150.14 | 98.3% | 100.3% | 37.0% | 0.0% |
| scifact_llama70b | heuristic_rules_full | Heuristic Rules | full | 0.586836 | 0.556833 | 0.266407 | 0.574866 | 2233.7 | 101.7% | 101.0% | 34.5% | 0.0% |
| scifact_llama70b | heuristic_rules_evidence_ngram_neighbors | Heuristic Rules + Compact Evidence | evidence_ngram_neighbors | 0.586836 | 0.556833 | 0.262447 | 0.574786 | 1453.02 | 100.2% | 101.0% | 57.4% | 0.0% |
| scifact_llama70b | answer_aware_fallback | Safe Adaptive Context | compact_then_full_fallback | 0.560244 | 0.541667 | 0.265189 | 0.568971 | 933.48 | 101.3% | 99.9% | 72.6% | 5.0% |
| scifact_mistral | fixed_10_full | Fixed Full Context | full | 0.602129 | 0.562762 | 0.20353 | 0.363887 | 3781.14 | 100.0% | 100.0% | 0.0% | 0.0% |
| scifact_mistral | fixed_10_evidence_ngram_neighbors | Compressed Fixed Full Context | evidence_ngram_neighbors | 0.602129 | 0.562762 | 0.247221 | 0.407015 | 2660.7 | 121.5% | 111.9% | 29.6% | 0.0% |
| scifact_mistral | heuristic_rules_full | Heuristic Rules | full | 0.586836 | 0.556833 | 0.26454 | 0.437337 | 2651.75 | 130.0% | 120.2% | 29.9% | 0.0% |
| scifact_mistral | heuristic_rules_evidence_ngram_neighbors | Heuristic Rules + Compact Evidence | evidence_ngram_neighbors | 0.586836 | 0.556833 | 0.25804 | 0.425665 | 1782.8 | 126.8% | 117.0% | 52.9% | 0.0% |
| scifact_mistral | answer_aware_fallback | Safe Adaptive Context | compact_then_full_fallback | 0.560244 | 0.541667 | 0.252913 | 0.426009 | 985.01 | 124.3% | 117.1% | 73.9% | 5.0% |
| hotpotqa_llama70b | fixed_10_full | Fixed Full Context | full | 0.74224 | 0.812429 | 0.764539 | 0.768282 | 1536.68 | 100.0% | 100.0% | 0.0% | 0.0% |
| hotpotqa_llama70b | fixed_10_evidence_ngram_neighbors | Compressed Fixed Full Context | evidence_ngram_neighbors | 0.74224 | 0.812429 | 0.752039 | 0.755009 | 1430.34 | 98.4% | 98.3% | 6.9% | 0.0% |
| hotpotqa_llama70b | heuristic_rules_full | Heuristic Rules | full | 0.691003 | 0.809095 | 0.707539 | 0.710674 | 944.98 | 92.5% | 92.5% | 38.5% | 0.0% |
| hotpotqa_llama70b | heuristic_rules_evidence_ngram_neighbors | Heuristic Rules + Compact Evidence | evidence_ngram_neighbors | 0.691003 | 0.809095 | 0.695706 | 0.700245 | 902.18 | 91.0% | 91.1% | 41.3% | 0.0% |
| hotpotqa_llama70b | answer_aware_fallback | Safe Adaptive Context | compact_then_full_fallback | 0.74224 | 0.812429 | 0.736539 | 0.740117 | 1065.72 | 96.3% | 96.3% | 30.6% | 0.0% |
