| Dataset | Model | Retriever | Baseline | F1 diff (95% CI) | Coverage diff (95% CI) | Semantic diff (95% CI) | Token reduction (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ASQA | Llama-70B | TF-IDF | Heuristic Rules | +0.024 [+0.008, +0.040] | -0.003 [-0.025, +0.017] | +0.001 [-0.011, +0.012] | 12.8% [-3.5, 25.2] |
| BioASQ | Llama-70B | Cross-encoder | Heuristic Rules | +0.002 [-0.014, +0.019] | -0.013 [-0.036, +0.009] | +0.002 [-0.011, +0.015] | 47.7% [40.9, 53.0] |
| BioASQ | Llama-70B | TF-IDF | Heuristic Rules | -0.007 [-0.021, +0.008] | -0.005 [-0.033, +0.022] | -0.001 [-0.014, +0.012] | 14.7% [6.4, 22.4] |
| BioASQ | Mistral | TF-IDF | Heuristic Rules | +0.006 [-0.022, +0.034] | +0.002 [-0.032, +0.038] | +0.013 [-0.014, +0.043] | 14.5% [6.2, 22.1] |
| HotpotQA | Llama-70B | Cross-encoder | Heuristic Rules | -0.016 [-0.051, +0.015] | -0.012 [-0.055, +0.028] | -0.016 [-0.048, +0.009] | -54.0% [-64.8, -44.1] |
| HotpotQA | Llama-70B | TF-IDF | Heuristic Rules | +0.029 [-0.003, +0.068] | +0.037 [+0.007, +0.073] | +0.029 [-0.001, +0.066] | -27.9% [-38.4, -18.0] |
| HotpotQA | Mistral | TF-IDF | Heuristic Rules | +0.058 [+0.006, +0.113] | +0.040 [-0.020, +0.105] | +0.057 [+0.006, +0.113] | -31.8% [-43.1, -21.0] |
| SciFact | Llama-70B | Cross-encoder | Heuristic Rules | -0.002 [-0.012, +0.009] | -0.008 [-0.029, +0.013] | +0.006 [-0.005, +0.017] | 44.3% [33.2, 52.1] |
| SciFact | Llama-70B | TF-IDF | Heuristic Rules | -0.001 [-0.012, +0.010] | -0.008 [-0.029, +0.014] | -0.006 [-0.019, +0.007] | 57.1% [49.8, 62.7] |
| SciFact | Mistral | TF-IDF | Heuristic Rules | -0.012 [-0.032, +0.009] | -0.022 [-0.069, +0.024] | -0.011 [-0.037, +0.014] | 60.6% [57.8, 63.3] |
