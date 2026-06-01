# Ablation Results

This folder contains ablation tables used to explain why Safe Adaptive Context
works, beyond the main final result tables.

## Tables

### Table B: Safe Adaptive Context component ablation

Folder:

`table_b_scifact_llama70b_component_ablation_25/`

This run isolates the main components of Safe Adaptive Context on SciFact:

- adaptive document budget
- compact evidence compression
- fallback expansion

The run uses 25 held-out SciFact evaluation queries, the hosted
`meta-llama/Llama-3.3-70B-Instruct` model, provider-reported token usage, and
`max-output-tokens 80`.

### Context compression ablation

Folder:

`context_compression/`

This table compares full context against compact evidence variants across the
completed SciFact and HotpotQA runs. It is a broader compression-focused
ablation, while Table B is the focused component ablation.

