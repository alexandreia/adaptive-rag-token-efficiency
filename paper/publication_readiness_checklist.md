# Publication Readiness Checklist

This checklist tracks what is already strong in the expanded TACER paper and what still needs polishing before a real workshop/conference submission.

## Already Strong

- Clear method evolution: Safe Adaptive Context -> TACER.
- Five-task evaluation: SciFact, BioASQ, HotpotQA, MS MARCO, ASQA.
- Two model scales: Llama-3.3-70B and Mistral.
- Retriever robustness: TF-IDF, TF-IDF + cross-encoder, Dense + cross-encoder.
- Strong baseline comparison: fixed top-k, heuristic rules, Adaptive-k, Safe Adaptive Context.
- Main argument is credible: Adaptive-k is efficient but can under-select on distributed-evidence tasks.
- TACER has a defensible role: stable quality-efficiency routing, not maximum compression in every case.
- Human annotation study supports the interpretation of automatic metrics.

## Needs Before Submission

- Convert manual references into BibTeX and use ACL citation style properly.
- Add one Pareto figure from `saved_results/paper_summary/`.
- Add 2-3 qualitative examples:
  - HotpotQA case where Adaptive-k under-selects evidence.
  - SciFact or BioASQ case where compact evidence removes distractors.
  - ASQA case where long-form context needs broader evidence.
- Add a small direct human evaluation comparing TACER vs Adaptive-k if time allows.
- Decide whether TACER or Safe Adaptive Context is the headline result in each dataset.
- Tighten the Method section after inspecting the exact implementation one more time.
- Replace any broad claims like "task-aware" with exact signals used by the controller.
- Confirm all final tables match the exact files under `saved_results/paper_summary/`.
- Compile in Overleaf with `acl.sty` uploaded.
- Check page length after compilation and move overflow tables to appendix.

## Framing To Preserve

The paper should not claim that TACER wins every metric. The stronger and more publishable claim is:

> Retrieval-score-only context adaptation is efficient but brittle. TACER uses task-aware evidence routing to provide a more stable quality-efficiency trade-off across datasets, models, and retrievers.

