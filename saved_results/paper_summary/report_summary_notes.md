# Report Summary Notes

## Main Pattern

TACER is best framed as the most stable quality-efficiency policy, not as the winner of every cell.
Adaptive-k is a strong score-only baseline, but it can under-select context on tasks where evidence is distributed.
Safe Adaptive Context is the conservative quality-preserving sibling of TACER.

## Key Results To Discuss

- HotpotQA is the clearest failure case for score-only Adaptive-k. Even with dense + cross-encoder retrieval, Adaptive-k retains only 79.1% F1, while TACER retains 97.2%.
- SciFact is the clearest TACER compression case. Under dense + cross-encoder retrieval, TACER retains 102.9% F1 and 100.9% semantic similarity while reducing tokens by 82.1%.
- ASQA shows convergence under strong retrieval. With dense + cross-encoder retrieval, all methods are close, and TACER retains 99.8% F1 with 54.4% token reduction.
- BioASQ shows near-lossless aggressive compression. TACER gives high token savings with small quality loss; Safe Adaptive is the safer quality-preserving point.
- MS MARCO shows that strong reranking plus Fixed Top-3 can be hard to beat, which is an important limitation and makes the paper more credible.

## Suggested Claim

Adaptive-k answers how many documents to keep. TACER chooses a context policy: compress aggressively when evidence is concentrated, and route to safer broader context when the task is likely to need distributed evidence.

## Suggested Limitation

TACER does not dominate every dataset. On short-answer web QA with very strong retrieval, small fixed-k baselines can be as good or better. The contribution is robustness across task types rather than universal superiority.
