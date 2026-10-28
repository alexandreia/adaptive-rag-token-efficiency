# TACER Targeted Human Evaluation

This workform compares Fixed Top-10, Adaptive-k, Safe Adaptive Context, and TACER on saved Llama-70B outputs.
The sample is targeted, not random: it prioritizes HotpotQA and ASQA queries where TACER differs from Adaptive-k, because those are the cases that test whether task-aware routing helps.

- Queries: 15 HotpotQA + 15 ASQA
- Answers to annotate: 120
- Systems are blinded as System A/B/C/D inside each query group.
- Hidden method labels are kept in the CSV for analysis; remove that column before sending to annotators if needed.

Suggested labels:
- CORRECT: answer contains the required information and does not contradict the reference/evidence.
- PARTIALLY_CORRECT: answer contains some required information but is incomplete, vague, or misses a condition.
- INCORRECT: answer contradicts the reference/evidence or gives a different answer.
- NOT_ENOUGH_INFO: annotator cannot confidently judge from the query, reference, and evidence shown.
