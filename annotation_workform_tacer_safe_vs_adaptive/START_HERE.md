# Start Here

Open `index.html` in a browser.

You will annotate 120 blinded answers:

- 15 HotpotQA questions x 4 systems
- 15 ASQA questions x 4 systems
- Systems are shown only as System A, B, C, or D

For each answer, choose:

- `CORRECT`
- `PARTIALLY_CORRECT`
- `INCORRECT`
- `NOT_ENOUGH_INFO`

Use the query, reference answer, model answer, and retrieved evidence shown in the page. The hidden method labels are kept in `annotation_items.csv` for later analysis, but they are not shown in the browser interface.

This evaluation is designed to compare Fixed Top-10, Adaptive-k, Safe Adaptive Context, and TACER on factual answer quality and grounding.
