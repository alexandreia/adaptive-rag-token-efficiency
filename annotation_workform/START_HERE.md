# RAG Annotation — Getting Started

You have been asked to rate AI-generated answers for quality.

This takes approximately **60–90 minutes** for all 95 items.

---

## What you are doing

You will review 95 answers produced by an AI system given a question and retrieved
scientific/medical documents. For each answer you will choose one of four ratings.
You are judging whether the AI answer correctly reflects the reference answer,
given the question and the retrieved evidence shown.

---

## Step 1 — Open the annotation form

**Double-click `index.html`** to open the form in your browser.

You will see:
- Tabs across the top of the sidebar: **All (95) · SciFact (40) · HotpotQA (40) · BioASQ (40)**
- A list of items on the left; click any item to open it
- The annotation panel on the right

> **Important:** use your own browser, not one shared with another annotator.
> Your progress is saved automatically in the browser — you can close and return later.

---

## Step 2 — Rate each item

For every item you will see:

| Field | What it is |
|-------|-----------|
| **Query / Claim** | The question or scientific claim the AI had to answer |
| **Reference Answer** | The correct answer according to the dataset |
| **Model Answer** | The AI-generated answer you are rating |
| **Retrieved Evidence** | The documents the AI could use (collapse if not needed) |

Pick a **Rating** from the dropdown. Adding a **Note** is optional but helpful when the item is borderline.

### Rating labels

| Rating | Use when… |
|--------|-----------|
| `CORRECT` | The AI answer contains the required information and does not contradict the reference |
| `PARTIALLY_CORRECT` | The AI answer contains some required information but is incomplete, vague, or misses an important condition |
| `INCORRECT` | The AI answer contradicts the reference or gives a different answer |
| `NOT_ENOUGH_INFO` | You cannot judge the answer confidently from the question, reference, and evidence shown |

---

## Step 3 — Export when done

Click **Export CSV** (top-right corner) to download `rag_annotations.csv`.
Send that file to the person who is doing the Annotation results reporting.

---

## Resuming work / importing previous annotations

If the form is updated with new samples and you want to carry over your previous ratings:

1. Click **Import CSV** and pick your previously exported `rag_annotations.csv`.
2. Your old ratings and notes will be restored automatically.
3. Only items that exist in both the file and the form are updated — new samples stay blank.

You can also import a JSON file using **Import JSON** if you exported in that format.

---

## Prefer a spreadsheet? Please consult with the other annotator(s) as well as the project members to ensure the data formatting matches between annotators

Open **`backup_files/annotation_backup.xlsx`**.
Each dataset is a separate sheet (SciFact, HotpotQA, BioASQ).
Fill only the **rating** column (dropdown provided) and the **notes** column.
Return the completed Excel file to the project lead.
!! Note that you will ned to convert the excel to CSV if you or the other Annotators
 want to use the Kappa automated calculator page built wihtin kappa.html

---

## Tips

- The **Filter** dropdown (sidebar) lets you show only unstarted or already-annotated items.
- The **Search** box searches query text and your notes.
- Metric scores (F1, Semantic, nDCG) are shown for reference only — base your rating on the answer content, not on these numbers.
- If you are unsure, use `NOT_ENOUGH_INFO` rather than guessing.
