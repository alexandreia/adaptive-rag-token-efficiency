#!/usr/bin/env python3
"""
Build the human annotation workform.

Samples 15 answers per dataset (8 best-scored + 7 worst-scored) from the
Main Adaptive Method (answer_aware_fallback) only, pooled across all available
models (Llama-70B, Mistral), from three datasets: SciFact, HotpotQA, BioASQ.

Total: 45 items.

Outputs (written to --out-dir, default annotation_workform/final)
------
  annotation_items.csv            all 45 items
  annotation_items_scifact.csv    SciFact subset (15 items)
  annotation_items_hotpotqa.csv   HotpotQA subset (15 items)
  annotation_items_bioasq.csv     BioASQ subset (15 items)
  source_files.csv                input file paths used
  annotation_backup.xlsx          Excel backup with per-dataset sheets
  README.md

Also writes:
  annotation_workform/data.js     browser form data
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

METHOD = "answer_aware_fallback"
METHOD_DISPLAY = "Main Adaptive Method"

MODEL_CONFIGS = {
    "llama70b": {"display": "Llama-70B"},
    "mistral": {"display": "Mistral"},
}

DATASET_DISPLAY = {
    "scifact": "SciFact",
    "hotpotqa": "HotpotQA",
    "bioasq": "BioASQ",
}

# 15 items per dataset: 8 best-scored + 7 worst-scored, pooled across models.
DATASET_CONFIGS = {
    "scifact": {
        "best_items": 8,
        "worst_items": 7,
        "documents_candidates": [
            "data/scifact/documents.jsonl",
        ],
        "queries_candidates": [
            "data/scifact/queries_150_seed0_llm_gold_v2.jsonl",
            "data/scifact/queries_150_seed0_llm_gold.jsonl",
            "data/scifact/queries_150_seed0.jsonl",
            "data/scifact/queries_all.jsonl",
        ],
        "answers_by_model": {
            "llama70b": [
                "saved_results/scifact_llama70b_final_eval100/llm_answers_by_query.csv",
                "outputs/scifact_llama70b_merged_eval100/llm_answers_by_query.csv",
            ],
            "mistral": [
                "saved_results/scifact_mistral_final_eval100/llm_answers_by_query.csv",
            ],
        },
    },
    "hotpotqa": {
        "best_items": 8,
        "worst_items": 7,
        "documents_candidates": [
            "data/hotpotqa/documents.jsonl",
            "data/hotpotqa_classmate/documents.jsonl",
            "data/hotpotqa_final/documents.jsonl",
        ],
        "queries_candidates": [
            "data/hotpotqa/queries.jsonl",
            "data/hotpotqa_classmate/queries_150.jsonl",
            "data/hotpotqa_final/queries_150.jsonl",
        ],
        "answers_by_model": {
            "llama70b": [
                "saved_results/hotpotqa_llama70b_final_eval100/llm_answers_by_query.csv",
                "saved_results/hotpotqa_llama70b_final_eval/llm_answers_by_query.csv",
                "saved_results/hotpotqa_llama70b_final/llm_answers_by_query.csv",
            ],
            "mistral": [
                "saved_results/hotpotqa_mistral_final_eval100/llm_answers_by_query.csv",
            ],
        },
    },
    "bioasq": {
        "best_items": 8,
        "worst_items": 7,
        "documents_candidates": [
            "data/bioasq_candidate/documents.jsonl",
        ],
        "queries_candidates": [
            "data/bioasq_candidate/queries.jsonl",
        ],
        "answers_by_model": {
            "llama70b": [
                "saved_results/bioasq_llama70b_final_eval100/llm_answers_by_query.csv",
                "outputs/bioasq_llama70b_80_eval100_merged/llm_answers_by_query.csv",
            ],
            "mistral": [
                "saved_results/bioasq_mistral_final_eval100/llm_answers_by_query.csv",
            ],
        },
    },
}

RATING_LABELS = [
    "CORRECT",
    "PARTIALLY_CORRECT",
    "INCORRECT",
    "NOT_ENOUGH_INFO",
]

# Columns to include in the Excel backup (evidence is truncated to fit cells).
EXCEL_COLUMNS = [
    "annotation_id",
    "dataset_name",
    "model_name",
    "system",
    "selection_reason",
    "query_text",
    "reference_answer",
    "model_answer",
    "evidence",
    "answer_f1",
    "semantic_similarity",
    "rating",
    "notes",
]

MAX_EVIDENCE_CHARS = 3000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def find_existing_path(candidates: list[str], label: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
    )


def read_queries(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def read_documents(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row["doc_id"])] = clean_text(row.get("text", ""))
    return rows


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_selected_doc_ids(value) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def build_selected_document_text(documents: dict[str, str], doc_ids: list[str]) -> str:
    blocks = []
    for i, doc_id in enumerate(doc_ids, start=1):
        text = documents.get(doc_id, "")
        if not text:
            text = "[document text not found in local dataset file]"
        blocks.append(f"[{i}] {doc_id}: {text}")
    return "\n\n".join(blocks)


def numeric_value(row, column: str) -> float:
    try:
        value = row.get(column, 0.0)
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def answer_quality_score(row) -> float:
    """Composite score used only for selecting best/worst examples."""
    return (
        0.50 * numeric_value(row, "answer_f1")
        + 0.30 * numeric_value(row, "semantic_similarity")
        + 0.20 * numeric_value(row, "answer_coverage")
    )


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def sample_dataset_rows(
    answers_by_model: dict[str, pd.DataFrame],
    best_count: int,
    worst_count: int,
    seed: int,
) -> pd.DataFrame:
    """Pool adaptive-method answers from all available models, score, return best+worst rows."""
    frames = []
    for model_family, df in answers_by_model.items():
        tagged = df[df["mode"] == METHOD].copy()
        tagged["_model_family"] = model_family
        tagged["_model_name"] = MODEL_CONFIGS[model_family]["display"]
        frames.append(tagged)

    candidates = pd.concat(frames, ignore_index=True)
    n_needed = best_count + worst_count
    if len(candidates) < n_needed:
        raise RuntimeError(
            f"Only {len(candidates)} candidate answer rows; need {n_needed}"
        )

    candidates["_quality_score"] = candidates.apply(answer_quality_score, axis=1)
    candidates["_pair_id"] = (
        candidates["query_id"].astype(str) + "||" + candidates["_model_family"].astype(str)
    )
    used_pairs: set[str] = set()

    def take_rows(frame: pd.DataFrame, count: int, reason: str) -> list[dict]:
        rows: list[dict] = []
        for _, row in frame.iterrows():
            pair = str(row["_pair_id"])
            if pair in used_pairs:
                continue
            item = row.drop(labels=["_quality_score", "_pair_id"]).to_dict()
            item["_selection_reason"] = reason
            item["_selection_score"] = round(float(row["_quality_score"]), 6)
            rows.append(item)
            used_pairs.add(pair)
            if len(rows) == count:
                break
        return rows

    selected: list[dict] = []
    selected.extend(
        take_rows(
            candidates.sort_values(
                ["_quality_score", "query_id", "mode"],
                ascending=[False, True, True],
            ),
            best_count,
            "best_metric_score",
        )
    )
    selected.extend(
        take_rows(
            candidates.sort_values(
                ["_quality_score", "query_id", "mode"],
                ascending=[True, True, True],
            ),
            worst_count,
            "worst_metric_score",
        )
    )

    result = pd.DataFrame(selected)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-dataset build
# --------------------------------------------------------------------------- #

def build_dataset_rows(
    dataset: str,
    config: dict,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    documents_path = find_existing_path(config["documents_candidates"], f"{dataset} documents")
    queries_path = find_existing_path(config["queries_candidates"], f"{dataset} queries")
    print(f"\n[{dataset}]")
    print(f"  Documents : {documents_path}")
    print(f"  Queries   : {queries_path}")

    documents = read_documents(documents_path)
    queries = read_queries(queries_path)

    answers_by_model: dict[str, pd.DataFrame] = {}
    answer_paths: dict[str, str] = {}
    for model_family, candidates in config["answers_by_model"].items():
        try:
            path = find_existing_path(candidates, f"{dataset} {model_family} answers")
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc} — skipping {model_family}")
            continue
        print(f"  Answers ({MODEL_CONFIGS[model_family]['display']}): {path}")
        df = pd.read_csv(path)
        missing = {"query_id", "mode", "answer"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df["query_id"] = df["query_id"].astype(str)
        df["mode"] = df["mode"].astype(str)
        answers_by_model[model_family] = df
        answer_paths[model_family] = str(path)

    if not answers_by_model:
        raise RuntimeError(f"No answer files found for dataset '{dataset}'")

    sampled = sample_dataset_rows(
        answers_by_model,
        best_count=config["best_items"],
        worst_count=config["worst_items"],
        seed=seed,
    )
    expected = config["best_items"] + config["worst_items"]
    if len(sampled) != expected:
        raise RuntimeError(f"Sampled {len(sampled)} rows for {dataset}; expected {expected}")
    print(f"  Sampled   : {len(sampled)} items "
          f"({config['best_items']} best + {config['worst_items']} worst)")

    annotation_rows: list[dict] = []

    for local_idx, ans in enumerate(sampled.to_dict("records"), start=1):
        qid = str(ans["query_id"])
        qrow = queries.get(qid, {})

        if qrow:
            query_text = clean_text(qrow.get("text", ""))
            reference_answer = clean_text(qrow.get("reference_answer", ""))
            relevant_doc_ids = qrow.get("relevant_doc_ids", [])
        else:
            query_text = (
                f"[Missing source metadata for query {qid}. "
                "The saved result was produced with a different prepared dataset file.]"
            )
            reference_answer = (
                "[Missing reference answer — restore the matching prepared dataset before annotation.]"
            )
            relevant_doc_ids = []

        model_family = str(ans.get("_model_family", ""))
        model_name = str(ans.get("_model_name", ""))
        annotation_id = f"{dataset}_{local_idx:03d}"
        selected_doc_ids = parse_selected_doc_ids(ans.get("selected_doc_ids", ""))

        annotation_rows.append(
            {
                "annotation_id": annotation_id,
                "candidate_id": annotation_id,
                "dataset": dataset,
                "dataset_name": DATASET_DISPLAY.get(dataset, dataset),
                "model_family": model_family,
                "model_name": model_name,
                "method": METHOD_DISPLAY,
                "query_id": qid,
                "selection_reason": ans.get("_selection_reason", f"{dataset}_sample"),
                "selection_score": ans.get("_selection_score", ""),
                "query_text": query_text,
                "reference_answer": reference_answer,
                "model_answer": clean_text(ans.get("answer", "")),
                "selected_doc_ids": json.dumps(selected_doc_ids, ensure_ascii=False),
                "selected_document_text": build_selected_document_text(documents, selected_doc_ids),
                "relevant_doc_ids": json.dumps(relevant_doc_ids, ensure_ascii=False),
                "answer_f1": ans.get("answer_f1", ""),
                "answer_coverage": ans.get("answer_coverage", ""),
                "semantic_similarity": ans.get("semantic_similarity", ""),
                "ndcg_at_10": ans.get("ndcg_at_10", ""),
                "mrr_at_10": ans.get("mrr_at_10", ""),
                "docs_used": ans.get("docs_used", ""),
                "total_tokens": ans.get("total_tokens", ""),
                "fallback_used": ans.get("fallback_used", ""),
                "fallback_reason": clean_text(ans.get("fallback_reason", "")),
                "rating": "",
                "notes": "",
            }
        )

    metadata = {
        "dataset": dataset,
        "documents_path": str(documents_path),
        "queries_path": str(queries_path),
        "answers_paths": json.dumps(answer_paths, sort_keys=True),
        "n_best": config["best_items"],
        "n_worst": config["worst_items"],
        "total_items": len(annotation_rows),
    }

    return annotation_rows, metadata


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def write_browser_data(path: Path, rows: list[dict]) -> None:
    """Write data.js consumed by the browser annotation form."""
    browser_rows = [
        {
            "candidate_id": r["candidate_id"],
            "annotation_id": r["annotation_id"],
            "dataset": r["dataset"],
            "dataset_name": r["dataset_name"],
            "model_family": r["model_family"],
            "model_name": r["model_name"],
            "method": r["method"],
            "selection_reason": r["selection_reason"],
            "query_id": r["query_id"],
            "selection_score": r["selection_score"],
            "query_text": r["query_text"],
            "reference_answer": r["reference_answer"],
            "model_answer": r["model_answer"],
            "selected_doc_ids": r["selected_doc_ids"],
            "relevant_doc_ids": r["relevant_doc_ids"],
            "selected_document_text": r["selected_document_text"],
            "answer_f1": r["answer_f1"],
            "answer_coverage": r["answer_coverage"],
            "semantic_similarity": r["semantic_similarity"],
            "ndcg_at_10": r["ndcg_at_10"],
            "mrr_at_10": r["mrr_at_10"],
            "docs_used": r["docs_used"],
            "total_tokens": r["total_tokens"],
            "fallback_used": r["fallback_used"],
            "fallback_reason": r["fallback_reason"],
        }
        for r in rows
    ]
    payload = json.dumps(browser_rows, ensure_ascii=False, indent=2)
    path.write_text(f"window.ANNOTATION_SAMPLES = {payload};\n", encoding="utf-8")


def _col_letter(col_index: int) -> str:
    """Convert 1-based column index to Excel letter (A, B, ..., Z, AA, ...)."""
    result = ""
    while col_index > 0:
        col_index, remainder = divmod(col_index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_excel_backup(path: Path, annotation_df: pd.DataFrame) -> None:
    """Write an Excel workbook with one sheet per dataset plus an All sheet."""
    wb = Workbook()
    wb.remove(wb.active)

    rating_formula = '"' + ",".join(RATING_LABELS) + '"'
    sheets = [
        ("All Datasets", annotation_df),
    ] + [
        (DATASET_DISPLAY.get(ds, ds), annotation_df[annotation_df["dataset"] == ds])
        for ds in DATASET_CONFIGS
    ]

    for sheet_name, df in sheets:
        ws = wb.create_sheet(sheet_name)

        edf = df.copy()
        edf["evidence"] = edf["selected_document_text"].apply(
            lambda x: (str(x)[:MAX_EVIDENCE_CHARS] + " [truncated…]")
            if len(str(x)) > MAX_EVIDENCE_CHARS
            else str(x)
        )
        edf["rating"] = ""
        edf["notes"] = ""

        present_cols = [c for c in EXCEL_COLUMNS if c in edf.columns]
        ws.append(present_cols)

        for _, row in edf.iterrows():
            ws.append(
                [
                    str(row.get(c, ""))[:MAX_EVIDENCE_CHARS] if c not in ("rating", "notes") else ""
                    for c in present_cols
                ]
            )

        if "rating" in present_cols:
            rating_col = _col_letter(present_cols.index("rating") + 1)
            dv = DataValidation(
                type="list",
                formula1=rating_formula,
                allow_blank=True,
                showDropDown=False,
            )
            dv.sqref = f"{rating_col}2:{rating_col}{len(edf) + 1}"
            ws.add_data_validation(dv)

    wb.save(path)


def write_readme(path: Path, annotation_df: pd.DataFrame, out_dir: Path) -> None:
    total = len(annotation_df)
    dataset_counts = annotation_df["dataset"].value_counts()
    model_counts = annotation_df["model_family"].value_counts()

    models_str = ", ".join(
        f"{MODEL_CONFIGS[m]['display']} ({model_counts.get(m, 0)})"
        for m in MODEL_CONFIGS
        if m in model_counts.index
    )

    table_rows = "\n".join(
        f"| {DATASET_DISPLAY.get(ds, ds)} | {dataset_counts.get(ds, 0)} |"
        for ds in DATASET_CONFIGS
    )

    readme = f"""# Human Annotation Workform

This folder contains the annotation materials for human evaluation of the **Main Adaptive Method**.

## Design

**{total} items total — 15 per dataset (8 best-scored + 7 worst-scored, pooled across models).**

Method: {METHOD_DISPLAY} (`{METHOD}`)

| Dataset | Items |
|---------|-------|
{table_rows}

Models included: {models_str}

Scores were used only to select examples (best 8, worst 7 per dataset).
**Human ratings must be based on the answer content, not on metric values.**

Quality score (selection only):
`selection_score = 0.50 × answer_f1 + 0.30 × semantic_similarity + 0.20 × answer_coverage`

## Files

| File | Contents |
|------|----------|
| `annotation_items.csv` | All {total} items |
| `annotation_items_scifact.csv` | SciFact subset ({dataset_counts.get('scifact', 0)} items) |
| `annotation_items_hotpotqa.csv` | HotpotQA subset ({dataset_counts.get('hotpotqa', 0)} items) |
| `annotation_items_bioasq.csv` | BioASQ subset ({dataset_counts.get('bioasq', 0)} items) |
| `annotation_backup.xlsx` | Excel backup with per-dataset sheets and rating dropdown |
| `source_files.csv` | Input file paths used |

## Browser annotation workflow

1. Open `annotation_workform/index.html` in your browser.
2. Choose a dataset tab: **All ({total})**, **SciFact (15)**, **HotpotQA (15)**, or **BioASQ (15)**.
3. For each item read the query, reference answer, model answer, and retrieved evidence.
4. Fill **Rating** (required) and **Notes** (optional), then move to the next item.
5. Click **Export CSV** to download all {total} items with your ratings.

**Each annotator must use their own browser profile. Do not share a browser profile between annotators.**
Progress is saved in browser localStorage. Export before switching machines or browsers.

## Excel / CSV backup

Use `annotation_backup.xlsx` or a per-dataset CSV if you prefer a spreadsheet.
Fill only the **rating** column (dropdown provided) and **notes** column.
The evidence column is truncated at {MAX_EVIDENCE_CHARS} characters; use the browser form for full evidence.

## Rating labels

| Label | When to use |
|-------|-------------|
| `CORRECT` | Answer contains the required information and does not contradict the reference |
| `PARTIALLY_CORRECT` | Answer contains some required information but is incomplete, vague, or misses an important condition |
| `INCORRECT` | Answer contradicts the reference or gives a different answer |
| `NOT_ENOUGH_INFO` | Answer cannot be judged confidently from the query, reference, and retrieved evidence |

## Kappa workflow (after both annotators finish)

1. Open `annotation_workform/kappa.html`.
2. Drop Annotator 1's exported CSV into the first upload box.
3. Drop Annotator 2's exported CSV into the second upload box.
4. Item identifier column: `candidate_id` — Rating column: `rating`.
5. Click **Calculate**.
"""
    path.write_text(readme, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="annotation_workform/csv_backups",
        help="Output directory for CSV backups (default: annotation_workform/csv_backups)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_annotation_rows: list[dict] = []
    metadata_rows: list[dict] = []

    for dataset, config in DATASET_CONFIGS.items():
        annotation_rows, metadata = build_dataset_rows(dataset, config, args.seed)
        all_annotation_rows.extend(annotation_rows)
        metadata_rows.append(metadata)

    annotation_df = pd.DataFrame(all_annotation_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    # CSV backups
    annotation_df.to_csv(out_dir / "all_items.csv", index=False)
    for dataset in DATASET_CONFIGS:
        subset = annotation_df[annotation_df["dataset"] == dataset]
        subset.to_csv(out_dir / f"{dataset}.csv", index=False)
    metadata_df.to_csv(out_dir / "source_files.csv", index=False)

    # Excel backup (placed at annotation_workform root for easy access)
    excel_path = Path("annotation_workform/annotation_backup.xlsx")
    write_excel_backup(excel_path, annotation_df)
    print(f"\nExcel backup  : {excel_path}")

    # Browser data
    browser_data_path = Path("annotation_workform/data.js")
    write_browser_data(browser_data_path, all_annotation_rows)

    total = len(annotation_df)
    print(f"CSV backups   : {out_dir}")
    print(f"Browser data  : {browser_data_path}")
    print(f"Total items   : {total}")
    print("\nItems by dataset:")
    print(annotation_df["dataset"].value_counts().to_string())
    print("\nItems by model:")
    print(annotation_df["model_family"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
