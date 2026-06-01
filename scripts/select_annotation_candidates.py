#!/usr/bin/env python3
"""
Select low-scoring examples for human annotation.

The script reads an experiment output CSV, joins each row back to the original
dataset query and selected documents, then writes a smaller CSV that is ready to
inspect or paste into an annotation sheet.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl_by_id(path: Path, id_field: str) -> dict[str, dict[str, object]]:
    rows = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row[id_field])] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No rows selected.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_doc_ids(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(doc_id) for doc_id in parsed]


def short_text(text: object, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def sort_key(row: dict[str, str], metric: str) -> tuple[float, str, str]:
    return (as_float(row, metric), row.get("query_id", ""), row.get("mode", ""))


def add_unique(
    selected: list[tuple[str, dict[str, str]]],
    seen: set[tuple[str, str]],
    reason: str,
    rows: list[dict[str, str]],
    limit: int,
) -> None:
    for row in rows:
        key = (row.get("mode", ""), row.get("query_id", ""))
        if key in seen:
            continue
        selected.append((reason, row))
        seen.add(key)
        if len([item for item in selected if item[0] == reason]) >= limit:
            break


def select_rows(rows: list[dict[str, str]], metric: str, per_bucket: int, total_limit: int) -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()

    add_unique(
        selected,
        seen,
        f"lowest_{metric}",
        sorted(rows, key=lambda row: sort_key(row, metric)),
        per_bucket,
    )
    add_unique(
        selected,
        seen,
        "lowest_semantic_similarity",
        sorted(rows, key=lambda row: sort_key(row, "semantic_similarity")),
        per_bucket,
    )

    good_retrieval_bad_answer = [
        row for row in rows if as_float(row, "ndcg_at_10") >= 0.8 and as_float(row, "answer_f1") <= 0.3
    ]
    add_unique(
        selected,
        seen,
        "good_retrieval_bad_answer",
        sorted(good_retrieval_bad_answer, key=lambda row: (as_float(row, "answer_f1"), -as_float(row, "ndcg_at_10"))),
        per_bucket,
    )

    bad_retrieval_bad_answer = [
        row for row in rows if as_float(row, "ndcg_at_10") <= 0.3 and as_float(row, "answer_f1") <= 0.3
    ]
    add_unique(
        selected,
        seen,
        "bad_retrieval_bad_answer",
        sorted(bad_retrieval_bad_answer, key=lambda row: (as_float(row, "ndcg_at_10"), as_float(row, "answer_f1"))),
        per_bucket,
    )

    high_quality_controls = [
        row for row in rows if as_float(row, "answer_f1") >= 0.7 and as_float(row, "ndcg_at_10") >= 0.8
    ]
    add_unique(
        selected,
        seen,
        "high_quality_control",
        sorted(high_quality_controls, key=lambda row: (-as_float(row, "answer_f1"), -as_float(row, "ndcg_at_10"))),
        max(3, per_bucket // 2),
    )

    return selected[:total_limit]


def select_best_worst_rows(
    rows: list[dict[str, str]],
    metric: str,
    best_count: int,
    worst_count: int,
) -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()

    sorted_rows = sorted(rows, key=lambda row: sort_key(row, metric))
    add_unique(selected, seen, f"worst_{metric}", sorted_rows, worst_count)
    add_unique(selected, seen, f"best_{metric}", reversed(sorted_rows), best_count)

    return selected


def build_candidate_rows(
    selected: list[tuple[str, dict[str, str]]],
    queries_by_id: dict[str, dict[str, object]],
    docs_by_id: dict[str, dict[str, object]],
    max_doc_chars: int,
) -> list[dict[str, object]]:
    candidates = []
    for index, (reason, row) in enumerate(selected, start=1):
        query_id = str(row.get("query_id", ""))
        query = queries_by_id.get(query_id, {})
        selected_doc_ids = parse_doc_ids(row.get("selected_doc_ids", ""))
        selected_docs = []
        for rank, doc_id in enumerate(selected_doc_ids, start=1):
            doc = docs_by_id.get(doc_id, {})
            selected_docs.append(f"[{rank}] {doc_id}: {short_text(doc.get('text', ''), max_doc_chars)}")

        candidates.append(
            {
                "candidate_id": f"ANN-{index:03d}",
                "selection_reason": reason,
                "mode": row.get("mode", ""),
                "method_name": row.get("method_name", ""),
                "query_id": query_id,
                "query_text": query.get("text", ""),
                "reference_answer": query.get("reference_answer", ""),
                "relevant_doc_ids": ", ".join(str(doc_id) for doc_id in query.get("relevant_doc_ids", [])),
                "selected_doc_ids": ", ".join(selected_doc_ids),
                "selected_document_text": "\n\n".join(selected_docs),
                "model_answer": row.get("answer", ""),
                "answer_f1": row.get("answer_f1", ""),
                "answer_coverage": row.get("answer_coverage", ""),
                "semantic_similarity": row.get("semantic_similarity", ""),
                "ndcg_at_10": row.get("ndcg_at_10", ""),
                "mrr_at_10": row.get("mrr_at_10", ""),
                "docs_used": row.get("docs_used", ""),
                "total_tokens": row.get("total_tokens", ""),
                "annotator_1_label": "",
                "annotator_1_notes": "",
                "annotator_2_label": "",
                "annotator_2_notes": "",
                "adjudicated_label": "",
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull low-scoring evaluation examples for annotation.")
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=Path("saved_results/scifact_llama70b_final_eval100/llm_answers_by_query.csv"),
        help="Detailed evaluation CSV produced by scripts/run_experiment.py.",
    )
    parser.add_argument("--queries", type=Path, default=Path("data/scifact/queries_150_seed0_llm_gold_v2.jsonl"))
    parser.add_argument("--documents", type=Path, default=Path("data/scifact/documents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/annotation_candidates.csv"))
    parser.add_argument(
        "--mode",
        default="answer_aware_fallback",
        help='Evaluation mode to select from. Use "all" to include every mode.',
    )
    parser.add_argument(
        "--metric",
        default="answer_f1",
        choices=["answer_f1", "answer_coverage", "semantic_similarity", "ndcg_at_10", "mrr_at_10"],
        help="Main metric to sort by.",
    )
    parser.add_argument(
        "--preset",
        choices=["diagnostic", "best_worst"],
        default="diagnostic",
        help="Selection strategy. diagnostic pulls error-analysis buckets; best_worst pulls the best and worst rows by metric.",
    )
    parser.add_argument("--best-count", type=int, default=50, help="Rows to pull from the top of the metric ranking.")
    parser.add_argument("--worst-count", type=int, default=50, help="Rows to pull from the bottom of the metric ranking.")
    parser.add_argument("--per-bucket", type=int, default=10, help="Rows to pull from each diagnostic bucket.")
    parser.add_argument("--total-limit", type=int, default=50, help="Maximum number of output rows.")
    parser.add_argument("--max-doc-chars", type=int, default=900, help="Maximum characters per selected document.")
    args = parser.parse_args()

    rows = read_csv(args.eval_csv)
    if args.mode != "all":
        rows = [row for row in rows if row.get("mode") == args.mode]

    if not rows:
        raise SystemExit(f"No evaluation rows found for mode={args.mode!r} in {args.eval_csv}")

    queries_by_id = read_jsonl_by_id(args.queries, "query_id")
    docs_by_id = read_jsonl_by_id(args.documents, "doc_id")

    if args.preset == "best_worst":
        selected = select_best_worst_rows(rows, args.metric, args.best_count, args.worst_count)
    else:
        selected = select_rows(rows, args.metric, args.per_bucket, args.total_limit)
    candidates = build_candidate_rows(selected, queries_by_id, docs_by_id, args.max_doc_chars)
    write_csv(args.output, candidates)

    print(f"Selected {len(candidates)} annotation candidates.")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
