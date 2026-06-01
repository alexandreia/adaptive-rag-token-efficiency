#!/usr/bin/env python3
"""Build annotation_form/data.js from an annotation candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)


FIELDS = [
    "candidate_id",
    "selection_reason",
    "mode",
    "method_name",
    "query_id",
    "query_text",
    "reference_answer",
    "relevant_doc_ids",
    "selected_doc_ids",
    "selected_document_text",
    "model_answer",
    "answer_f1",
    "answer_coverage",
    "semantic_similarity",
    "ndcg_at_10",
    "mrr_at_10",
    "docs_used",
    "total_tokens",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static data for the annotation web form.")
    parser.add_argument("--input", type=Path, default=Path("outputs/annotation_best_worst_100.csv"))
    parser.add_argument("--output", type=Path, default=Path("annotation_form/data.js"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as file:
        rows = [
            {field: row.get(field, "") for field in FIELDS}
            for row in csv.DictReader(file)
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    args.output.write_text(f"window.ANNOTATION_SAMPLES = {payload};\n", encoding="utf-8")
    print(f"Wrote {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
