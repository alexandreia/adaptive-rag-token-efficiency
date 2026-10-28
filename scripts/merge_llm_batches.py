#!/usr/bin/env python3
"""Merge batched LLM experiment outputs into one final table.

Each batch directory must contain llm_answers_by_query.csv from
scripts/run_experiment.py. The script concatenates the detailed rows,
recomputes method summaries, and writes final_table.csv/final_table.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment import FINAL_TABLE_ROWS, build_final_table, write_csv, write_markdown


NUMERIC_FIELDS = [
    "docs_used",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "generation_time_ms",
    "answer_f1",
    "answer_coverage",
    "semantic_similarity",
    "ndcg_at_10",
    "mrr_at_10",
    "first_pass_tokens",
    "fallback_tokens",
]


def read_rows(batch_dirs: list[Path]) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for batch_dir in batch_dirs:
        path = batch_dir / "llm_answers_by_query.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = (row["mode"], row["query_id"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else 0.0


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def token_source_summary(rows: list[dict[str, str]]) -> str:
    return ",".join(sorted({row.get("token_source", "") for row in rows if row.get("token_source", "")}))


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    mode_order = []
    for row in rows:
        mode = row["mode"]
        if mode not in grouped:
            mode_order.append(mode)
        grouped[mode].append(row)

    fixed_10_rows = grouped.get("fixed_10_full", [])
    fixed_10_tokens = average([as_float(row, "total_tokens") for row in fixed_10_rows])

    summary = []
    for mode in mode_order:
        selected = grouped[mode]
        total_tokens = average([as_float(row, "total_tokens") for row in selected])
        token_reduction = 1 - (total_tokens / fixed_10_tokens) if fixed_10_tokens else 0.0
        summary.append(
            {
                "method_name": selected[0].get("method_name", FINAL_TABLE_ROWS.get(mode, mode)),
                "mode": mode,
                "docs_used": round(average([as_float(row, "docs_used") for row in selected]), 6),
                "prompt_tokens": round(average([as_float(row, "prompt_tokens") for row in selected]), 6),
                "completion_tokens": round(average([as_float(row, "completion_tokens") for row in selected]), 6),
                "total_tokens": round(total_tokens, 6),
                "token_reduction_vs_fixed_10": round(token_reduction, 6),
                "token_source": token_source_summary(selected),
                "generation_time_ms": round(average([as_float(row, "generation_time_ms") for row in selected]), 6),
                "fallback_rate": round(average([1.0 if as_bool(row.get("fallback_used", "")) else 0.0 for row in selected]), 6),
                "first_pass_tokens": round(average([as_float(row, "first_pass_tokens") for row in selected]), 6),
                "fallback_tokens": round(average([as_float(row, "fallback_tokens") for row in selected]), 6),
                "answer_f1": round(average([as_float(row, "answer_f1") for row in selected]), 6),
                "answer_coverage": round(average([as_float(row, "answer_coverage") for row in selected]), 6),
                "semantic_similarity": round(average([as_float(row, "semantic_similarity") for row in selected]), 6),
                "ndcg_at_10": round(average([as_float(row, "ndcg_at_10") for row in selected]), 6),
                "mrr_at_10": round(average([as_float(row, "mrr_at_10") for row in selected]), 6),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge batched LLM experiment outputs.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("batch_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = read_rows(args.batch_dirs)
    summary = summarize_rows(rows)
    final_rows = build_final_table(summary, args.dataset_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "llm_answers_by_query.csv", rows)
    write_csv(args.output_dir / "llm_summary.csv", summary)
    write_csv(args.output_dir / "final_table.csv", final_rows)
    write_markdown(args.output_dir / "final_table.md", final_rows)

    print(f"Merged {len(rows)} detailed rows from {len(args.batch_dirs)} batches.")
    print(f"Wrote merged outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
