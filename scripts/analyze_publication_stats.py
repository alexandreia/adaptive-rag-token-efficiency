#!/usr/bin/env python3
"""Publication-oriented statistics from saved per-query experiment outputs.

This script does not call any model. It reads existing
saved_results/*/llm_answers_by_query.csv files and computes:

- bootstrap confidence intervals for method-level means
- paired Safe Adaptive vs Fixed Top-10 comparisons
- paired Safe Adaptive vs Heuristic Rules comparisons

The goal is to strengthen the current paper claims before running any new
expensive LLM experiments.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


QUALITY_METRICS = ["answer_f1", "answer_coverage", "semantic_similarity"]
EFFICIENCY_METRICS = ["total_tokens"]
ALL_METRICS = QUALITY_METRICS + EFFICIENCY_METRICS

SAFE_MODE = "answer_aware_fallback"
FIXED_MODE = "fixed_10_full"
HEURISTIC_MODE = "heuristic_rules_full"

METHOD_NAMES = {
    "no_retrieval_full": "No Retrieval",
    "fixed_3_full": "Fixed Top-3",
    "fixed_5_full": "Fixed Top-5",
    "fixed_7_full": "Fixed Top-7",
    "fixed_10_full": "Fixed Top-10",
    "heuristic_rules_full": "Heuristic Rules",
    "answer_aware_fallback": "Safe Adaptive Context",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * pct
    low = int(index)
    high = min(low + 1, len(sorted_values) - 1)
    weight = index - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_mean_ci(values: list[float], iterations: int, rng: random.Random) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    estimates = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(mean(sample))
    estimates.sort()
    return mean(values), percentile(estimates, 0.025), percentile(estimates, 0.975)


def infer_run_name(path: Path) -> str:
    return path.parent.name


def infer_dataset(run_name: str) -> str:
    lowered = run_name.lower()
    if "scifact" in lowered:
        return "SciFact"
    if "hotpotqa" in lowered:
        return "HotpotQA"
    if "bioasq" in lowered:
        return "BioASQ"
    if "asqa" in lowered:
        return "ASQA"
    return run_name


def infer_model(run_name: str) -> str:
    lowered = run_name.lower()
    if "mistral" in lowered:
        return "Mistral"
    if "llama70b" in lowered or "llama-70b" in lowered:
        return "Llama-70B"
    return "Unknown"


def infer_retriever(run_name: str) -> str:
    return "Cross-encoder" if "crossenc" in run_name.lower() else "TF-IDF"


def load_runs(saved_results: Path) -> dict[str, list[dict[str, str]]]:
    runs = {}
    for path in sorted(saved_results.glob("*/llm_answers_by_query.csv")):
        rows = read_csv(path)
        if rows:
            runs[infer_run_name(path)] = rows
    return runs


def method_mean_rows(
    runs: dict[str, list[dict[str, str]]],
    iterations: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for run_name, rows in runs.items():
        by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_mode[row["mode"]].append(row)

        for mode, mode_rows in sorted(by_mode.items()):
            for metric in ALL_METRICS:
                values = [v for row in mode_rows if (v := as_float(row, metric)) is not None]
                estimate, low, high = bootstrap_mean_ci(values, iterations, rng)
                output_rows.append(
                    {
                        "run": run_name,
                        "dataset": infer_dataset(run_name),
                        "model": infer_model(run_name),
                        "retriever": infer_retriever(run_name),
                        "method": METHOD_NAMES.get(mode, mode),
                        "mode": mode,
                        "metric": metric,
                        "n": len(values),
                        "mean": round(estimate, 6),
                        "ci_low": round(low, 6),
                        "ci_high": round(high, 6),
                    }
                )
    return output_rows


def paired_rows_for_run(
    run_name: str,
    rows: list[dict[str, str]],
    baseline_mode: str,
    baseline_label: str,
    iterations: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    by_query_mode: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        by_query_mode[(row["query_id"], row["mode"])] = row

    query_ids = sorted(
        query_id
        for query_id, mode in by_query_mode
        if mode == SAFE_MODE and (query_id, baseline_mode) in by_query_mode
    )
    if not query_ids:
        return []

    output_rows: list[dict[str, object]] = []
    for metric in QUALITY_METRICS:
        diffs = []
        for query_id in query_ids:
            safe_value = as_float(by_query_mode[(query_id, SAFE_MODE)], metric)
            base_value = as_float(by_query_mode[(query_id, baseline_mode)], metric)
            if safe_value is not None and base_value is not None:
                diffs.append(safe_value - base_value)
        estimate, low, high = bootstrap_mean_ci(diffs, iterations, rng)
        output_rows.append(
            {
                "run": run_name,
                "dataset": infer_dataset(run_name),
                "model": infer_model(run_name),
                "retriever": infer_retriever(run_name),
                "comparison": f"Safe Adaptive - {baseline_label}",
                "metric": metric,
                "n": len(diffs),
                "mean_difference": round(estimate, 6),
                "ci_low": round(low, 6),
                "ci_high": round(high, 6),
                "interpretation": "positive favors Safe Adaptive",
            }
        )

    token_reductions = []
    token_diffs = []
    for query_id in query_ids:
        safe_tokens = as_float(by_query_mode[(query_id, SAFE_MODE)], "total_tokens")
        base_tokens = as_float(by_query_mode[(query_id, baseline_mode)], "total_tokens")
        if safe_tokens is not None and base_tokens is not None and base_tokens:
            token_diffs.append(safe_tokens - base_tokens)
            token_reductions.append(1 - (safe_tokens / base_tokens))

    estimate, low, high = bootstrap_mean_ci(token_diffs, iterations, rng)
    output_rows.append(
        {
            "run": run_name,
            "dataset": infer_dataset(run_name),
            "model": infer_model(run_name),
            "retriever": infer_retriever(run_name),
            "comparison": f"Safe Adaptive - {baseline_label}",
            "metric": "total_tokens",
            "n": len(token_diffs),
            "mean_difference": round(estimate, 6),
            "ci_low": round(low, 6),
            "ci_high": round(high, 6),
            "interpretation": "negative means Safe Adaptive uses fewer tokens",
        }
    )

    estimate, low, high = bootstrap_mean_ci(token_reductions, iterations, rng)
    output_rows.append(
        {
            "run": run_name,
            "dataset": infer_dataset(run_name),
            "model": infer_model(run_name),
            "retriever": infer_retriever(run_name),
            "comparison": f"Safe Adaptive vs {baseline_label}",
            "metric": "token_reduction",
            "n": len(token_reductions),
            "mean_difference": round(estimate, 6),
            "ci_low": round(low, 6),
            "ci_high": round(high, 6),
            "interpretation": "positive means Safe Adaptive saves tokens",
        }
    )
    return output_rows


def paired_comparison_rows(
    runs: dict[str, list[dict[str, str]]],
    iterations: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for run_name, rows in runs.items():
        output_rows.extend(
            paired_rows_for_run(run_name, rows, FIXED_MODE, "Fixed Top-10", iterations, rng)
        )
        output_rows.extend(
            paired_rows_for_run(run_name, rows, HEURISTIC_MODE, "Heuristic Rules", iterations, rng)
        )
    return output_rows


def format_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def format_ci(row: dict[str, object], percent: bool = False) -> str:
    mean_value = float(row["mean_difference"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if percent:
        return f"{mean_value * 100:.1f}% [{low * 100:.1f}, {high * 100:.1f}]"
    return f"{format_signed(mean_value)} [{format_signed(low)}, {format_signed(high)}]"


def compact_publication_rows(
    paired_rows: list[dict[str, object]],
    baseline_label: str,
) -> list[dict[str, object]]:
    by_run: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in paired_rows:
        comparison = str(row["comparison"])
        metric = str(row["metric"])
        if baseline_label == "Fixed Top-10":
            wanted = comparison in {
                "Safe Adaptive - Fixed Top-10",
                "Safe Adaptive vs Fixed Top-10",
            }
        else:
            wanted = comparison in {
                "Safe Adaptive - Heuristic Rules",
                "Safe Adaptive vs Heuristic Rules",
            }
        if wanted:
            by_run[str(row["run"])][metric] = row

    output_rows = []
    for run_name in sorted(by_run):
        metrics = by_run[run_name]
        required = {"answer_f1", "answer_coverage", "semantic_similarity", "token_reduction"}
        if not required.issubset(metrics):
            continue
        output_rows.append(
            {
                "Dataset": infer_dataset(run_name),
                "Model": infer_model(run_name),
                "Retriever": infer_retriever(run_name),
                "Baseline": baseline_label,
                "F1 diff (95% CI)": format_ci(metrics["answer_f1"]),
                "Coverage diff (95% CI)": format_ci(metrics["answer_coverage"]),
                "Semantic diff (95% CI)": format_ci(metrics["semantic_similarity"]),
                "Token reduction (95% CI)": format_ci(metrics["token_reduction"], percent=True),
            }
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-results", type=Path, default=Path("saved_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("saved_results/publication_stats"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    runs = load_runs(args.saved_results)
    if not runs:
        raise SystemExit(f"No llm_answers_by_query.csv files found under {args.saved_results}")

    mean_rows = method_mean_rows(runs, args.bootstrap_iterations, rng)
    paired_rows = paired_comparison_rows(runs, args.bootstrap_iterations, rng)

    write_csv(args.output_dir / "method_mean_confidence_intervals.csv", mean_rows)
    write_markdown(args.output_dir / "method_mean_confidence_intervals.md", mean_rows)
    write_csv(args.output_dir / "paired_safe_adaptive_comparisons.csv", paired_rows)
    write_markdown(args.output_dir / "paired_safe_adaptive_comparisons.md", paired_rows)

    fixed_rows = compact_publication_rows(paired_rows, "Fixed Top-10")
    heuristic_rows = compact_publication_rows(paired_rows, "Heuristic Rules")
    all_compact_rows = fixed_rows + heuristic_rows
    write_csv(args.output_dir / "publication_table_safe_vs_fixed.csv", fixed_rows)
    write_markdown(args.output_dir / "publication_table_safe_vs_fixed.md", fixed_rows)
    write_csv(args.output_dir / "publication_table_safe_vs_heuristic.csv", heuristic_rows)
    write_markdown(args.output_dir / "publication_table_safe_vs_heuristic.md", heuristic_rows)
    write_csv(args.output_dir / "publication_table_all_safe_comparisons.csv", all_compact_rows)
    write_markdown(args.output_dir / "publication_table_all_safe_comparisons.md", all_compact_rows)

    print(f"Read {len(runs)} saved runs.")
    print(f"Wrote method CIs: {args.output_dir / 'method_mean_confidence_intervals.csv'}")
    print(f"Wrote paired comparisons: {args.output_dir / 'paired_safe_adaptive_comparisons.csv'}")
    print(f"Wrote compact Fixed Top-10 table: {args.output_dir / 'publication_table_safe_vs_fixed.csv'}")
    print(f"Wrote compact Heuristic Rules table: {args.output_dir / 'publication_table_safe_vs_heuristic.csv'}")
    print(f"Wrote combined compact table: {args.output_dir / 'publication_table_all_safe_comparisons.csv'}")


if __name__ == "__main__":
    main()
