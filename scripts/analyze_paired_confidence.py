#!/usr/bin/env python3
"""Paired confidence checks for paper-facing RAG results.

The script compares methods query-by-query and reports paired bootstrap
confidence intervals plus simple win/tie/loss counts. It intentionally uses
only the Python standard library so it can run in the project environment
without extra dependencies.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "saved_results" / "paper_summary"
DATASETS = ["scifact", "bioasq", "hotpotqa", "msmarco", "asqa"]
DISPLAY_DATASET = {
    "scifact": "SciFact",
    "bioasq": "BioASQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "asqa": "ASQA",
}
COMPARISONS = [
    ("TACER vs Adaptive-k", "task_aware_coverage_ultra", "adaptive_k_full"),
    ("Safe Adaptive vs Adaptive-k", "answer_aware_fallback", "adaptive_k_full"),
    ("TACER vs Fixed Top-10", "task_aware_coverage_ultra", "fixed_10_full"),
    ("Safe Adaptive vs Fixed Top-10", "answer_aware_fallback", "fixed_10_full"),
]
METRICS = ["answer_f1", "answer_coverage", "semantic_similarity", "total_tokens"]


@dataclass(frozen=True)
class PairedSummary:
    dataset: str
    comparison: str
    n: int
    delta_f1: float
    f1_ci_low: float
    f1_ci_high: float
    f1_wins: int
    f1_ties: int
    f1_losses: int
    delta_coverage: float
    coverage_ci_low: float
    coverage_ci_high: float
    delta_semantic: float
    semantic_ci_low: float
    semantic_ci_high: float
    delta_tokens: float
    tokens_ci_low: float
    tokens_ci_high: float
    token_wins: int
    token_ties: int
    token_losses: int


def read_rows(dataset: str) -> dict[str, dict[str, dict[str, float]]]:
    path = (
        ROOT
        / "saved_results"
        / "final_main"
        / "llama70b_tfidf"
        / dataset
        / "llm_answers_by_query.csv"
    )
    by_mode: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_mode.setdefault(row["mode"], {})[row["query_id"]] = {
                metric: float(row[metric]) for metric in METRICS
            }
    return by_mode


def bootstrap_ci(values: list[float], *, reps: int = 5000, seed: int = 13) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low = means[int(0.025 * reps)]
    high = means[min(reps - 1, int(0.975 * reps))]
    return low, high


def win_tie_loss(values: list[float], *, eps: float = 1e-9, lower_is_better: bool = False) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for value in values:
        score = -value if lower_is_better else value
        if score > eps:
            wins += 1
        elif score < -eps:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses


def summarize(dataset: str, comparison: str, method: str, baseline: str) -> PairedSummary:
    rows = read_rows(dataset)
    method_rows = rows[method]
    baseline_rows = rows[baseline]
    query_ids = sorted(set(method_rows) & set(baseline_rows))
    deltas = {
        metric: [method_rows[q][metric] - baseline_rows[q][metric] for q in query_ids]
        for metric in METRICS
    }
    f1_low, f1_high = bootstrap_ci(deltas["answer_f1"])
    cov_low, cov_high = bootstrap_ci(deltas["answer_coverage"])
    sem_low, sem_high = bootstrap_ci(deltas["semantic_similarity"])
    tok_low, tok_high = bootstrap_ci(deltas["total_tokens"])
    f1_wins, f1_ties, f1_losses = win_tie_loss(deltas["answer_f1"])
    token_wins, token_ties, token_losses = win_tie_loss(
        deltas["total_tokens"], lower_is_better=True
    )
    return PairedSummary(
        dataset=DISPLAY_DATASET[dataset],
        comparison=comparison,
        n=len(query_ids),
        delta_f1=mean(deltas["answer_f1"]),
        f1_ci_low=f1_low,
        f1_ci_high=f1_high,
        f1_wins=f1_wins,
        f1_ties=f1_ties,
        f1_losses=f1_losses,
        delta_coverage=mean(deltas["answer_coverage"]),
        coverage_ci_low=cov_low,
        coverage_ci_high=cov_high,
        delta_semantic=mean(deltas["semantic_similarity"]),
        semantic_ci_low=sem_low,
        semantic_ci_high=sem_high,
        delta_tokens=mean(deltas["total_tokens"]),
        tokens_ci_low=tok_low,
        tokens_ci_high=tok_high,
        token_wins=token_wins,
        token_ties=token_ties,
        token_losses=token_losses,
    )


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_ci(value: float, low: float, high: float) -> str:
    return f"{value:+.3f} [{low:+.3f}, {high:+.3f}]"


def fmt_tokens(value: float, low: float, high: float) -> str:
    return f"{value:+.1f} [{low:+.1f}, {high:+.1f}]"


def write_csv(rows: list[PairedSummary]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "paired_confidence_main_tfidf.csv"
    fieldnames = list(PairedSummary.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(rows: list[PairedSummary]) -> None:
    path = OUT_DIR / "paired_confidence_main_tfidf.md"
    headers = [
        "dataset",
        "comparison",
        "n",
        "delta_f1_ci",
        "f1_w/t/l",
        "delta_coverage_ci",
        "delta_semantic_ci",
        "delta_tokens_ci",
        "token_w/t/l",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.dataset,
                    row.comparison,
                    str(row.n),
                    fmt_ci(row.delta_f1, row.f1_ci_low, row.f1_ci_high),
                    f"{row.f1_wins}/{row.f1_ties}/{row.f1_losses}",
                    fmt_ci(row.delta_coverage, row.coverage_ci_low, row.coverage_ci_high),
                    fmt_ci(row.delta_semantic, row.semantic_ci_low, row.semantic_ci_high),
                    fmt_tokens(row.delta_tokens, row.tokens_ci_low, row.tokens_ci_high),
                    f"{row.token_wins}/{row.token_ties}/{row.token_losses}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_latex_tacer_vs_adaptive(rows: list[PairedSummary]) -> None:
    path = OUT_DIR / "paired_confidence_tacer_vs_adaptive_latex.tex"
    selected = [row for row in rows if row.comparison == "TACER vs Adaptive-k"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Paired TACER--Adaptive-$k$ confidence check in the main Llama-70B TF-IDF setting. F1 confidence intervals are paired bootstrap 95\\% intervals over queries. Token deltas are TACER minus Adaptive-$k$, so negative values mean TACER uses fewer tokens.}",
        "\\label{tab:paired_confidence}",
        "\\resizebox{0.95\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & $\\Delta$F1 [95\\% CI] & F1 W/T/L & $\\Delta$Tokens [95\\% CI] & Token W/T/L \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row.dataset} & "
            f"{fmt_ci(row.delta_f1, row.f1_ci_low, row.f1_ci_high)} & "
            f"{row.f1_wins}/{row.f1_ties}/{row.f1_losses} & "
            f"{fmt_tokens(row.delta_tokens, row.tokens_ci_low, row.tokens_ci_high)} & "
            f"{row.token_wins}/{row.token_ties}/{row.token_losses} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    rows = [
        summarize(dataset, comparison, method, baseline)
        for dataset in DATASETS
        for comparison, method, baseline in COMPARISONS
    ]
    write_csv(rows)
    write_markdown(rows)
    write_latex_tacer_vs_adaptive(rows)
    print(f"Wrote paired confidence artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
