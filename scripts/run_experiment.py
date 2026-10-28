#!/usr/bin/env python3
"""
Main experiment runner.

This file is the one we run from the terminal.

It does not contain the full model logic. The heavy work is inside:

    src/adaptive_retrieval/llm_budget.py

This file only does the experiment steps:

1. Load documents and queries.
2. Choose which methods we want to compare.
3. Configure the LLM.
4. Run the experiment.
5. Save the final tables.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path


# Let Python find our project code inside src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_retrieval.data import load_documents, load_queries
from adaptive_retrieval.llm_budget import (
    LLMConfig,
    load_precomputed_rankings,
    run_llm_budget_experiment,
    write_llm_outputs,
)


# These are the systems we actually run.
#
# We removed Basic Adaptive from the final experiment because it was mainly an
# early prototype. The useful comparisons now are:
# - fixed baselines
# - heuristic baseline
# - Adaptive-k baseline
# - Safe Adaptive Context, which now expands progressively inside one model
METHODS_TO_RUN = [
    "no_retrieval",
    "fixed_3",
    "fixed_5",
    "fixed_7",
    "fixed_10",
    "heuristic_rules",
    "adaptive_k",
    "adaptive_k_paper",
    "adaptive_k_official",
    "llmlingua2_top10",
    "llmlingua2_adaptive_k_official",
    "flare_lite",
    "guarded_adaptive_k",
    "guarded_predicate_compact",
    "discourse_preserving_compact",
    "answer_aware_fallback",
    "safe_adaptive_v2",
    "coverage_guided_adaptive",
    "coverage_guided_ultra",
    "task_aware_coverage_ultra",
    "routed_predicate_adaptive",
    "routed_guarded_adaptive",
    "routed_safe_guarded_adaptive",
    "merged_evidence_brief",
    "hybrid_safe_adaptive",
]


# These are the rows we want in the clean final table.
# The detailed CSV files still keep more information.
FINAL_TABLE_ROWS = {
    "no_retrieval_full": "No Retrieval",
    "fixed_3_full": "Fixed Top-3",
    "fixed_5_full": "Fixed Top-5",
    "fixed_7_full": "Fixed Top-7",
    "fixed_10_full": "Fixed Top-10",
    "heuristic_rules_full": "Heuristic Rules",
    "adaptive_k_full": "Adaptive-k",
    "adaptive_k_paper_full": "Adaptive-k (paper)",
    "adaptive_k_official_full": "Adaptive-k (paper)",
    "llmlingua2_top10": "Fixed Top-10 + LLMLingua-2",
    "llmlingua2_adaptive_k_official": "Adaptive-k (paper) + LLMLingua-2",
    "flare_lite": "FLARE-lite",
    "guarded_adaptive_k_full": "Guarded Adaptive-k",
    "guarded_adaptive_k_evidence_ngram_neighbors": "Guarded Adaptive-k + Compact Evidence",
    "guarded_predicate_compact": "Guarded Predicate Compact",
    "discourse_preserving_compact": "Discourse-Preserving Compact",
    "answer_aware_fallback": "Safe Adaptive Context",
    "safe_adaptive_v2": "Safe Adaptive Context v2",
    "coverage_guided_adaptive": "Coverage-Guided Safe Adaptive",
    "coverage_guided_ultra": "Coverage-Guided Ultra",
    "task_aware_coverage_ultra": "TACER",
    "routed_predicate_adaptive": "Routed Predicate Adaptive",
    "routed_guarded_adaptive": "Routed Guarded Adaptive Context",
    "routed_safe_guarded_adaptive": "Routed Safe Guarded Adaptive Context",
    "merged_evidence_brief": "Merged Evidence Brief",
    "hybrid_safe_adaptive": "Hybrid Safe Adaptive",
}


def as_float(row, key):
    # CSV/table values can be strings, so this converts them safely.
    return float(row[key])


def percent(value):
    # Format decimals as percentages.
    return f"{value * 100:.1f}%"


def find_row(rows, mode):
    # Find the result row for one method.
    for row in rows:
        if row["mode"] == mode:
            return row
    raise RuntimeError(f"Missing result row: {mode}")


def available_modes(rows):
    return {row["mode"] for row in rows}


def build_final_table(answer_summary, dataset_name):
    # Make one small table for the report.
    baseline = find_row(answer_summary, "fixed_10_full")

    baseline_f1 = as_float(baseline, "answer_f1")
    baseline_semantic_similarity = as_float(baseline, "semantic_similarity")
    baseline_tokens = as_float(baseline, "total_tokens")

    rows = []
    present_modes = available_modes(answer_summary)
    for mode, method_name in FINAL_TABLE_ROWS.items():
        if mode not in present_modes:
            continue
        row = find_row(answer_summary, mode)

        f1 = as_float(row, "answer_f1")
        coverage = as_float(row, "answer_coverage")
        similarity = as_float(row, "semantic_similarity")
        tokens = as_float(row, "total_tokens")
        fallback_rate = as_float(row, "fallback_rate")

        rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "code_mode": mode,
                "ndcg_at_10": round(as_float(row, "ndcg_at_10"), 6),
                "mrr_at_10": round(as_float(row, "mrr_at_10"), 6),
                "answer_f1": round(f1, 6),
                "answer_coverage": round(as_float(row, "answer_coverage"), 6),
                "semantic_similarity": row.get("semantic_similarity", ""),
                "f1_retained_vs_top10": percent(f1 / baseline_f1 if baseline_f1 else 0),
                "answer_coverage": round(coverage, 6),
                "semantic_similarity": round(similarity, 6),
                "semantic_similarity_retained_vs_top10": percent(
                    similarity / baseline_semantic_similarity if baseline_semantic_similarity else 0
                ),
                "total_tokens": round(tokens, 2),
                "token_reduction_vs_top10": percent(1 - (tokens / baseline_tokens) if baseline_tokens else 0),
                "fallback_rate": percent(fallback_rate),
            }
        )

    return rows


def write_csv(path, rows):
    # Save a table as CSV so it can be opened in Excel or Google Sheets.
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    # Save the same table as Markdown for the report.
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows):
    # Print the final table in the terminal.
    if not rows:
        return
    headers = list(rows[0].keys())
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]) for header in headers))


def set_seed(seed):
    # Make every Python-side choice deterministic.
    #
    # The current experiment is already mostly deterministic because:
    # - the query sample file is fixed
    # - the train/eval split sorts by query id
    # - the learned budget model has no random initialization
    #
    # This seed is still useful because it documents reproducibility and protects
    # future changes if random sampling is added later.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(description="Run the real Adaptive RAG experiment.")

    # Dataset paths.
    parser.add_argument("--documents", type=Path, default=Path("data/scifact/documents.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/scifact/queries_150_seed0.jsonl"))
    parser.add_argument(
        "--dev-queries",
        type=Path,
        default=None,
        help="Optional calibration/dev query file. If set, --queries is used entirely as eval data.",
    )
    parser.add_argument("--dataset-name", default="SciFact")
    parser.add_argument(
        "--rankings-file",
        type=Path,
        default=None,
        help="Optional precomputed retrieval_rankings.csv file from build_retrieval_rankings.py.",
    )
    parser.add_argument(
        "--retriever-name",
        default=None,
        help="Retriever row name inside --rankings-file, e.g. tfidf_cross_encoder, dense, dense_cross_encoder.",
    )

    # Output folder.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scifact_run"))

    # LLM settings.
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--api-url", default="http://localhost:11434/v1")
    parser.add_argument("--no-api-key", action="store_true", default=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--prompt-style", choices=["default", "concise", "anchor"], default="default")
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-provider-tokens",
        action="store_true",
        help="Use this for final runs so token numbers must come from Ollama/provider usage.",
    )

    # Experiment size.
    parser.add_argument("--max-eval-queries", type=int, default=50)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS_TO_RUN,
        choices=METHODS_TO_RUN,
        help="Run only these methods. Useful for cheaper batch runs.",
    )
    parser.add_argument(
        "--compression-modes",
        nargs="+",
        default=["full", "evidence_ngram_neighbors"],
        choices=["full", "evidence_ngram_neighbors"],
        help=(
            "Compression modes for ordinary baselines. Use 'full' for cheap "
            "experiments when compact baseline rows are not needed."
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed)

    print("Loading data...")
    documents = load_documents(args.documents)
    queries = load_queries(args.queries)
    dev_queries = load_queries(args.dev_queries) if args.dev_queries else None
    eval_ranked_override = None
    if args.rankings_file or args.retriever_name:
        if not args.rankings_file or not args.retriever_name:
            raise SystemExit("--rankings-file and --retriever-name must be used together.")
        print(f"Loading precomputed rankings: {args.retriever_name}")
        eval_ranked_override = load_precomputed_rankings(args.rankings_file, documents, args.retriever_name)

    print("Configuring model...")
    config = LLMConfig(
        model=args.model,
        api_url=args.api_url,
        api_key_env=args.api_key_env,
        require_api_key=not args.no_api_key,
        dry_run=args.dry_run,
        prompt_style=args.prompt_style,
        max_output_tokens=args.max_output_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        require_provider_tokens=args.require_provider_tokens,
    )

    print("Running experiment...")
    answer_rows, answer_summary, retrieval_summary = run_llm_budget_experiment(
        documents=documents,
        queries=queries,
        dev_ratio=1 / 3,
        config=config,
        max_eval_queries=args.max_eval_queries,
        eval_start_index=args.eval_start_index,
        modes=args.methods,
        compression_modes=args.compression_modes,
        oracle_strategy="minimum_sufficient",
        sufficiency_ratio=0.95,
        threshold_strategy="heuristic",
        dev_queries_override=dev_queries,
        eval_ranked_override=eval_ranked_override,
    )

    # Save detailed outputs from the real model pipeline.
    write_llm_outputs(args.output_dir, answer_rows, answer_summary, retrieval_summary)

    # Save the clean final table.
    final_rows = build_final_table(answer_summary, args.dataset_name)
    write_csv(args.output_dir / "final_table.csv", final_rows)
    write_markdown(args.output_dir / "final_table.md", final_rows)

    print("\nFinal table")
    print_table(final_rows)
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
