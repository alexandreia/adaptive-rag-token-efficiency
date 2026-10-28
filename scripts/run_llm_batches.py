#!/usr/bin/env python3
"""Run LLM experiments in small saved batches, then merge completed batches."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run run_experiment.py in numbered batches.")
    parser.add_argument("--documents", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--dev-queries", type=Path, default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--rankings-file", type=Path, default=None)
    parser.add_argument("--retriever-name", default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--prompt-style", choices=["default", "concise", "anchor"], default="default")
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-provider-tokens", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--compression-modes",
        nargs="+",
        default=["full", "evidence_ngram_neighbors"],
        choices=["full", "evidence_ngram_neighbors"],
        help="Compression modes for ordinary baselines. Use 'full' to avoid hidden extra baseline calls.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
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
            "task_aware_coverage_ultra",
        ],
        choices=[
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
        ],
    )
    args = parser.parse_args()

    total_queries = count_jsonl(args.queries)
    end_index = min(total_queries, args.start_index + (args.total if args.total is not None else total_queries))
    completed_dirs = []
    failed = []

    for start in range(args.start_index, end_index, args.batch_size):
        stop = min(start + args.batch_size, end_index)
        batch_dir = args.output_root / f"batch_{start + 1:03d}_{stop:03d}"
        final_table = batch_dir / "final_table.csv"
        answers_file = batch_dir / "llm_answers_by_query.csv"
        if final_table.exists() and answers_file.exists():
            print(f"\n=== Skipping completed {args.dataset_name} examples {start + 1}-{stop} ===")
            completed_dirs.append(batch_dir)
            continue

        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_experiment.py"),
            "--documents",
            str(args.documents),
            "--queries",
            str(args.queries),
            "--dataset-name",
            args.dataset_name,
            "--output-dir",
            str(batch_dir),
            "--model",
            args.model,
            "--api-url",
            args.api_url,
            "--api-key-env",
            args.api_key_env,
            "--prompt-style",
            args.prompt_style,
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--max-eval-queries",
            str(stop - start),
            "--eval-start-index",
            str(start),
            "--seed",
            str(args.seed),
            "--methods",
            *args.methods,
            "--compression-modes",
            *args.compression_modes,
        ]
        if args.rankings_file or args.retriever_name:
            if not args.rankings_file or not args.retriever_name:
                raise SystemExit("--rankings-file and --retriever-name must be used together.")
            command.extend(["--rankings-file", str(args.rankings_file)])
            command.extend(["--retriever-name", args.retriever_name])
        if args.dev_queries:
            command.extend(["--dev-queries", str(args.dev_queries)])
        if args.require_provider_tokens:
            command.append("--require-provider-tokens")
        if args.dry_run:
            command.append("--dry-run")

        print(f"\n=== Running {args.dataset_name} examples {start + 1}-{stop} ===")
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode == 0:
            completed_dirs.append(batch_dir)
        else:
            failed.append(batch_dir)
            print(f"Batch failed: {batch_dir}")
            if args.stop_on_error:
                break

    if completed_dirs:
        merged_dir = args.output_root / "merged"
        merge_command = [
            sys.executable,
            str(ROOT / "scripts" / "merge_llm_batches.py"),
            "--dataset-name",
            args.dataset_name,
            "--output-dir",
            str(merged_dir),
            *[str(path) for path in completed_dirs],
        ]
        print("\n=== Merging completed batches ===")
        subprocess.run(merge_command, cwd=ROOT, check=True)

    print(f"\nCompleted batches: {len(completed_dirs)}")
    if failed:
        print("Failed batches:")
        for path in failed:
            print(f"- {path}")


if __name__ == "__main__":
    main()
