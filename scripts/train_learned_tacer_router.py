#!/usr/bin/env python3
"""Train the first learned TACER evidence router.

This script is intentionally modest. It turns the existing saved experiment
tables into an oracle routing dataset, then trains a small multiclass linear
model that predicts the oracle action from pre-generation features.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_retrieval.learned_router import (  # noqa: E402
    ACTION_DISPLAY_NAMES,
    DEFAULT_ACTIONS,
    RouterExample,
    build_router_examples,
    deterministic_split,
    evaluate_router,
    save_model,
    train_router,
)


DATASETS = ["scifact", "bioasq", "hotpotqa", "msmarco", "asqa"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--model", default="llama70b_tfidf")
    parser.add_argument("--retriever", default="tfidf")
    parser.add_argument("--quality-floor", type=float, default=0.95)
    parser.add_argument("--semantic-margin", type=float, default=0.03)
    parser.add_argument("--token-weight", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--leave-one-dataset-out", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "saved_results" / "learned_router",
    )
    args = parser.parse_args()

    examples = load_all_examples(args)
    if not examples:
        raise SystemExit("No training examples found. Check saved result and ranking paths.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_examples(args.output_dir / "router_training_examples.csv", examples)

    if args.leave_one_dataset_out:
        summaries = run_leave_one_dataset_out(args, examples)
        model_examples = examples
        summary_path = args.output_dir / "learned_router_leave_one_dataset_out_summary.json"
    else:
        train_examples, test_examples = deterministic_split(examples, test_ratio=args.test_ratio)
        model = train_router(
            train_examples,
            actions=DEFAULT_ACTIONS,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        save_model(model, args.output_dir / "learned_tacer_router.json")
        summaries = [
            {"split": "train", **evaluate_router(model, train_examples)},
            {"split": "test", **evaluate_router(model, test_examples)},
        ]
        model_examples = train_examples
        summary_path = args.output_dir / "learned_router_summary.json"

    write_summary(summary_path, summaries, examples)
    write_label_counts(args.output_dir / "router_label_counts.csv", examples)
    print_summary(summaries, examples)
    print(f"\nWrote examples to: {args.output_dir / 'router_training_examples.csv'}")
    print(f"Wrote summary to: {summary_path}")
    if not args.leave_one_dataset_out:
        print(f"Wrote model to: {args.output_dir / 'learned_tacer_router.json'}")
    print(f"Training rows used for final model path: {len(model_examples)}")


def load_all_examples(args: argparse.Namespace) -> list[RouterExample]:
    all_examples: list[RouterExample] = []
    for dataset in DATASETS:
        queries_path = args.repo_root / "data" / f"{dataset}_250" / "queries_eval_200.jsonl"
        rankings_path = (
            args.repo_root
            / "saved_results"
            / "retrieval_rankings"
            / f"{dataset}_200"
            / "retrieval_rankings.csv"
        )
        outcomes_path = (
            args.repo_root
            / "saved_results"
            / "final_main"
            / args.model
            / dataset
            / "llm_answers_by_query.csv"
        )
        if not queries_path.exists() or not rankings_path.exists() or not outcomes_path.exists():
            print(f"Skipping {dataset}: missing one or more inputs", file=sys.stderr)
            continue
        dataset_examples = build_router_examples(
            dataset=dataset,
            queries_path=queries_path,
            rankings_path=rankings_path,
            outcomes_path=outcomes_path,
            actions=DEFAULT_ACTIONS,
            quality_floor=args.quality_floor,
            semantic_margin=args.semantic_margin,
            token_weight=args.token_weight,
        )
        print(f"Loaded {len(dataset_examples)} examples from {dataset}", file=sys.stderr)
        all_examples.extend(dataset_examples)
    return all_examples


def run_leave_one_dataset_out(
    args: argparse.Namespace,
    examples: list[RouterExample],
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    datasets = sorted({example.dataset for example in examples})
    for heldout in datasets:
        train_examples = [example for example in examples if example.dataset != heldout]
        test_examples = [example for example in examples if example.dataset == heldout]
        model = train_router(
            train_examples,
            actions=DEFAULT_ACTIONS,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        save_model(model, args.output_dir / f"learned_tacer_router_without_{heldout}.json")
        summaries.append({"split": f"heldout_{heldout}", **evaluate_router(model, test_examples)})
    return summaries


def write_examples(path: Path, examples: list[RouterExample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "dataset",
            "query_id",
            "label",
            "fixed_f1",
            "chosen_f1",
            "chosen_tokens",
            "fixed_tokens",
            "features_json",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "dataset": example.dataset,
                    "query_id": example.query_id,
                    "label": example.label,
                    "fixed_f1": f"{example.fixed_f1:.6f}",
                    "chosen_f1": f"{example.chosen_f1:.6f}",
                    "chosen_tokens": f"{example.chosen_tokens:.1f}",
                    "fixed_tokens": f"{example.fixed_tokens:.1f}",
                    "features_json": json.dumps(example.features),
                }
            )


def write_summary(path: Path, summaries: list[dict[str, float | int | str]], examples: list[RouterExample]) -> None:
    payload = {
        "examples": len(examples),
        "actions": {action: ACTION_DISPLAY_NAMES.get(action, action) for action in DEFAULT_ACTIONS},
        "summaries": summaries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_label_counts(path: Path, examples: list[RouterExample]) -> None:
    counts = Counter((example.dataset, example.label) for example in examples)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "label", "label_name", "count"])
        for dataset, label in sorted(counts):
            writer.writerow([dataset, label, ACTION_DISPLAY_NAMES.get(label, label), counts[(dataset, label)]])


def print_summary(summaries: list[dict[str, float | int | str]], examples: list[RouterExample]) -> None:
    print(f"\nBuilt {len(examples)} learned-router examples.")
    for summary in summaries:
        accuracy = float(summary.get("accuracy", 0.0))
        token_ratio = float(summary.get("oracle_token_ratio", 0.0))
        f1_delta = float(summary.get("oracle_f1_delta", 0.0))
        print(
            f"{summary['split']}: n={summary['examples']} "
            f"accuracy={accuracy:.3f} oracle_token_ratio={token_ratio:.3f} "
            f"oracle_f1_delta={f1_delta:+.4f}"
        )


if __name__ == "__main__":
    main()
