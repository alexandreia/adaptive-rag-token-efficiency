#!/usr/bin/env python3
"""Run Table B component ablations for Safe Adaptive Context.

This script is intentionally separate from run_experiment.py. It does not
change the main system; it only runs targeted variants that isolate the parts
of Safe Adaptive Context:

- adaptive budget
- compact evidence compression
- fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_retrieval.data import Document, Query, load_documents, load_queries
from adaptive_retrieval.learned_budget import (
    build_examples,
    evaluate_learned_budget,
    split_queries,
    summarize as summarize_retrieval_metrics,
    train_centroid_model,
)
from adaptive_retrieval.llm_budget import (
    LLMConfig,
    LLMRunRow,
    answer_aware_fallback_run,
    answer_needs_fallback,
    compress_documents,
    config_for_answer_call,
    context_mrr_at_10,
    context_ndcg_at_10,
    generate_answer,
    summarize_llm_rows,
    write_llm_outputs,
)
from adaptive_retrieval.metrics import answer_coverage, semantic_similarity, token_f1


TABLE_B_ROWS = {
    "fixed_top10_full": "Fixed Top-10",
    "adaptive_compact_only": "Adaptive Compact Only",
    "adaptive_full_only": "Adaptive Full Only",
    "fixed_top5_compact_fallback": "Fixed Top-5 Compact + Fallback",
    "full_safe_adaptive": "Full Safe Adaptive Context",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def find_row(rows: list[dict[str, object]], mode: str) -> dict[str, object]:
    for row in rows:
        if row["mode"] == mode:
            return row
    raise RuntimeError(f"Missing summary row: {mode}")


def build_table_b(summary_rows: list[dict[str, object]], dataset_name: str) -> list[dict[str, object]]:
    baseline = find_row(summary_rows, "fixed_top10_full")
    baseline_f1 = float(baseline["answer_f1"])
    baseline_similarity = float(baseline["semantic_similarity"])
    baseline_tokens = float(baseline["total_tokens"])

    rows = []
    for mode, component in TABLE_B_ROWS.items():
        row = find_row(summary_rows, mode)
        f1 = float(row["answer_f1"])
        coverage = float(row["answer_coverage"])
        similarity = float(row["semantic_similarity"])
        tokens = float(row["total_tokens"])
        fallback_rate = float(row["fallback_rate"])

        rows.append(
            {
                "dataset": dataset_name,
                "component_variant": component,
                "code_mode": mode,
                "ndcg_at_10": round(float(row["ndcg_at_10"]), 6),
                "mrr_at_10": round(float(row["mrr_at_10"]), 6),
                "answer_f1": round(f1, 6),
                "answer_coverage": round(coverage, 6),
                "semantic_similarity": round(similarity, 6),
                "f1_retained_vs_top10": percent(f1 / baseline_f1 if baseline_f1 else 0.0),
                "semantic_similarity_retained_vs_top10": percent(
                    similarity / baseline_similarity if baseline_similarity else 0.0
                ),
                "total_tokens": round(tokens, 2),
                "token_reduction_vs_top10": percent(1 - tokens / baseline_tokens if baseline_tokens else 0.0),
                "fallback_rate": percent(fallback_rate),
            }
        )
    return rows


def make_row(
    *,
    mode: str,
    method_name: str,
    budget_mode: str,
    compression_mode: str,
    query: Query,
    selected_docs: list[Document],
    answer,
    fallback_used: bool = False,
    fallback_reason: str = "",
    first_pass_tokens: int = 0,
    fallback_tokens: int = 0,
) -> LLMRunRow:
    return LLMRunRow(
        mode=mode,
        method_name=method_name,
        budget_mode=budget_mode,
        compression_mode=compression_mode,
        query_id=query.query_id,
        docs_used=len(selected_docs),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        total_tokens=answer.total_tokens,
        token_source=answer.token_source,
        generation_time_ms=answer.generation_time_ms,
        answer_f1=round(token_f1(answer.text, query.reference_answer), 6),
        answer_coverage=round(answer_coverage(answer.text, query.reference_answer), 6),
        semantic_similarity=round(semantic_similarity(answer.text, query.reference_answer), 6),
        ndcg_at_10=context_ndcg_at_10(selected_docs, query),
        mrr_at_10=context_mrr_at_10(selected_docs, query),
        selected_doc_ids=json.dumps([doc.doc_id for doc in selected_docs]),
        answer=answer.text,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        first_pass_tokens=first_pass_tokens,
        fallback_tokens=fallback_tokens,
    )


def generate_with_retries(query: Query, selected_docs: list[Document], config: LLMConfig, args):
    for attempt in range(1, args.max_attempts + 1):
        try:
            answer = generate_answer(query, selected_docs, config)
            if args.sleep_between_calls:
                time.sleep(args.sleep_between_calls)
            return answer
        except RuntimeError as error:
            retryable = "429" in str(error) or "502" in str(error) or "temporarily unreachable" in str(error)
            if not retryable or attempt == args.max_attempts:
                raise
            print(
                f"Temporary LLM API problem; waiting {args.retry_wait_seconds:.0f}s "
                f"before retry {attempt + 1}/{args.max_attempts}."
            )
            time.sleep(args.retry_wait_seconds)
    raise RuntimeError("Unexpected retry loop exit.")


def fixed_top5_compact_fallback(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
    args,
) -> tuple[object, list[Document], bool, str, int, int]:
    full_docs = [doc for doc, _score in ranked_docs[:5]]
    compact_docs = compress_documents(query, full_docs, "evidence_ngram_neighbors")
    compact_config = config_for_answer_call(config, "evidence_ngram_neighbors")
    first_answer = generate_with_retries(query, compact_docs, compact_config, args)

    should_expand, reason = answer_needs_fallback(query, first_answer.text, compact_docs)
    if not should_expand:
        return first_answer, compact_docs, False, "", first_answer.total_tokens, 0

    full_config = config_for_answer_call(config, "full")
    final_answer = generate_with_retries(query, full_docs, full_config, args)

    combined_answer = type(first_answer)(
        text=final_answer.text,
        prompt_tokens=first_answer.prompt_tokens + final_answer.prompt_tokens,
        completion_tokens=first_answer.completion_tokens + final_answer.completion_tokens,
        total_tokens=first_answer.total_tokens + final_answer.total_tokens,
        token_source=",".join(sorted({first_answer.token_source, final_answer.token_source})),
        generation_time_ms=first_answer.generation_time_ms + final_answer.generation_time_ms,
    )
    return (
        combined_answer,
        full_docs,
        True,
        f"top_5_compact:{reason}",
        first_answer.total_tokens,
        final_answer.total_tokens,
    )


def run_table_b(args) -> tuple[list[LLMRunRow], list[dict[str, object]], list[dict[str, object]]]:
    documents = load_documents(args.documents)
    queries = load_queries(args.queries)

    dev_queries, eval_queries = split_queries(queries, args.dev_ratio)
    if args.eval_start_index:
        eval_queries = eval_queries[args.eval_start_index :]
    if args.max_eval_queries is not None:
        eval_queries = eval_queries[: args.max_eval_queries]

    dev_examples, _dev_ranked = build_examples(
        documents,
        dev_queries,
        oracle_strategy="minimum_sufficient",
        sufficiency_ratio=0.95,
    )
    eval_examples, eval_ranked = build_examples(
        documents,
        eval_queries,
        oracle_strategy="minimum_sufficient",
        sufficiency_ratio=0.95,
    )
    model = train_centroid_model(dev_examples, threshold_strategy="heuristic")
    retrieval_metrics, predictions = evaluate_learned_budget(eval_queries, eval_examples, eval_ranked, model)
    prediction_by_query = {str(row["query_id"]): row for row in predictions}

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

    answer_rows: list[LLMRunRow] = []
    for query_index, query in enumerate(eval_queries, start=1):
        print(f"Running Table B query {query_index}/{len(eval_queries)}: {query.query_id}")
        ranked_docs = eval_ranked[query.query_id]
        prediction = prediction_by_query[query.query_id]
        sequential_budget = min(max(int(prediction["sequential_budget"]), 3), 10)

        top10_docs = [doc for doc, _score in ranked_docs[:10]]
        answer = generate_with_retries(query, top10_docs, config_for_answer_call(config, "full"), args)
        answer_rows.append(
            make_row(
                mode="fixed_top10_full",
                method_name=TABLE_B_ROWS["fixed_top10_full"],
                budget_mode="fixed_top10",
                compression_mode="full",
                query=query,
                selected_docs=top10_docs,
                answer=answer,
            )
        )

        adaptive_full_docs = [doc for doc, _score in ranked_docs[:sequential_budget]]
        adaptive_compact_docs = compress_documents(query, adaptive_full_docs, "evidence_ngram_neighbors")
        answer = generate_with_retries(
            query,
            adaptive_compact_docs,
            config_for_answer_call(config, "evidence_ngram_neighbors"),
            args,
        )
        answer_rows.append(
            make_row(
                mode="adaptive_compact_only",
                method_name=TABLE_B_ROWS["adaptive_compact_only"],
                budget_mode="adaptive_budget",
                compression_mode="evidence_ngram_neighbors",
                query=query,
                selected_docs=adaptive_compact_docs,
                answer=answer,
            )
        )

        answer = generate_with_retries(query, adaptive_full_docs, config_for_answer_call(config, "full"), args)
        answer_rows.append(
            make_row(
                mode="adaptive_full_only",
                method_name=TABLE_B_ROWS["adaptive_full_only"],
                budget_mode="adaptive_budget",
                compression_mode="full",
                query=query,
                selected_docs=adaptive_full_docs,
                answer=answer,
            )
        )

        answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
            fixed_top5_compact_fallback(query, ranked_docs, config, args)
        )
        answer_rows.append(
            make_row(
                mode="fixed_top5_compact_fallback",
                method_name=TABLE_B_ROWS["fixed_top5_compact_fallback"],
                budget_mode="fixed_top5",
                compression_mode="compact_then_full_fallback",
                query=query,
                selected_docs=selected_docs,
                answer=answer,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                first_pass_tokens=first_pass_tokens,
                fallback_tokens=fallback_tokens,
            )
        )

        answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
            answer_aware_fallback_run(query, ranked_docs, sequential_budget, config)
        )
        if args.sleep_between_calls:
            time.sleep(args.sleep_between_calls)
        answer_rows.append(
            make_row(
                mode="full_safe_adaptive",
                method_name=TABLE_B_ROWS["full_safe_adaptive"],
                budget_mode="safe_adaptive",
                compression_mode="compact_then_full_fallback",
                query=query,
                selected_docs=selected_docs,
                answer=answer,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                first_pass_tokens=first_pass_tokens,
                fallback_tokens=fallback_tokens,
            )
        )

    return answer_rows, summarize_llm_rows(answer_rows), summarize_retrieval_metrics(retrieval_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table B Safe Adaptive Context component ablation.")
    parser.add_argument("--documents", type=Path, default=Path("data/scifact/documents.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/scifact/queries_150_seed0_llm_gold_v2.jsonl"))
    parser.add_argument("--dataset-name", default="scifact")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/table_b_scifact_llama70b_ablation_10"))
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--api-url", default="https://api.berget.ai/v1")
    parser.add_argument("--api-key-env", default="BERGET_API_KEY")
    parser.add_argument("--no-api-key", action="store_true")
    parser.add_argument("--prompt-style", choices=["default", "concise", "anchor"], default="default")
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-provider-tokens", action="store_true")
    parser.add_argument("--max-eval-queries", type=int, default=10)
    parser.add_argument("--eval-start-index", type=int, default=0)
    parser.add_argument("--dev-ratio", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-between-calls", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-wait-seconds", type=float, default=65.0)
    args = parser.parse_args()

    set_seed(args.seed)
    answer_rows, summary_rows, retrieval_metrics = run_table_b(args)

    write_llm_outputs(args.output_dir, answer_rows, summary_rows, retrieval_metrics)
    table_b_rows = build_table_b(summary_rows, args.dataset_name)
    write_csv(args.output_dir / "table_b_component_ablation.csv", table_b_rows)
    write_markdown(args.output_dir / "table_b_component_ablation.md", table_b_rows)

    print("\nTable B")
    for row in table_b_rows:
        print(row)
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
