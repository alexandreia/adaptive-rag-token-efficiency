#!/usr/bin/env python3
"""Prepare MS MARCO QA in the project JSONL format.

MS MARCO is useful here because it is web-search QA: each question comes with
candidate passages, human answers, and passage-level selected-evidence labels.
That makes it a good stress test for adaptive evidence shaping without turning
the project into conversational QA engineering.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


NO_ANSWER = {"no answer present.", "no answer present", ""}


def clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_answer(row: dict[str, object]) -> str:
    answers = row.get("answers") or []
    if isinstance(answers, str):
        answers = [answers]
    for answer in answers:
        cleaned = clean(answer)
        if cleaned.lower() not in NO_ANSWER:
            return cleaned
    well_formed = row.get("wellFormedAnswers") or []
    if isinstance(well_formed, str):
        well_formed = [well_formed]
    for answer in well_formed:
        cleaned = clean(answer)
        if cleaned.lower() not in NO_ANSWER:
            return cleaned
    return ""


def passage_rows(row: dict[str, object]) -> list[tuple[int, str, str]]:
    passages = row.get("passages") or {}
    if not isinstance(passages, dict):
        return []

    texts = passages.get("passage_text") or []
    selected = passages.get("is_selected") or []
    urls = passages.get("url") or []

    rows = []
    for index, text in enumerate(texts):
        passage = clean(text)
        if len(passage.split()) < 8:
            continue
        is_selected = int(selected[index]) if index < len(selected) else 0
        url = clean(urls[index]) if index < len(urls) else ""
        rows.append((is_selected, url, passage))
    return rows


def split_queries(
    queries: list[dict[str, object]],
    calibration_size: int,
    eval_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    calibration = shuffled[:calibration_size]
    evaluation = shuffled[calibration_size : calibration_size + eval_size]
    return calibration, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MS MARCO QA data.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/msmarco_250"))
    parser.add_argument("--dataset", default="microsoft/ms_marco")
    parser.add_argument("--config", default="v1.1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--calibration-size", type=int, default=50)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    documents: list[dict[str, str]] = []
    queries: list[dict[str, object]] = []
    seen_doc_ids: set[str] = set()

    for row in dataset:
        query = clean(row.get("query"))
        answer = first_answer(row)
        query_id = clean(row.get("query_id"))
        passages = passage_rows(row)

        if not query or not answer or not query_id or not passages:
            continue

        relevant_doc_ids: list[str] = []
        candidate_doc_ids: list[str] = []
        for index, (is_selected, url, passage) in enumerate(passages):
            doc_id = f"msmarco_{query_id}_p{index}"
            doc_text = f"{url}. {passage}".strip() if url else passage
            candidate_doc_ids.append(doc_id)
            if doc_id not in seen_doc_ids:
                documents.append({"doc_id": doc_id, "text": doc_text})
                seen_doc_ids.add(doc_id)
            if is_selected:
                relevant_doc_ids.append(doc_id)

        if not relevant_doc_ids:
            continue

        queries.append(
            {
                "query_id": f"msmarco_{query_id}",
                "text": query,
                "reference_answer": answer,
                "relevant_doc_ids": sorted(set(relevant_doc_ids)),
            }
        )

        if len(queries) >= args.n:
            break

    calibration, evaluation = split_queries(
        queries,
        calibration_size=args.calibration_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "documents.jsonl", documents)
    write_jsonl(args.out_dir / "queries.jsonl", queries)
    write_jsonl(args.out_dir / "queries_calibration_50.jsonl", calibration)
    write_jsonl(args.out_dir / "queries_eval_200.jsonl", evaluation)

    print("Wrote:")
    print(args.out_dir / "documents.jsonl")
    print(args.out_dir / "queries.jsonl")
    print(args.out_dir / "queries_calibration_50.jsonl")
    print(args.out_dir / "queries_eval_200.jsonl")
    print("docs:", len(documents))
    print("queries:", len(queries))
    print("calibration:", len(calibration))
    print("evaluation:", len(evaluation))
    if queries:
        print("example:", queries[0])


if __name__ == "__main__":
    main()
