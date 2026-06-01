#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset


def stable_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_question(row):
    return clean(row.get("ambiguous_question") or row.get("question") or "")


def iter_annotations(row):
    annotations = row.get("annotations") or []
    if isinstance(annotations, dict):
        annotations = [annotations]
    for ann in annotations:
        if isinstance(ann, dict):
            yield ann


def get_long_answer(row):
    for ann in iter_annotations(row):
        answer = clean(ann.get("long_answer"))
        if answer:
            return answer
    return ""


def iter_evidence(row):
    for ann in iter_annotations(row):
        knowledge = ann.get("knowledge") or []
        for item in knowledge:
            if not isinstance(item, dict):
                continue
            title = clean(item.get("wikipage") or item.get("title") or "")
            text = clean(item.get("content") or item.get("context") or item.get("text") or "")
            if text and text.lower() != "no context provided":
                yield title, text

    for pair in row.get("qa_pairs") or []:
        if not isinstance(pair, dict):
            continue
        title = clean(pair.get("wikipage") or "")
        text = clean(pair.get("context") or "")
        if text and text.lower() != "no context provided":
            yield title, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/asqa_candidate")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("din0s/asqa", split=args.split, streaming=True)

    docs_by_id = {}
    queries = []

    for row in ds:
        question = get_question(row)
        answer = get_long_answer(row)
        evidence = list(iter_evidence(row))

        if not question or not answer or len(answer.split()) < 8 or not evidence:
            continue

        relevant_doc_ids = []
        for title, text in evidence[:10]:
            full_text = f"{title}. {text}".strip() if title else text
            if len(full_text.split()) < 15:
                continue
            doc_id = f"asqa_{stable_id(full_text)}"
            docs_by_id[doc_id] = {"doc_id": doc_id, "text": full_text}
            relevant_doc_ids.append(doc_id)

        relevant_doc_ids = sorted(set(relevant_doc_ids))
        if not relevant_doc_ids:
            continue

        queries.append({
            "query_id": f"asqa_{len(queries)}",
            "text": question,
            "reference_answer": answer,
            "relevant_doc_ids": relevant_doc_ids,
        })

        if len(queries) >= args.n:
            break

    with (out / "documents.jsonl").open("w", encoding="utf-8") as f:
        for row in docs_by_id.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Wrote:")
    print(out / "documents.jsonl")
    print(out / "queries.jsonl")
    print("docs:", len(docs_by_id))
    print("queries:", len(queries))


if __name__ == "__main__":
    main()
