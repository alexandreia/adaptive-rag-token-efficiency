#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/bioasq_candidate")
    parser.add_argument("--n", type=int, default=150)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qa_ds = load_dataset(
        "enelpol/rag-mini-bioasq",
        "question-answer-passages",
        split="train",
        streaming=True,
    )
    corpus_ds = load_dataset(
        "enelpol/rag-mini-bioasq",
        "text-corpus",
        split="test",
        streaming=True,
    )

    corpus = {}
    for row in corpus_ds:
        pid = row.get("id")
        text = str(row.get("passage") or row.get("text") or "").strip()
        title = str(row.get("title", "")).strip()
        if pid is not None and text:
            corpus[int(pid)] = {
                "doc_id": f"bioasq_{pid}",
                "text": f"{title}. {text}".strip() if title else text,
            }

    docs_by_id = {}
    queries = []

    for row in qa_ds:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        rel_ids = row.get("relevant_passage_ids") or []

        if not question or not answer or not rel_ids:
            continue

        relevant_doc_ids = []
        for pid in rel_ids:
            pid = int(pid)
            if pid in corpus:
                doc = corpus[pid]
                docs_by_id[doc["doc_id"]] = doc
                relevant_doc_ids.append(doc["doc_id"])

        relevant_doc_ids = sorted(set(relevant_doc_ids))
        if not relevant_doc_ids:
            continue

        queries.append({
            "query_id": f"bioasq_{row['id']}",
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
    if queries:
        print("example:", queries[0])


if __name__ == "__main__":
    main()
