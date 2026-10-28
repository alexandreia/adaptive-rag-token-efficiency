#!/usr/bin/env python3
"""Build retrieval rankings for publication expansion.

This script prepares retrieval-only rankings and metrics for three settings:

1. TF-IDF
2. TF-IDF candidates reranked by a cross-encoder
3. Dense retrieval candidates reranked by a cross-encoder

It does not call the answer-generation LLM. The goal is to decide which strong
retrieval settings are worth running through the full RAG experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_retrieval.data import Document, Query, load_documents, load_queries  # noqa: E402
from adaptive_retrieval.metrics import ndcg_at_k, reciprocal_rank  # noqa: E402
from adaptive_retrieval.retriever import retrieve  # noqa: E402
from adaptive_retrieval.text import build_idf, tfidf_vector  # noqa: E402


def batched(items: list, batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_query(text: str) -> str:
    return f"query: {text}"


def normalize_document(text: str) -> str:
    return f"passage: {text}"


def build_tfidf_rankings(
    documents: list[Document],
    queries: list[Query],
    candidate_k: int,
) -> dict[str, list[tuple[Document, float]]]:
    idf = build_idf(documents)
    doc_vectors = {doc.doc_id: tfidf_vector(doc.text, idf) for doc in documents}
    return {
        query.query_id: retrieve(query, documents, doc_vectors, idf, None, candidate_k)
        for query in queries
    }


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install sentence-transformers first: python -m pip install sentence-transformers") from error
    return SentenceTransformer(model_name)


def load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise SystemExit("Install sentence-transformers first: python -m pip install sentence-transformers") from error
    return CrossEncoder(model_name)


def dense_rankings(
    documents: list[Document],
    queries: list[Query],
    model_name: str,
    candidate_k: int,
    batch_size: int,
) -> dict[str, list[tuple[Document, float]]]:
    model = load_sentence_transformer(model_name)
    doc_texts = [normalize_document(doc.text) for doc in documents]
    query_texts = [normalize_query(query.text) for query in queries]

    doc_embeddings = model.encode(
        doc_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    rankings: dict[str, list[tuple[Document, float]]] = {}
    for query, query_embedding in zip(queries, query_embeddings):
        scores = doc_embeddings @ query_embedding
        top_indexes = scores.argsort()[-candidate_k:][::-1]
        rankings[query.query_id] = [(documents[int(index)], float(scores[int(index)])) for index in top_indexes]
    return rankings


def rerank_with_cross_encoder(
    rankings: dict[str, list[tuple[Document, float]]],
    queries: list[Query],
    model_name: str,
    batch_size: int,
) -> dict[str, list[tuple[Document, float]]]:
    model = load_cross_encoder(model_name)
    query_by_id = {query.query_id: query for query in queries}

    pairs: list[tuple[str, str]] = []
    meta: list[tuple[str, Document]] = []
    for query_id, ranked_docs in rankings.items():
        query = query_by_id[query_id]
        for doc, _score in ranked_docs:
            pairs.append((query.text, doc.text))
            meta.append((query_id, doc))

    scores = []
    for batch in batched(pairs, batch_size):
        scores.extend(float(score) for score in model.predict(batch))

    reranked: dict[str, list[tuple[Document, float]]] = {query.query_id: [] for query in queries}
    for (query_id, doc), score in zip(meta, scores):
        reranked[query_id].append((doc, score))
    for query_id in reranked:
        reranked[query_id].sort(key=lambda item: item[1], reverse=True)
    return reranked


def ranking_rows(name: str, rankings: dict[str, list[tuple[Document, float]]], queries: list[Query], top_k: int) -> list[dict]:
    rows = []
    for query in queries:
        ranked_docs = rankings[query.query_id][:top_k]
        rows.append(
            {
                "retriever": name,
                "query_id": query.query_id,
                "ranked_doc_ids": json.dumps([doc.doc_id for doc, _score in ranked_docs]),
                "scores": json.dumps([round(float(score), 6) for _doc, score in ranked_docs]),
                "ndcg_at_10": round(ndcg_at_k([doc.doc_id for doc, _score in ranked_docs], query.relevant_doc_ids, 10), 6),
                "mrr_at_10": round(reciprocal_rank([doc.doc_id for doc, _score in ranked_docs], query.relevant_doc_ids), 6),
                "recall_at_10": round(
                    len({doc.doc_id for doc, _score in ranked_docs[:10]} & set(query.relevant_doc_ids))
                    / len(query.relevant_doc_ids)
                    if query.relevant_doc_ids
                    else 0.0,
                    6,
                ),
            }
        )
    return rows


def summary_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["retriever"], []).append(row)

    summaries = []
    for retriever, selected in grouped.items():
        summaries.append(
            {
                "retriever": retriever,
                "queries": len(selected),
                "ndcg_at_10": round(sum(float(row["ndcg_at_10"]) for row in selected) / len(selected), 6),
                "mrr_at_10": round(sum(float(row["mrr_at_10"]) for row in selected) / len(selected), 6),
                "recall_at_10": round(sum(float(row["recall_at_10"]) for row in selected) / len(selected), 6),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--tfidf-candidate-k", type=int, default=50)
    parser.add_argument("--dense-candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-model", default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-cross-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    queries = load_queries(args.queries)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    all_rows: list[dict] = []

    print("Building TF-IDF rankings...")
    tfidf = build_tfidf_rankings(documents, queries, candidate_k=args.tfidf_candidate_k)
    all_rows.extend(ranking_rows("tfidf", tfidf, queries, args.top_k))

    if not args.skip_cross_encoder:
        print("Reranking TF-IDF candidates with cross-encoder...")
        tfidf_ce = rerank_with_cross_encoder(tfidf, queries, args.cross_encoder_model, args.batch_size)
        all_rows.extend(ranking_rows("tfidf_cross_encoder", tfidf_ce, queries, args.top_k))

    if not args.skip_dense:
        print("Building dense rankings...")
        dense = dense_rankings(documents, queries, args.dense_model, args.dense_candidate_k, args.batch_size)
        all_rows.extend(ranking_rows("dense", dense, queries, args.top_k))

        if not args.skip_cross_encoder:
            print("Reranking dense candidates with cross-encoder...")
            dense_ce = rerank_with_cross_encoder(dense, queries, args.cross_encoder_model, args.batch_size)
            all_rows.extend(ranking_rows("dense_cross_encoder", dense_ce, queries, args.top_k))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "retrieval_rankings.csv", all_rows)
    write_jsonl(args.output_dir / "retrieval_rankings.jsonl", all_rows)
    summaries = summary_rows(all_rows)
    write_csv(args.output_dir / "retrieval_summary.csv", summaries)

    print("\nRetrieval summary")
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
