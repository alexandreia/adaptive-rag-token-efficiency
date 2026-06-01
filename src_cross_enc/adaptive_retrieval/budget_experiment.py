"""
Small retrieval-metric helper used by the adaptive budget model.

The current GitHub project only needs one function from the older budget
experiments:

    _record_metrics(...)

This function takes:

- the query
- the selected documents
- the full ranked list

and returns the retrieval/answer proxy metrics used to train and evaluate the
basic adaptive budget model.

Keeping only this helper makes the code easier to understand while preserving
the behavior of the real model.
"""

from __future__ import annotations

from adaptive_retrieval.data import Document, Query
from adaptive_retrieval.metrics import (
    RunMetrics,
    answer_coverage,
    precision_at_k,
    recall,
    ndcg_at_k,
    reciprocal_rank,
    token_f1,
)
from adaptive_retrieval.text import estimate_tokens


def _record_metrics(
    mode: str,
    iteration: int,
    query: Query,
    selected: list[tuple[Document, float]],
    ranked: list[tuple[Document, float]],
) -> RunMetrics:
    # Convert selected documents into ids and text.
    selected_docs = [doc for doc, _score in selected]
    selected_doc_ids = [doc.doc_id for doc in selected_docs]
    ranked_doc_ids = [doc.doc_id for doc, _score in ranked]

    # Estimate context size for retrieval-side budget comparisons.
    tokens_used = sum(estimate_tokens(doc.text) for doc in selected_docs)

    # The proxy answer context is just the selected document text joined together.
    # This is not an LLM answer. It is used only to create/evaluate budget labels.
    context = " ".join(doc.text for doc in selected_docs)

    # Context precision asks: what fraction of selected documents are relevant?
    context_precision = precision_at_k(selected_doc_ids, query.relevant_doc_ids)

    return RunMetrics(
        mode=mode,
        iteration=iteration,
        query_id=query.query_id,
        precision_at_k=context_precision,
        recall=recall(selected_doc_ids, query.relevant_doc_ids),
        mrr=reciprocal_rank(ranked_doc_ids, query.relevant_doc_ids),
        ndcg_at_10=ndcg_at_k(selected_doc_ids, query.relevant_doc_ids, k=10),
        docs_used=len(selected_doc_ids),
        tokens_used=tokens_used,
        context_precision=context_precision,
        context_noise_tokens=tokens_used * (1 - context_precision),
        answer_f1=token_f1(context, query.reference_answer),
        answer_coverage=answer_coverage(context, query.reference_answer),
    )
