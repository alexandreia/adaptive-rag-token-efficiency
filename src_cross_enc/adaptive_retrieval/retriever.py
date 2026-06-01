"""
Retrieval logic.

This file ranks documents for each query using TF-IDF cosine similarity.

The retrieval stage is separate from the context-budget stage:

- Retrieval asks: which documents look relevant?
- Budgeting asks: how many of those documents, or how much evidence from them,
  should we send to the LLM?

This distinction matters because retrieval can look broadly at top-10 candidates,
while the adaptive controller decides how much of that retrieved material should
actually enter the prompt.
"""

from __future__ import annotations

from adaptive_retrieval.data import Document, Query
from adaptive_retrieval.text import cosine_similarity, tfidf_vector, tokenize
from sentence_transformers import CrossEncoder
import hashlib
from collections import defaultdict

# This file contains the simple retrieval logic.
# It ranks documents using TF-IDF cosine similarity.
# Later parts of the project decide how much of the retrieved context to send to the LLM.


def _score_gap(ranked_docs: list[tuple[Document, float]], first: int, second: int) -> float:
    # Helper for older adaptive rules.
    # It measures how much higher one rank score is compared with another.
    if len(ranked_docs) <= second:
        return 0.0
    top_score = ranked_docs[0][1]
    if top_score <= 0:
        return 0.0
    return (ranked_docs[first][1] - ranked_docs[second][1]) / top_score

def make_key(query_text, doc_id):
    return hashlib.md5(
        (query_text + "||" + doc_id).encode("utf-8")
    ).hexdigest()


def retrieve(
    query: Query,
    documents: list[Document],
    doc_vectors,
    idf,
    doc_weights,
    top_k: int,
    candidate_k: int = 15,
):
    assert idf is not None, "idf is None — build_idf() not called or not passed correctly"
    query_vector = tfidf_vector(query.text, idf)

    scored = []
    for doc in documents:
        base_score = cosine_similarity(query_vector, doc_vectors[doc.doc_id])
        weighted = base_score * (doc_weights or {}).get(doc.doc_id, 1.0)
        scored.append((doc, weighted))

    candidates = sorted(scored, key=lambda x: x[1], reverse=True)[:candidate_k]

    return candidates[:top_k]



def build_all_candidates(queries, documents, doc_vectors, idf, doc_weights, candidate_k):
    all_pairs = []
    meta = []

    for query in queries:
        query_vector = tfidf_vector(query.text, idf)

        scored = []
        for doc in documents:
            base = cosine_similarity(query_vector, doc_vectors[doc.doc_id])
            weighted = base * (doc_weights or {}).get(doc.doc_id, 1.0)
            scored.append((doc, weighted))

        candidates = sorted(scored, key=lambda x: x[1], reverse=True)[:candidate_k]

        for doc, _ in candidates:
            all_pairs.append((query.text, doc.text))
            meta.append((query.query_id, doc))

    return all_pairs, meta



def group_by_query(meta, scores):
    grouped = defaultdict(list)

    for (query_id, doc), score in zip(meta, scores):
        grouped[query_id].append((doc, float(score)))

    # sort each query’s results once
    for qid in grouped:
        grouped[qid].sort(key=lambda x: x[1], reverse=True)

    return grouped

def retrieve_all_with_cross_encoder(
    queries,
    documents,
    idf,
    doc_vectors,
    doc_weights,
    candidate_k,
    cross_encoder
):
    # 1. build all TF-IDF candidates
    all_pairs, meta = build_all_candidates(
        queries,
        documents,
        doc_vectors,
        idf,
        doc_weights,
        candidate_k
    )

    # 2. single CE call
    scores = run_cross_encoder(all_pairs, cross_encoder)

    # 3. group results
    ranked_by_query = group_by_query(meta, scores)

    return ranked_by_query
    
def choose_adaptive_subset(
    ranked_docs: list[tuple[Document, float]],
    min_docs: int,
    max_docs: int,
    threshold_ratio: float,
) -> list[tuple[Document, float]]:
    # Older adaptive baseline:
    # keep documents whose score is close enough to the top document score.
    if not ranked_docs:
        return []

    top_score = ranked_docs[0][1]
    threshold = top_score * threshold_ratio
    selected = [item for item in ranked_docs if item[1] >= threshold][:max_docs]
    if len(selected) < min_docs:
        selected = ranked_docs[:min_docs]
    return selected


def choose_budget_subset(ranked_docs: list[tuple[Document, float]]) -> tuple[list[tuple[Document, float]], str]:
    """Choose context size from score-shape confidence.

    The rule is deliberately simple and explainable:
    - strong top-3 separation: keep 3 docs
    - moderate top-5 separation: keep 5 docs
    - otherwise keep 8 docs

    This tests whether a query should decide its own context budget instead
    of forcing every query into the same 3-5 document window.
    """
    if not ranked_docs:
        return [], "empty"

    # If score drops sharply after rank 3, we trust the top 3.
    gap_3_to_4 = _score_gap(ranked_docs, 2, 3)

    # If score drops after rank 5, we keep 5.
    gap_5_to_6 = _score_gap(ranked_docs, 4, 5)

    if gap_3_to_4 >= 0.12:
        return ranked_docs[:3], "high_confidence_keep_3"
    if gap_5_to_6 >= 0.06:
        return ranked_docs[:5], "medium_confidence_keep_5"
    return ranked_docs[:8], "low_confidence_keep_8"


def choose_query_aware_budget_subset(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
) -> tuple[list[tuple[Document, float]], str]:
    """Choose budget using query specificity plus retrieval confidence.

    Short keyword-style queries are often broader, so they receive a larger
    budget when the score shape is uncertain. Longer claim-style queries are
    treated as more specific and use aggressive filtering unless the ranking
    is very ambiguous.
    """
    if not ranked_docs:
        return [], "empty"

    # Query length acts as a simple signal:
    # short queries can be broad, longer queries are often more specific.
    query_terms = tokenize(query.text)
    gap_3_to_4 = _score_gap(ranked_docs, 2, 3)
    gap_5_to_6 = _score_gap(ranked_docs, 4, 5)

    if len(query_terms) <= 6:
        if gap_3_to_4 >= 0.14:
            return ranked_docs[:3], "short_high_confidence_keep_3"
        if gap_5_to_6 >= 0.07:
            return ranked_docs[:5], "short_medium_confidence_keep_5"
        return ranked_docs[:8], "short_broad_keep_8"

    if gap_3_to_4 >= 0.08:
        return ranked_docs[:3], "long_high_confidence_keep_3"
    return ranked_docs[:5], "long_specific_keep_5"


def update_weights(
    doc_weights: dict[str, float],
    selected_doc_ids: list[str],
    relevant_doc_ids: frozenset[str],
    reward: float,
    penalty: float,
) -> None:
    # Older online-learning style experiment:
    # reward documents that were relevant and penalize selected non-relevant documents.
    for doc_id in selected_doc_ids:
        if doc_id in relevant_doc_ids:
            doc_weights[doc_id] = min(2.5, doc_weights.get(doc_id, 1.0) + reward)
        else:
            doc_weights[doc_id] = max(0.35, doc_weights.get(doc_id, 1.0) - penalty)
