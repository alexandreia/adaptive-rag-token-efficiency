"""
Text utilities.

This file contains the simple text-processing functions used by retrieval and
evaluation:

- tokenize text
- estimate tokens when provider token counts are unavailable
- build TF-IDF vectors
- compute cosine similarity

The functions are intentionally lightweight. The project is not about building
the most advanced retriever. It is about testing whether adaptive context
selection can reduce LLM prompt cost.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from adaptive_retrieval.data import Document


# This file has the simple text utilities used by the retrieval system.
# We keep these lightweight on purpose, because the project is about the
# adaptive context controller, not about building a complicated retriever.

def tokenize(text: str) -> list[str]:
    # Basic lowercase tokenizer.
    # It keeps letters and numbers, and removes punctuation.
    return re.findall(r"[a-z0-9]+", text.lower())


def estimate_tokens(text: str) -> int:
    # This is an approximate token counter.
    # Real LLM tokenizers split words differently, but this gives a stable local estimate.
    # We multiply by 1.3 because LLM token counts are often larger than plain word counts.
    return max(1, math.ceil(len(tokenize(text)) * 1.3))


def build_idf(documents: Iterable[Document]) -> dict[str, float]:
    # Build inverse document frequency values for TF-IDF retrieval.
    # Rare terms get higher weight, common terms get lower weight.
    docs = list(documents)
    document_frequency: Counter[str] = Counter()
    for doc in docs:
        document_frequency.update(set(tokenize(doc.text)))

    return {
        term: math.log((1 + len(docs)) / (1 + count)) + 1
        for term, count in document_frequency.items()
    }

def build_index(documents, idf):
    doc_vectors = {
        doc.doc_id: tfidf_vector(doc.text, idf)
        for doc in documents
    }
    return doc_vectors

def build_doc_vectors(documents, idf):
    # Precompute TF-IDF vectors for all documents.
    # This makes retrieval faster because we don't have to recompute them for every query.
    return {
        doc.doc_id: tfidf_vector(doc.text, idf)
        for doc in documents
    }

def tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    # Convert text into a sparse TF-IDF vector:
    # {term: weight}
    # This is what the retriever compares against query vectors.
    terms = tokenize(text)
    counts = Counter(terms)
    if not counts:
        return {}
    return {term: (count / len(terms)) * idf.get(term, 1.0) for term, count in counts.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    # Compare two sparse vectors.
    # Higher value means the query and document share stronger weighted terms.
    common_terms = set(left) & set(right)
    numerator = sum(left[term] * right[term] for term in common_terms)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
