"""
Dataset loading and asic data objects.

Every dataset is converted into two JSONL files:

1. documents.jsonl
   One document per line.
   Each document has:
   - doc_id
   - text

2. queries.jsonl
   One query per line.
   Each query has:
   - query_id
   - text
   - relevant_doc_ids
   - reference_answer

The rest of the project does not need to know the original BEIR format.
It only works with these simple Document and Query objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# This file only defines the basic data structures that the rest of the project uses.
# Every dataset is converted into:
# - documents.jsonl
# - queries.jsonl
# That way all experiments can run on the same format.

@dataclass(frozen=True)
class Document:
    # doc_id is the dataset id for the retrieved document.
    doc_id: str
    # text is the full text that can be retrieved and passed to the LLM.
    text: str


@dataclass(frozen=True)
class Query:
    # query_id is the dataset id for the question/query.
    query_id: str
    # text is the actual user/search question.
    text: str
    # relevant_doc_ids are the gold documents from the dataset.
    relevant_doc_ids: frozenset[str]
    # reference_answer is used by our proxy F1/coverage evaluation.
    reference_answer: str


def load_documents(path: Path) -> list[Document]:
    # Read a JSONL file where each line is one document.
    # Expected shape:
    # {"doc_id": "...", "text": "..."}
    documents = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            documents.append(Document(doc_id=row["doc_id"], text=row["text"]))
    return documents


def load_queries(path: Path) -> list[Query]:
    # Read a JSONL file where each line is one query.
    # Expected shape:
    # {"query_id": "...", "text": "...", "relevant_doc_ids": [...], "reference_answer": "..."}
    queries = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            queries.append(
                Query(
                    query_id=row["query_id"],
                    text=row["text"],
                    relevant_doc_ids=frozenset(row["relevant_doc_ids"]),
                    reference_answer=row["reference_answer"],
                )
            )
    return queries
