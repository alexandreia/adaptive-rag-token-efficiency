"""
Main LLM experiment and Safe Adaptive Context model.

This is the most important file in the GitHub project.

It connects the retrieval system to an actual LLM through an OpenAI-compatible
API. For our local experiments, that API is Ollama running Mistral.

What this file does:

1. Builds prompts from retrieved documents.
2. Compresses documents into evidence spans when needed.
3. Calls the LLM or runs a dry-run fake answer.
4. Records provider token counts when the model returns them.
5. Computes answer quality metrics.
6. Runs fixed baselines, adaptive budgets, and Safe Adaptive Context.

The key model:

    answer_aware_fallback_run(...)

Safe Adaptive Context works like this:

1. Choose an adaptive document budget.
2. Try those documents as compact evidence.
3. Score whether the answer looks weak or unsupported.
4. If weak, retry the same documents without compression.
5. If still weak, expand to more documents.
6. Count the full cost, including every attempt.

This is the real model we are evaluating, not a toy imitation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from socket import timeout as SocketTimeout
from typing import Iterable

from adaptive_retrieval.budget_experiment import _record_metrics
from adaptive_retrieval.data import Document, Query
from adaptive_retrieval.learned_budget import (
    BUDGETS,
    TrainingExample,
    build_examples,
    evaluate_learned_budget,
    extract_features,
    oracle_budget_for_query,
    split_queries,
    sufficiency_risk_score,
    summarize as summarize_retrieval_metrics,
    train_centroid_model,
)
from adaptive_retrieval.metrics import answer_coverage, ndcg_at_k, semantic_similarity, token_f1
from adaptive_retrieval.text import estimate_tokens, tokenize

PROMPT_STYLES = {"default", "concise", "anchor"}

ANSWER_AWARE_FALLBACK_MODE = "answer_aware_fallback"
SAFE_ADAPTIVE_V2_MODE = "safe_adaptive_v2"
ADAPTIVE_K_PAPER_MODE = "adaptive_k_paper"
ADAPTIVE_K_OFFICIAL_MODE = "adaptive_k_official"
LLMLINGUA2_TOP10_MODE = "llmlingua2_top10"
LLMLINGUA2_ADAPTIVE_K_OFFICIAL_MODE = "llmlingua2_adaptive_k_official"
FLARE_LITE_MODE = "flare_lite"
COVERAGE_GUIDED_ADAPTIVE_MODE = "coverage_guided_adaptive"
COVERAGE_GUIDED_ULTRA_MODE = "coverage_guided_ultra"
TASK_AWARE_COVERAGE_ULTRA_MODE = "task_aware_coverage_ultra"
ROUTED_PREDICATE_ADAPTIVE_MODE = "routed_predicate_adaptive"
GUARDED_ADAPTIVE_K_MODE = "guarded_adaptive_k"
ROUTED_GUARDED_ADAPTIVE_MODE = "routed_guarded_adaptive"
ROUTED_SAFE_GUARDED_ADAPTIVE_MODE = "routed_safe_guarded_adaptive"
GUARDED_PREDICATE_COMPACT_MODE = "guarded_predicate_compact"
DISCOURSE_PRESERVING_COMPACT_MODE = "discourse_preserving_compact"
MERGED_EVIDENCE_BRIEF_MODE = "merged_evidence_brief"
HYBRID_SAFE_ADAPTIVE_MODE = "hybrid_safe_adaptive"


def load_precomputed_rankings(
    rankings_path: Path,
    documents: list[Document],
    retriever_name: str,
) -> dict[str, list[tuple[Document, float]]]:
    doc_by_id = {doc.doc_id: doc for doc in documents}
    rankings: dict[str, list[tuple[Document, float]]] = {}
    with rankings_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("retriever") != retriever_name:
                continue
            doc_ids = json.loads(row["ranked_doc_ids"])
            scores = json.loads(row["scores"])
            ranked: list[tuple[Document, float]] = []
            for doc_id, score in zip(doc_ids, scores):
                doc = doc_by_id.get(str(doc_id))
                if doc is not None:
                    ranked.append((doc, float(score)))
            rankings[str(row["query_id"])] = ranked
    if not rankings:
        raise ValueError(f"No rankings found for retriever '{retriever_name}' in {rankings_path}")
    return rankings


def examples_from_rankings(
    queries: list[Query],
    ranked_by_query: dict[str, list[tuple[Document, float]]],
    oracle_strategy: str,
    sufficiency_ratio: float,
) -> list[TrainingExample]:
    examples = []
    for query in queries:
        if query.query_id not in ranked_by_query:
            raise ValueError(f"Missing precomputed ranking for query_id={query.query_id}")
        ranked = ranked_by_query[query.query_id]
        examples.append(
            TrainingExample(
                query_id=query.query_id,
                label=oracle_budget_for_query(
                    query,
                    ranked,
                    oracle_strategy=oracle_strategy,
                    sufficiency_ratio=sufficiency_ratio,
                ),
                features=extract_features(query, ranked),
            )
        )
    return examples

_SENTENCE_EMBEDDER = None
_LLMLINGUA2_COMPRESSOR = None


@dataclass(frozen=True)
class TaskAdaptiveContextPolicy:
    shape: str
    complexity: str
    brief_kind: str
    brief_token_budget: int
    brief_max_sentences: int
    brief_min_sentences: int
    min_prompt_tokens: int
    pack_token_budget: int
    min_sources: int


WEAK_ANSWER_PHRASES = {
    "insufficient evidence",
    "not enough evidence",
    "not enough information",
    "cannot determine",
    "can't determine",
    "not mentioned",
    "not provided",
    "not stated",
    "no information",
    "no evidence",
    "unclear",
    "unknown",
}

RISK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class LLMConfig:
    # Model name used by the remote LLM API.
    model: str = "gpt-4o-mini"
    # Temperature is kept at zero so repeated runs are easier to compare.
    temperature: float = 0.0
    # Upper bound on generated answer length. This keeps completion-token cost controlled.
    max_output_tokens: int = 220
    # Local models can be slow, especially with long fixed_10 prompts.
    request_timeout_seconds: int = 300
    # API URL for OpenAI-compatible chat-completions providers.
    api_url: str = "https://api.openai.com/v1/chat/completions"
    # Environment variable that stores the API key. Ollama can leave this unset.
    api_key_env: str = "OPENAI_API_KEY"
    # Local providers such as Ollama do not require an Authorization header.
    require_api_key: bool = True
    # Dry-run mode avoids network calls and uses a simple extractive answer instead.
    dry_run: bool = True
    # Compression controls how much of each selected document is shown to the answer model.
    compression_mode: str = "full"
    # Prompt style controls the answer format without changing retrieval/context selection.
    prompt_style: str = "default"
    # Final research runs should use real provider token counts.
    # If this is True and the API does not return usage, the run stops instead
    # of silently using local token estimates.
    require_provider_tokens: bool = False


@dataclass(frozen=True)
class LLMRunRow:
    # Budget mode plus compression mode, for example fixed_10_full or learned_budget_query_overlap.
    mode: str
    # Human-readable method name for reports and tables.
    method_name: str
    # Which retrieval/budgeting strategy selected documents before compression.
    budget_mode: str
    # How selected documents were shortened before being placed in the prompt.
    compression_mode: str
    # Query identifier from the dataset.
    query_id: str
    # Number of documents passed to the answer generator.
    docs_used: int
    # Estimated prompt tokens for the question plus selected context.
    prompt_tokens: int
    # Estimated/generated answer tokens.
    completion_tokens: int
    # Prompt plus completion tokens, useful as the main cost proxy.
    total_tokens: int
    # Whether token counts came from the model/API response or the local estimator.
    token_source: str
    # End-to-end answer-generation time for this strategy/query pair.
    generation_time_ms: int
    # Token-overlap F1 between the generated answer and the dataset reference.
    answer_f1: float
    # Reference-answer term coverage by the generated answer.
    answer_coverage: float
    # Meaning similarity between generated answer and reference answer.
    semantic_similarity: float
    # nDCG@10 of the documents that enter the prompt/context.
    ndcg_at_10: float
    # MRR@10 of the documents that enter the prompt/context.
    mrr_at_10: float
    # Document ids selected for the prompt, stored as JSON for easy inspection.
    selected_doc_ids: str
    # The actual answer text generated by the dry-run extractor or LLM.
    answer: str
    # Whether an answer-aware method had to expand after its compact first pass.
    fallback_used: bool = False
    # Human-readable reason for expansion; empty for normal one-shot modes.
    fallback_reason: str = ""
    # Token cost of the compact first answer attempt, if the mode uses one.
    first_pass_tokens: int = 0
    # Extra token cost spent by fallback expansion, if any.
    fallback_tokens: int = 0


@dataclass(frozen=True)
class GeneratedAnswer:
    # Text returned by the dry-run extractor or LLM.
    text: str
    # Prompt tokens, either from provider usage or the local estimator.
    prompt_tokens: int
    # Completion tokens, either from provider usage or the local estimator.
    completion_tokens: int
    # Total tokens, either from provider usage or prompt + completion estimates.
    total_tokens: int
    # "provider" means the model/API reported usage; "estimated" means local approximation.
    token_source: str
    # Time spent generating this answer.
    generation_time_ms: int


def evidence_candidate_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        if cleaned and not cleaned.endswith(":"):
            lines.append(cleaned)
    return lines or split_sentences(text)


def anchor_score(query: Query, candidate: str, doc_index: int) -> float:
    query_tokens = tokenize(query.text)
    query_terms = set(query_tokens)
    candidate_tokens = tokenize(candidate)
    candidate_terms = set(candidate_tokens)
    if not query_terms or not candidate_terms:
        return 0.0

    overlap = len(query_terms & candidate_terms) / len(query_terms)
    phrase_overlap = phrase_overlap_score(query_tokens, candidate_tokens)
    density = len(query_terms & candidate_terms) / len(candidate_terms)
    doc_position_bonus = 1 / (1 + doc_index)
    return (0.45 * overlap) + (0.35 * phrase_overlap) + (0.15 * density) + (0.05 * doc_position_bonus)


def select_anchor_evidence(query: Query, selected_docs: list[Document]) -> str:
    best_score = float("-inf")
    best_candidate = ""
    for doc_index, doc in enumerate(selected_docs):
        for candidate in evidence_candidate_lines(doc.text):
            score = anchor_score(query, candidate, doc_index)
            if score > best_score:
                best_score = score
                best_candidate = candidate
    return best_candidate or (selected_docs[0].text if selected_docs else "No evidence provided.")


def build_prompt(query: Query, selected_docs: list[Document], prompt_style: str = "default") -> str:
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    # Each document is numbered and tagged with its doc id so a model can ground its answer.
    context_blocks = [
        f"[Document {index} | {doc.doc_id}]\n{doc.text}"
        for index, doc in enumerate(selected_docs, start=1)
    ]
    context = "\n\n".join(context_blocks)

    if prompt_style == "concise":
        return (
            "Use only the evidence below to answer the question.\n"
            "Write only the final answer.\n"
            "Use as few words as possible, usually 1 to 5 words.\n"
            "Use the same key terms as the evidence when possible.\n"
            "Do not explain your reasoning.\n"
            "Do not repeat the question.\n"
            "If the evidence does not answer the question, write exactly: insufficient evidence.\n\n"
            f"Question:\n{query.text}\n\n"
            f"Evidence:\n{context}\n\n"
            "Short answer:"
        )

    if prompt_style == "anchor":
        anchor = select_anchor_evidence(query, selected_docs)
        return (
            "Use only the evidence below to answer the question.\n"
            "The key evidence is the most important span. Use it as the main anchor for your answer.\n"
            "Use the supporting evidence only if it helps clarify or confirm the key evidence.\n"
            "Give one concise answer sentence. Do not add background information.\n"
            "If the evidence is insufficient, write exactly: insufficient evidence.\n\n"
            f"Question:\n{query.text}\n\n"
            f"Key evidence:\n{anchor}\n\n"
            f"Supporting evidence:\n{context}\n\n"
            "Answer:"
        )

    # The default prompt is intentionally plain: the experiment should test
    # context budgeting, not prompt-engineering tricks.
    return (
        "Answer the question using only the provided documents. "
        "If the documents do not contain enough evidence, say that the evidence is insufficient.\n\n"
        f"Question:\n{query.text}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer:"
    )


def llmlingua2_compressor():
    global _LLMLINGUA2_COMPRESSOR
    if _LLMLINGUA2_COMPRESSOR is not None:
        return _LLMLINGUA2_COMPRESSOR
    try:
        import torch
        from llmlingua import PromptCompressor
    except ImportError as error:
        raise RuntimeError(
            "LLMLingua-2 baseline requested, but the llmlingua package is not installed. "
            "Install it with: python -m pip install llmlingua"
        ) from error

    model_name = os.environ.get(
        "LLMLINGUA2_MODEL",
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    )
    device_map = os.environ.get("LLMLINGUA2_DEVICE")
    if not device_map:
        if torch.cuda.is_available():
            device_map = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"
    _LLMLINGUA2_COMPRESSOR = PromptCompressor(
        model_name=model_name,
        device_map=device_map,
        use_llmlingua2=True,
    )
    return _LLMLINGUA2_COMPRESSOR


def build_prompt_with_context(query: Query, context: str, prompt_style: str) -> str:
    """Build a generation prompt from an already prepared context string."""
    if prompt_style == "concise":
        return (
            "Use only the evidence below to answer the question.\n"
            "Write only the final answer.\n"
            "Use as few words as possible, usually 1 to 5 words.\n"
            "Use the same key terms as the evidence when possible.\n"
            "Do not explain your reasoning.\n"
            "Do not repeat the question.\n"
            "If the evidence does not answer the question, write exactly: insufficient evidence.\n\n"
            f"Question:\n{query.text}\n\n"
            f"Evidence:\n{context}\n\n"
            "Short answer:"
        )

    return (
        "Answer the question using only the provided documents. "
        "If the documents do not contain enough evidence, say that the evidence is insufficient.\n\n"
        f"Question:\n{query.text}\n\n"
        f"Documents:\n{context}\n\n"
        "Answer:"
    )


def llmlingua2_context_blocks(selected_docs: list[Document], max_block_tokens: int = 360) -> list[str]:
    """Split retrieved documents into compressor-friendly evidence blocks."""
    blocks: list[str] = []
    for index, doc in enumerate(selected_docs, start=1):
        header = f"[Document {index} | {doc.doc_id}]"
        sentences = split_sentences(doc.text) or [doc.text]
        current = header
        for sentence in sentences:
            candidate = f"{current}\n{sentence}" if current != header else f"{header}\n{sentence}"
            if estimate_tokens(candidate) > max_block_tokens and current != header:
                blocks.append(current)
                current = f"{header}\n{sentence}"
            else:
                current = candidate
        if current.strip() and current != header:
            blocks.append(current)
    return blocks or ["No retrieved evidence."]


def compress_context_with_llmlingua2(query: Query, selected_docs: list[Document]) -> str:
    rate = float(os.environ.get("LLMLINGUA2_RATE", "0.33"))
    compressor = llmlingua2_compressor()
    context_blocks = llmlingua2_context_blocks(selected_docs)
    result = compressor.compress_prompt(
        context_blocks,
        question=query.text,
        rate=rate,
        iterative_size=128,
        concate_question=False,
        keep_first_sentence=1,
        force_context_number=min(1, len(context_blocks)),
        force_tokens=["\n", "?", "[", "]", ":", "|"],
        force_reserve_digit=True,
        chunk_end_tokens=[".", "\n"],
    )
    compressed_prompt = result.get("compressed_prompt")
    if not isinstance(compressed_prompt, str) or not compressed_prompt.strip():
        compressed_list = result.get("compressed_prompt_list")
        if isinstance(compressed_list, list):
            compressed_prompt = "\n\n".join(str(item) for item in compressed_list if str(item).strip())
    if not isinstance(compressed_prompt, str) or not compressed_prompt.strip():
        raise RuntimeError("LLMLingua-2 returned empty compressed context.")
    return compressed_prompt


def split_sentences(text: str) -> list[str]:
    # A small sentence splitter is enough for the first compression experiment.
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    return sentences or [text]


def sentence_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if len(tokens) < size:
        return set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def phrase_overlap_score(query_tokens: list[str], sentence_tokens: list[str]) -> float:
    query_bigrams = ngrams(query_tokens, 2)
    query_trigrams = ngrams(query_tokens, 3)
    if not query_bigrams and not query_trigrams:
        return 0.0

    sentence_bigrams = ngrams(sentence_tokens, 2)
    sentence_trigrams = ngrams(sentence_tokens, 3)
    bigram_overlap = len(query_bigrams & sentence_bigrams) / len(query_bigrams) if query_bigrams else 0.0
    trigram_overlap = len(query_trigrams & sentence_trigrams) / len(query_trigrams) if query_trigrams else 0.0
    return (0.40 * bigram_overlap) + (0.60 * trigram_overlap)


def evidence_sentences(query: Query, doc: Document, max_sentences: int = 4) -> list[str]:
    sentences = split_sentences(doc.text)
    query_tokens = tokenize(query.text)
    query_terms = set(query_tokens)
    if not query_terms:
        return sentences[:max_sentences]

    document_term_counts = Counter(tokenize(doc.text))
    scored_sentences = []
    selected_term_sets: list[set[str]] = []
    for index, sentence in enumerate(sentences):
        sentence_tokens = tokenize(sentence)
        sentence_terms = set(sentence_tokens)
        if not sentence_terms:
            continue
        overlap_terms = query_terms & sentence_terms
        exact_overlap = len(overlap_terms) / len(query_terms)
        rare_overlap = sum(1 / math.sqrt(document_term_counts[term]) for term in overlap_terms)
        rare_overlap = rare_overlap / len(query_terms)
        phrase_overlap = phrase_overlap_score(query_tokens, sentence_tokens)
        position_bonus = 1 / (1 + index)
        density = len(overlap_terms) / len(sentence_terms)
        score = (
            (0.40 * exact_overlap)
            + (0.25 * rare_overlap)
            + (0.20 * phrase_overlap)
            + (0.10 * density)
            + (0.05 * position_bonus)
        )
        scored_sentences.append((score, index, sentence, sentence_terms))

    if not scored_sentences:
        return sentences[:max_sentences]

    selected = []
    for score, index, sentence, sentence_terms in sorted(scored_sentences, reverse=True):
        if score <= 0 and selected:
            continue
        # Prefer sentences that add at least some new query evidence. This keeps
        # compression focused instead of repeating similar sentences.
        already_covered = set().union(*selected_term_sets) if selected_term_sets else set()
        new_query_terms = (query_terms & sentence_terms) - already_covered
        if selected and not new_query_terms and len(selected) < max_sentences:
            continue
        selected.append((index, sentence))
        selected_term_sets.append(sentence_terms)
        if len(selected) >= max_sentences:
            break

    if not selected:
        return sentences[:1]
    return [sentence for _index, sentence in sorted(selected)]


def balanced_evidence_sentences(query: Query, doc: Document) -> list[str]:
    sentences = split_sentences(doc.text)
    evidence = evidence_sentences(query, doc, max_sentences=6)
    if not evidence:
        return sentences[:3]

    # Keep the opening sentence when possible because scientific abstracts often
    # define the topic or population there, while later sentences carry evidence.
    selected = []
    if sentences and sentences[0] not in evidence:
        selected.append(sentences[0])
    selected.extend(evidence)
    deduped = []
    for sentence in selected:
        if sentence not in deduped:
            deduped.append(sentence)
    return deduped[:7]


def ngram_neighbor_evidence_sentences(query: Query, doc: Document) -> list[str]:
    sentences = split_sentences(doc.text)
    core_evidence = set(evidence_sentences(query, doc, max_sentences=5))
    if not core_evidence:
        return sentences[:4]

    selected_indexes = set()
    for index, sentence in enumerate(sentences):
        if sentence in core_evidence:
            selected_indexes.update({index - 1, index, index + 1})

    valid_indexes = sorted(index for index in selected_indexes if 0 <= index < len(sentences))
    selected = [sentences[index] for index in valid_indexes]

    # Keep the beginning of the abstract/document when it is not already covered;
    # this often gives the LLM the missing topic definition for scientific text.
    if sentences and sentences[0] not in selected:
        selected.insert(0, sentences[0])

    deduped = []
    for sentence in selected:
        if sentence not in deduped:
            deduped.append(sentence)
    return deduped[:10]


def sentence_embedder():
    global _SENTENCE_EMBEDDER
    if _SENTENCE_EMBEDDER is not None:
        return _SENTENCE_EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer

        _SENTENCE_EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        _SENTENCE_EMBEDDER = False
    return _SENTENCE_EMBEDDER


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def hybrid_evidence_sentences(query: Query, doc: Document, max_sentences: int = 7) -> list[str]:
    """Select evidence with lexical, semantic, signal, and novelty scoring."""
    sentences = split_sentences(doc.text)
    if len(sentences) <= 2:
        return sentences

    query_tokens = tokenize(query.text)
    query_terms = content_word_set(query.text)
    signal_terms = query_signal_terms(query)
    lexical_scores = []
    for index, sentence in enumerate(sentences):
        sentence_tokens = tokenize(sentence)
        sentence_terms = content_word_set(sentence)
        exact_overlap = overlap_ratio(query_terms, sentence_terms)
        signal_overlap = overlap_ratio(signal_terms, sentence_terms)
        phrase_overlap = phrase_overlap_score(query_tokens, sentence_tokens)
        density = len(query_terms & sentence_terms) / len(sentence_terms) if sentence_terms else 0.0
        position_bonus = 1 / (1 + index)
        entity_bonus = 0.15 if capitalized_terms(query.text) & capitalized_terms(sentence) else 0.0
        number_bonus = 0.15 if has_number(query.text) and has_number(sentence) else 0.0
        lexical_scores.append(
            (0.34 * exact_overlap)
            + (0.24 * signal_overlap)
            + (0.20 * phrase_overlap)
            + (0.08 * density)
            + (0.04 * position_bonus)
            + entity_bonus
            + number_bonus
        )

    semantic_scores = [0.0 for _sentence in sentences]
    embedder = sentence_embedder()
    if embedder:
        try:
            embeddings = embedder.encode([query.text, *sentences], normalize_embeddings=True)
            query_embedding = embeddings[0]
            semantic_scores = [
                float(cosine_similarity(query_embedding, sentence_embedding))
                for sentence_embedding in embeddings[1:]
            ]
        except Exception:
            semantic_scores = [0.0 for _sentence in sentences]

    candidates = []
    for index, sentence in enumerate(sentences):
        sentence_terms = content_word_set(sentence)
        score = (0.62 * lexical_scores[index]) + (0.38 * max(0.0, semantic_scores[index]))
        candidates.append(
            {
                "index": index,
                "sentence": sentence,
                "terms": sentence_terms,
                "score": score,
            }
        )

    selected = []
    selected_terms: list[set[str]] = []
    covered_signals: set[str] = set()
    for item in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if item["score"] <= 0 and selected:
            continue
        if any(sentence_similarity(item["terms"], terms) > 0.78 for terms in selected_terms):
            continue

        new_signals = (item["terms"] & signal_terms) - covered_signals
        if selected and not new_signals and item["score"] < 0.30:
            continue

        selected.append(item)
        selected_terms.append(item["terms"])
        covered_signals.update(item["terms"] & signal_terms)

        if len(selected) >= max_sentences:
            break
        if overlap_ratio(signal_terms, covered_signals) >= 0.70 and len(selected) >= 3:
            break

    if not selected:
        return ngram_neighbor_evidence_sentences(query, doc)

    selected_indexes = set()
    for item in selected:
        selected_indexes.update({item["index"] - 1, item["index"], item["index"] + 1})
    valid_indexes = sorted(index for index in selected_indexes if 0 <= index < len(sentences))
    evidence = [sentences[index] for index in valid_indexes]

    if sentences and sentences[0] not in evidence:
        evidence.insert(0, sentences[0])

    deduped = []
    for sentence in evidence:
        if sentence not in deduped:
            deduped.append(sentence)
    return deduped[:10]


def compress_document(query: Query, doc: Document, compression_mode: str) -> Document:
    # full is the no-compression baseline.
    if compression_mode == "full":
        return doc

    sentences = split_sentences(doc.text)
    if compression_mode == "first_sentence":
        compressed_text = sentences[0]
    elif compression_mode == "first_2_sentences":
        compressed_text = " ".join(sentences[:2])
    elif compression_mode == "query_overlap":
        query_terms = set(re.findall(r"[a-z0-9]+", query.text.lower()))
        selected_sentences = [
            sentence
            for sentence in sentences
            if query_terms & set(re.findall(r"[a-z0-9]+", sentence.lower()))
        ]
        compressed_text = " ".join(selected_sentences[:3] or sentences[:1])
    elif compression_mode == "evidence":
        evidence = evidence_sentences(query, doc)
        compressed_text = "Evidence snippets:\n" + "\n".join(f"- {sentence}" for sentence in evidence)
    elif compression_mode == "evidence_balanced":
        evidence = balanced_evidence_sentences(query, doc)
        compressed_text = "Focused evidence:\n" + "\n".join(f"- {sentence}" for sentence in evidence)
    elif compression_mode == "evidence_ngram_neighbors":
        evidence = ngram_neighbor_evidence_sentences(query, doc)
        compressed_text = "Phrase-aware evidence spans:\n" + "\n".join(f"- {sentence}" for sentence in evidence)
    elif compression_mode == "evidence_hybrid":
        evidence = hybrid_evidence_sentences(query, doc)
        compressed_text = "Hybrid evidence spans:\n" + "\n".join(f"- {sentence}" for sentence in evidence)
    else:
        raise ValueError(f"Unknown compression mode: {compression_mode}")

    return Document(doc_id=doc.doc_id, text=compressed_text)


def compress_documents(query: Query, selected_docs: list[Document], compression_mode: str) -> list[Document]:
    # Compression is applied after budget selection, so the experiment can test both levers:
    # how many documents are selected and how much text from each document is kept.
    return [compress_document(query, doc, compression_mode) for doc in selected_docs]


def estimate_prompt_tokens(query: Query, selected_docs: list[Document], prompt_style: str = "default") -> int:
    # This uses the project's existing rough token estimator so cost numbers are consistent
    # with the retrieval-only experiments.
    return estimate_tokens(build_prompt(query, selected_docs, prompt_style))


def dry_run_answer(query: Query, selected_docs: list[Document]) -> str:
    # Dry-run mode lets you test the complete experiment without calling an LLM.
    # It returns the shortest selected document that overlaps with the reference answer;
    # if no selected document overlaps, it falls back to the first selected document.
    if not selected_docs:
        return "The evidence is insufficient."

    reference_terms = set(query.reference_answer.lower().split())
    best_doc = selected_docs[0]
    best_overlap = -1
    for doc in selected_docs:
        overlap = len(reference_terms & set(doc.text.lower().split()))
        if overlap > best_overlap:
            best_doc = doc
            best_overlap = overlap
    return best_doc.text


def call_openai_chat(prompt: str, config: LLMConfig) -> GeneratedAnswer:
    # The API key is read at call time so users can set it in the shell before running.
    # Local OpenAI-compatible servers such as Ollama can skip this entirely.
    api_key = os.environ.get(config.api_key_env)
    if config.require_api_key and not api_key:
        raise RuntimeError(f"Set {config.api_key_env} before running without --dry-run.")

    # This body follows the OpenAI-compatible chat completions shape.
    request_body = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful retrieval-augmented QA assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.perf_counter()
    payload = None
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            chat_completions_url(config.api_url),
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        if api_key:
            request.add_unredirected_header("Authorization", f"Bearer {api_key}")

        try:
            with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            if error.code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                retry_after = error.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds = int(retry_after)
                else:
                    wait_seconds = min(65, 10 * attempt) if error.code == 429 else min(45, 5 * attempt)
                error_label = "Rate limit" if error.code == 429 else f"Provider error {error.code}"
                print(
                    f"{error_label}; waiting {wait_seconds}s before retry "
                    f"({attempt}/{max_attempts})."
                )
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"LLM API request failed: {error.code} {details}") from error
        except TimeoutError as error:
            raise RuntimeError(
                f"LLM API request timed out after {config.request_timeout_seconds} seconds. "
                "For local Ollama runs, try a smaller --max-eval-queries value, fewer modes, "
                "a smaller model, or a larger --request-timeout-seconds value."
            ) from error
        except SocketTimeout as error:
            raise RuntimeError(
                f"LLM API request timed out after {config.request_timeout_seconds} seconds. "
                "For local Ollama runs, try a smaller --max-eval-queries value, fewer modes, "
                "a smaller model, or a larger --request-timeout-seconds value."
            ) from error

    if payload is None:
        raise RuntimeError("LLM API request failed after repeated rate-limit retries.")

    # Chat completions return the answer in choices[0].message.content.
    answer = payload["choices"][0]["message"]["content"].strip()
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        if not isinstance(total_tokens, int):
            total_tokens = prompt_tokens + completion_tokens
        token_source = "provider"
    else:
        if config.require_provider_tokens:
            raise RuntimeError(
                "The LLM provider did not return token usage, but require_provider_tokens=True. "
                "For final project results, use a provider/model that returns prompt_tokens and "
                "completion_tokens, or rerun without the provider-token requirement and clearly "
                "label tokens as estimated."
            )
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(answer)
        total_tokens = prompt_tokens + completion_tokens
        token_source = "estimated"

    return GeneratedAnswer(
        text=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        token_source=token_source,
        generation_time_ms=elapsed_ms,
    )


def chat_completions_url(api_url: str) -> str:
    # Accept either a full endpoint or a base URL. This makes Ollama convenient:
    # --api-url http://localhost:11434/v1 becomes /v1/chat/completions.
    cleaned_url = api_url.rstrip("/")
    if cleaned_url.endswith("/chat/completions"):
        return cleaned_url
    return f"{cleaned_url}/chat/completions"


def generate_answer(query: Query, selected_docs: list[Document], config: LLMConfig) -> GeneratedAnswer:
    # This single function makes it easy to switch between dry-run and real LLM mode.
    if config.dry_run:
        started = time.perf_counter()
        answer = dry_run_answer(query, selected_docs)
        prompt_tokens = estimate_prompt_tokens(query, selected_docs, config.prompt_style)
        completion_tokens = estimate_tokens(answer)
        return GeneratedAnswer(
            text=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            token_source="estimated",
            generation_time_ms=round((time.perf_counter() - started) * 1000),
        )
    return call_openai_chat(build_prompt(query, selected_docs, config.prompt_style), config)


def generate_answer_from_prompt(
    query: Query,
    selected_docs: list[Document],
    prompt: str,
    config: LLMConfig,
) -> GeneratedAnswer:
    if config.dry_run:
        started = time.perf_counter()
        answer = dry_run_answer(query, selected_docs)
        prompt_tokens = estimate_tokens(prompt)
        completion_tokens = estimate_tokens(answer)
        return GeneratedAnswer(
            text=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            token_source="estimated",
            generation_time_ms=round((time.perf_counter() - started) * 1000),
        )
    return call_openai_chat(prompt, config)


def generate_llmlingua2_answer(
    query: Query,
    selected_docs: list[Document],
    config: LLMConfig,
) -> GeneratedAnswer:
    prompt = build_prompt(query, selected_docs, config.prompt_style)
    if config.dry_run:
        compressed_prompt = prompt
    else:
        compressed_context = compress_context_with_llmlingua2(query, selected_docs)
        compressed_prompt = build_prompt_with_context(query, compressed_context, config.prompt_style)
    return generate_answer_from_prompt(query, selected_docs, compressed_prompt, config)


def config_for_answer_call(config: LLMConfig, compression_mode: str, prompt_style: str | None = None) -> LLMConfig:
    # LLMConfig is frozen, so this helper creates a modified copy for one answer call.
    return LLMConfig(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        request_timeout_seconds=config.request_timeout_seconds,
        api_url=config.api_url,
        api_key_env=config.api_key_env,
        require_api_key=config.require_api_key,
        dry_run=config.dry_run,
        compression_mode=compression_mode,
        prompt_style=prompt_style or config.prompt_style,
        require_provider_tokens=config.require_provider_tokens,
    )


def content_word_set(text: str) -> set[str]:
    # Risk checks should ignore common function words and focus on meaningful overlap.
    return {term for term in tokenize(text) if len(term) > 2 and term not in RISK_STOPWORDS}


def overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def answer_risk_score(query: Query, answer: str, selected_docs: list[Document]) -> tuple[int, list[str]]:
    """Score whether an answer looks risky using only runtime information.

    This does not look at the gold answer. It only uses:
    - the query
    - the selected context
    - the answer generated by the LLM

    The score is simple on purpose:
    more weak signals = more reason to expand context.
    """
    cleaned_answer = answer.strip()
    lowered_answer = cleaned_answer.lower()
    if not cleaned_answer:
        return 10, ["empty_answer"]

    risk = 0
    reasons = []
    for phrase in WEAK_ANSWER_PHRASES:
        if phrase in lowered_answer:
            risk += 3
            reasons.append(f"weak_phrase:{phrase}")
            break

    answer_terms = content_word_set(cleaned_answer)
    query_terms = content_word_set(query.text)
    context_terms = content_word_set(" ".join(doc.text for doc in selected_docs))

    # If the answer uses terms that barely appear in the provided evidence, it may
    # be hallucinating or failing to anchor on the compact snippets.
    answer_context_overlap = overlap_ratio(answer_terms, context_terms)
    if answer_context_overlap < 0.08:
        risk += 3
        reasons.append("low_answer_context_overlap")

    # If the selected context does not cover much of the query vocabulary, the
    # compressed evidence may be missing part of the question.
    context_query_coverage = overlap_ratio(query_terms, context_terms)
    if context_query_coverage < 0.35:
        risk += 2
        reasons.append("low_context_query_coverage")

    # Short answers are common in QA datasets such as HotpotQA: names, places,
    # years, and yes/no answers can be correct. Only treat a short answer as risky
    # when it also has weak grounding in the selected context.
    if len(answer_terms) < 5 and answer_context_overlap < 0.20:
        risk += 2
        reasons.append("short_answer_with_weak_context_overlap")

    # If the answer barely touches the query vocabulary, it may be too generic.
    answer_query_overlap = overlap_ratio(query_terms, answer_terms)
    if answer_query_overlap < 0.04:
        risk += 1
        reasons.append("low_answer_query_overlap")

    # Very long answers are often bad for short-answer QA datasets. This is only
    # a soft signal because scientific datasets may need longer answers.
    if len(answer_terms) > 24 and answer_context_overlap < 0.90:
        risk += 1
        reasons.append("verbose_answer")

    return risk, reasons


def answer_needs_fallback(query: Query, answer: str, selected_docs: list[Document]) -> tuple[bool, str]:
    """Return whether the answer is risky enough to expand context."""
    risk, reasons = answer_risk_score(query, answer, selected_docs)
    if risk >= 3:
        return True, ",".join(reasons)

    return False, ""


def short_answer_pre_generation_budget(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
) -> int:
    """Choose one concise-QA budget before calling the LLM.

    This is cheaper than answer-level fallback because it avoids paying for a
    failed first generation. For short-answer QA, keep full documents, but
    slightly reduce budget 8 to budget 7 when top-7 already looks sufficient.
    """
    budget = min(max(sequential_budget, 3), 10)

    if budget == 8:
        features = extract_features(query, ranked_docs)
        risk_7 = sufficiency_risk_score(query, features, ranked_docs, 7)
        if risk_7 <= 0.50:
            return 7

    return budget


def short_answer_needs_fallback(query: Query, answer: str, selected_docs: list[Document]) -> tuple[bool, str]:
    """Risk check for short exact answers.

    Keep this deliberately soft. Strong grounding checks after generation can
    improve quality, but they require a second LLM call and can remove token
    savings. Here we only expand for clearly broken answer shapes.
    """
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        return True, "empty_answer"

    answer_terms = content_word_set(cleaned_answer)
    context_terms = content_word_set(" ".join(doc.text for doc in selected_docs))
    answer_context_overlap = overlap_ratio(answer_terms, context_terms)

    if len(answer_terms) > 18 and answer_context_overlap < 0.80:
        return True, "verbose_short_answer"

    return False, ""


def title_from_doc_id(doc_id: str) -> str:
    """Return the readable title part used by converted paragraph datasets."""
    return doc_id.split("::")[-1]


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def capitalized_terms(text: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][a-zA-Z0-9'’-]+\b", text))


def sentence_pack_score(
    query: Query,
    doc: Document,
    sentence: str,
    doc_score: float,
    top_score: float,
) -> float:
    """Score a sentence for short-answer QA packing."""
    query_terms = content_word_set(query.text)
    sentence_terms = content_word_set(sentence)
    title_terms = content_word_set(title_from_doc_id(doc.doc_id))

    overlap = len(query_terms & sentence_terms)
    title_overlap = len(query_terms & title_terms)
    entity_bonus = min(3, len(capitalized_terms(sentence))) * 0.15
    number_bonus = 0.25 if has_number(sentence) else 0.0
    relevance = doc_score / top_score if top_score > 0 else 0.0

    return (2.0 * relevance) + (1.2 * overlap) + (0.8 * title_overlap) + entity_bonus + number_bonus


def grouped_sentence_pack(units: list[tuple[str, str, str]]) -> list[Document]:
    """Group selected sentence units back into compact pseudo-documents."""
    by_doc: dict[str, list[str]] = {}
    order: list[str] = []

    for doc_id, title, sentence in units:
        if doc_id not in by_doc:
            by_doc[doc_id] = [title]
            order.append(doc_id)
        if sentence not in by_doc[doc_id]:
            by_doc[doc_id].append(sentence)

    return [
        Document(doc_id=doc_id, text=". ".join(by_doc[doc_id]))
        for doc_id in order
    ]


def token_budget_greedy_pack(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    token_budget: int = 950,
    pool_size: int = 10,
    force_first_per_doc: bool = False,
) -> list[Document]:
    """Build a sentence-level context pack for short-answer / multi-hop QA.

    This keeps broad top-10 retrieval coverage available and compresses inside
    that pool. Sentences are selected by utility per token.
    """
    pool = ranked_docs[:pool_size]
    top_score = pool[0][1] if pool and pool[0][1] > 0 else 1.0
    candidates = []

    for rank, (doc, doc_score) in enumerate(pool, start=1):
        title = title_from_doc_id(doc.doc_id)
        for index, sentence in enumerate(split_sentences(doc.text)):
            score = sentence_pack_score(query, doc, sentence, doc_score, top_score)
            token_len = max(1, estimate_tokens(sentence))
            utility = score / (token_len ** 0.55)

            if index == 0:
                utility += 0.35

            candidates.append({
                "doc": doc,
                "title": title,
                "sentence": sentence,
                "utility": utility,
                "rank": rank,
                "index": index,
                "tokens": token_len,
            })

    selected = []
    used = set()
    total = 0

    if force_first_per_doc:
        for item in candidates:
            key = (item["doc"].doc_id, item["index"])
            if item["index"] == 0 and key not in used:
                cost = estimate_tokens(item["title"]) + item["tokens"] + 4
                if total + cost <= token_budget:
                    selected.append(item)
                    used.add(key)
                    total += cost

    for item in sorted(candidates, key=lambda item: item["utility"], reverse=True):
        key = (item["doc"].doc_id, item["index"])
        if key in used:
            continue

        cost = estimate_tokens(item["title"]) + item["tokens"] + 4
        if total + cost > token_budget:
            continue

        selected.append(item)
        used.add(key)
        total += cost

    selected.sort(key=lambda item: (item["rank"], item["index"]))
    units = [
        (item["doc"].doc_id, item["title"], item["sentence"])
        for item in selected
    ]
    return grouped_sentence_pack(units)


def merged_evidence_sentence_score(
    query: Query,
    doc: Document,
    sentence: str,
    doc_score: float,
    top_score: float,
    rank: int,
) -> float:
    query_tokens = tokenize(query.text)
    query_terms = content_word_set(query.text)
    sentence_tokens = tokenize(sentence)
    sentence_terms = content_word_set(sentence)
    if not sentence_terms:
        return 0.0

    overlap = overlap_ratio(query_terms, sentence_terms)
    phrase_overlap = phrase_overlap_score(query_tokens, sentence_tokens)
    title_overlap = overlap_ratio(query_terms, content_word_set(title_from_doc_id(doc.doc_id)))
    entity_bonus = min(3, len(capitalized_terms(sentence) & capitalized_terms(query.text))) * 0.12
    number_bonus = 0.18 if has_number(query.text) and has_number(sentence) else 0.0
    retrieval_relevance = doc_score / top_score if top_score > 0 else 0.0
    rank_bonus = 1 / (1 + rank)

    return (
        (1.50 * overlap)
        + (0.85 * phrase_overlap)
        + (0.55 * retrieval_relevance)
        + (0.35 * title_overlap)
        + entity_bonus
        + number_bonus
        + (0.15 * rank_bonus)
    )


def sentence_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


NATIONALITY_VALUE_TERMS = {
    "american",
    "british",
    "english",
    "scottish",
    "welsh",
    "irish",
    "canadian",
    "australian",
    "french",
    "german",
    "italian",
    "spanish",
    "swedish",
    "norwegian",
    "danish",
    "dutch",
    "belgian",
    "swiss",
    "austrian",
    "polish",
    "russian",
    "ukrainian",
    "chinese",
    "japanese",
    "korean",
    "indian",
    "pakistani",
    "iranian",
    "iraqi",
    "israeli",
    "egyptian",
    "turkish",
    "greek",
    "mexican",
    "brazilian",
    "argentine",
    "south",
    "african",
}


def is_title_fragment(doc: Document, sentence: str) -> bool:
    """Detect snippets that only repeat the page title and carry no evidence."""
    title_terms = content_word_set(title_from_doc_id(doc.doc_id))
    sentence_terms = content_word_set(sentence)
    if not sentence_terms:
        return True
    return sentence_terms <= title_terms or len(sentence_terms) <= 2


def is_non_predicate_fragment(sentence: str) -> bool:
    """Detect short name/list fragments that do not state a relation."""
    sentence_terms = content_word_set(sentence)
    lowered = sentence.lower()
    has_predicate = re.search(
        r"\b(is|was|are|were|born|became|served|played|located|founded|created|directed|written|produced|released)\b",
        lowered,
    )
    if has_predicate or has_number(sentence):
        return False
    return len(sentence_terms) <= 5


def is_descriptive_answer_sentence(query: Query, sentence: str) -> bool:
    lowered = sentence.lower()
    if re.search(r"\b(is|was|are|were|born|became|served|played|located|founded|created|directed)\b", lowered):
        return True
    if "nationality" in query.text.lower() and (content_word_set(sentence) & NATIONALITY_VALUE_TERMS):
        return True
    if has_number(query.text) and has_number(sentence):
        return True
    return False


def is_comparison_query(query: Query) -> bool:
    lowered = query.text.lower()
    return any(
        cue in lowered
        for cue in [
            "same",
            "both",
            "older",
            "younger",
            "larger",
            "smaller",
            "more",
            "less",
            "between",
            "compared",
            "which",
        ]
    )


def needs_bridge_evidence(query: Query, prompt_style: str) -> bool:
    """Detect questions where isolated answer candidates are risky.

    These are cases where the model usually needs a small predicate-bearing
    sentence for each anchor entity, not just a likely answer string.
    """
    if prompt_style != "concise":
        return False
    text = query.text.lower()
    if is_comparison_query(query):
        return True
    if " and " in text and len(capitalized_terms(query.text)) >= 2:
        return True
    compositional_cues = [
        "director",
        "author",
        "writer",
        "producer",
        "founder",
        "creator",
        "member",
        "based",
        "located",
        "born",
        "city",
        "country",
        "organization",
    ]
    if any(cue in text for cue in compositional_cues) and (
        len(capitalized_terms(query.text)) >= 1 or '"' in query.text or "'" in query.text
    ):
        return True
    return False


def is_conversational_query(query: Query) -> bool:
    text = query.text.lower()
    if "conversation so far:" in text or " current question:" in text:
        return True
    if re.search(r"\b(q:|a:)\b", text):
        return True
    pronouns = {"he", "she", "it", "they", "him", "her", "them", "his", "hers", "their", "its", "that", "there"}
    return bool(content_word_set(query.text) & pronouns)


def first_descriptive_sentence(query: Query, doc: Document) -> tuple[int, str]:
    sentences = split_sentences(doc.text)
    if not sentences:
        return 0, doc.text
    for index, sentence in enumerate(sentences):
        if not is_title_fragment(doc, sentence) and is_descriptive_answer_sentence(query, sentence):
            return index, sentence
    for index, sentence in enumerate(sentences):
        if not is_title_fragment(doc, sentence):
            return index, sentence
    return 0, sentences[0]


def bridge_packet_has_required_predicate(query: Query, selected_docs: list[Document]) -> bool:
    """Check whether a bridge packet kept the relation asked by the query."""
    query_text = query.text.lower()
    context_text = " ".join(doc.text.lower() for doc in selected_docs)
    predicate_groups = [
        (["based"], ["based", "located", "headquartered", "lives", "resides"]),
        (["located"], ["located", "based", "situated"]),
        (["born"], ["born", "birth"]),
        (["nationality"], list(NATIONALITY_VALUE_TERMS)),
        (["director"], ["directed", "director", "written and directed"]),
        (["author", "writer"], ["author", "writer", "written", "wrote"]),
        (["founder"], ["founder", "founded"]),
        (["producer"], ["producer", "produced"]),
    ]
    for query_cues, evidence_cues in predicate_groups:
        if any(cue in query_text for cue in query_cues):
            return any(cue in context_text for cue in evidence_cues)
    return True


def answer_type_for_query(query: Query) -> str:
    text = query.text.lower()
    if re.search(r"\b(city|town|village|county|state|province|where)\b", text):
        return "location"
    if re.search(r"\b(country|nation|nationality)\b", text):
        return "country"
    if re.search(r"\b(when|year|date|age|old)\b", text):
        return "date"
    if re.search(r"\b(who|person|director|author|writer|founder|producer|member)\b", text):
        return "person"
    if re.search(r"\b(company|organization|publisher|label|studio|school|university)\b", text):
        return "organization"
    return ""


def sentence_has_answer_type_candidate(query: Query, sentence: str) -> bool:
    answer_type = answer_type_for_query(query)
    if not answer_type:
        return False
    lowered = sentence.lower()
    if answer_type == "location":
        if re.search(r"\b(city|town|village|county|state|province|capital|located|based|headquartered)\b", lowered):
            return True
        return bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+(?:City|Village|Town|County|State)\b", sentence))
    if answer_type == "country":
        return bool(content_word_set(sentence) & NATIONALITY_VALUE_TERMS) or re.search(
            r"\b(country|nation|nationality|republic|kingdom|states)\b",
            lowered,
        )
    if answer_type == "date":
        return has_number(sentence) or bool(re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", lowered))
    if answer_type == "person":
        return bool(capitalized_terms(sentence)) and bool(re.search(r"\b(is|was|born|directed|written|founded|created|produced|member)\b", lowered))
    if answer_type == "organization":
        return bool(re.search(r"\b(company|organization|studio|label|university|school|publisher|founded|based)\b", lowered))
    return False


def merged_evidence_brief(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    token_budget: int = 850,
    pool_size: int = 10,
    max_sentences: int = 14,
    min_sentences: int = 5,
) -> list[Document]:
    """Build one compact evidence brief from useful sentences across top docs."""
    pool = ranked_docs[:pool_size]
    if not pool:
        return []

    top_score = pool[0][1] if pool and pool[0][1] > 0 else 1.0
    candidates = []
    for rank, (doc, doc_score) in enumerate(pool, start=1):
        title = title_from_doc_id(doc.doc_id)
        for sentence_index, sentence in enumerate(split_sentences(doc.text)):
            score = merged_evidence_sentence_score(query, doc, sentence, doc_score, top_score, rank)
            if score <= 0:
                continue
            token_len = max(1, estimate_tokens(sentence))
            candidates.append(
                {
                    "doc_id": doc.doc_id,
                    "title": title,
                    "sentence": sentence,
                    "sentence_terms": content_word_set(sentence),
                    "score": score,
                    "utility": score / (token_len ** 0.55),
                    "rank": rank,
                    "sentence_index": sentence_index,
                    "tokens": token_len,
                }
            )

    if not candidates:
        return compress_documents(query, [doc for doc, _score in pool[:3]], "evidence_ngram_neighbors")

    selected = []
    selected_terms: list[set[str]] = []
    covered_query_terms: set[str] = set()
    query_terms = content_word_set(query.text)
    total_tokens = 0
    seen_sentences = set()

    for item in sorted(candidates, key=lambda row: row["utility"], reverse=True):
        normalized_sentence = " ".join(item["sentence"].lower().split())
        if normalized_sentence in seen_sentences:
            continue
        if any(sentence_similarity(item["sentence_terms"], terms) > 0.82 for terms in selected_terms):
            continue

        new_query_terms = (item["sentence_terms"] & query_terms) - covered_query_terms
        novelty_bonus = 35 if new_query_terms else 0
        estimated_cost = item["tokens"] + estimate_tokens(item["title"]) + 8
        if total_tokens + estimated_cost > token_budget and selected:
            continue

        selected.append(item)
        selected_terms.append(item["sentence_terms"])
        covered_query_terms.update(item["sentence_terms"] & query_terms)
        seen_sentences.add(normalized_sentence)
        total_tokens += estimated_cost + novelty_bonus

        if len(selected) >= max_sentences:
            break
        if overlap_ratio(query_terms, covered_query_terms) >= 0.72 and len(selected) >= min_sentences:
            break

    selected.sort(key=lambda row: (row["rank"], row["sentence_index"]))
    by_doc: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    order: list[str] = []
    for index, item in enumerate(selected, start=1):
        doc_id = item["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = []
            titles[doc_id] = item["title"]
            order.append(doc_id)
        by_doc[doc_id].append(f"[{index}] {item['sentence']}")

    return [
        Document(
            doc_id=doc_id,
            text=f"Merged evidence brief from {titles[doc_id]}:\n" + "\n".join(by_doc[doc_id]),
        )
        for doc_id in order
    ]


def answer_candidate_score(
    query: Query,
    doc: Document,
    sentence: str,
    doc_score: float,
    top_score: float,
    rank: int,
    sentence_index: int,
) -> float:
    query_terms = content_word_set(query.text)
    sentence_terms = content_word_set(sentence)
    title_terms = content_word_set(title_from_doc_id(doc.doc_id))
    if not sentence_terms:
        return 0.0
    if is_title_fragment(doc, sentence) and (
        is_comparison_query(query) or "nationality" in query.text.lower()
    ):
        return -1.0
    if needs_bridge_evidence(query, "concise") and is_non_predicate_fragment(sentence):
        return -1.0

    query_entities = capitalized_terms(query.text)
    sentence_entities = capitalized_terms(sentence)
    title_entities = capitalized_terms(title_from_doc_id(doc.doc_id))

    overlap = len(query_terms & sentence_terms)
    title_overlap = len(query_terms & title_terms)
    entity_count = len(sentence_entities - query_entities)
    title_entity_link = len((query_entities | title_entities) & sentence_entities)
    number_signal = 1 if has_number(sentence) else 0
    retrieval_relevance = doc_score / top_score if top_score > 0 else 0.0
    early_sentence_bonus = 1 / (1 + sentence_index)
    short_answer_density = min(1.0, (entity_count + number_signal) / max(1, len(sentence_terms) / 8))
    descriptive_bonus = 1.35 if is_descriptive_answer_sentence(query, sentence) else 0.0
    answer_type_bonus = 1.10 if sentence_has_answer_type_candidate(query, sentence) else 0.0
    title_fragment_penalty = 4.0 if is_title_fragment(doc, sentence) else 0.0
    nationality_bonus = (
        1.25
        if "nationality" in query.text.lower() and (sentence_terms & NATIONALITY_VALUE_TERMS)
        else 0.0
    )

    return (
        (1.25 * retrieval_relevance)
        + (0.95 * overlap)
        + (0.80 * title_overlap)
        + (0.55 * min(entity_count, 4))
        + (0.45 * title_entity_link)
        + (0.55 * number_signal)
        + (0.35 * short_answer_density)
        + (0.20 * early_sentence_bonus)
        + descriptive_bonus
        + answer_type_bonus
        + nationality_bonus
        - title_fragment_penalty
    )


def candidate_answer_brief(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    token_budget: int = 650,
    pool_size: int = 10,
    max_sentences: int = 12,
    min_sentences: int = 4,
) -> list[Document]:
    """Build a compact brief for short-answer tasks.

    Unlike the general evidence brief, this prioritizes answer-like spans:
    entities, dates, numbers, titles, and bridge sentences that connect query
    anchors to likely answer candidates.
    """
    pool = ranked_docs[:pool_size]
    if not pool:
        return []

    top_score = pool[0][1] if pool and pool[0][1] > 0 else 1.0
    candidates = []
    for rank, (doc, doc_score) in enumerate(pool, start=1):
        title = title_from_doc_id(doc.doc_id)
        for sentence_index, sentence in enumerate(split_sentences(doc.text)):
            score = answer_candidate_score(
                query,
                doc,
                sentence,
                doc_score,
                top_score,
                rank,
                sentence_index,
            )
            if score <= 0:
                continue
            token_len = max(1, estimate_tokens(sentence))
            candidates.append(
                {
                    "doc_id": doc.doc_id,
                    "title": title,
                    "sentence": sentence,
                    "terms": content_word_set(sentence),
                    "score": score,
                    "utility": score / (token_len ** 0.62),
                    "rank": rank,
                    "sentence_index": sentence_index,
                    "tokens": token_len,
                }
            )

    if not candidates:
        return token_budget_greedy_pack(query, ranked_docs, token_budget=token_budget, pool_size=pool_size)

    selected = []
    selected_terms: list[set[str]] = []
    covered_docs = set()
    selected_signal_terms: set[str] = set()
    signals = query_signal_terms(query)
    target_coverage = 0.52 if needs_bridge_evidence(query, "concise") else 0.42
    total = 0
    for item in sorted(candidates, key=lambda row: row["utility"], reverse=True):
        if any(sentence_similarity(item["terms"], terms) > 0.78 for terms in selected_terms):
            continue

        diversity_bonus = 25 if item["doc_id"] not in covered_docs else 0
        cost = estimate_tokens(item["title"]) + item["tokens"] + 7
        if total + cost > token_budget and selected:
            continue

        selected.append(item)
        selected_terms.append(item["terms"])
        covered_docs.add(item["doc_id"])
        selected_signal_terms.update(item["terms"] & signals)
        total += cost + diversity_bonus

        if len(selected) >= max_sentences:
            break
        if (
            len(selected) >= min_sentences
            and len(covered_docs) >= 3
            and overlap_ratio(signals, selected_signal_terms) >= target_coverage
        ):
            break

    selected.sort(key=lambda row: (row["rank"], row["sentence_index"]))
    by_doc: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    order: list[str] = []
    for index, item in enumerate(selected, start=1):
        doc_id = item["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = []
            titles[doc_id] = item["title"]
            order.append(doc_id)
        by_doc[doc_id].append(f"[candidate {index}] {item['sentence']}")

    return [
        Document(
            doc_id=doc_id,
            text=f"Candidate answer evidence from {titles[doc_id]}:\n" + "\n".join(by_doc[doc_id]),
        )
        for doc_id in order
    ]


def bridge_preserving_candidate_brief(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    token_budget: int = 720,
    pool_size: int = 10,
    max_sentences: int = 14,
    min_sources: int = 3,
) -> list[Document]:
    """Compact short-answer context that preserves multi-hop bridge anchors."""
    pool = ranked_docs[:pool_size]
    if not pool:
        return []

    top_score = pool[0][1] if pool and pool[0][1] > 0 else 1.0
    selected = []
    selected_keys = set()
    selected_terms: list[set[str]] = []
    total = 0

    # Keep one lead/anchor sentence from the strongest sources so the model can
    # connect entities across documents instead of seeing isolated candidates.
    for rank, (doc, _doc_score) in enumerate(pool[:min_sources], start=1):
        sentence_index, sentence = first_descriptive_sentence(query, doc)
        title = title_from_doc_id(doc.doc_id)
        cost = estimate_tokens(title) + estimate_tokens(sentence) + 7
        if total + cost > token_budget and selected:
            continue
        selected.append(
            {
                "doc_id": doc.doc_id,
                "title": title,
                "sentence": sentence,
                "terms": content_word_set(sentence),
                "rank": rank,
                "sentence_index": sentence_index,
                "label": "bridge",
            }
        )
        selected_keys.add((doc.doc_id, sentence_index))
        selected_terms.append(content_word_set(sentence))
        total += cost

    candidates = []
    for rank, (doc, doc_score) in enumerate(pool, start=1):
        title = title_from_doc_id(doc.doc_id)
        for sentence_index, sentence in enumerate(split_sentences(doc.text)):
            if (doc.doc_id, sentence_index) in selected_keys:
                continue
            score = answer_candidate_score(
                query,
                doc,
                sentence,
                doc_score,
                top_score,
                rank,
                sentence_index,
            )
            if score <= 0:
                continue
            token_len = max(1, estimate_tokens(sentence))
            candidates.append(
                {
                    "doc_id": doc.doc_id,
                    "title": title,
                    "sentence": sentence,
                    "terms": content_word_set(sentence),
                    "rank": rank,
                    "sentence_index": sentence_index,
                    "tokens": token_len,
                    "label": "candidate",
                    "utility": score / (token_len ** 0.60),
                }
            )

    signals = query_signal_terms(query)
    covered = content_word_set(" ".join(item["sentence"] for item in selected)) & signals
    for item in sorted(candidates, key=lambda row: row["utility"], reverse=True):
        if len(selected) >= max_sentences:
            break
        if needs_bridge_evidence(query, "concise"):
            descriptive_count = sum(
                1
                for row in selected
                if is_descriptive_answer_sentence(query, row["sentence"])
            )
            has_answer_candidate = (
                not answer_type_for_query(query)
                or any(sentence_has_answer_type_candidate(query, row["sentence"]) for row in selected)
            )
            if (
                descriptive_count >= min_sources
                and has_answer_candidate
                and source_diversity(grouped_sentence_pack([
                    (row["doc_id"], row["title"], row["sentence"]) for row in selected
                ])) >= min_sources
            ):
                break
        if overlap_ratio(signals, covered) >= 0.62 and source_diversity(grouped_sentence_pack([
            (row["doc_id"], row["title"], row["sentence"]) for row in selected
        ])) >= min_sources:
            break
        if any(sentence_similarity(item["terms"], terms) > 0.80 for terms in selected_terms):
            continue
        cost = estimate_tokens(item["title"]) + item["tokens"] + 7
        if total + cost > token_budget and selected:
            continue

        selected.append(item)
        selected_terms.append(item["terms"])
        selected_keys.add((item["doc_id"], item["sentence_index"]))
        covered.update(item["terms"] & signals)
        total += cost

    selected.sort(key=lambda row: (row["rank"], row["sentence_index"]))
    by_doc: dict[str, list[str]] = {}
    titles: dict[str, str] = {}
    order: list[str] = []
    for index, item in enumerate(selected, start=1):
        doc_id = item["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = []
            titles[doc_id] = item["title"]
            order.append(doc_id)
        by_doc[doc_id].append(f"[{item['label']} {index}] {item['sentence']}")

    return [
        Document(
            doc_id=doc_id,
            text=f"Bridge-preserving answer evidence from {titles[doc_id]}:\n" + "\n".join(by_doc[doc_id]),
        )
        for doc_id in order
    ]


def query_shape(query: Query, prompt_style: str) -> str:
    """Classify the query shape using only the query text.

    This is deliberately lightweight. It is not trying to solve the task; it
    only chooses a safer context shape before generation.
    """
    text = query.text.strip().lower()
    if prompt_style == "concise":
        multi_hop_cues = [
            "what is the",
            "who is the",
            "which",
            "where",
            "when",
            "whose",
            "both",
            "and",
        ]
        if any(cue in text for cue in multi_hop_cues) and len(content_word_set(text)) >= 6:
            return "multi_hop"
        return "factoid"
    if "?" in text:
        return "factoid"
    return "claim"


def query_complexity(query: Query, prompt_style: str) -> str:
    """Estimate evidence complexity without using dataset-specific rules."""
    terms = content_word_set(query.text)
    text = query.text.lower()
    connective_count = sum(1 for cue in [" and ", " or ", " between ", " compared ", " versus "] if cue in text)
    entity_count = len(capitalized_terms(query.text))
    number_count = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", query.text))

    if prompt_style == "concise" and (connective_count or entity_count >= 2) and len(terms) >= 6:
        return "distributed"
    if len(terms) >= 12 or connective_count >= 2 or entity_count >= 3 or number_count >= 2:
        return "distributed"
    if len(terms) >= 7 or entity_count >= 1:
        return "moderate"
    return "focused"


def task_adaptive_context_policy(query: Query, prompt_style: str) -> TaskAdaptiveContextPolicy:
    """Choose a context shape from task signals, not dataset names."""
    shape = query_shape(query, prompt_style)
    complexity = query_complexity(query, prompt_style)

    if prompt_style == "concise":
        if complexity == "distributed" or shape == "multi_hop":
            return TaskAdaptiveContextPolicy(
                shape=shape,
                complexity=complexity,
                brief_kind="candidate_answer",
                brief_token_budget=720,
                brief_max_sentences=14,
                brief_min_sentences=6,
                min_prompt_tokens=0,
                pack_token_budget=950,
                min_sources=3,
            )
        return TaskAdaptiveContextPolicy(
            shape=shape,
            complexity=complexity,
            brief_kind="candidate_answer",
            brief_token_budget=540,
            brief_max_sentences=9,
            brief_min_sentences=4,
            min_prompt_tokens=0,
            pack_token_budget=800,
            min_sources=2,
        )

    if shape == "factoid" and complexity == "focused":
        return TaskAdaptiveContextPolicy(
            shape=shape,
            complexity="moderate",
            brief_kind="merged_evidence",
            brief_token_budget=760,
            brief_max_sentences=12,
            brief_min_sentences=7,
            min_prompt_tokens=520,
            pack_token_budget=900,
            min_sources=2,
        )

    if complexity == "distributed":
        return TaskAdaptiveContextPolicy(
            shape=shape,
            complexity=complexity,
            brief_kind="merged_evidence",
            brief_token_budget=950,
            brief_max_sentences=16,
            brief_min_sentences=9,
            min_prompt_tokens=680,
            pack_token_budget=1150,
            min_sources=3,
        )

    if complexity == "moderate":
        return TaskAdaptiveContextPolicy(
            shape=shape,
            complexity=complexity,
            brief_kind="merged_evidence",
            brief_token_budget=780,
            brief_max_sentences=12,
            brief_min_sentences=7,
            min_prompt_tokens=520,
            pack_token_budget=950,
            min_sources=2,
        )

    return TaskAdaptiveContextPolicy(
        shape=shape,
        complexity=complexity,
        brief_kind="merged_evidence",
        brief_token_budget=560,
        brief_max_sentences=8,
        brief_min_sentences=5,
        min_prompt_tokens=0,
        pack_token_budget=760,
        min_sources=1,
    )


def task_adaptive_brief(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    prompt_style: str,
    allow_synthesis_pack: bool = False,
) -> tuple[list[Document], str]:
    """Build the cheapest useful brief for the detected task shape."""
    policy = task_adaptive_context_policy(query, prompt_style)
    if policy.brief_kind == "candidate_answer":
        docs = candidate_answer_brief(
            query,
            ranked_docs,
            token_budget=policy.brief_token_budget,
            pool_size=10,
            max_sentences=policy.brief_max_sentences,
            min_sentences=policy.brief_min_sentences,
        )
        return docs, f"task_adaptive_candidate_{policy.complexity}"

    docs = merged_evidence_brief(
        query,
        ranked_docs,
        token_budget=policy.brief_token_budget,
        pool_size=10,
        max_sentences=policy.brief_max_sentences,
        min_sentences=policy.brief_min_sentences,
    )
    if (
        allow_synthesis_pack
        and policy.min_prompt_tokens
        and context_estimated_tokens(query, docs, prompt_style) < policy.min_prompt_tokens
    ):
        return (
            token_budget_greedy_pack(
                query,
                ranked_docs,
                token_budget=policy.pack_token_budget,
                pool_size=10,
                force_first_per_doc=True,
            ),
            f"task_adaptive_synthesis_pack_{policy.complexity}",
        )
    return docs, f"task_adaptive_merged_{policy.complexity}"


def query_signal_terms(query: Query) -> set[str]:
    """Terms that selected evidence should ideally cover before generation."""
    terms = content_word_set(query.text)
    terms.update(term.lower() for term in capitalized_terms(query.text))
    terms.update(re.findall(r"\b\d+(?:\.\d+)?%?\b", query.text))
    return {term for term in terms if len(term) > 2 or has_number(term)}


def evidence_signal_coverage(query: Query, selected_docs: list[Document]) -> float:
    signals = query_signal_terms(query)
    context_terms = content_word_set(" ".join(doc.text for doc in selected_docs))
    return overlap_ratio(signals, context_terms)


def coverage_threshold_for_shape(shape: str) -> float:
    if shape == "distributed":
        return 0.62
    if shape == "moderate":
        return 0.52
    return 0.42


def evidence_coverage_curve(query: Query, ranked_docs: list[tuple[Document, float]], max_budget: int = 10) -> list[float]:
    curve = []
    for budget in range(1, min(max_budget, len(ranked_docs)) + 1):
        docs = [doc for doc, _score in ranked_docs[:budget]]
        compact_docs = compress_documents(query, docs, "evidence_ngram_neighbors")
        curve.append(evidence_signal_coverage(query, compact_docs))
    return curve


def marginal_gain(curve: list[float], budget: int) -> float:
    if budget <= 1 or budget > len(curve):
        return curve[0] if curve else 0.0
    return curve[budget - 1] - curve[budget - 2]


def source_diversity(selected_docs: list[Document]) -> int:
    return len({doc.doc_id for doc in selected_docs})


def average_rank_score(selected_docs: list[Document], ranked_docs: list[tuple[Document, float]]) -> float:
    rank_by_id = {doc.doc_id: rank for rank, (doc, _score) in enumerate(ranked_docs[:10], start=1)}
    ranks = [rank_by_id.get(doc.doc_id, 10) for doc in selected_docs]
    if not ranks:
        return 0.0
    return average(1 / rank for rank in ranks)


def context_estimated_tokens(query: Query, selected_docs: list[Document], prompt_style: str) -> int:
    return max(1, estimate_prompt_tokens(query, selected_docs, prompt_style))


def portfolio_context_candidates(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    budget: int,
    prompt_style: str,
) -> list[tuple[str, list[Document]]]:
    adaptive_budget = max(1, adaptive_k_budget(ranked_docs, max_budget=10))
    compact_budget = min(max(budget, adaptive_budget), len(ranked_docs), 10)
    policy = task_adaptive_context_policy(query, prompt_style)
    candidates: list[tuple[str, list[Document]]] = []

    adaptive_docs = [doc for doc, _score in ranked_docs[:adaptive_budget]]
    candidates.append((
        "portfolio_adaptive_k_compact",
        compress_documents(query, adaptive_docs, "evidence_ngram_neighbors"),
    ))

    for candidate_budget in sorted({2, 3, 5, compact_budget}):
        if candidate_budget <= len(ranked_docs):
            docs = [doc for doc, _score in ranked_docs[:candidate_budget]]
            candidates.append((
                f"portfolio_top_{candidate_budget}_compact",
                compress_documents(query, docs, "evidence_ngram_neighbors"),
            ))

    task_brief_docs, task_brief_name = task_adaptive_brief(
        query,
        ranked_docs,
        prompt_style,
        allow_synthesis_pack=True,
    )
    candidates.append((f"portfolio_{task_brief_name}", task_brief_docs))

    if policy.complexity == "distributed":
        candidates.append((
            "portfolio_sentence_pack",
            token_budget_greedy_pack(
                query,
                ranked_docs,
                token_budget=policy.pack_token_budget,
                pool_size=10,
                force_first_per_doc=True,
            ),
        ))

    deduped = []
    seen = set()
    for name, docs in candidates:
        signature = tuple(doc.doc_id + ":" + doc.text[:80] for doc in docs)
        if docs and signature not in seen:
            deduped.append((name, docs))
            seen.add(signature)
    return deduped


def portfolio_candidate_score(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    selected_docs: list[Document],
    prompt_style: str,
) -> tuple[bool, float, float, int]:
    policy = task_adaptive_context_policy(query, prompt_style)
    complexity = policy.complexity
    coverage = evidence_signal_coverage(query, selected_docs)
    diversity = source_diversity(selected_docs)
    tokens = context_estimated_tokens(query, selected_docs, prompt_style)
    rank_score = average_rank_score(selected_docs, ranked_docs)

    required_diversity = policy.min_sources
    required_coverage = coverage_threshold_for_shape(complexity)
    passes = coverage >= required_coverage and diversity >= required_diversity

    utility = (
        (2.8 * coverage)
        + (0.45 * min(diversity, 4))
        + (0.65 * rank_score)
        - (0.55 * math.log1p(tokens) / math.log(2000))
    )
    return passes, utility, coverage, tokens


def choose_portfolio_context(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    budget: int,
    prompt_style: str,
) -> tuple[list[Document], str]:
    candidates = portfolio_context_candidates(query, ranked_docs, budget, prompt_style)
    if not candidates:
        return [], "coverage_guided_empty"

    scored = []
    for name, docs in candidates:
        passes, utility, coverage, tokens = portfolio_candidate_score(query, ranked_docs, docs, prompt_style)
        scored.append(
            {
                "name": name,
                "docs": docs,
                "passes": passes,
                "utility": utility,
                "coverage": coverage,
                "tokens": tokens,
                "diversity": source_diversity(docs),
            }
        )

    passing = [row for row in scored if row["passes"]]
    if passing:
        policy = task_adaptive_context_policy(query, prompt_style)
        if prompt_style == "concise" and policy.complexity != "distributed":
            passing.sort(key=lambda row: (row["tokens"], -row["coverage"], -row["diversity"]))
        else:
            token_ceiling = policy.pack_token_budget + 250
            viable = [row for row in passing if row["tokens"] <= token_ceiling] or passing
            viable.sort(key=lambda row: (-row["utility"], row["tokens"]))
            passing = viable
        best = passing[0]
    else:
        scored.sort(key=lambda row: (-row["utility"], row["tokens"]))
        best = scored[0]

    return best["docs"], best["name"]


def coverage_guided_budget(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    prompt_style: str,
) -> int:
    """Start from Adaptive-k, then use evidence-coverage saturation to adjust."""
    if not ranked_docs:
        return 0

    policy = task_adaptive_context_policy(query, prompt_style)
    complexity = policy.complexity
    adaptive_budget = max(1, adaptive_k_budget(ranked_docs, max_budget=10))
    if complexity == "distributed":
        budget = max(3, adaptive_budget)
        budget_cap = 8
    elif complexity == "moderate":
        budget = max(2, adaptive_budget)
        budget_cap = 6
    else:
        budget = adaptive_budget
        budget_cap = 5

    threshold = coverage_threshold_for_shape(complexity)
    budget = min(budget, len(ranked_docs), 10)
    budget_cap = min(budget_cap, len(ranked_docs), 10)
    coverage_curve = evidence_coverage_curve(query, ranked_docs, max_budget=budget_cap)

    while budget < budget_cap:
        current_coverage = coverage_curve[budget - 1] if budget - 1 < len(coverage_curve) else 0.0
        next_gain = marginal_gain(coverage_curve, budget + 1)
        if current_coverage >= threshold and next_gain < 0.08:
            break
        if current_coverage >= 0.70 and next_gain < 0.04:
            break
        budget += 1

    return min(budget, 10)


def coverage_guided_context(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    budget: int,
    prompt_style: str,
) -> tuple[list[Document], str]:
    """Choose the best compact evidence portfolio after budget estimation."""
    return choose_portfolio_context(query, ranked_docs, budget, prompt_style)


def missing_signal_repair_context(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    prompt_style: str,
    ultra: bool = False,
) -> tuple[list[Document], str]:
    """Start from Adaptive-k, then add only evidence units covering missing signals."""
    if not ranked_docs:
        return [], "coverage_guided_empty"

    policy = task_adaptive_context_policy(query, prompt_style)
    adaptive_budget = max(1, adaptive_k_budget(ranked_docs, max_budget=10))
    seed_docs = [doc for doc, _score in ranked_docs[:adaptive_budget]]

    if prompt_style == "concise":
        if needs_bridge_evidence(query, prompt_style):
            if ultra:
                selected_docs = bridge_preserving_candidate_brief(
                    query,
                    ranked_docs,
                    token_budget=min(policy.brief_token_budget, 520),
                    pool_size=10,
                    max_sentences=min(policy.brief_max_sentences, 8),
                    min_sources=min(3, max(2, policy.min_sources)),
                )
                context_shape = f"missing_signal_ultra_bridge_candidate_seed_{adaptive_budget}"
            else:
                selected_docs = bridge_preserving_candidate_brief(
                    query,
                    ranked_docs,
                    token_budget=policy.brief_token_budget,
                    pool_size=10,
                    max_sentences=policy.brief_max_sentences,
                    min_sources=policy.min_sources,
                )
                context_shape = f"missing_signal_bridge_candidate_seed_{adaptive_budget}"
        elif policy.complexity == "distributed" and not ultra:
            selected_docs = bridge_preserving_candidate_brief(
                query,
                ranked_docs,
                token_budget=policy.brief_token_budget,
                pool_size=10,
                max_sentences=policy.brief_max_sentences,
                min_sources=policy.min_sources,
            )
            context_shape = f"missing_signal_bridge_candidate_seed_{adaptive_budget}"
        else:
            selected_docs = candidate_answer_brief(
                query,
                ranked_docs[: max(adaptive_budget, policy.min_sources)],
                token_budget=policy.brief_token_budget,
                pool_size=max(adaptive_budget, policy.min_sources),
                max_sentences=policy.brief_max_sentences,
                min_sentences=policy.brief_min_sentences,
            )
            context_shape = f"missing_signal_candidate_seed_{adaptive_budget}"
    else:
        selected_docs = compress_documents(query, seed_docs, "evidence_ngram_neighbors")
        context_shape = f"missing_signal_compact_seed_{adaptive_budget}"

    target_coverage = coverage_threshold_for_shape(policy.complexity)
    token_ceiling = policy.brief_token_budget + 260
    if policy.complexity == "distributed":
        token_ceiling = policy.pack_token_budget
    elif prompt_style != "concise" and query_shape(query, prompt_style) == "factoid":
        token_ceiling = min(policy.pack_token_budget, 900)
    if ultra:
        target_coverage = max(0.34, target_coverage - 0.14)
        token_ceiling = min(token_ceiling, max(260, int(policy.brief_token_budget * 0.62)))
        if prompt_style == "concise":
            token_ceiling = min(token_ceiling, 520)

    selected_units: list[tuple[str, str, str]] = []
    selected_keys = set()
    for doc in selected_docs:
        title = title_from_doc_id(doc.doc_id)
        for index, sentence in enumerate(split_sentences(doc.text)):
            key = (doc.doc_id, "seed", index)
            selected_units.append((doc.doc_id, title, sentence))
            selected_keys.add(key)

    signals = query_signal_terms(query)
    covered = content_word_set(" ".join(doc.text for doc in selected_docs)) & signals
    total_tokens = context_estimated_tokens(query, selected_docs, prompt_style)
    top_score = ranked_docs[0][1] if ranked_docs and ranked_docs[0][1] > 0 else 1.0

    candidates = []
    for rank, (doc, doc_score) in enumerate(ranked_docs[:10], start=1):
        title = title_from_doc_id(doc.doc_id)
        for index, sentence in enumerate(split_sentences(doc.text)):
            sentence_terms = content_word_set(sentence)
            new_signals = (sentence_terms & signals) - covered
            if not new_signals and rank <= adaptive_budget:
                continue
            token_len = max(1, estimate_tokens(sentence))
            relevance = doc_score / top_score if top_score > 0 else 0.0
            phrase = phrase_overlap_score(tokenize(query.text), tokenize(sentence))
            title_overlap = overlap_ratio(signals, content_word_set(title))
            novelty = len(new_signals)
            score = (
                (2.6 * novelty)
                + (1.2 * overlap_ratio(signals, sentence_terms))
                + (0.85 * phrase)
                + (0.55 * relevance)
                + (0.35 * title_overlap)
                + (0.15 / rank)
            )
            if score <= 0:
                continue
            candidates.append(
                {
                    "doc_id": doc.doc_id,
                    "title": title,
                    "sentence": sentence,
                    "terms": sentence_terms,
                    "new_signals": new_signals,
                    "rank": rank,
                    "index": index,
                    "tokens": token_len,
                    "utility": score / (token_len ** 0.58),
                }
            )

    selected_terms = [content_word_set(sentence) for _doc_id, _title, sentence in selected_units]
    for item in sorted(candidates, key=lambda row: row["utility"], reverse=True):
        if overlap_ratio(signals, covered) >= target_coverage and source_diversity(grouped_sentence_pack(selected_units)) >= policy.min_sources:
            break
        key = (item["doc_id"], item["rank"], item["index"])
        if key in selected_keys:
            continue
        if any(sentence_similarity(item["terms"], terms) > 0.82 for terms in selected_terms):
            continue

        estimated_next = total_tokens + item["tokens"] + estimate_tokens(item["title"]) + 8
        if estimated_next > token_ceiling and selected_units:
            continue

        selected_units.append((item["doc_id"], item["title"], item["sentence"]))
        selected_terms.append(item["terms"])
        selected_keys.add(key)
        covered.update(item["terms"] & signals)
        total_tokens = estimated_next

    repaired_docs = grouped_sentence_pack(selected_units)
    if not repaired_docs:
        return selected_docs, context_shape

    if repaired_docs != selected_docs:
        context_shape += "_repaired"
    if ultra:
        context_shape += "_ultra"
    return repaired_docs, context_shape


def ultra_needs_fallback(query: Query, answer: str, selected_docs: list[Document], prompt_style: str) -> tuple[bool, str]:
    """Very conservative fallback for the ultra-cheap variant."""
    cleaned_answer = answer.strip()
    if not cleaned_answer:
        return True, "empty_answer"
    lowered_answer = cleaned_answer.lower()
    weak_answer = any(phrase in lowered_answer for phrase in WEAK_ANSWER_PHRASES)
    if weak_answer and evidence_signal_coverage(query, selected_docs) < 0.25:
        return True, "weak_answer_very_low_coverage"
    if prompt_style == "concise":
        return short_answer_needs_fallback(query, answer, selected_docs)
    return False, ""


def coverage_guided_needs_fallback(
    query: Query,
    answer: str,
    selected_docs: list[Document],
    prompt_style: str,
) -> tuple[bool, str]:
    """Fallback gate that combines answer risk with pre-generation coverage."""
    risk, reasons = answer_risk_score(query, answer, selected_docs)
    coverage = evidence_signal_coverage(query, selected_docs)
    complexity = task_adaptive_context_policy(query, prompt_style).complexity
    threshold = coverage_threshold_for_shape(complexity)

    if not answer.strip():
        return True, "empty_answer"

    weak_answer = any(phrase in answer.lower() for phrase in WEAK_ANSWER_PHRASES)
    if weak_answer and coverage < threshold:
        return True, f"weak_answer_low_coverage:{coverage:.2f}"
    if complexity == "focused" and risk < 5:
        return False, ""
    if coverage < threshold and risk >= 3:
        return True, f"low_coverage:{coverage:.2f}," + ",".join(reasons)
    if risk >= 5:
        return True, ",".join(reasons)
    return False, ""


def coverage_guided_adaptive_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Coverage-Guided Safe Adaptive Context.

    Experimental variant:
    1. Use the Adaptive-k score-shape cutoff as a cheap starting point.
    2. Expand before generation if compact evidence misses query signals.
    3. Use task-shaped context: compact evidence for claims/factoids and
       sentence packing for multi-hop concise QA.
    4. Fallback only when answer risk and evidence coverage justify it.
    """
    budget = coverage_guided_budget(query, ranked_docs, config.prompt_style)
    selected_docs, context_shape = missing_signal_repair_context(query, ranked_docs, config.prompt_style)

    first_config = config_for_answer_call(config, context_shape)
    first_answer = generate_answer(query, selected_docs, first_config)

    if config.prompt_style == "concise":
        should_expand, reason = short_answer_needs_fallback(query, first_answer.text, selected_docs)
    else:
        should_expand, reason = coverage_guided_needs_fallback(
            query,
            first_answer.text,
            selected_docs,
            config.prompt_style,
        )
    if not should_expand:
        return first_answer, selected_docs, False, "", first_answer.total_tokens, 0

    expanded_budget = min(10, max(budget + 2, 5))
    policy = task_adaptive_context_policy(query, config.prompt_style)
    complexity = policy.complexity
    if complexity == "distributed":
        fallback_docs = token_budget_greedy_pack(
            query,
            ranked_docs,
            token_budget=policy.pack_token_budget,
            pool_size=10,
            force_first_per_doc=True,
        )
        fallback_shape = "coverage_guided_expanded_sentence_pack"
    else:
        fallback_docs, fallback_shape = choose_portfolio_context(
            query,
            ranked_docs,
            expanded_budget,
            config.prompt_style,
        )
        if evidence_signal_coverage(query, fallback_docs) < coverage_threshold_for_shape(complexity):
            fallback_docs = compress_documents(
                query,
                [doc for doc, _score in ranked_docs[:expanded_budget]],
                "evidence_ngram_neighbors",
            )
            fallback_shape = "coverage_guided_expanded_compact"

    fallback_config = config_for_answer_call(config, fallback_shape)
    fallback_answer = generate_answer(query, fallback_docs, fallback_config)
    token_source = combine_token_sources([first_answer.token_source, fallback_answer.token_source])
    combined_answer = GeneratedAnswer(
        text=fallback_answer.text,
        prompt_tokens=first_answer.prompt_tokens + fallback_answer.prompt_tokens,
        completion_tokens=first_answer.completion_tokens + fallback_answer.completion_tokens,
        total_tokens=first_answer.total_tokens + fallback_answer.total_tokens,
        token_source=token_source,
        generation_time_ms=first_answer.generation_time_ms + fallback_answer.generation_time_ms,
    )
    return combined_answer, fallback_docs, True, reason, first_answer.total_tokens, fallback_answer.total_tokens


def coverage_guided_ultra_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Ultra-cheap Coverage-Guided variant.

    This tests whether starting from Adaptive-k and sending compressed repaired
    evidence can beat Adaptive-k's full-document context on tokens while keeping
    most answer quality.
    """
    selected_docs, context_shape = missing_signal_repair_context(
        query,
        ranked_docs,
        config.prompt_style,
        ultra=True,
    )
    first_config = config_for_answer_call(config, context_shape)
    first_answer = generate_answer(query, selected_docs, first_config)

    should_expand, reason = ultra_needs_fallback(query, first_answer.text, selected_docs, config.prompt_style)
    if not should_expand:
        return first_answer, selected_docs, False, "", first_answer.total_tokens, 0

    fallback_docs, fallback_shape = task_adaptive_brief(
        query,
        ranked_docs,
        config.prompt_style,
        allow_synthesis_pack=False,
    )
    fallback_config = config_for_answer_call(config, fallback_shape)
    fallback_answer = generate_answer(query, fallback_docs, fallback_config)
    combined_answer = GeneratedAnswer(
        text=fallback_answer.text,
        prompt_tokens=first_answer.prompt_tokens + fallback_answer.prompt_tokens,
        completion_tokens=first_answer.completion_tokens + fallback_answer.completion_tokens,
        total_tokens=first_answer.total_tokens + fallback_answer.total_tokens,
        token_source=combine_token_sources([first_answer.token_source, fallback_answer.token_source]),
        generation_time_ms=first_answer.generation_time_ms + fallback_answer.generation_time_ms,
    )
    return combined_answer, fallback_docs, True, reason, first_answer.total_tokens, fallback_answer.total_tokens


def task_aware_coverage_ultra_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Use Coverage-Guided Ultra only when task/evidence shape is safe.

    Ultra is efficient for focused or concentrated evidence. For multi-hop,
    distributed, or long-answer settings, this routes to the original Safe
    Adaptive ladder instead of trying to compress away necessary context.
    """
    use_safe, route_reason = task_should_use_safe_adaptive(query, ranked_docs, config)
    if use_safe:
        answer, docs, fallback_used, fallback_reason, first_tokens, fallback_tokens = answer_aware_fallback_run(
            query=query,
            ranked_docs=ranked_docs,
            sequential_budget=sequential_budget,
            config=config,
        )
        reason = f"safe_route:{route_reason}"
        if fallback_reason:
            reason += f";{fallback_reason}"
        return answer, docs, fallback_used, reason, first_tokens, fallback_tokens

    answer, docs, fallback_used, fallback_reason, first_tokens, fallback_tokens = coverage_guided_ultra_run(
        query=query,
        ranked_docs=ranked_docs,
        config=config,
    )
    reason = f"ultra_route:{route_reason}"
    if fallback_reason:
        reason += f";{fallback_reason}"
    return answer, docs, fallback_used, reason, first_tokens, fallback_tokens


def routed_predicate_should_use_ultra(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    ultra_docs: list[Document],
    adaptive_docs: list[Document],
    guarded_docs: list[Document],
    config: LLMConfig,
) -> tuple[bool, str]:
    """Route between Adaptive-k full context and predicate-preserving Ultra.

    The rule is intentionally query-shape based rather than dataset based:
    short factoid/relation questions can benefit from predicate compression,
    while long-answer synthesis and claim verification usually need fuller
    Adaptive-k context.
    """
    if not ranked_docs or not ultra_docs:
        return False, "no_ultra_context"

    ultra_tokens = context_estimated_tokens(query, ultra_docs, config.prompt_style)
    adaptive_tokens = context_estimated_tokens(query, adaptive_docs, config.prompt_style)
    guarded_tokens = context_estimated_tokens(query, guarded_docs, config.prompt_style)
    shape = query_shape(query, config.prompt_style)
    complexity = query_complexity(query, config.prompt_style)
    coverage = evidence_signal_coverage(query, ultra_docs)
    diversity = source_diversity(ultra_docs)

    if config.max_output_tokens > 120:
        return False, "long_answer_setting"
    if shape == "claim" and config.prompt_style != "concise":
        return False, "claim_needs_fuller_context"
    if needs_bridge_evidence(query, config.prompt_style):
        if not bridge_packet_has_required_predicate(query, ultra_docs):
            return False, "bridge_packet_missing_predicate"
        if ultra_tokens > guarded_tokens * 0.80:
            return False, "guarded_bridge_safer"
        if diversity >= 2 and coverage >= 0.32:
            return True, "bridge_predicate_packet"
        return False, "bridge_packet_too_thin"

    if shape == "factoid" or config.prompt_style == "concise":
        threshold = 0.30 if complexity == "focused" else 0.36
        if coverage >= threshold and ultra_tokens <= adaptive_tokens * 1.12:
            return True, "factoid_predicate_packet"
        if ultra_tokens <= adaptive_tokens * 0.62 and coverage >= 0.24:
            return True, "very_cheap_factoid_packet"
        return False, "factoid_packet_too_thin"

    if ultra_tokens >= adaptive_tokens * 0.96:
        return False, "ultra_not_cheaper"

    if ultra_tokens <= adaptive_tokens * 0.55 and coverage >= 0.40:
        return True, "cheap_supported_packet"
    return False, "adaptive_k_safer"


def routed_predicate_adaptive_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Adaptive-k with predicate-preserving evidence routing.

    Adaptive-k chooses the candidate region. The router then decides whether
    that region should be sent as full documents or as a compact predicate
    packet before generation.
    """
    adaptive_budget = max(1, adaptive_k_budget(ranked_docs, max_budget=10))
    adaptive_docs = [doc for doc, _score in ranked_docs[:adaptive_budget]]
    guarded_budget = guarded_adaptive_k_budget(
        query,
        ranked_docs,
        config.prompt_style,
        config.max_output_tokens,
        max_budget=10,
    )
    guarded_docs = [doc for doc, _score in ranked_docs[:guarded_budget]]
    ultra_docs, ultra_shape = missing_signal_repair_context(
        query,
        ranked_docs,
        config.prompt_style,
        ultra=True,
    )
    use_ultra, route_reason = routed_predicate_should_use_ultra(
        query,
        ranked_docs,
        ultra_docs,
        adaptive_docs,
        guarded_docs,
        config,
    )

    if use_ultra:
        selected_docs = ultra_docs
        context_shape = f"routed_ultra:{route_reason}:{ultra_shape}"
    elif needs_bridge_evidence(query, config.prompt_style) or config.prompt_style == "concise":
        selected_docs = guarded_docs
        context_shape = f"routed_guarded_adaptive_k_full:{route_reason}"
    else:
        selected_docs = adaptive_docs
        context_shape = f"routed_adaptive_k_full:{route_reason}"

    answer_config = config_for_answer_call(config, context_shape)
    answer = generate_answer(query, selected_docs, answer_config)
    return answer, selected_docs, False, route_reason, answer.total_tokens if use_ultra else 0, 0


def routed_guarded_should_use_compact(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    full_docs: list[Document],
    compact_docs: list[Document],
    config: LLMConfig,
) -> tuple[bool, str]:
    """Route guarded Adaptive-k between full documents and compact evidence.

    The route is intentionally general: use compact evidence only when it is
    meaningfully cheaper and still covers the query signals. Multi-hop or
    spread-evidence questions default to fuller context unless compact evidence
    looks unusually well supported.
    """
    if not full_docs:
        return False, "no_context"

    full_tokens = context_estimated_tokens(query, full_docs, config.prompt_style)
    compact_tokens = context_estimated_tokens(query, compact_docs, config.prompt_style)
    if compact_tokens >= full_tokens * 0.92:
        return False, "compact_not_cheaper"

    shape = query_shape(query, config.prompt_style)
    complexity = query_complexity(query, config.prompt_style)
    compact_coverage = evidence_signal_coverage(query, compact_docs)
    compact_diversity = source_diversity(compact_docs)
    required_sources = min(3, max(1, len(full_docs)))
    token_saving_ratio = 1.0 - (compact_tokens / max(1, full_tokens))

    if config.prompt_style == "concise" and (
        needs_bridge_evidence(query, config.prompt_style)
        or shape == "multi_hop"
        or complexity == "distributed"
    ):
        if compact_coverage >= 0.72 and compact_diversity >= required_sources:
            return True, "compact_supported_multihop"
        return False, "full_for_multihop"

    if config.max_output_tokens > 120 or complexity == "distributed":
        if compact_coverage >= 0.62 and compact_diversity >= min(2, required_sources):
            return True, "compact_supported_synthesis"
        return False, "full_for_distributed_evidence"

    if compact_coverage >= coverage_threshold_for_shape(complexity):
        return True, "compact_signal_coverage"

    if token_saving_ratio >= 0.35 and compact_coverage >= 0.34:
        return True, "cheap_compact_with_partial_coverage"

    return False, "full_for_low_compact_coverage"


def routed_guarded_adaptive_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Final routed guarded context policy.

    Guarded Adaptive-k first chooses how much evidence is safe to include. The
    route then chooses whether to send that evidence as full documents or as
    compact evidence spans.
    """
    guarded_budget = guarded_adaptive_k_budget(
        query,
        ranked_docs,
        config.prompt_style,
        config.max_output_tokens,
        max_budget=10,
    )
    full_docs = [doc for doc, _score in ranked_docs[:guarded_budget]]
    compact_docs = compress_documents(query, full_docs, "evidence_ngram_neighbors")
    use_compact, route_reason = routed_guarded_should_use_compact(
        query,
        ranked_docs,
        full_docs,
        compact_docs,
        config,
    )

    if use_compact:
        selected_docs = compact_docs
        context_shape = f"routed_guarded_compact:{route_reason}"
    else:
        selected_docs = full_docs
        context_shape = f"routed_guarded_full:{route_reason}"

    answer_config = config_for_answer_call(config, context_shape)
    answer = generate_answer(query, selected_docs, answer_config)
    return answer, selected_docs, False, route_reason, 0, 0


def run_answer_checked_stages(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    stages: list[tuple[str, int, str]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Run a Safe-Adaptive-style stage ladder with answer-aware fallback."""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_time_ms = 0
    token_sources = []

    first_pass_tokens = 0
    fallback_tokens = 0
    fallback_used = False
    fallback_reasons = []
    final_answer = None
    final_docs = []

    for stage_index, (stage_name, budget, compression_mode) in enumerate(stages):
        full_docs = [doc for doc, _score in ranked_docs[:budget]]
        if compression_mode == "full":
            selected_docs = full_docs
        elif compression_mode == "token_budget_greedy_pack":
            selected_docs = token_budget_greedy_pack(query, ranked_docs, token_budget=950, pool_size=10)
        else:
            selected_docs = compress_documents(query, full_docs, compression_mode)

        prompt_style = "default" if config.prompt_style == "anchor" else config.prompt_style
        answer_config = config_for_answer_call(config, compression_mode, prompt_style)
        answer = generate_answer(query, selected_docs, answer_config)

        total_prompt_tokens += answer.prompt_tokens
        total_completion_tokens += answer.completion_tokens
        total_tokens += answer.total_tokens
        total_time_ms += answer.generation_time_ms
        token_sources.append(answer.token_source)

        if stage_index == 0:
            first_pass_tokens = answer.total_tokens
        else:
            fallback_used = True
            fallback_tokens += answer.total_tokens

        final_answer = answer
        final_docs = selected_docs

        if config.prompt_style == "concise":
            should_expand, reason = short_answer_needs_fallback(query, answer.text, selected_docs)
        else:
            should_expand, reason = answer_needs_fallback(query, answer.text, selected_docs)
        if not should_expand or stage_index == len(stages) - 1:
            if reason:
                fallback_reasons.append(f"{stage_name}:{reason}")
            break

        fallback_reasons.append(f"{stage_name}:{reason}")

    combined_answer = GeneratedAnswer(
        text=final_answer.text,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        token_source=combine_token_sources(token_sources),
        generation_time_ms=total_time_ms,
    )
    return (
        combined_answer,
        final_docs,
        fallback_used,
        ";".join(fallback_reasons),
        first_pass_tokens,
        fallback_tokens,
    )


def retrieval_evidence_is_concentrated(ranked_docs: list[tuple[Document, float]], max_budget: int = 10) -> bool:
    """Return whether the top of the ranking looks clearly dominant."""
    scores = [score for _doc, score in ranked_docs[:max_budget]]
    if len(scores) < 3:
        return True
    top_score = scores[0]
    if top_score <= 0:
        return False
    normalized = [score / top_score for score in scores]
    second_ratio = normalized[1]
    top3_drop = normalized[0] - normalized[2]
    return second_ratio < 0.55 or top3_drop > 0.22


def task_should_use_safe_adaptive(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[bool, str]:
    """Decide when the safer Safe Adaptive ladder is the right starting policy.

    This uses task/evidence shape, not dataset names. Distributed questions and
    flat rankings are risky for very aggressive Adaptive-k-style compression.
    """
    shape = query_shape(query, config.prompt_style)
    complexity = query_complexity(query, config.prompt_style)
    concentrated = retrieval_evidence_is_concentrated(ranked_docs)

    if config.prompt_style == "concise" and (
        needs_bridge_evidence(query, config.prompt_style)
        or shape == "multi_hop"
        or complexity == "distributed"
    ):
        return True, "concise_distributed_or_bridge"

    if config.max_output_tokens > 120 and complexity != "focused":
        return True, "long_answer_setting"

    if shape == "claim" and config.max_output_tokens <= 120:
        return False, "claim_or_focused_evidence"

    if complexity == "distributed" and not concentrated:
        return True, "distributed_flat_ranking"

    return False, "focused_or_concentrated"


def routed_safe_guarded_adaptive_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Task-aware route between Safe Adaptive and guarded compact context.

    Distributed or multi-hop tasks use the safer answer-aware ladder. Focused or
    concentrated-evidence tasks start from guarded Adaptive-k and compact
    evidence, with answer-aware expansion if the first answer looks weak.
    """
    guarded_budget = guarded_adaptive_k_budget(
        query,
        ranked_docs,
        config.prompt_style,
        config.max_output_tokens,
        max_budget=10,
    )
    use_safe_adaptive, task_route_reason = task_should_use_safe_adaptive(query, ranked_docs, config)
    if use_safe_adaptive:
        return answer_aware_fallback_run(
            query=query,
            ranked_docs=ranked_docs,
            sequential_budget=max(guarded_budget, 3),
            config=config,
        )

    start_budget = guarded_budget
    start_budget = min(max(1, start_budget), len(ranked_docs), 10)

    full_docs = [doc for doc, _score in ranked_docs[:start_budget]]
    compact_docs = compress_documents(query, full_docs, "evidence_ngram_neighbors")
    use_compact, route_reason = routed_guarded_should_use_compact(
        query,
        ranked_docs,
        full_docs,
        compact_docs,
        config,
    )

    stages: list[tuple[str, int, str]] = []
    seen_stages: set[tuple[int, str]] = set()

    def add_stage(name: str, budget: int, compression_mode: str) -> None:
        budget = min(max(1, budget), len(ranked_docs), 10)
        key = (budget, compression_mode)
        if key not in seen_stages:
            stages.append((name, budget, compression_mode))
            seen_stages.add(key)

    if use_compact:
        add_stage(
            f"guarded_{start_budget}_compact:{task_route_reason}:{route_reason}",
            start_budget,
            "evidence_ngram_neighbors",
        )
        add_stage(f"guarded_{start_budget}_full", start_budget, "full")
        if config.prompt_style != "concise":
            for candidate in [5, 8, 10]:
                if candidate > start_budget:
                    add_stage(f"top_{candidate}_compact", candidate, "evidence_ngram_neighbors")
                    break
    else:
        add_stage(f"guarded_{start_budget}_full:{task_route_reason}:{route_reason}", start_budget, "full")

    if config.prompt_style == "concise":
        for candidate in [7, 10]:
            if candidate > start_budget:
                add_stage(f"top_{candidate}_full", candidate, "full")
    elif start_budget < 10:
        add_stage("top_10_full", 10, "full")

    return run_answer_checked_stages(
        query=query,
        ranked_docs=ranked_docs,
        stages=stages,
        config=config,
    )


def guarded_predicate_compact_context(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[list[Document], str]:
    """Compress only the safer Guarded Adaptive-k document set."""
    guarded_budget = guarded_adaptive_k_budget(
        query,
        ranked_docs,
        config.prompt_style,
        config.max_output_tokens,
        max_budget=10,
    )
    guarded_ranked = ranked_docs[:guarded_budget]
    guarded_docs = [doc for doc, _score in guarded_ranked]

    if not guarded_docs:
        return [], "guarded_predicate_empty"

    if config.prompt_style == "concise":
        pool_size = max(1, guarded_budget)
        if needs_bridge_evidence(query, config.prompt_style):
            docs = bridge_preserving_candidate_brief(
                query,
                guarded_ranked,
                token_budget=430,
                pool_size=pool_size,
                max_sentences=min(8, max(4, guarded_budget * 2 + 1)),
                min_sources=min(max(2, guarded_budget), 3),
            )
            return docs, f"guarded_bridge_compact_k_{guarded_budget}"
        docs = candidate_answer_brief(
            query,
            guarded_ranked,
            token_budget=390,
            pool_size=pool_size,
            max_sentences=min(7, max(4, guarded_budget * 2 + 1)),
            min_sentences=min(4, max(2, guarded_budget)),
        )
        return docs, f"guarded_candidate_compact_k_{guarded_budget}"

    if config.max_output_tokens > 120:
        return guarded_docs, f"guarded_full_long_answer_k_{guarded_budget}"

    docs = merged_evidence_brief(
        query,
        guarded_ranked,
        token_budget=560,
        pool_size=guarded_budget,
        max_sentences=min(9, max(5, guarded_budget * 2 + 1)),
        min_sentences=min(5, max(2, guarded_budget + 1)),
    )
    return docs, f"guarded_merged_compact_k_{guarded_budget}"


def guarded_predicate_compact_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Guarded Adaptive-k with task-shaped compact evidence."""
    selected_docs, context_shape = guarded_predicate_compact_context(query, ranked_docs, config)
    answer_config = config_for_answer_call(config, context_shape)
    answer = generate_answer(query, selected_docs, answer_config)
    return answer, selected_docs, False, context_shape, answer.total_tokens, 0


def conversational_current_question(query: Query) -> str:
    marker = "Current question:"
    if marker in query.text:
        return query.text.split(marker, 1)[1].strip()
    return query.text


def discourse_sentence_score(
    query: Query,
    sentence: str,
    index: int,
) -> float:
    current_question = conversational_current_question(query)
    question_terms = content_word_set(current_question)
    full_terms = content_word_set(query.text)
    sentence_terms = content_word_set(sentence)
    if not sentence_terms:
        return 0.0
    overlap = overlap_ratio(question_terms, sentence_terms)
    full_overlap = overlap_ratio(full_terms, sentence_terms)
    phrase = phrase_overlap_score(tokenize(current_question), tokenize(sentence))
    entity_bonus = 0.20 if capitalized_terms(query.text) & capitalized_terms(sentence) else 0.0
    answer_type_bonus = 0.25 if sentence_has_answer_type_candidate(query, sentence) else 0.0
    early_bonus = 1 / (1 + index)
    return (
        (1.4 * overlap)
        + (0.55 * full_overlap)
        + (0.70 * phrase)
        + entity_bonus
        + answer_type_bonus
        + (0.18 * early_bonus)
    )


def discourse_preserving_compact_context(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[list[Document], str]:
    """Keep discourse continuity for conversational/context-dependent QA."""
    if not ranked_docs:
        return [], "discourse_empty"

    adaptive_budget = max(1, adaptive_k_budget(ranked_docs, max_budget=10))
    budget = max(2, adaptive_budget) if is_conversational_query(query) else adaptive_budget
    budget = min(budget, len(ranked_docs), 5)
    token_budget = 620 if config.max_output_tokens <= 120 else 820

    units: list[tuple[str, str, str]] = []
    used = set()
    total = 0

    for rank, (doc, _score) in enumerate(ranked_docs[:budget], start=1):
        title = title_from_doc_id(doc.doc_id)
        sentences = split_sentences(doc.text)
        if not sentences:
            continue

        # Preserve the story/document lead; conversational answers often need
        # discourse setup rather than only the local answer sentence.
        for index in [0, 1]:
            if index < len(sentences):
                key = (doc.doc_id, index)
                cost = estimate_tokens(title) + estimate_tokens(sentences[index]) + 7
                if key not in used and total + cost <= token_budget:
                    units.append((doc.doc_id, title, sentences[index]))
                    used.add(key)
                    total += cost

        scored = []
        for index, sentence in enumerate(sentences):
            score = discourse_sentence_score(query, sentence, index)
            if score > 0:
                scored.append((score, index, sentence))

        for _score, index, _sentence in sorted(scored, reverse=True)[:3]:
            for neighbor in [index - 1, index, index + 1]:
                if not 0 <= neighbor < len(sentences):
                    continue
                key = (doc.doc_id, neighbor)
                if key in used:
                    continue
                sentence = sentences[neighbor]
                cost = estimate_tokens(title) + estimate_tokens(sentence) + 7
                if total + cost > token_budget and units:
                    continue
                units.append((doc.doc_id, title, sentence))
                used.add(key)
                total += cost

    if not units:
        return [doc for doc, _score in ranked_docs[:budget]], f"discourse_full_fallback_k_{budget}"
    return grouped_sentence_pack(units), f"discourse_compact_k_{budget}"


def discourse_preserving_compact_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    selected_docs, context_shape = discourse_preserving_compact_context(query, ranked_docs, config)
    answer_config = config_for_answer_call(config, context_shape)
    answer = generate_answer(query, selected_docs, answer_config)
    return answer, selected_docs, False, context_shape, answer.total_tokens, 0


def merged_evidence_brief_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Generate from a task-adaptive cross-document evidence brief."""
    selected_docs, context_shape = task_adaptive_brief(
        query,
        ranked_docs,
        config.prompt_style,
        allow_synthesis_pack=False,
    )
    answer_config = config_for_answer_call(config, context_shape)
    answer = generate_answer(query, selected_docs, answer_config)
    return answer, selected_docs, False, "", answer.total_tokens, 0


def answer_aware_fallback_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
    config: LLMConfig,
    compact_compression_mode: str = "evidence_ngram_neighbors",
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Safe Adaptive Context.

    The model starts cheap and only expands if the answer looks risky.

    Step 1: decide the task style.
    Step 2: choose a context policy from that task style.
    Step 3: generate an answer and check if it looks weak.
    Step 4: expand only when needed.

    This keeps the idea as one model:
    - short-answer tasks start with the smallest full context, because exact
      names/dates/places can be lost by compression and distracted by top-10
    - evidence-heavy tasks use the adaptive budget and try compact evidence
      first, because full documents can be noisy and expensive
    - fallback checks if the first choice was too weak
    - full top-10 is the final safety fallback
    """
    # The prompt style tells us the type of task:
    # concise = short exact answers, so start small and keep exact wording.
    # default = evidence-style answers, so use adaptive k and compact evidence.
    short_answer_task = config.prompt_style == "concise"
    if short_answer_task:
        # Short-answer / multi-hop tasks should keep broad retrieval coverage,
        # but full top-10 is expensive. Pack high-utility sentences from top-10
        # first, and only fall back to full top-10 if the answer is broken.
        stages = [
            ("top_10_token_budget_pack", 10, "token_budget_greedy_pack"),
            ("top_10_full", 10, "full"),
        ]
    else:
        first_budget = min(max(sequential_budget, 3), 10)
        stages = [
            # name, number of docs, compression mode
            (f"top_{first_budget}_compact", first_budget, compact_compression_mode),
            (f"top_{first_budget}_full", first_budget, "full"),
        ]
        # If the first budget was not already 10, try the next larger compact budget.
        for candidate in [5, 8, 10]:
            if candidate > first_budget:
                stages.append((f"top_{candidate}_compact", candidate, compact_compression_mode))
                break

        # Full top-10 is always the final safety option.
        already_has_top_10_full = any(
            budget == 10 and compression_mode == "full"
            for _name, budget, compression_mode in stages
        )
        if first_budget != 10 and not already_has_top_10_full:
            stages.append(("top_10_full", 10, "full"))

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_time_ms = 0
    token_sources = []

    first_pass_tokens = 0
    fallback_tokens = 0
    fallback_used = False
    fallback_reasons = []

    final_answer = None
    final_docs = []

    for stage_index, (stage_name, budget, compression_mode) in enumerate(stages):
        full_docs = [doc for doc, _score in ranked_docs[:budget]]
        if compression_mode == "full":
            selected_docs = full_docs
        elif compression_mode == "token_budget_greedy_pack":
            selected_docs = token_budget_greedy_pack(query, ranked_docs, token_budget=950, pool_size=10)
        else:
            selected_docs = compress_documents(query, full_docs, compression_mode)

        prompt_style = "default" if config.prompt_style == "anchor" else config.prompt_style
        answer_config = config_for_answer_call(config, compression_mode, prompt_style)
        answer = generate_answer(query, selected_docs, answer_config)

        total_prompt_tokens += answer.prompt_tokens
        total_completion_tokens += answer.completion_tokens
        total_tokens += answer.total_tokens
        total_time_ms += answer.generation_time_ms
        token_sources.append(answer.token_source)

        if stage_index == 0:
            first_pass_tokens = answer.total_tokens
        else:
            fallback_used = True
            fallback_tokens += answer.total_tokens

        final_answer = answer
        final_docs = selected_docs

        if short_answer_task:
            should_expand, reason = short_answer_needs_fallback(query, answer.text, selected_docs)
        else:
            should_expand, reason = answer_needs_fallback(query, answer.text, selected_docs)
        if not should_expand or stage_index == len(stages) - 1:
            if reason:
                fallback_reasons.append(f"{stage_name}:{reason}")
            break

        fallback_reasons.append(f"{stage_name}:{reason}")

    combined_answer = GeneratedAnswer(
        text=final_answer.text,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        token_source=combine_token_sources(token_sources),
        generation_time_ms=total_time_ms,
    )
    return (
        combined_answer,
        final_docs,
        fallback_used,
        ";".join(fallback_reasons),
        first_pass_tokens,
        fallback_tokens,
    )


def safe_adaptive_v2_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
    compact_compression_mode: str = "evidence_ngram_neighbors",
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Safe Adaptive Context with guarded starting budget and capped fallback.

    This keeps the original Safe Adaptive idea but folds in the clean lesson
    from the Adaptive-k experiments: use a guarded retrieval-score budget as the
    first evidence budget, and avoid expensive full top-10 recovery for focused
    claim/factoid cases unless the task evidence looks distributed.
    """
    if config.prompt_style == "concise":
        return answer_aware_fallback_run(
            query=query,
            ranked_docs=ranked_docs,
            sequential_budget=10,
            config=config,
            compact_compression_mode=compact_compression_mode,
        )

    guarded_budget = guarded_adaptive_k_budget(
        query,
        ranked_docs,
        config.prompt_style,
        config.max_output_tokens,
        max_budget=10,
    )
    first_budget = min(max(guarded_budget, 3), len(ranked_docs), 10)
    complexity = query_complexity(query, config.prompt_style)
    shape = query_shape(query, config.prompt_style)
    concentrated = retrieval_evidence_is_concentrated(ranked_docs)

    stages: list[tuple[str, int, str]] = [
        (f"guarded_{first_budget}_compact", first_budget, compact_compression_mode),
        (f"guarded_{first_budget}_full", first_budget, "full"),
    ]

    if complexity == "distributed" or (shape == "factoid" and not concentrated) or config.max_output_tokens > 120:
        for candidate in [5, 8, 10]:
            if candidate > first_budget:
                stages.append((f"top_{candidate}_compact", candidate, compact_compression_mode))
                break
        if first_budget < 10:
            stages.append(("top_10_full", 10, "full"))
    else:
        for candidate in [5, 8]:
            if candidate > first_budget:
                stages.append((f"top_{candidate}_compact", candidate, compact_compression_mode))
                break

    return run_answer_checked_stages(
        query=query,
        ranked_docs=ranked_docs,
        stages=stages,
        config=config,
    )


def flare_lite_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """FLARE-style orientation baseline for APIs without token uncertainty.

    Official FLARE actively retrieves while generating when the next sentence
    looks uncertain. Our hosted API path does not expose a comparable token-level
    uncertainty signal, so this lightweight baseline tests the closest fair
    control idea inside this codebase: answer from very small evidence first,
    then expand only if the generated answer looks unsupported by that evidence.

    This should be reported as FLARE-lite / FLARE-style, not as official FLARE.
    """
    official_budget = adaptive_k_paper_budget(
        ranked_docs,
        max_budget=10,
        buffer_docs=5,
        search_fraction=0.9,
    )
    official_budget = min(max(official_budget, 3), len(ranked_docs), 10)

    stages: list[tuple[str, int, str]] = [("top_1_full", 1, "full")]
    if official_budget > 1:
        stages.append((f"official_adaptive_k_{official_budget}_full", official_budget, "full"))
    if official_budget < 10 and len(ranked_docs) >= 10:
        stages.append(("top_10_full", 10, "full"))

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_time_ms = 0
    token_sources = []

    first_pass_tokens = 0
    fallback_tokens = 0
    fallback_used = False
    fallback_reasons = []
    final_answer = None
    final_docs: list[Document] = []

    for stage_index, (stage_name, budget, compression_mode) in enumerate(stages):
        selected_docs = [doc for doc, _score in ranked_docs[:budget]]
        prompt_style = "default" if config.prompt_style == "anchor" else config.prompt_style
        answer_config = config_for_answer_call(config, compression_mode, prompt_style)
        answer = generate_answer(query, selected_docs, answer_config)

        total_prompt_tokens += answer.prompt_tokens
        total_completion_tokens += answer.completion_tokens
        total_tokens += answer.total_tokens
        total_time_ms += answer.generation_time_ms
        token_sources.append(answer.token_source)

        if stage_index == 0:
            first_pass_tokens = answer.total_tokens
        else:
            fallback_used = True
            fallback_tokens += answer.total_tokens

        final_answer = answer
        final_docs = selected_docs

        should_expand, reason = answer_needs_fallback(query, answer.text, selected_docs)
        if not should_expand or stage_index == len(stages) - 1:
            if reason:
                fallback_reasons.append(f"{stage_name}:{reason}")
            break
        fallback_reasons.append(f"{stage_name}:{reason}")

    combined_answer = GeneratedAnswer(
        text=final_answer.text,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        token_source=combine_token_sources(token_sources),
        generation_time_ms=total_time_ms,
    )
    return (
        combined_answer,
        final_docs,
        fallback_used,
        ";".join(fallback_reasons),
        first_pass_tokens,
        fallback_tokens,
    )


def hybrid_safe_adaptive_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
    """Safe Adaptive Context with hybrid semantic/lexical evidence selection."""
    return answer_aware_fallback_run(
        query=query,
        ranked_docs=ranked_docs,
        sequential_budget=sequential_budget,
        config=config,
        compact_compression_mode="evidence_hybrid",
    )


def combine_token_sources(sources: list[str]) -> str:
    return ",".join(sorted(set(sources)))


def selected_doc_ids_for_metric(selected_docs: list[Document]) -> list[str]:
    # Fallback methods can concatenate compact first-pass docs with full fallback
    # docs. Deduplicate ids while keeping their first-seen order for ranking metrics.
    seen = set()
    doc_ids = []
    for doc in selected_docs:
        if doc.doc_id not in seen:
            seen.add(doc.doc_id)
            doc_ids.append(doc.doc_id)
    return doc_ids


def context_ndcg_at_10(selected_docs: list[Document], query: Query) -> float:
    return round(ndcg_at_k(selected_doc_ids_for_metric(selected_docs), query.relevant_doc_ids, k=10), 6)


def context_mrr_at_10(selected_docs: list[Document], query: Query) -> float:
    doc_ids = selected_doc_ids_for_metric(selected_docs)[:10]
    for index, doc_id in enumerate(doc_ids, start=1):
        if doc_id in query.relevant_doc_ids:
            return round(1 / index, 6)
    return 0.0


def method_display_name(mode: str, budget_mode: str, compression_mode: str) -> str:
    # These names are for the report/presentation.
    # We keep the technical mode too, but the method name makes tables easier to read.
    if mode == ANSWER_AWARE_FALLBACK_MODE:
        return "Safe Adaptive Context"
    if mode == SAFE_ADAPTIVE_V2_MODE:
        return "Safe Adaptive Context v2"
    if mode == LLMLINGUA2_TOP10_MODE:
        return "Fixed Top-10 + LLMLingua-2"
    if mode == LLMLINGUA2_ADAPTIVE_K_OFFICIAL_MODE:
        return "Adaptive-k (paper) + LLMLingua-2"
    if mode == FLARE_LITE_MODE:
        return "FLARE-lite"
    if mode == COVERAGE_GUIDED_ADAPTIVE_MODE:
        return "Coverage-Guided Safe Adaptive"
    if mode == COVERAGE_GUIDED_ULTRA_MODE:
        return "Coverage-Guided Ultra"
    if mode == TASK_AWARE_COVERAGE_ULTRA_MODE:
        return "TACER"
    if mode == ROUTED_PREDICATE_ADAPTIVE_MODE:
        return "Routed Predicate Adaptive"
    if mode == ROUTED_GUARDED_ADAPTIVE_MODE:
        return "Routed Guarded Adaptive Context"
    if mode == ROUTED_SAFE_GUARDED_ADAPTIVE_MODE:
        return "Routed Safe Guarded Adaptive Context"
    if mode == GUARDED_ADAPTIVE_K_MODE or budget_mode == GUARDED_ADAPTIVE_K_MODE:
        return "Guarded Adaptive-k"
    if mode == GUARDED_PREDICATE_COMPACT_MODE:
        return "Guarded Predicate Compact"
    if mode == DISCOURSE_PRESERVING_COMPACT_MODE:
        return "Discourse-Preserving Compact"
    if mode == MERGED_EVIDENCE_BRIEF_MODE:
        return "Merged Evidence Brief"
    if mode == HYBRID_SAFE_ADAPTIVE_MODE:
        return "Hybrid Safe Adaptive"

    if budget_mode == "no_retrieval":
        return "No Retrieval"
    if budget_mode == "fixed_3":
        return "Fixed Small Context" if compression_mode == "full" else "Fixed Small + Compact Evidence"
    if budget_mode == "fixed_5":
        return "Fixed Medium Context" if compression_mode == "full" else "Fixed Medium + Compact Evidence"
    if budget_mode == "fixed_7":
        return "Fixed Large Context" if compression_mode == "full" else "Fixed Large + Compact Evidence"
    if budget_mode == "fixed_10":
        return "Fixed Full Context" if compression_mode == "full" else "Compressed Fixed Full Context"
    if budget_mode == "heuristic_rules":
        return "Heuristic Rules" if compression_mode == "full" else "Heuristic Rules + Compact Evidence"
    if budget_mode == "adaptive_k":
        return "Adaptive-k" if compression_mode == "full" else "Adaptive-k + Compact Evidence"

    if budget_mode == "learned_budget":
        return "Basic Adaptive Budget" if compression_mode == "full" else "Basic Adaptive + Compact Evidence"
    if budget_mode == "learned_compensated_budget":
        return (
            "Compensated Adaptive Budget"
            if compression_mode == "full"
            else "Compensated Adaptive + Compact Evidence"
        )
    if budget_mode == "sequential_sufficiency_budget":
        return "Sequential Adaptive Budget" if compression_mode == "full" else "Compact Adaptive Context"
    if budget_mode == "oracle_dynamic_budget":
        return "Oracle Dynamic Budget" if compression_mode == "full" else "Oracle Dynamic + Compact Evidence"

    return mode


def adaptive_k_budget(ranked_docs: list[tuple[Document, float]], max_budget: int = 10) -> int:
    """Select k from the largest gap in the retrieval-score distribution.

    This follows the default Adaptive-k "largest_gap" thresholding rule:
    given one sorted candidate list, choose the cutoff at the largest adjacent
    similarity-score drop. It uses only retrieval scores, requires no tuning,
    and makes no extra LLM call.
    """
    scores = [score for _doc, score in ranked_docs[:max_budget]]
    if not scores:
        return 0
    if len(scores) == 1:
        return min(len(scores), max_budget)

    gaps = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
    largest_gap = max(gaps)
    if largest_gap <= 0:
        return min(len(scores), max_budget)
    return gaps.index(largest_gap) + 1


def adaptive_k_paper_budget(
    ranked_docs: list[tuple[Document, float]],
    max_budget: int = 10,
    buffer_docs: int = 5,
    search_fraction: float = 0.9,
) -> int:
    """Paper-faithful Adaptive-k cutoff with the paper's fixed buffer.

    Taguchi et al. choose the largest adjacent similarity-score gap, then add
    a small fixed buffer after the cutoff. They set the buffer to 5 in their
    experiments and restrict gap search away from the least-relevant tail. This
    implementation adapts that rule to our top-10 candidate lists.
    """
    limit = min(len(ranked_docs), max_budget)
    if limit <= 1:
        return limit

    scores = [score for _doc, score in ranked_docs[:limit]]
    gap_count = max(1, min(limit - 1, math.ceil((limit - 1) * search_fraction)))
    gaps = [scores[index] - scores[index + 1] for index in range(gap_count)]
    largest_gap = max(gaps)
    if largest_gap <= 0:
        cutoff = limit
    else:
        cutoff = gaps.index(largest_gap) + 1
    return min(limit, cutoff + buffer_docs)


def guarded_adaptive_k_budget(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    prompt_style: str,
    max_output_tokens: int,
    max_budget: int = 10,
) -> int:
    """Adaptive-k with general guards for task shape and rank uncertainty.

    The base cutoff is still the largest retrieval-score gap. The guards only
    prevent obviously fragile budgets, such as k=1 for relation questions or
    long-answer settings.
    """
    if not ranked_docs:
        return 0

    budget = max(1, adaptive_k_budget(ranked_docs, max_budget=max_budget))
    limit = min(len(ranked_docs), max_budget)
    scores = [score for _doc, score in ranked_docs[:limit]]
    top_score = scores[0] if scores else 0.0
    shape = query_shape(query, prompt_style)
    complexity = query_complexity(query, prompt_style)

    if needs_bridge_evidence(query, prompt_style):
        budget = max(budget, 3)
    elif prompt_style == "concise" and complexity in {"moderate", "distributed"}:
        budget = max(budget, 2)

    if max_output_tokens > 120:
        budget = max(budget, 3)
    if shape == "claim" and complexity != "focused":
        budget = max(budget, 3)

    if len(scores) >= 5 and top_score > 0:
        normalized = [score / top_score for score in scores]
        top3_spread = normalized[0] - normalized[2]
        top5_mass = sum(max(0.0, value) for value in normalized[:5])
        selected_mass = sum(max(0.0, value) for value in normalized[:budget])
        if top3_spread < 0.08:
            budget = max(budget, 3)
        if budget < 5 and top5_mass > 0 and selected_mass / top5_mass < 0.58:
            budget += 1

    if top_score > 0 and len(scores) >= 2:
        second_ratio = scores[1] / top_score
        if second_ratio < 0.35 and not needs_bridge_evidence(query, prompt_style) and max_output_tokens <= 120:
            budget = min(budget, 2)

    return min(max(1, budget), limit)


def selected_docs_for_mode(
    mode: str,
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    predicted_budget: int,
    compensated_budget: int,
    sequential_budget: int,
    oracle_budget: int,
    prompt_style: str = "default",
    max_output_tokens: int = 80,
) -> list[Document]:
    if mode == "no_retrieval":
        return []
    # Fixed modes pass the first k retrieved documents.
    if mode.startswith("fixed_"):
        budget = int(mode.removeprefix("fixed_"))
    # heuristic_rules is the simple hand-written baseline:
    # if the ranking is confident after 3 or 5 documents, stop early;
    # otherwise keep 7 documents. This is intentionally explainable.
    elif mode == "heuristic_rules":
        scores = [score for _doc, score in ranked_docs[:10]]
        top_score = scores[0] if scores else 0.0
        gap_3_to_4 = (scores[2] - scores[3]) / top_score if len(scores) > 3 and top_score > 0 else 0.0
        gap_5_to_6 = (scores[4] - scores[5]) / top_score if len(scores) > 5 and top_score > 0 else 0.0
        query_length = len(tokenize(query.text))
        if gap_3_to_4 >= 0.10 and query_length <= 12:
            budget = 3
        elif gap_5_to_6 >= 0.05:
            budget = 5
        else:
            budget = 7
    elif mode == "adaptive_k":
        budget = adaptive_k_budget(ranked_docs, max_budget=10)
    elif mode == ADAPTIVE_K_PAPER_MODE:
        budget = adaptive_k_paper_budget(ranked_docs, max_budget=10)
    elif mode == ADAPTIVE_K_OFFICIAL_MODE:
        budget = adaptive_k_paper_budget(
            ranked_docs,
            max_budget=10,
            buffer_docs=5,
            search_fraction=0.9,
        )
    elif mode == GUARDED_ADAPTIVE_K_MODE:
        budget = guarded_adaptive_k_budget(
            query,
            ranked_docs,
            prompt_style=prompt_style,
            max_output_tokens=max_output_tokens,
            max_budget=10,
        )
    # learned_budget uses the budget predicted by the robust binary controller.
    elif mode == "learned_budget":
        budget = predicted_budget
    elif mode == "learned_compensated_budget":
        budget = compensated_budget
    elif mode == "sequential_sufficiency_budget":
        budget = sequential_budget
    # oracle_dynamic_budget is an upper-bound comparison, not a deployable strategy.
    elif mode == "oracle_dynamic_budget":
        budget = oracle_budget
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return [doc for doc, _score in ranked_docs[:budget]]


def run_llm_budget_experiment(
    documents: list[Document],
    queries: list[Query],
    dev_ratio: float,
    config: LLMConfig,
    max_eval_queries: int | None = None,
    eval_start_index: int = 0,
    modes: list[str] | None = None,
    compression_modes: list[str] | None = None,
    oracle_strategy: str = "minimum_sufficient",
    sufficiency_ratio: float = 0.95,
    threshold_strategy: str = "heuristic",
    dev_queries_override: list[Query] | None = None,
    eval_ranked_override: dict[str, list[tuple[Document, float]]] | None = None,
) -> tuple[list[LLMRunRow], list[dict[str, object]], list[dict[str, object]]]:
    # Reuse the learned-budget training/evaluation path so the LLM experiment
    # tests exactly the same budget controller as run_learned_budget.py.
    if dev_queries_override is None:
        dev_queries, eval_queries = split_queries(queries, dev_ratio)
    else:
        dev_queries = dev_queries_override
        eval_queries = list(queries)
    if eval_start_index:
        eval_queries = eval_queries[eval_start_index:]
    if max_eval_queries is not None:
        eval_queries = eval_queries[:max_eval_queries]
    dev_examples, _dev_ranked = build_examples(
        documents,
        dev_queries,
        oracle_strategy=oracle_strategy,
        sufficiency_ratio=sufficiency_ratio,
    )
    if eval_ranked_override is None:
        eval_examples, eval_ranked = build_examples(
            documents,
            eval_queries,
            oracle_strategy=oracle_strategy,
            sufficiency_ratio=sufficiency_ratio,
        )
    else:
        eval_ranked = {
            query.query_id: eval_ranked_override[query.query_id]
            for query in eval_queries
            if query.query_id in eval_ranked_override
        }
        missing_query_ids = [query.query_id for query in eval_queries if query.query_id not in eval_ranked]
        if missing_query_ids:
            preview = ", ".join(missing_query_ids[:5])
            raise ValueError(f"Missing precomputed rankings for {len(missing_query_ids)} eval queries: {preview}")
        eval_examples = examples_from_rankings(
            eval_queries,
            eval_ranked,
            oracle_strategy=oracle_strategy,
            sufficiency_ratio=sufficiency_ratio,
        )
    model = train_centroid_model(dev_examples, threshold_strategy=threshold_strategy)
    retrieval_metrics, predictions = evaluate_learned_budget(eval_queries, eval_examples, eval_ranked, model)

    # Predictions are keyed by query id so each mode can reuse the learned and oracle budgets.
    prediction_by_query = {str(row["query_id"]): row for row in predictions}

    answer_rows: list[LLMRunRow] = []
    selected_modes = modes or [
        *(f"fixed_{budget}" for budget in BUDGETS),
        "learned_budget",
        "learned_compensated_budget",
        "sequential_sufficiency_budget",
        "oracle_dynamic_budget",
    ]
    selected_compression_modes = compression_modes or [config.compression_mode]
    for query in eval_queries:
        ranked_docs = eval_ranked[query.query_id]
        prediction = prediction_by_query[query.query_id]
        predicted_budget = int(prediction["predicted_budget"])
        compensated_budget = int(prediction["compensated_budget"])
        sequential_budget = int(prediction["sequential_budget"])
        oracle_budget = int(prediction["oracle_budget"])

        for mode in selected_modes:
            if mode == ANSWER_AWARE_FALLBACK_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    answer_aware_fallback_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        sequential_budget=sequential_budget,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=ANSWER_AWARE_FALLBACK_MODE,
                        method_name=method_display_name(
                            ANSWER_AWARE_FALLBACK_MODE,
                            ANSWER_AWARE_FALLBACK_MODE,
                            "compact_then_full_fallback",
                        ),
                        budget_mode=ANSWER_AWARE_FALLBACK_MODE,
                        compression_mode="compact_then_full_fallback",
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
                )
                continue

            if mode in {LLMLINGUA2_TOP10_MODE, LLMLINGUA2_ADAPTIVE_K_OFFICIAL_MODE}:
                if mode == LLMLINGUA2_TOP10_MODE:
                    selected_docs = [doc for doc, _score in ranked_docs[:10]]
                    compression_mode = "llmlingua2_top10"
                else:
                    budget = adaptive_k_paper_budget(
                        ranked_docs,
                        max_budget=10,
                        buffer_docs=5,
                        search_fraction=0.9,
                    )
                    selected_docs = [doc for doc, _score in ranked_docs[:budget]]
                    compression_mode = "llmlingua2_adaptive_k_official"
                answer = generate_llmlingua2_answer(query, selected_docs, config)
                answer_rows.append(
                    LLMRunRow(
                        mode=mode,
                        method_name=method_display_name(mode, mode, compression_mode),
                        budget_mode=mode,
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
                    )
                )
                continue

            if mode == FLARE_LITE_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    flare_lite_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=FLARE_LITE_MODE,
                        method_name=method_display_name(
                            FLARE_LITE_MODE,
                            FLARE_LITE_MODE,
                            "answer_triggered_retrieval",
                        ),
                        budget_mode=FLARE_LITE_MODE,
                        compression_mode="answer_triggered_retrieval",
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
                )
                continue

            if mode == SAFE_ADAPTIVE_V2_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    safe_adaptive_v2_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=SAFE_ADAPTIVE_V2_MODE,
                        method_name=method_display_name(
                            SAFE_ADAPTIVE_V2_MODE,
                            SAFE_ADAPTIVE_V2_MODE,
                            "guarded_safe_fallback",
                        ),
                        budget_mode=SAFE_ADAPTIVE_V2_MODE,
                        compression_mode="guarded_safe_fallback",
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
                )
                continue

            if mode == COVERAGE_GUIDED_ADAPTIVE_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    coverage_guided_adaptive_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=COVERAGE_GUIDED_ADAPTIVE_MODE,
                        method_name=method_display_name(
                            COVERAGE_GUIDED_ADAPTIVE_MODE,
                            COVERAGE_GUIDED_ADAPTIVE_MODE,
                            "coverage_guided",
                        ),
                        budget_mode=COVERAGE_GUIDED_ADAPTIVE_MODE,
                        compression_mode="coverage_guided",
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
                )
                continue

            if mode == COVERAGE_GUIDED_ULTRA_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    coverage_guided_ultra_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=COVERAGE_GUIDED_ULTRA_MODE,
                        method_name=method_display_name(
                            COVERAGE_GUIDED_ULTRA_MODE,
                            COVERAGE_GUIDED_ULTRA_MODE,
                            "coverage_guided_ultra",
                        ),
                        budget_mode=COVERAGE_GUIDED_ULTRA_MODE,
                        compression_mode="coverage_guided_ultra",
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
                )
                continue

            if mode == TASK_AWARE_COVERAGE_ULTRA_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    task_aware_coverage_ultra_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        sequential_budget=sequential_budget,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=TASK_AWARE_COVERAGE_ULTRA_MODE,
                        method_name=method_display_name(
                            TASK_AWARE_COVERAGE_ULTRA_MODE,
                            TASK_AWARE_COVERAGE_ULTRA_MODE,
                            "task_aware_ultra",
                        ),
                        budget_mode=TASK_AWARE_COVERAGE_ULTRA_MODE,
                        compression_mode="task_aware_ultra",
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
                )
                continue

            if mode == ROUTED_PREDICATE_ADAPTIVE_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    routed_predicate_adaptive_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=ROUTED_PREDICATE_ADAPTIVE_MODE,
                        method_name=method_display_name(
                            ROUTED_PREDICATE_ADAPTIVE_MODE,
                            ROUTED_PREDICATE_ADAPTIVE_MODE,
                            "routed_predicate",
                        ),
                        budget_mode=ROUTED_PREDICATE_ADAPTIVE_MODE,
                        compression_mode="routed_predicate",
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
                )
                continue

            if mode == ROUTED_GUARDED_ADAPTIVE_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    routed_guarded_adaptive_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=ROUTED_GUARDED_ADAPTIVE_MODE,
                        method_name=method_display_name(
                            ROUTED_GUARDED_ADAPTIVE_MODE,
                            ROUTED_GUARDED_ADAPTIVE_MODE,
                            "routed_guarded",
                        ),
                        budget_mode=ROUTED_GUARDED_ADAPTIVE_MODE,
                        compression_mode="routed_guarded",
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
                )
                continue

            if mode == ROUTED_SAFE_GUARDED_ADAPTIVE_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    routed_safe_guarded_adaptive_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=ROUTED_SAFE_GUARDED_ADAPTIVE_MODE,
                        method_name=method_display_name(
                            ROUTED_SAFE_GUARDED_ADAPTIVE_MODE,
                            ROUTED_SAFE_GUARDED_ADAPTIVE_MODE,
                            "routed_safe_guarded",
                        ),
                        budget_mode=ROUTED_SAFE_GUARDED_ADAPTIVE_MODE,
                        compression_mode="routed_safe_guarded",
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
                )
                continue

            if mode == GUARDED_PREDICATE_COMPACT_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    guarded_predicate_compact_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=GUARDED_PREDICATE_COMPACT_MODE,
                        method_name=method_display_name(
                            GUARDED_PREDICATE_COMPACT_MODE,
                            GUARDED_PREDICATE_COMPACT_MODE,
                            "guarded_predicate_compact",
                        ),
                        budget_mode=GUARDED_PREDICATE_COMPACT_MODE,
                        compression_mode="guarded_predicate_compact",
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
                )
                continue

            if mode == DISCOURSE_PRESERVING_COMPACT_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    discourse_preserving_compact_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=DISCOURSE_PRESERVING_COMPACT_MODE,
                        method_name=method_display_name(
                            DISCOURSE_PRESERVING_COMPACT_MODE,
                            DISCOURSE_PRESERVING_COMPACT_MODE,
                            "discourse_preserving_compact",
                        ),
                        budget_mode=DISCOURSE_PRESERVING_COMPACT_MODE,
                        compression_mode="discourse_preserving_compact",
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
                )
                continue

            if mode == HYBRID_SAFE_ADAPTIVE_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    hybrid_safe_adaptive_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        sequential_budget=sequential_budget,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=HYBRID_SAFE_ADAPTIVE_MODE,
                        method_name=method_display_name(
                            HYBRID_SAFE_ADAPTIVE_MODE,
                            HYBRID_SAFE_ADAPTIVE_MODE,
                            "hybrid_then_full_fallback",
                        ),
                        budget_mode=HYBRID_SAFE_ADAPTIVE_MODE,
                        compression_mode="hybrid_then_full_fallback",
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
                )
                continue

            if mode == MERGED_EVIDENCE_BRIEF_MODE:
                answer, selected_docs, fallback_used, fallback_reason, first_pass_tokens, fallback_tokens = (
                    merged_evidence_brief_run(
                        query=query,
                        ranked_docs=ranked_docs,
                        config=config,
                    )
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=MERGED_EVIDENCE_BRIEF_MODE,
                        method_name=method_display_name(
                            MERGED_EVIDENCE_BRIEF_MODE,
                            MERGED_EVIDENCE_BRIEF_MODE,
                            "merged_evidence_brief",
                        ),
                        budget_mode=MERGED_EVIDENCE_BRIEF_MODE,
                        compression_mode="merged_evidence_brief",
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
                )
                continue

            full_docs = selected_docs_for_mode(
                mode,
                query,
                ranked_docs,
                predicted_budget,
                compensated_budget,
                sequential_budget,
                oracle_budget,
                prompt_style=config.prompt_style,
                max_output_tokens=config.max_output_tokens,
            )
            for compression_mode in selected_compression_modes:
                selected_docs = compress_documents(query, full_docs, compression_mode)
                answer_config = LLMConfig(
                    model=config.model,
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    request_timeout_seconds=config.request_timeout_seconds,
                    api_url=config.api_url,
                    api_key_env=config.api_key_env,
                    require_api_key=config.require_api_key,
                    dry_run=config.dry_run,
                    compression_mode=compression_mode,
                    prompt_style=config.prompt_style,
                    require_provider_tokens=config.require_provider_tokens,
                )
                answer = generate_answer(query, selected_docs, answer_config)
                strategy_name = f"{mode}_{compression_mode}"
                answer_rows.append(
                    LLMRunRow(
                        mode=strategy_name,
                        method_name=method_display_name(strategy_name, mode, compression_mode),
                        budget_mode=mode,
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
                    )
                )

    # Retrieval summary is useful side-by-side with LLM answer results.
    retrieval_summary = summarize_retrieval_metrics(retrieval_metrics)
    answer_summary = summarize_llm_rows(answer_rows)
    return answer_rows, answer_summary, retrieval_summary


def summarize_llm_rows(rows: list[LLMRunRow]) -> list[dict[str, object]]:
    # Aggregate answer quality and token cost by budget mode.
    modes = []
    for row in rows:
        if row.mode not in modes:
            modes.append(row.mode)

    summary_rows = []
    fixed_10_tokens = average(row.total_tokens for row in rows if row.mode == "fixed_10_full")
    if not fixed_10_tokens:
        fixed_10_tokens = average(row.total_tokens for row in rows if row.budget_mode == "fixed_10")
    for mode in modes:
        selected = [row for row in rows if row.mode == mode]
        total_tokens = average(row.total_tokens for row in selected)
        token_reduction = 1 - (total_tokens / fixed_10_tokens) if fixed_10_tokens else 0.0
        summary_rows.append(
            {
                "method_name": selected[0].method_name,
                "mode": mode,
                "docs_used": round(average(row.docs_used for row in selected), 6),
                "prompt_tokens": round(average(row.prompt_tokens for row in selected), 6),
                "completion_tokens": round(average(row.completion_tokens for row in selected), 6),
                "total_tokens": round(total_tokens, 6),
                "token_reduction_vs_fixed_10": round(token_reduction, 6),
                "token_source": token_source_summary(selected),
                "generation_time_ms": round(average(row.generation_time_ms for row in selected), 6),
                "fallback_rate": round(average(1.0 if row.fallback_used else 0.0 for row in selected), 6),
                "first_pass_tokens": round(average(row.first_pass_tokens for row in selected), 6),
                "fallback_tokens": round(average(row.fallback_tokens for row in selected), 6),
                "answer_f1": round(average(row.answer_f1 for row in selected), 6),
                "answer_coverage": round(average(row.answer_coverage for row in selected), 6),
                "semantic_similarity": round(average(row.semantic_similarity for row in selected), 6),
                "ndcg_at_10": round(average(row.ndcg_at_10 for row in selected), 6),
                "mrr_at_10": round(average(row.mrr_at_10 for row in selected), 6),
            }
        )
    return summary_rows


def token_source_summary(rows: list[LLMRunRow]) -> str:
    sources = sorted({row.token_source for row in rows})
    return ",".join(sources)


def average(values: Iterable[float]) -> float:
    # Convert generators to a list once so they can be safely counted and summed.
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def write_llm_outputs(
    output_dir: Path,
    answer_rows: list[LLMRunRow],
    answer_summary: list[dict[str, object]],
    retrieval_summary: list[dict[str, object]],
) -> None:
    # Keep detailed answers and aggregate summaries in separate files.
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "llm_answers_by_query.csv", [asdict(row) for row in answer_rows])
    write_csv(output_dir / "llm_summary.csv", answer_summary)
    write_csv(output_dir / "retrieval_summary.csv", retrieval_summary)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    # Shared CSV writer for all outputs in this module.
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
