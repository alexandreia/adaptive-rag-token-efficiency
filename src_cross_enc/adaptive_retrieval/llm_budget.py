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

<<<<<<< HEAD
1. Start with compact adaptive evidence.
2. Generate a first answer.
3. Check if the answer looks weak or unsupported.
4. If weak, expand to full top-10 documents and generate again.
5. Count the full cost, including the first pass and fallback pass.
=======
1. Choose an adaptive document budget.
2. Try those documents as compact evidence.
3. Score whether the answer looks weak or unsupported.
4. If weak, retry the same documents without compression.
5. If still weak, expand to more documents.
6. Count the full cost, including every attempt.
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

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
    build_examples,
    evaluate_learned_budget,
<<<<<<< HEAD
    split_queries,
    summarize as summarize_retrieval_metrics,
    train_centroid_model,
)
from adaptive_retrieval.metrics import answer_coverage, ndcg_at_k, token_f1
from adaptive_retrieval.text import estimate_tokens, tokenize, build_idf, build_index, tfidf_vector, build_doc_vectors
=======
    extract_features,
    split_queries,
    sufficiency_risk_score,
    summarize as summarize_retrieval_metrics,
    train_centroid_model,
)
from adaptive_retrieval.metrics import answer_coverage, ndcg_at_k, semantic_similarity, token_f1
from adaptive_retrieval.text import estimate_tokens, tokenize
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

PROMPT_STYLES = {"default", "concise", "anchor"}

ANSWER_AWARE_FALLBACK_MODE = "answer_aware_fallback"
<<<<<<< HEAD
PRE_GENERATION_ROUTING_MODE = "pre_generation_routing"
=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

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

<<<<<<< HEAD
NEGATION_OR_COMPLEXITY_TERMS = {
    "absent",
    "absence",
    "decrease",
    "decreased",
    "decreases",
    "inhibit",
    "inhibited",
    "inhibits",
    "lack",
    "lacks",
    "never",
    "no",
    "not",
    "reduce",
    "reduced",
    "reduces",
    "without",
}

=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD
    model: str = "mistral"
=======
    model: str = "gpt-4o-mini"
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    # Temperature is kept at zero so repeated runs are easier to compare.
    temperature: float = 0.0
    # Upper bound on generated answer length. This keeps completion-token cost controlled.
    max_output_tokens: int = 220
    # Local models can be slow, especially with long fixed_10 prompts.
<<<<<<< HEAD
    request_timeout_seconds: int = 30000
     # 👇 Ollama OpenAI-compatible endpoint
    api_url: str = "http://localhost:11434/v1/chat/completions"
    # Environment variable that stores the API key. Ollama can leave this unset.
    # 👇 Ollama does NOT use API keys
    api_key_env: str = ""
    require_api_key: bool = False
    # ⚠️ IMPORTANT: turn this OFF or it will skip real calls
    dry_run: bool = False
=======
    request_timeout_seconds: int = 300
    # API URL for OpenAI-compatible chat-completions providers.
    api_url: str = "https://api.openai.com/v1/chat/completions"
    # Environment variable that stores the API key. Ollama can leave this unset.
    api_key_env: str = "OPENAI_API_KEY"
    # Local providers such as Ollama do not require an Authorization header.
    require_api_key: bool = True
    # Dry-run mode avoids network calls and uses a simple extractive answer instead.
    dry_run: bool = True
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD
=======
    # Meaning similarity between generated answer and reference answer.
    semantic_similarity: float
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD
            "Write one short answer sentence.\n"
            "Use the same key terms as the evidence when possible.\n"
            "Do not explain your reasoning.\n"
=======
            "Write only the final answer.\n"
            "Use as few words as possible, usually 1 to 5 words.\n"
            "Use the same key terms as the evidence when possible.\n"
            "Do not explain your reasoning.\n"
            "Do not repeat the question.\n"
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD

=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        chat_completions_url(config.api_url),
        data=json.dumps(request_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
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


<<<<<<< HEAD
def answer_needs_fallback(query: Query, answer: str, selected_docs: list[Document]) -> tuple[bool, str]:
    """Return whether the compact answer looks risky enough to expand context.

    This is intentionally heuristic and cheap: a deployable controller cannot use
    gold labels, so it looks for signals available at runtime only.
=======
def answer_risk_score(query: Query, answer: str, selected_docs: list[Document]) -> tuple[int, list[str]]:
    """Score whether an answer looks risky using only runtime information.

    This does not look at the gold answer. It only uses:
    - the query
    - the selected context
    - the answer generated by the LLM

    The score is simple on purpose:
    more weak signals = more reason to expand context.
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    """
    cleaned_answer = answer.strip()
    lowered_answer = cleaned_answer.lower()
    if not cleaned_answer:
<<<<<<< HEAD
        return True, "empty_answer"

    for phrase in WEAK_ANSWER_PHRASES:
        if phrase in lowered_answer:
            return True, f"weak_phrase:{phrase}"
=======
        return 10, ["empty_answer"]

    risk = 0
    reasons = []
    for phrase in WEAK_ANSWER_PHRASES:
        if phrase in lowered_answer:
            risk += 3
            reasons.append(f"weak_phrase:{phrase}")
            break
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

    answer_terms = content_word_set(cleaned_answer)
    query_terms = content_word_set(query.text)
    context_terms = content_word_set(" ".join(doc.text for doc in selected_docs))

<<<<<<< HEAD
    # Very short answers are often refusals, fragments, or underspecified outputs.
    if len(answer_terms) < 5:
        return True, "very_short_answer"

=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    # If the answer uses terms that barely appear in the provided evidence, it may
    # be hallucinating or failing to anchor on the compact snippets.
    answer_context_overlap = overlap_ratio(answer_terms, context_terms)
    if answer_context_overlap < 0.08:
<<<<<<< HEAD
        return True, "low_answer_context_overlap"
=======
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
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

    # If the answer barely touches the query vocabulary, it may be too generic.
    answer_query_overlap = overlap_ratio(query_terms, answer_terms)
    if answer_query_overlap < 0.04:
<<<<<<< HEAD
        return True, "low_answer_query_overlap"
=======
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
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

    return False, ""


<<<<<<< HEAD
=======
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


>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
def answer_aware_fallback_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str, int, int]:
<<<<<<< HEAD
    """Generate from compact evidence first, then expand to full top-10 if risky.

    The first pass uses the best compressed strategy we found so far:
    sequential_sufficiency_budget + evidence_ngram_neighbors. The fallback uses
    full fixed_10 context because that is the strongest quality baseline in the
    real Mistral runs.
    """
    first_full_docs = [doc for doc, _score in ranked_docs[:sequential_budget]]
    first_docs = compress_documents(query, first_full_docs, "evidence_ngram_neighbors")

    # The anchor prompt failed in the real run, so the fallback controller avoids
    # inheriting it accidentally. Concise/default remain useful prompt ablations.
    first_prompt_style = "default" if config.prompt_style == "anchor" else config.prompt_style
    first_config = config_for_answer_call(config, "evidence_ngram_neighbors", first_prompt_style)
    first_answer = generate_answer(query, first_docs, first_config)

    should_fallback, fallback_reason = answer_needs_fallback(query, first_answer.text, first_docs)
    if not should_fallback:
        return first_answer, first_docs, False, "", first_answer.total_tokens, 0

    fallback_docs = [doc for doc, _score in ranked_docs[:10]]
    fallback_config = config_for_answer_call(config, "full", "default")
    fallback_answer = generate_answer(query, fallback_docs, fallback_config)

    combined_answer = GeneratedAnswer(
        text=fallback_answer.text,
        prompt_tokens=first_answer.prompt_tokens + fallback_answer.prompt_tokens,
        completion_tokens=first_answer.completion_tokens + fallback_answer.completion_tokens,
        total_tokens=first_answer.total_tokens + fallback_answer.total_tokens,
        token_source=combine_token_sources([first_answer.token_source, fallback_answer.token_source]),
        generation_time_ms=first_answer.generation_time_ms + fallback_answer.generation_time_ms,
    )
    return (
        combined_answer,
        first_docs + fallback_docs,
        True,
        fallback_reason,
        first_answer.total_tokens,
        fallback_answer.total_tokens,
    )


def retrieval_score_entropy(scores: list[float]) -> float:
    # High entropy means relevance is spread across documents instead of dominated
    # by one clear candidate, which is a cheap signal that compact context is risky.
    total = sum(score for score in scores if score > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for score in scores:
        if score <= 0:
            continue
        probability = score / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(len(scores)) if len(scores) > 1 else 0.0


def pre_generation_routing_decision(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
) -> tuple[bool, str, float]:
    """Choose compact or full context before generation using cheap signals.

    This tests the alternative to answer-aware fallback: instead of generating
    once, judging the answer, and maybe generating again, route hard-looking
    queries to full context before the first LLM call.
    """
    top_scores = [score for _doc, score in ranked_docs[:10]]
    top_five_scores = top_scores[:5]
    top_score = top_scores[0] if top_scores else 0.0
    second_score = top_scores[1] if len(top_scores) > 1 else 0.0
    score_gap = top_score - second_score
    top_five_mass = sum(top_five_scores)
    top_doc_ratio = top_score / top_five_mass if top_five_mass > 0 else 0.0
    entropy = retrieval_score_entropy(top_scores)

    compact_full_docs = [doc for doc, _score in ranked_docs[:sequential_budget]]
    compact_docs = compress_documents(query, compact_full_docs, "evidence_ngram_neighbors")
    full_top_10_docs = [doc for doc, _score in ranked_docs[:10]]
    compact_tokens = sum(estimate_tokens(doc.text) for doc in compact_docs)
    full_tokens = sum(estimate_tokens(doc.text) for doc in full_top_10_docs)
    compression_ratio = compact_tokens / full_tokens if full_tokens > 0 else 1.0

    query_terms = content_word_set(query.text)
    has_negation = bool(query_terms & NEGATION_OR_COMPLEXITY_TERMS)
    query_length = len(tokenize(query.text))

    risk_score = 0.0
    reasons = []
    if score_gap < 0.05:
        risk_score += 1.0
        reasons.append("small_score_gap")
    if top_doc_ratio < 0.25:
        risk_score += 0.8
        reasons.append("low_top_doc_ratio")
    if entropy > 0.90:
        risk_score += 0.8
        reasons.append("high_retrieval_entropy")
    if compression_ratio < 0.15:
        risk_score += 0.7
        reasons.append("heavy_compression")
    if query_length > 12:
        risk_score += 0.4
        reasons.append("long_query")
    if has_negation:
        risk_score += 0.6
        reasons.append("negation_or_polarity")

    route_to_full = risk_score >= 1.5
    reason = ",".join(reasons) if reasons else "low_risk"
    return route_to_full, reason, risk_score


def pre_generation_routing_run(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    sequential_budget: int,
    config: LLMConfig,
) -> tuple[GeneratedAnswer, list[Document], bool, str]:
    route_to_full, route_reason, risk_score = pre_generation_routing_decision(
        query=query,
        ranked_docs=ranked_docs,
        sequential_budget=sequential_budget,
    )
    if route_to_full:
        selected_docs = [doc for doc, _score in ranked_docs[:10]]
        answer_config = config_for_answer_call(config, "full", "default")
        route_label = f"routed_full:{route_reason};risk={risk_score:.2f}"
    else:
        full_docs = [doc for doc, _score in ranked_docs[:sequential_budget]]
        selected_docs = compress_documents(query, full_docs, "evidence_ngram_neighbors")
        prompt_style = "default" if config.prompt_style == "anchor" else config.prompt_style
        answer_config = config_for_answer_call(config, "evidence_ngram_neighbors", prompt_style)
        route_label = f"routed_compact:{route_reason};risk={risk_score:.2f}"

    answer = generate_answer(query, selected_docs, answer_config)
    return answer, selected_docs, route_to_full, route_label


=======
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
            (f"top_{first_budget}_compact", first_budget, "evidence_ngram_neighbors"),
            (f"top_{first_budget}_full", first_budget, "full"),
        ]
        # If the first budget was not already 10, try the next larger compact budget.
        for candidate in [5, 8, 10]:
            if candidate > first_budget:
                stages.append((f"top_{candidate}_compact", candidate, "evidence_ngram_neighbors"))
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


>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD
    if mode == PRE_GENERATION_ROUTING_MODE:
        return "Risk-Routed Context"
=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d

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


def selected_docs_for_mode(
    mode: str,
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    predicted_budget: int,
    compensated_budget: int,
    sequential_budget: int,
    oracle_budget: int,
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
<<<<<<< HEAD
=======
    eval_start_index: int = 0,
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    modes: list[str] | None = None,
    compression_modes: list[str] | None = None,
    oracle_strategy: str = "minimum_sufficient",
    sufficiency_ratio: float = 0.95,
    threshold_strategy: str = "heuristic",
) -> tuple[list[LLMRunRow], list[dict[str, object]], list[dict[str, object]]]:
    # Reuse the learned-budget training/evaluation path so the LLM experiment
    # tests exactly the same budget controller as run_learned_budget.py.
<<<<<<< HEAD

    idf = build_idf(documents)
    doc_vectors = build_doc_vectors(documents, idf)

    dev_queries, eval_queries = split_queries(queries, dev_ratio)
=======
    dev_queries, eval_queries = split_queries(queries, dev_ratio)
    if eval_start_index:
        eval_queries = eval_queries[eval_start_index:]
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    if max_eval_queries is not None:
        eval_queries = eval_queries[:max_eval_queries]
    dev_examples, _dev_ranked = build_examples(
        documents,
        dev_queries,
<<<<<<< HEAD
        idf=idf,
        doc_vectors=doc_vectors,
=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
        oracle_strategy=oracle_strategy,
        sufficiency_ratio=sufficiency_ratio,
    )
    eval_examples, eval_ranked = build_examples(
        documents,
        eval_queries,
<<<<<<< HEAD
        idf=idf,
        doc_vectors=doc_vectors,
=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD
=======
                        semantic_similarity=round(semantic_similarity(answer.text, query.reference_answer), 6),
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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

<<<<<<< HEAD
            if mode == PRE_GENERATION_ROUTING_MODE:
                answer, selected_docs, routed_full, route_reason = pre_generation_routing_run(
                    query=query,
                    ranked_docs=ranked_docs,
                    sequential_budget=sequential_budget,
                    config=config,
                )
                answer_rows.append(
                    LLMRunRow(
                        mode=PRE_GENERATION_ROUTING_MODE,
                        method_name=method_display_name(
                            PRE_GENERATION_ROUTING_MODE,
                            PRE_GENERATION_ROUTING_MODE,
                            "compact_or_full_route",
                        ),
                        budget_mode=PRE_GENERATION_ROUTING_MODE,
                        compression_mode="compact_or_full_route",
                        query_id=query.query_id,
                        docs_used=len(selected_docs),
                        prompt_tokens=answer.prompt_tokens,
                        completion_tokens=answer.completion_tokens,
                        total_tokens=answer.total_tokens,
                        token_source=answer.token_source,
                        generation_time_ms=answer.generation_time_ms,
                        answer_f1=round(token_f1(answer.text, query.reference_answer), 6),
                        answer_coverage=round(answer_coverage(answer.text, query.reference_answer), 6),
                        ndcg_at_10=context_ndcg_at_10(selected_docs, query),
                        mrr_at_10=context_mrr_at_10(selected_docs, query),
                        selected_doc_ids=json.dumps([doc.doc_id for doc in selected_docs]),
                        answer=answer.text,
                        fallback_used=routed_full,
                        fallback_reason=route_reason,
                    )
                )
                continue

=======
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
            full_docs = selected_docs_for_mode(
                mode,
                query,
                ranked_docs,
                predicted_budget,
                compensated_budget,
                sequential_budget,
                oracle_budget,
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
<<<<<<< HEAD
=======
                        semantic_similarity=round(semantic_similarity(answer.text, query.reference_answer), 6),
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
<<<<<<< HEAD
=======
                "semantic_similarity": round(average(row.semantic_similarity for row in selected), 6),
>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
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
