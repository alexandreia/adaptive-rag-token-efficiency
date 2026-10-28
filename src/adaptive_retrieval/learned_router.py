"""
Learned TACER router foundation.

The submitted TACER system uses transparent task/evidence rules. This module is
the first step toward the next version: a small learned router that predicts
which context policy to use from pre-generation signals only.

Training is teacher-student:

1. Existing saved experiment outputs define the teacher/oracle action per query.
2. Runtime-available query and retrieval-score features become the student input.
3. A lightweight multiclass linear model learns to choose the action.

The model intentionally avoids dataset identifiers and gold retrieval metrics.
It should learn evidence-routing behavior from score shape and query form, not
memorize which benchmark it is seeing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from adaptive_retrieval.text import tokenize


DEFAULT_ACTIONS = [
    "adaptive_k_full",
    "answer_aware_fallback",
    "coverage_guided_ultra",
    "task_aware_coverage_ultra",
    "fixed_10_full",
]

ACTION_DISPLAY_NAMES = {
    "adaptive_k_full": "Adaptive-k",
    "answer_aware_fallback": "Safe Adaptive Context",
    "coverage_guided_ultra": "Coverage-Guided Ultra",
    "task_aware_coverage_ultra": "TACER",
    "fixed_10_full": "Fixed Top-10",
}

FEATURE_NAMES = [
    "query_terms",
    "unique_query_terms",
    "unique_query_ratio",
    "has_question_mark",
    "numeric_token_count",
    "capitalized_token_count",
    "connective_cue_count",
    "multi_hop_cue_count",
    "long_form_cue_count",
    "top_score",
    "mean_score",
    "score_std",
    "score_range",
    "gap_1_2",
    "gap_2_3",
    "gap_3_4",
    "gap_5_6",
    "gap_1_5",
    "largest_adjacent_gap",
    "largest_gap_rank",
    "top1_mass",
    "top3_mass",
    "top5_mass",
    "tail5_mass",
    "top1_to_top5",
    "top3_to_top8",
    "score_entropy",
    "normalized_score_entropy",
]

CONNECTIVE_CUES = {
    "and",
    "both",
    "between",
    "after",
    "before",
    "while",
    "during",
    "whose",
    "which",
}

MULTI_HOP_CUES = {
    "which",
    "whose",
    "both",
    "between",
    "part",
    "conference",
    "born",
    "played",
}

LONG_FORM_CUES = {
    "explain",
    "describe",
    "summarize",
    "why",
    "how",
    "relationship",
    "compare",
}


@dataclass(frozen=True)
class RouterExample:
    dataset: str
    query_id: str
    label: str
    features: list[float]
    fixed_f1: float
    chosen_f1: float
    chosen_tokens: float
    fixed_tokens: float


@dataclass
class LearnedRouterModel:
    actions: list[str]
    feature_names: list[str]
    means: list[float]
    stdevs: list[float]
    weights: list[list[float]]
    biases: list[float]

    def predict_proba(self, features: list[float]) -> dict[str, float]:
        vector = standardize(features, self.means, self.stdevs)
        logits = [
            bias + sum(weight * value for weight, value in zip(row, vector))
            for row, bias in zip(self.weights, self.biases)
        ]
        probs = softmax(logits)
        return dict(zip(self.actions, probs))

    def predict(self, features: list[float]) -> str:
        probs = self.predict_proba(features)
        return max(probs.items(), key=lambda item: item[1])[0]

    def to_json(self) -> dict:
        return asdict(self)


def save_model(model: LearnedRouterModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: Path) -> LearnedRouterModel:
    row = json.loads(path.read_text(encoding="utf-8"))
    return LearnedRouterModel(
        actions=list(row["actions"]),
        feature_names=list(row["feature_names"]),
        means=[float(value) for value in row["means"]],
        stdevs=[float(value) for value in row["stdevs"]],
        weights=[[float(value) for value in weights] for weights in row["weights"]],
        biases=[float(value) for value in row["biases"]],
    )


def load_query_texts(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            queries[str(row["query_id"])] = str(row["text"])
    return queries


def load_retrieval_scores(path: Path, retriever: str = "tfidf") -> dict[str, list[float]]:
    scores_by_query: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("retriever") != retriever:
                continue
            scores_by_query[str(row["query_id"])] = [float(value) for value in json.loads(row["scores"])]
    return scores_by_query


def load_outcome_rows(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    rows: dict[str, dict[str, dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.setdefault(str(row["query_id"]), {})[str(row["mode"])] = row
    return rows


def build_router_examples(
    dataset: str,
    queries_path: Path,
    rankings_path: Path,
    outcomes_path: Path,
    actions: list[str] | None = None,
    quality_floor: float = 0.95,
    semantic_margin: float = 0.03,
    token_weight: float = 0.08,
) -> list[RouterExample]:
    actions = actions or DEFAULT_ACTIONS
    query_texts = load_query_texts(queries_path)
    scores_by_query = load_retrieval_scores(rankings_path)
    outcomes_by_query = load_outcome_rows(outcomes_path)
    examples: list[RouterExample] = []

    for query_id, outcomes in outcomes_by_query.items():
        fixed = outcomes.get("fixed_10_full")
        if fixed is None or query_id not in query_texts or query_id not in scores_by_query:
            continue
        label = choose_oracle_action(
            outcomes,
            actions,
            quality_floor=quality_floor,
            semantic_margin=semantic_margin,
            token_weight=token_weight,
        )
        if label is None:
            continue
        chosen = outcomes[label]
        examples.append(
            RouterExample(
                dataset=dataset,
                query_id=query_id,
                label=label,
                features=extract_router_features(query_texts[query_id], scores_by_query[query_id]),
                fixed_f1=parse_float(fixed, "answer_f1"),
                chosen_f1=parse_float(chosen, "answer_f1"),
                chosen_tokens=parse_float(chosen, "total_tokens"),
                fixed_tokens=parse_float(fixed, "total_tokens"),
            )
        )
    return examples


def choose_oracle_action(
    outcomes: dict[str, dict[str, str]],
    actions: list[str],
    quality_floor: float,
    semantic_margin: float,
    token_weight: float,
) -> str | None:
    fixed = outcomes.get("fixed_10_full")
    if fixed is None:
        return None
    fixed_f1 = parse_float(fixed, "answer_f1")
    fixed_semantic = parse_float(fixed, "semantic_similarity")
    fixed_tokens = max(1.0, parse_float(fixed, "total_tokens"))

    candidates = [outcomes[action] for action in actions if action in outcomes]
    if not candidates:
        return None

    sufficient = [
        row
        for row in candidates
        if parse_float(row, "answer_f1") >= fixed_f1 * quality_floor
        and parse_float(row, "semantic_similarity") >= fixed_semantic - semantic_margin
    ]
    if sufficient:
        return min(sufficient, key=lambda row: parse_float(row, "total_tokens"))["mode"]

    def utility(row: dict[str, str]) -> float:
        token_ratio = parse_float(row, "total_tokens") / fixed_tokens
        return (
            parse_float(row, "answer_f1")
            + 0.35 * parse_float(row, "answer_coverage")
            + 0.25 * parse_float(row, "semantic_similarity")
            - token_weight * token_ratio
        )

    return max(candidates, key=utility)["mode"]


def extract_router_features(query_text: str, scores: list[float]) -> list[float]:
    terms = tokenize(query_text)
    unique_terms = set(terms)
    lower_terms = set(terms)
    positive_scores = [max(0.0, score) for score in scores[:10]]
    score_sum = sum(positive_scores)
    mean_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    variance = (
        sum((score - mean_score) ** 2 for score in positive_scores) / len(positive_scores)
        if positive_scores
        else 0.0
    )
    adjacent_gaps = [
        normalized_gap(positive_scores, index, index + 1)
        for index in range(max(0, len(positive_scores) - 1))
    ]
    largest_gap = max(adjacent_gaps) if adjacent_gaps else 0.0
    largest_gap_rank = float(adjacent_gaps.index(largest_gap) + 1) if adjacent_gaps else 0.0

    return [
        float(len(terms)),
        float(len(unique_terms)),
        len(unique_terms) / len(terms) if terms else 0.0,
        1.0 if "?" in query_text else 0.0,
        float(sum(1 for term in terms if any(char.isdigit() for char in term))),
        float(len(re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", query_text))),
        float(len(lower_terms & CONNECTIVE_CUES)),
        float(len(lower_terms & MULTI_HOP_CUES)),
        float(len(lower_terms & LONG_FORM_CUES)),
        positive_scores[0] if positive_scores else 0.0,
        mean_score,
        math.sqrt(variance),
        (max(positive_scores) - min(positive_scores)) if positive_scores else 0.0,
        normalized_gap(positive_scores, 0, 1),
        normalized_gap(positive_scores, 1, 2),
        normalized_gap(positive_scores, 2, 3),
        normalized_gap(positive_scores, 4, 5),
        normalized_gap(positive_scores, 0, 4),
        largest_gap,
        largest_gap_rank,
        mass(positive_scores, 0, 1, score_sum),
        mass(positive_scores, 0, 3, score_sum),
        mass(positive_scores, 0, 5, score_sum),
        mass(positive_scores, 5, 10, score_sum),
        ratio(positive_scores, 0, 4),
        sum(positive_scores[:3]) / sum(positive_scores[:8]) if sum(positive_scores[:8]) else 0.0,
        entropy(positive_scores),
        normalized_entropy(positive_scores),
    ]


def train_router(
    examples: list[RouterExample],
    actions: list[str] | None = None,
    epochs: int = 600,
    learning_rate: float = 0.08,
    l2: float = 0.001,
) -> LearnedRouterModel:
    if not examples:
        raise ValueError("No router examples were provided.")
    actions = actions or sorted({example.label for example in examples})
    action_to_index = {action: index for index, action in enumerate(actions)}
    means, stdevs = feature_stats([example.features for example in examples])
    train_vectors = [standardize(example.features, means, stdevs) for example in examples]
    labels = [action_to_index[example.label] for example in examples]
    weights = [[0.0 for _ in FEATURE_NAMES] for _ in actions]
    biases = [0.0 for _ in actions]

    for _epoch in range(epochs):
        for vector, label in zip(train_vectors, labels):
            logits = [
                bias + sum(weight * value for weight, value in zip(row, vector))
                for row, bias in zip(weights, biases)
            ]
            probs = softmax(logits)
            for action_index in range(len(actions)):
                target = 1.0 if action_index == label else 0.0
                error = probs[action_index] - target
                biases[action_index] -= learning_rate * error
                for feature_index, value in enumerate(vector):
                    gradient = error * value + l2 * weights[action_index][feature_index]
                    weights[action_index][feature_index] -= learning_rate * gradient

    return LearnedRouterModel(
        actions=actions,
        feature_names=FEATURE_NAMES,
        means=means,
        stdevs=stdevs,
        weights=weights,
        biases=biases,
    )


def evaluate_router(model: LearnedRouterModel, examples: list[RouterExample]) -> dict[str, float | int]:
    if not examples:
        return {"examples": 0, "accuracy": 0.0, "token_ratio": 0.0, "f1_delta": 0.0}
    correct = 0
    token_ratios = []
    f1_deltas = []
    for example in examples:
        prediction = model.predict(example.features)
        if prediction == example.label:
            correct += 1
        # These are oracle-label outcome summaries, not predicted-action outcome
        # summaries. They tell us the target operating point represented by the
        # teacher labels in this first scaffold.
        token_ratios.append(example.chosen_tokens / example.fixed_tokens if example.fixed_tokens else 1.0)
        f1_deltas.append(example.chosen_f1 - example.fixed_f1)
    return {
        "examples": len(examples),
        "accuracy": correct / len(examples),
        "oracle_token_ratio": sum(token_ratios) / len(token_ratios),
        "oracle_f1_delta": sum(f1_deltas) / len(f1_deltas),
    }


def deterministic_split(
    examples: list[RouterExample],
    test_ratio: float = 0.2,
    seed: str = "learned-tacer-router",
) -> tuple[list[RouterExample], list[RouterExample]]:
    train: list[RouterExample] = []
    test: list[RouterExample] = []
    cutoff = int(test_ratio * 10_000)
    for example in examples:
        key = f"{seed}:{example.dataset}:{example.query_id}".encode("utf-8")
        bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 10_000
        if bucket < cutoff:
            test.append(example)
        else:
            train.append(example)
    return train, test


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def normalized_gap(scores: list[float], left: int, right: int) -> float:
    if len(scores) <= right or not scores or scores[0] <= 0:
        return 0.0
    return (scores[left] - scores[right]) / scores[0]


def mass(scores: list[float], start: int, stop: int, total: float) -> float:
    return sum(scores[start:stop]) / total if total else 0.0


def ratio(scores: list[float], numerator_index: int, denominator_index: int) -> float:
    if len(scores) <= denominator_index or scores[denominator_index] <= 0:
        return 0.0
    return scores[numerator_index] / scores[denominator_index]


def entropy(scores: list[float]) -> float:
    total = sum(scores)
    if total <= 0:
        return 0.0
    return -sum((score / total) * math.log(score / total) for score in scores if score > 0)


def normalized_entropy(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    return entropy(scores) / math.log(len(scores))


def feature_stats(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    means = [sum(column) / len(column) for column in zip(*vectors)]
    stdevs = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / len(vectors)
        stdevs.append(math.sqrt(variance) or 1.0)
    return means, stdevs


def standardize(features: list[float], means: list[float], stdevs: list[float]) -> list[float]:
    return [(value - mean) / stdev for value, mean, stdev in zip(features, means, stdevs)]


def softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    max_logit = max(logits)
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]

