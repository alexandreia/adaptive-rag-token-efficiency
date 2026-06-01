"""
Basic learned budget model.

This file implements the simple adaptive budget predictor used in the project.

The goal is to predict how much context a query needs before sending context
to the LLM. It keeps the model simple:

- no neural network
- no external training service
- only cheap retrieval/query features
- nearest-centroid style classifier

Important idea:

The oracle is not deployable because it looks at evaluation information.
But the oracle can act as a teacher. This file trains a small student model
to imitate that teacher using features available at runtime.

The final Safe Adaptive Context method builds on this idea, but adds a safety
fallback after generation.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from sentence_transformers import CrossEncoder


from adaptive_retrieval.budget_experiment import _record_metrics
from adaptive_retrieval.data import Document, Query
from adaptive_retrieval.metrics import RunMetrics
from adaptive_retrieval.retriever import retrieve
from adaptive_retrieval.text import build_idf, tfidf_vector, tokenize
from collections import defaultdict

BUDGETS = [3, 5, 8, 10]
SMALL_BUDGETS = [3, 5]
LARGE_BUDGETS = [8, 10]
SMALL_CLASS = "small"
LARGE_CLASS = "large"
ORACLE_STRATEGIES = {"best_utility", "minimum_sufficient"}
THRESHOLD_STRATEGIES = {"heuristic", "calibrated"}


# This file contains the basic adaptive budget model.
# It is intentionally simple:
# - no neural networks
# - uses retrieval/query features
# - predicts whether the query needs a small or large context budget
# The newer LLM script also uses these predictions as part of the stronger models.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

cross_encoder_cahce = {}

all_pairs = []
pair_metadata = []  # to reconstruct results
query_order = []

@dataclass(frozen=True)
class TrainingExample:
    # One training row for one query.
    query_id: str
    # Oracle budget label, one of 3, 5, 8, 10.
    label: int
    # Numeric query/retrieval features.
    features: list[float]


@dataclass
class CentroidModel:
    # Nearest-centroid classifier parameters.
    # This is the simple model used instead of a neural network.
    labels: list[str]
    means: list[float]
    stdevs: list[float]
    centroids: dict[str, list[float]]
    class_counts: dict[str, int]
    class_priors: dict[str, float]
    large_threshold: float
    very_large_threshold: float
    low_entropy_small_default: bool



def split_queries(queries: list[Query], dev_ratio: float) -> tuple[list[Query], list[Query]]:
    # Deterministic split:
    # sort by query id, then use the first part for dev/training.
    sorted_queries = sorted(queries, key=lambda query: query.query_id)
    dev_size = max(1, min(len(sorted_queries) - 1, round(len(sorted_queries) * dev_ratio)))
    return sorted_queries[:dev_size], sorted_queries[dev_size:]


def _score_values(ranked_docs: list[tuple[Document, float]]) -> list[float]:
    return [max(0.0, score) for _doc, score in ranked_docs]


def _gap(scores: list[float], left: int, right: int) -> float:
    if len(scores) <= right or scores[0] == 0:
        return 0.0
    return (scores[left] - scores[right]) / scores[0]


def _entropy(scores: list[float]) -> float:
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [score / total for score in scores if score > 0]
    return -sum(prob * math.log(prob) for prob in probs)


def _normalized_entropy(scores: list[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    return _entropy(scores) / math.log(len(scores))


def extract_features(query: Query, ranked_docs: list[tuple[Document, float]]) -> list[float]:
    # These are the cheap features available before generation.
    # They describe:
    # - query length
    # - retrieval confidence
    # - score gaps
    # - score entropy
    # - whether evidence is concentrated or spread out
    query_terms = tokenize(query.text)
    scores = _score_values(ranked_docs)
    top_score = scores[0] if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores) if scores else 0.0
    score_sum = sum(scores)
    top3_mass = sum(scores[:3]) / score_sum if score_sum else 0.0
    top5_mass = sum(scores[:5]) / score_sum if score_sum else 0.0
    top1_mass = scores[0] / score_sum if score_sum and scores else 0.0
    top1_to_top5 = scores[0] / scores[4] if len(scores) >= 5 and scores[4] > 0 else 0.0
    top3_to_top8 = sum(scores[:3]) / sum(scores[:8]) if sum(scores[:8]) else 0.0
    unique_terms = len(set(query_terms))

    return [
        float(len(query_terms)),
        float(unique_terms),
        unique_terms / len(query_terms) if query_terms else 0.0,
        top_score,
        mean_score,
        math.sqrt(variance),
        _gap(scores, 0, 1),
        _gap(scores, 2, 3),
        _gap(scores, 4, 5),
        _gap(scores, 0, 4),
        top3_mass,
        top5_mass,
        top1_mass,
        top1_to_top5,
        top3_to_top8,
        _entropy(scores),
        _normalized_entropy(scores),
    ]


def oracle_budget_for_query(
    query: Query,
    ranked_docs: list[tuple[Document, float]],
    oracle_strategy: str = "minimum_sufficient",
    sufficiency_ratio: float = 0.95,
) -> int:
    # The oracle is the teacher used for experiments.
    # It can look at evaluation information, so it is not deployable directly.
    # But it tells us what budget would have been best for a query.
    if oracle_strategy not in ORACLE_STRATEGIES:
        raise ValueError(f"Unknown oracle strategy: {oracle_strategy}")

    baseline = _record_metrics("fixed_10", 1, query, ranked_docs[:10], ranked_docs)
    budget_scores = []
    for budget in BUDGETS:
        row = _record_metrics(f"fixed_{budget}", 1, query, ranked_docs[:budget], ranked_docs)
        budget_scores.append((budget, oracle_objective(row, baseline.tokens_used)))

    best_budget, best_score = max(budget_scores, key=lambda item: item[1])
    if oracle_strategy == "best_utility":
        return best_budget

    # The minimum-sufficient oracle is the teacher policy we want the deployed
    # controller to imitate: it chooses the smallest budget that is close enough
    # to the best available utility, instead of paying tokens for tiny gains.
    threshold = best_score * sufficiency_ratio if best_score > 0 else best_score
    for budget, score in budget_scores:
        if score >= threshold:
            return budget
    return best_budget


def oracle_objective(row: RunMetrics, baseline_tokens: float) -> float:
    # This utility score balances answer/retrieval quality with token cost.
    # Higher answer quality is good.
    # More token usage is penalized.
    token_ratio = row.tokens_used / baseline_tokens if baseline_tokens else 1.0
    return (
        row.answer_f1
        + (0.6 * row.recall)
        + (0.5 * row.mrr)
        + (0.3 * row.precision_at_k)
        - (0.2 * token_ratio)
    )




def build_examples(
    documents: list[Document],
    queries: list[Query],
    oracle_strategy: str = "minimum_sufficient",
    sufficiency_ratio: float = 0.95,
    cross_encoder=cross_encoder,
    doc_vectors=None,
    idf=None,
) -> tuple[list[TrainingExample], dict[str, list[tuple[Document, float]]]]:

<<<<<<< HEAD
=======
    # If the final llm_budget.py calls this function without precomputed TF-IDF
    # objects, build them here. This keeps the cross-encoder pipeline compatible
    # with the final Safe Adaptive controller.
    if idf is None:
        idf = build_idf(documents)
    if doc_vectors is None:
        doc_vectors = {doc.doc_id: tfidf_vector(doc.text, idf) for doc in documents}

>>>>>>> f3bcb272f9407d130ab07b67ba0f2651e5f7b44d
    ranked_by_query = {}
    examples = []

    all_pairs = []
    meta = []

    # ---------------------------
    # STEP 1: build candidate pool
    # ---------------------------
    count = 0
    for query in queries:
        count += 1
        if count % 20 == 0:
            print("query", count)
        candidates = retrieve(
            query,
            documents,
            doc_vectors,
            idf,
            doc_weights=None,
            top_k=10,
            candidate_k=15
        )

        for doc, _ in candidates:
            all_pairs.append((query.text, doc.text))
            meta.append((query.query_id, doc))

    scores = cross_encoder.predict(all_pairs, batch_size=64, show_progress_bar=True)

    ranked_by_query = defaultdict(list)

    for (query_id, doc), score in zip(meta, scores):
        ranked_by_query[query_id].append((doc, float(score)))
    
    for qid in ranked_by_query:
        ranked_by_query[qid].sort(key=lambda x: x[1], reverse=True)

    for query in queries:
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


    return examples, ranked_by_query


def budget_class(budget: int) -> str:
    # Turn the 4 possible budgets into a binary problem.
    # This was added because predicting 3/5/8/10 directly was too brittle.
    return SMALL_CLASS if budget <= 5 else LARGE_CLASS


def _default_centroid(normalized_rows: list[list[float]], feature_count: int) -> list[float]:
    if not normalized_rows:
        return [0.0 for _index in range(feature_count)]
    return [
        sum(row[index] for row in normalized_rows) / len(normalized_rows)
        for index in range(feature_count)
    ]


def train_centroid_model(
    examples: list[TrainingExample],
    threshold_strategy: str = "heuristic",
) -> CentroidModel:
    # Train a simple nearest-centroid model.
    # Each class gets an average feature vector.
    # At prediction time, we compare the query to the small and large centroids.
    if threshold_strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(f"Unknown threshold strategy: {threshold_strategy}")

    feature_count = len(examples[0].features)
    means = [
        sum(example.features[index] for example in examples) / len(examples)
        for index in range(feature_count)
    ]
    stdevs = []
    for index in range(feature_count):
        variance = sum((example.features[index] - means[index]) ** 2 for example in examples) / len(examples)
        stdevs.append(math.sqrt(variance) or 1.0)

    normalized_rows = [normalize(example.features, means, stdevs) for example in examples]
    normalized_by_label: dict[str, list[list[float]]] = {}
    class_counts = {SMALL_CLASS: 0, LARGE_CLASS: 0}
    for example in examples:
        label = budget_class(example.label)
        class_counts[label] += 1
        normalized_by_label.setdefault(label, []).append(normalize(example.features, means, stdevs))

    centroids = {}
    for label in [SMALL_CLASS, LARGE_CLASS]:
        centroids[label] = _default_centroid(normalized_by_label.get(label, []), feature_count)

    total = len(examples)
    class_priors = {
        label: (class_counts[label] + 1) / (total + 2)
        for label in [SMALL_CLASS, LARGE_CLASS]
    }

    # The binary model is intentionally conservative: low-entropy datasets such
    # as SciFact should only choose a large budget when the retrieval features
    # give strong evidence that the query is not a small-budget case.
    large_rate = class_counts[LARGE_CLASS] / total
    small_rate = class_counts[SMALL_CLASS] / total
    low_entropy_small_default = small_rate >= 0.85
    if low_entropy_small_default:
        large_threshold = 0.985
    else:
        if threshold_strategy == "calibrated":
            large_threshold = calibrate_large_threshold(examples, means, stdevs, centroids, class_priors)
        else:
            large_threshold = min(0.92, max(0.55, 0.5 + (0.5 - large_rate) * 0.8))

    # k=10 is reserved for the clearest high-context cases in calibrated mode.
    # The default heuristic keeps the earlier quality-preserving behavior.
    if threshold_strategy == "calibrated":
        very_large_threshold = min(0.98, max(0.90, large_threshold + 0.12))
    else:
        very_large_threshold = 0.82

    return CentroidModel(
        labels=[SMALL_CLASS, LARGE_CLASS],
        means=means,
        stdevs=stdevs,
        centroids=centroids,
        class_counts=class_counts,
        class_priors=class_priors,
        large_threshold=large_threshold,
        very_large_threshold=very_large_threshold,
        low_entropy_small_default=low_entropy_small_default,
    )


def normalize(features: list[float], means: list[float], stdevs: list[float]) -> list[float]:
    # Put every feature on a similar scale.
    # This stops large-valued features from dominating the centroid distance.
    return [(value - mean) / stdev for value, mean, stdev in zip(features, means, stdevs)]


def _class_scores(model: CentroidModel, features: list[float]) -> dict[str, float]:
    # Compute how close this query is to each class centroid.
    # Higher score means more likely class.
    normalized = normalize(features, model.means, model.stdevs)
    scores = {}
    for label, centroid in model.centroids.items():
        distance = sum((value - center) ** 2 for value, center in zip(normalized, centroid))
        prior = model.class_priors.get(label, 0.5)
        scores[label] = math.exp(-0.5 * distance) * prior
    return scores


def _large_probability_from_parts(
    features: list[float],
    means: list[float],
    stdevs: list[float],
    centroids: dict[str, list[float]],
    class_priors: dict[str, float],
) -> float:
    normalized = normalize(features, means, stdevs)
    scores = {}
    for label, centroid in centroids.items():
        distance = sum((value - center) ** 2 for value, center in zip(normalized, centroid))
        scores[label] = math.exp(-0.5 * distance) * class_priors.get(label, 0.5)
    total = sum(scores.values())
    return scores.get(LARGE_CLASS, 0.0) / total if total else 0.0


def calibrate_large_threshold(
    examples: list[TrainingExample],
    means: list[float],
    stdevs: list[float],
    centroids: dict[str, list[float]],
    class_priors: dict[str, float],
) -> float:
    # Optional threshold calibration.
    # It tries different large-class thresholds and keeps the one that works best
    # on the training/dev labels.
    scored_examples = [
        (
            _large_probability_from_parts(example.features, means, stdevs, centroids, class_priors),
            budget_class(example.label),
        )
        for example in examples
    ]
    candidates = sorted({0.50, 0.60, 0.70, 0.80, 0.90, *(score for score, _label in scored_examples)})
    best_threshold = 0.70
    best_accuracy = -1.0
    best_large_predictions = len(examples) + 1

    for threshold in candidates:
        correct = 0
        large_predictions = 0
        for large_probability, gold_class in scored_examples:
            predicted_class = LARGE_CLASS if large_probability >= threshold else SMALL_CLASS
            correct += int(predicted_class == gold_class)
            large_predictions += int(predicted_class == LARGE_CLASS)
        accuracy = correct / len(scored_examples)

        # Ties prefer the threshold with fewer large predictions, because the
        # project optimizes for minimum sufficient context rather than maximum
        # context.
        if accuracy > best_accuracy or (
            accuracy == best_accuracy and large_predictions < best_large_predictions
        ):
            best_accuracy = accuracy
            best_threshold = threshold
            best_large_predictions = large_predictions

    return best_threshold


def predict_budget_class(model: CentroidModel, features: list[float]) -> tuple[str, float]:
    # Predict binary class:
    # small = k <= 5
    # large = k >= 8
    scores = _class_scores(model, features)
    total = sum(scores.values())
    large_probability = scores[LARGE_CLASS] / total if total else 0.0
    if large_probability >= model.large_threshold:
        return LARGE_CLASS, large_probability
    return SMALL_CLASS, large_probability


def predict_budget(model: CentroidModel, features: list[float]) -> int:
    # Convert the binary class back into an actual budget:
    # small -> 3 or 5
    # large -> 8 or 10
    predicted_class, large_probability = predict_budget_class(model, features)
    gap_3_to_4 = features[7]
    top3_mass = features[10]
    top5_mass = features[11]
    normalized_entropy = features[-1]

    if predicted_class == SMALL_CLASS:
        if model.low_entropy_small_default:
            return 3
        # Prefer k=3 unless the score distribution is flat enough that the
        # fourth and fifth documents still look useful.
        if large_probability >= 0.35 and gap_3_to_4 < 0.04 and top3_mass < 0.42:
            return 5
        return 3

    # Large-budget predictions are also conservative: reserve k=10 for very
    # diffuse rankings; otherwise k=8 captures the large class at lower cost.
    if model.low_entropy_small_default:
        return 8
    if large_probability >= model.very_large_threshold and normalized_entropy > 0.85 and top5_mass < 0.62:
        return 10
    return 8


def _doc_terms(docs: list[Document]) -> set[str]:
    terms: set[str] = set()
    for doc in docs:
        terms.update(tokenize(doc.text))
    return terms


def context_risk_score(query: Query, features: list[float], ranked_docs: list[tuple[Document, float]]) -> float:
    # This estimates how risky it is to keep a small context.
    # It looks at whether query terms are missing from top-3 and whether later
    # documents add new useful query terms.
    query_terms = set(tokenize(query.text))
    if not query_terms:
        return 0.0

    top3_docs = [doc for doc, _score in ranked_docs[:3]]
    next5_docs = [doc for doc, _score in ranked_docs[3:8]]
    top3_terms = _doc_terms(top3_docs)
    next5_terms = _doc_terms(next5_docs)

    missing_from_top3 = len(query_terms - top3_terms) / len(query_terms)
    new_terms_after_top3 = len((query_terms & next5_terms) - top3_terms) / len(query_terms)
    gap_1_to_5 = features[9]
    top3_mass = features[10]
    top3_to_top8 = features[14]
    normalized_entropy = features[-1]

    risk = 0.0
    if normalized_entropy > 0.82:
        risk += 0.25
    if top3_mass < 0.42:
        risk += 0.20
    if top3_to_top8 < 0.55:
        risk += 0.20
    if gap_1_to_5 < 0.20:
        risk += 0.15
    if missing_from_top3 > 0.35:
        risk += 0.10
    if new_terms_after_top3 > 0.10:
        risk += 0.10
    if len(query_terms) >= 8:
        risk += 0.05

    return min(1.0, risk)


def predict_compensated_budget(
    model: CentroidModel,
    query: Query,
    features: list[float],
    ranked_docs: list[tuple[Document, float]],
) -> tuple[int, float, str]:
    # This model starts from the learned budget and expands when risk looks high.
    # It was an intermediate step before the stronger answer-aware fallback system.
    base_budget = predict_budget(model, features)
    risk = context_risk_score(query, features, ranked_docs)

    if model.low_entropy_small_default:
        # In low-entropy domains such as SciFact, the main failure mode is
        # over-expansion. Keep the efficient learned decision instead of using
        # generic uncertainty signals that are too sensitive for this domain.
        return base_budget, risk, "low_entropy_keep_base"

    if base_budget == 3:
        if risk >= 0.70:
            return 8, risk, "high_risk_expand_3_to_8"
        if risk >= 0.45:
            return 5, risk, "medium_risk_expand_3_to_5"
        return 3, risk, "low_risk_keep_3"

    if base_budget == 5:
        if risk >= 0.65:
            return 8, risk, "high_risk_expand_5_to_8"
        return 5, risk, "keep_5"

    if base_budget == 8:
        if risk >= 0.90:
            return 10, risk, "very_high_risk_expand_8_to_10"
        return 8, risk, "keep_8"

    return base_budget, risk, "keep_10"


def sufficiency_risk_score(
    query: Query,
    features: list[float],
    ranked_docs: list[tuple[Document, float]],
    budget: int,
) -> float:
    # This asks:
    # "If we stop at this budget, does the remaining retrieved context still look important?"
    # High risk means we should keep expanding.
    query_terms = set(tokenize(query.text))
    if not query_terms:
        return 0.0

    selected_docs = [doc for doc, _score in ranked_docs[:budget]]
    remaining_docs = [doc for doc, _score in ranked_docs[budget:10]]
    selected_terms = _doc_terms(selected_docs)
    remaining_terms = _doc_terms(remaining_docs)
    scores = _score_values(ranked_docs)
    score_sum = sum(scores[:10])
    selected_score_mass = sum(scores[:budget]) / score_sum if score_sum else 1.0
    gap_after_budget = (
        (scores[budget - 1] - scores[budget]) / scores[0]
        if len(scores) > budget and scores[0] > 0
        else 1.0
    )

    missing_terms = len(query_terms - selected_terms) / len(query_terms)
    new_terms_after_budget = len((query_terms & remaining_terms) - selected_terms) / len(query_terms)
    normalized_entropy = features[-1]
    mass_floor = {3: 0.45, 5: 0.65, 8: 0.85}.get(budget, 0.90)
    gap_floor = {3: 0.05, 5: 0.04, 8: 0.03}.get(budget, 0.02)

    risk = 0.0
    if normalized_entropy > 0.82:
        risk += 0.18
    if selected_score_mass < mass_floor:
        risk += 0.25
    if gap_after_budget < gap_floor:
        risk += 0.15
    if missing_terms > 0.35:
        risk += 0.18
    if new_terms_after_budget > 0.10:
        risk += 0.18
    if len(query_terms) >= 8:
        risk += 0.06

    return min(1.0, risk)


def predict_sequential_sufficiency_budget(
    model: CentroidModel,
    query: Query,
    features: list[float],
    ranked_docs: list[tuple[Document, float]],
) -> tuple[int, float, str]:
    # Sequential budget policy:
    # start at 3, expand to 5, then 8, then 10 only if risk remains high.
    if model.low_entropy_small_default:
        return 3, 0.0, "low_entropy_stop_at_3"

    risk_3 = sufficiency_risk_score(query, features, ranked_docs, 3)
    if risk_3 <= 0.30:
        return 3, risk_3, "top3_sufficient"

    risk_5 = sufficiency_risk_score(query, features, ranked_docs, 5)
    if risk_5 <= 0.40:
        return 5, risk_5, "top5_sufficient"

    risk_8 = sufficiency_risk_score(query, features, ranked_docs, 8)
    if risk_8 <= 0.55:
        return 8, risk_8, "top8_sufficient"

    return 10, risk_8, "use_full_candidate_budget"


def evaluate_learned_budget(
    queries: list[Query],
    examples: list[TrainingExample],
    ranked_by_query: dict[str, list[tuple[Document, float]]],
    model: CentroidModel,
) -> tuple[list[RunMetrics], list[dict[str, object]]]:
    # Evaluate all retrieval-only budget modes on the same held-out queries.
    # This is used by run_learned_budget.py and also reused by the LLM experiment.
    query_by_id = {query.query_id: query for query in queries}
    example_by_id = {example.query_id: example for example in examples}
    metrics: list[RunMetrics] = []
    predictions = []

    for query_id, query in query_by_id.items():
        # For each query, get the precomputed top-10 ranking and prediction features.
        ranked = ranked_by_query[query_id]
        example = example_by_id[query_id]
        predicted_class, large_probability = predict_budget_class(model, example.features)
        predicted_budget = predict_budget(model, example.features)
        compensated_budget, compensation_risk, compensation_reason = predict_compensated_budget(
            model,
            query,
            example.features,
            ranked,
        )
        sequential_budget, sequential_risk, sequential_reason = predict_sequential_sufficiency_budget(
            model,
            query,
            example.features,
            ranked,
        )
        oracle_budget = example.label
        oracle_class = budget_class(oracle_budget)

        for budget in BUDGETS:
            # Fixed-k baselines.
            metrics.append(_record_metrics(f"fixed_{budget}", 1, query, ranked[:budget], ranked))

        # Basic learned budget.
        metrics.append(_record_metrics("learned_budget", 1, query, ranked[:predicted_budget], ranked))

        # Risk-compensated learned budget.
        metrics.append(
            _record_metrics(
                "learned_compensated_budget",
                1,
                query,
                ranked[:compensated_budget],
                ranked,
            )
        )

        # Sequential sufficiency budget.
        metrics.append(
            _record_metrics(
                "sequential_sufficiency_budget",
                1,
                query,
                ranked[:sequential_budget],
                ranked,
            )
        )

        # Oracle is the non-deployable upper-bound teacher.
        metrics.append(_record_metrics("oracle_dynamic_budget", 1, query, ranked[:oracle_budget], ranked))

        predictions.append(
            {
                "query_id": query_id,
                "oracle_budget": oracle_budget,
                "oracle_class": oracle_class,
                "predicted_budget": predicted_budget,
                "predicted_class": predicted_class,
                "compensated_budget": compensated_budget,
                "compensation_risk": round(compensation_risk, 6),
                "compensation_reason": compensation_reason,
                "sequential_budget": sequential_budget,
                "sequential_risk": round(sequential_risk, 6),
                "sequential_reason": sequential_reason,
                "large_probability": round(large_probability, 6),
                "correct": oracle_budget == predicted_budget,
                "compensated_correct": oracle_budget == compensated_budget,
                "sequential_correct": oracle_budget == sequential_budget,
                "class_correct": oracle_class == predicted_class,
                "compensated_class_correct": oracle_class == budget_class(compensated_budget),
                "sequential_class_correct": oracle_class == budget_class(sequential_budget),
            }
        )

    return metrics, predictions


def summarize(metrics: list[RunMetrics]) -> list[dict[str, object]]:
    rows = []
    modes = []
    for row in metrics:
        if row.mode not in modes:
            modes.append(row.mode)

    for mode in modes:
        selected = [row for row in metrics if row.mode == mode]
        count = len(selected)
        rows.append(
            {
                "mode": mode,
                "precision_at_k": round(sum(row.precision_at_k for row in selected) / count, 6),
                "recall": round(sum(row.recall for row in selected) / count, 6),
                "mrr_at_10": round(sum(row.mrr for row in selected) / count, 6),
                "ndcg_at_10": round(sum(row.ndcg_at_10 for row in selected) / count, 6),
                "docs_used": round(sum(row.docs_used for row in selected) / count, 6),
                "tokens_used": round(sum(row.tokens_used for row in selected) / count, 6),
                "noise_tokens": round(sum(row.context_noise_tokens for row in selected) / count, 6),
                "answer_f1": round(sum(row.answer_f1 for row in selected) / count, 6),
                "answer_coverage": round(sum(row.answer_coverage for row in selected) / count, 6),
            }
        )
    return rows


def label_distribution(examples: list[TrainingExample]) -> dict[int, int]:
    return dict(sorted(Counter(example.label for example in examples).items()))


def class_distribution(examples: list[TrainingExample]) -> dict[str, int]:
    counts = Counter(budget_class(example.label) for example in examples)
    return {label: counts.get(label, 0) for label in [SMALL_CLASS, LARGE_CLASS]}


def prediction_distribution(predictions: list[dict[str, object]], key: str) -> dict[object, int]:
    return dict(sorted(Counter(row[key] for row in predictions).items()))


def write_outputs(
    output_dir: Path,
    metrics: list[RunMetrics],
    summary_rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "metrics_by_query.csv", [asdict(row) for row in metrics])
    write_csv(output_dir / "predictions.csv", predictions)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
