# Learned TACER Router

This is the foundation for the next expansion of the paper: replacing the
hand-written TACER routing rules with a learned evidence-routing policy.

## Motivation

The submitted TACER system shows that task/evidence-aware routing works with
transparent rules. The learned-router extension asks a stronger question:

> Can a model learn when to use compact evidence, Safe Adaptive Context, TACER,
> Adaptive-k, or broad context from query and retrieval signals alone?

This reframes TACER as a learnable control problem rather than only a heuristic
controller.

## Current Scaffold

The initial implementation is intentionally lightweight:

- no neural network dependency
- no scikit-learn dependency
- pure-Python multiclass linear classifier
- teacher labels derived from existing saved experiment outputs
- runtime features limited to query text and retrieval score shape
- no dataset identifiers as model inputs

The model predicts one of:

- `adaptive_k_full`
- `answer_aware_fallback`
- `coverage_guided_ultra`
- `task_aware_coverage_ultra`
- `fixed_10_full`

## Teacher Labels

For each query, the script compares saved outcomes for the available policies.
It chooses the lowest-token method that stays close to Fixed Top-10 quality:

- answer F1 must be at least `quality_floor * Fixed Top-10 F1`
- semantic similarity must be within `semantic_margin` of Fixed Top-10

If no method meets that quality floor, the label falls back to a utility score
that trades off F1, coverage, semantic similarity, and token use.

This makes the first learned router a student of the empirical quality-efficiency
frontier already measured in the paper.

## Train

```bash
python3 scripts/train_learned_tacer_router.py
```

Outputs are written to:

```text
saved_results/learned_router/
```

The main model file is:

```text
saved_results/learned_router/learned_tacer_router.json
```

## Leave-One-Dataset-Out Check

This is the important generalization test for the next paper version:

```bash
python3 scripts/train_learned_tacer_router.py --leave-one-dataset-out
```

This trains on four datasets and evaluates policy prediction on the held-out
dataset. It directly tests whether the router is learning transferable evidence
signals rather than memorizing dataset-specific behavior.

## Next Integration Step

The current scaffold trains and saves the router. The next implementation step
is to add a new generation mode, for example:

```text
learned_tacer_router
```

At runtime, that mode should:

1. compute query/retrieval features with `extract_router_features(...)`
2. load `learned_tacer_router.json`
3. predict the action
4. dispatch to the existing context policy implementation

This keeps the learned extension clean: the model only learns the routing
decision, while the existing Safe Adaptive/TACER policy implementations remain
unchanged.

