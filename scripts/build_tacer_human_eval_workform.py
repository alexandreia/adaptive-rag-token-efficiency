#!/usr/bin/env python3
"""Build a targeted blind human-evaluation workform for TACER.

The workform compares Fixed Top-10, Adaptive-k, Safe Adaptive Context, and
TACER on HotpotQA and ASQA using already saved Llama-70B external-baseline outputs. It creates one
annotation item per system answer, with stable blind labels inside each query
group. No new model calls are made.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


METHODS = [
    ("fixed_10_full", "Fixed Top-10"),
    ("adaptive_k_official_full", "Adaptive-k"),
    ("answer_aware_fallback", "Safe Adaptive Context"),
    ("task_aware_coverage_ultra", "TACER"),
]

DATASETS = {
    "hotpotqa": {
        "display": "HotpotQA",
        "answers": "outputs/external_baselines/hotpotqa_llama70b_100/merged/llm_answers_by_query.csv",
        "queries": "data/hotpotqa_250/queries_eval_200.jsonl",
        "documents": "data/hotpotqa_250/documents.jsonl",
    },
    "asqa": {
        "display": "ASQA",
        "answers": "outputs/external_baselines/asqa_llama70b_100/merged/llm_answers_by_query.csv",
        "queries": "data/asqa_250/queries_eval_200.jsonl",
        "documents": "data/asqa_250/documents.jsonl",
    },
}

RATING_LABELS = ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "NOT_ENOUGH_INFO"]
MAX_EVIDENCE_CHARS = 2200


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_jsonl(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def read_answers(path: Path) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "")
            if mode not in {m[0] for m in METHODS}:
                continue
            grouped.setdefault(str(row["query_id"]), {})[mode] = row
    return grouped


def parse_doc_ids(value: str) -> list[str]:
    text = clean(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return []


def evidence_text(documents: dict[str, dict], doc_ids: list[str]) -> str:
    blocks = []
    for index, doc_id in enumerate(doc_ids, start=1):
        text = clean(documents.get(doc_id, {}).get("text", ""))
        if len(text) > MAX_EVIDENCE_CHARS:
            text = text[:MAX_EVIDENCE_CHARS].rstrip() + " ..."
        blocks.append(f"[{index}] {doc_id}: {text or '[document text not found]'}")
    return "\n\n".join(blocks)


def candidate_score(method_rows: dict[str, dict]) -> tuple[float, float, float]:
    fixed = method_rows["fixed_10_full"]
    adaptive = method_rows["adaptive_k_official_full"]
    tacer = method_rows["task_aware_coverage_ultra"]
    tacer_gain = as_float(tacer["answer_f1"]) - as_float(adaptive["answer_f1"])
    fixed_gap = as_float(fixed["answer_f1"]) - as_float(tacer["answer_f1"])
    token_delta = as_float(adaptive["total_tokens"]) - as_float(tacer["total_tokens"])
    return (tacer_gain, -abs(fixed_gap), token_delta)


def select_query_ids(grouped: dict[str, dict[str, dict]], count: int) -> list[str]:
    complete = {
        qid: rows
        for qid, rows in grouped.items()
        if all(mode in rows for mode, _ in METHODS)
    }
    ranked = sorted(
        complete.items(),
        key=lambda item: candidate_score(item[1]),
        reverse=True,
    )
    return [qid for qid, _ in ranked[:count]]


def build_items(repo: Path, per_dataset: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for dataset, config in DATASETS.items():
        queries = read_jsonl(repo / config["queries"], "query_id")
        documents = read_jsonl(repo / config["documents"], "doc_id")
        answers = read_answers(repo / config["answers"])
        selected_qids = select_query_ids(answers, per_dataset)

        for q_index, qid in enumerate(selected_qids, start=1):
            query = queries.get(qid, {})
            method_rows = answers[qid]
            blind_labels = [f"System {chr(ord('A') + i)}" for i in range(len(METHODS))]
            rng.shuffle(blind_labels)
            label_by_mode = {mode: blind_labels[i] for i, (mode, _) in enumerate(METHODS)}

            for mode, method_name in METHODS:
                row = method_rows[mode]
                doc_ids = parse_doc_ids(row.get("selected_doc_ids", ""))
                comparison_id = f"{dataset}_{q_index:02d}"
                annotation_id = f"{comparison_id}_{label_by_mode[mode].replace(' ', '').lower()}"
                items.append(
                    {
                        "candidate_id": annotation_id,
                        "annotation_id": annotation_id,
                        "comparison_id": comparison_id,
                        "dataset": dataset,
                        "dataset_name": config["display"],
                        "query_id": qid,
                        "blind_system": label_by_mode[mode],
                        "hidden_method": method_name,
                        "query_text": clean(query.get("text", "")),
                        "reference_answer": clean(query.get("reference_answer", "")),
                        "model_answer": clean(row.get("answer", "")),
                        "selected_doc_ids": json.dumps(doc_ids, ensure_ascii=False),
                        "selected_document_text": evidence_text(documents, doc_ids),
                        "answer_f1": row.get("answer_f1", ""),
                        "answer_coverage": row.get("answer_coverage", ""),
                        "semantic_similarity": row.get("semantic_similarity", ""),
                        "total_tokens": row.get("total_tokens", ""),
                        "rating": "",
                        "grounding_error": "",
                        "notes": "",
                    }
                )
    items.sort(key=lambda item: (item["dataset"], item["comparison_id"], item["blind_system"]))
    return items


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def without_hidden_fields(rows: list[dict]) -> list[dict]:
    hidden = {"hidden_method", "answer_f1", "answer_coverage", "semantic_similarity", "total_tokens"}
    return [{key: value for key, value in row.items() if key not in hidden} for row in rows]


def write_method_key(path: Path, rows: list[dict]) -> None:
    key_rows = [
        {
            "annotation_id": row["annotation_id"],
            "comparison_id": row["comparison_id"],
            "dataset": row["dataset"],
            "query_id": row["query_id"],
            "blind_system": row["blind_system"],
            "hidden_method": row["hidden_method"],
            "answer_f1": row["answer_f1"],
            "answer_coverage": row["answer_coverage"],
            "semantic_similarity": row["semantic_similarity"],
            "total_tokens": row["total_tokens"],
        }
        for row in rows
    ]
    write_csv(path, key_rows)


def write_data_js(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("window.ANNOTATION_SAMPLES = ")
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write(";\n")
        f.write("window.RATING_LABELS = ")
        json.dump(RATING_LABELS, f)
        f.write(";\n")


def write_readme(path: Path, per_dataset: int, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(
            [
                "# TACER Targeted Human Evaluation",
                "",
                "This workform compares Fixed Top-10, Adaptive-k, Safe Adaptive Context, and TACER on saved Llama-70B outputs.",
                "The sample is targeted, not random: it prioritizes HotpotQA and ASQA queries where TACER differs from Adaptive-k, because those are the cases that test whether task-aware routing helps.",
                "",
                f"- Queries: {per_dataset} HotpotQA + {per_dataset} ASQA",
                f"- Answers to annotate: {len(rows)}",
                "- Systems are blinded as System A/B/C/D inside each query group.",
                "- Hidden method labels are kept in the CSV for analysis; remove that column before sending to annotators if needed.",
                "",
                "Suggested labels:",
                "- CORRECT: answer contains the required information and does not contradict the reference/evidence.",
                "- PARTIALLY_CORRECT: answer contains some required information but is incomplete, vague, or misses a condition.",
                "- INCORRECT: answer contradicts the reference/evidence or gives a different answer.",
                "- NOT_ENOUGH_INFO: annotator cannot confidently judge from the query, reference, and evidence shown.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def copy_browser_assets(repo: Path, out_dir: Path) -> None:
    source = repo / "annotation_workform"
    for name in ["index.html", "app.js", "styles.css", "kappa.html", "kappa.js", "rag_annotation_guide.docx"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def sanitize_blind_app(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'els.sampleGroup.textContent = sample.selection_reason;',
        'els.sampleGroup.textContent = "Blind comparison";',
    )
    text = text.replace(
        'els.sampleMode.textContent = `${sample.dataset_name || sample.dataset || ""} · ${sample.model_name || sample.model_family || ""} · ${sample.method || ""}`;',
        'els.sampleMode.textContent = `${sample.dataset_name || sample.dataset || ""} · ${sample.comparison_id || ""} · ${sample.blind_system || ""}`;',
    )
    text = text.replace(
        '''els.metricStrip.innerHTML = [
      ["F1", sample.answer_f1],
      ["Semantic", sample.semantic_similarity],
      ["nDCG@10", sample.ndcg_at_10],
      ["MRR@10", sample.mrr_at_10],
      ["Docs", sample.docs_used],
      ["Tokens", sample.total_tokens],
    ]
      .map(([label, value]) => `<span class="metric-pill">${label}: ${escapeHtml(value ?? "")}</span>`)
      .join("");''',
        'els.metricStrip.innerHTML = "";',
    )
    path.write_text(text, encoding="utf-8")


def write_blind_packet(repo: Path, out_dir: Path, rows: list[dict], per_dataset: int) -> None:
    blind_dir = out_dir / "blind_annotation_packet"
    blind_dir.mkdir(parents=True, exist_ok=True)
    blind_rows = without_hidden_fields(rows)
    copy_browser_assets(repo, blind_dir)
    sanitize_blind_app(blind_dir / "app.js")
    write_csv(blind_dir / "annotation_items_blind.csv", blind_rows)
    write_data_js(blind_dir / "data.js", blind_rows)
    (blind_dir / "START_HERE.md").write_text(
        "\n".join(
            [
                "# Start Here: Blind Annotation Packet",
                "",
                "Open `index.html` in a browser and annotate only from this folder.",
                "",
                "This packet intentionally does not contain method names, automatic scores, or token counts.",
                "Each answer is shown only as System A/B/C/D within a query group.",
                "",
                f"Workload: {per_dataset} HotpotQA + {per_dataset} ASQA questions, with four blinded systems per question.",
                "",
                "Use the query, reference answer, model answer, and retrieved evidence shown in the page.",
                "Choose `CORRECT`, `PARTIALLY_CORRECT`, `INCORRECT`, or `NOT_ENOUGH_INFO`.",
                "",
                "Do not open files outside this folder while annotating.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("annotation_workform_tacer_safe_vs_adaptive"),
        help="Output directory, relative to repo unless absolute.",
    )
    parser.add_argument("--per-dataset", type=int, default=15)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_items(repo, args.per_dataset, args.seed)
    write_csv(out_dir / "annotation_items.csv", rows)
    write_method_key(out_dir / "method_key_do_not_open_during_annotation.csv", rows)
    write_data_js(out_dir / "data.js", rows)
    write_readme(out_dir / "README.md", args.per_dataset, rows)
    copy_browser_assets(repo, out_dir)
    write_blind_packet(repo, out_dir, rows, args.per_dataset)

    print(f"Wrote {len(rows)} annotation items to {out_dir}")


if __name__ == "__main__":
    main()
