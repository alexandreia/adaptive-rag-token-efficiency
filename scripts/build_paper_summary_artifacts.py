#!/usr/bin/env python3
"""Build report-facing summary tables and lightweight SVG plots.

This script intentionally uses only the Python standard library so the final
paper artifacts can be regenerated without installing plotting dependencies.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "saved_results" / "paper_summary"

LLAMA_TFIDF = ROOT / "saved_results" / "final_main" / "llama70b_tfidf" / "final_table_all_datasets.csv"
MISTRAL_TFIDF = ROOT / "saved_results" / "final_main" / "mistral_tfidf" / "final_table_all_datasets.csv"
TFIDF_CE = (
    ROOT
    / "saved_results"
    / "retriever_robustness"
    / "llama70b"
    / "combined"
    / "tfidf_cross_encoder_final_table_all_datasets.csv"
)
DENSE_CE = (
    ROOT
    / "saved_results"
    / "retriever_robustness"
    / "llama70b"
    / "combined"
    / "dense_cross_encoder_final_table_all_datasets.csv"
)

CORE_METHODS = [
    "Fixed Top-3",
    "Fixed Top-5",
    "Fixed Top-10",
    "Adaptive-k",
    "Safe Adaptive Context",
    "TACER",
]

PLOT_METHODS = ["Fixed Top-3", "Fixed Top-5", "Adaptive-k", "Safe Adaptive Context", "TACER"]
METHOD_COLORS = {
    "Fixed Top-3": "#7c8a99",
    "Fixed Top-5": "#a3772b",
    "Adaptive-k": "#d14f42",
    "Safe Adaptive Context": "#2f6f9f",
    "TACER": "#16805a",
}
DATASET_SHAPES = {
    "SciFact": "circle",
    "BioASQ": "square",
    "HotpotQA": "diamond",
    "MSMARCO": "triangle",
    "ASQA": "circle-open",
}


def read_rows(path: Path, model: str, retriever: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["model"] = model
        row["retriever"] = retriever
        row["answer_f1_float"] = float(row["answer_f1"])
        row["semantic_similarity_float"] = float(row["semantic_similarity"])
        row["total_tokens_float"] = float(row["total_tokens"])
        row["f1_retained_float"] = parse_percent(row["f1_retained_vs_top10"])
        row["semantic_retained_float"] = parse_percent(row["semantic_similarity_retained_vs_top10"])
        row["token_reduction_float"] = parse_percent(row["token_reduction_vs_top10"])
        row["fallback_rate_float"] = parse_percent(row["fallback_rate"])
    return rows


def parse_percent(value: str) -> float:
    return float(value.strip().rstrip("%"))


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset"], row["method"]): row for row in rows}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main_summary_table(llama_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    lookup = by_key(llama_rows)
    datasets = ["SciFact", "BioASQ", "HotpotQA", "MSMARCO", "ASQA"]
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        fixed = lookup[(dataset, "Fixed Top-10")]
        adaptive = lookup[(dataset, "Adaptive-k")]
        safe = lookup[(dataset, "Safe Adaptive Context")]
        tacer = lookup[(dataset, "TACER")]
        rows.append(
            {
                "Dataset": dataset,
                "Fixed F1": fmt(fixed["answer_f1_float"]),
                "Adaptive-k F1 / Red.": f"{fmt(adaptive['answer_f1_float'])} / {fmt_pct(adaptive['token_reduction_float'])}",
                "Safe F1 / Red.": f"{fmt(safe['answer_f1_float'])} / {fmt_pct(safe['token_reduction_float'])}",
                "TACER F1 / Red.": f"{fmt(tacer['answer_f1_float'])} / {fmt_pct(tacer['token_reduction_float'])}",
                "TACER vs Adaptive-k F1": f"{tacer['answer_f1_float'] - adaptive['answer_f1_float']:+.3f}",
                "Takeaway": dataset_takeaway(dataset, adaptive, safe, tacer),
            }
        )
    return rows


def dataset_takeaway(dataset: str, adaptive: dict[str, str], safe: dict[str, str], tacer: dict[str, str]) -> str:
    if dataset == "HotpotQA":
        return "Safe/TACER avoid score-only under-selection"
    if dataset == "SciFact":
        return "TACER is highly efficient on concentrated evidence"
    if dataset == "ASQA":
        return "TACER preserves long-form quality better than Adaptive-k"
    if dataset == "BioASQ":
        return "Safe is quality-preserving; TACER is aggressive"
    return "Small contexts are strong with good ranking"


def model_comparison_table(llama_rows: list[dict[str, str]], mistral_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    llama = by_key(llama_rows)
    mistral = by_key(mistral_rows)
    for dataset in ["SciFact", "BioASQ", "HotpotQA", "MSMARCO", "ASQA"]:
        for method in ["Adaptive-k", "Safe Adaptive Context", "TACER"]:
            lrow = llama[(dataset, method)]
            mrow = mistral[(dataset, method)]
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": method,
                    "Llama F1 / Red.": f"{fmt(lrow['answer_f1_float'])} / {fmt_pct(lrow['token_reduction_float'])}",
                    "Mistral F1 / Red.": f"{fmt(mrow['answer_f1_float'])} / {fmt_pct(mrow['token_reduction_float'])}",
                    "Mistral Fixed Top-10 F1": fmt(mistral[(dataset, "Fixed Top-10")]["answer_f1_float"]),
                }
            )
    return rows


def retriever_summary_table(all_llama_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in all_llama_rows:
        if row["method"] not in ["Fixed Top-3", "Fixed Top-5", "Fixed Top-10", "Adaptive-k", "Safe Adaptive Context", "TACER"]:
            continue
        rows.append(
            {
                "Retriever": row["retriever"],
                "Dataset": row["dataset"],
                "Method": row["method"],
                "F1": fmt(row["answer_f1_float"]),
                "F1 Retained": fmt_pct(row["f1_retained_float"]),
                "Semantic Retained": fmt_pct(row["semantic_retained_float"]),
                "Token Reduction": fmt_pct(row["token_reduction_float"]),
                "Fallback": fmt_pct(row["fallback_rate_float"]),
            }
        )
    return rows


def aggregate_table(all_llama_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_llama_rows:
        if row["method"] in ["Fixed Top-3", "Fixed Top-5", "Adaptive-k", "Safe Adaptive Context", "TACER"]:
            grouped[(row["retriever"], row["method"])].append(row)
    rows = []
    for (retriever, method), selected in sorted(grouped.items()):
        rows.append(
            {
                "Retriever": retriever,
                "Method": method,
                "Avg F1 Retained": fmt_pct(mean([r["f1_retained_float"] for r in selected])),
                "Avg Semantic Retained": fmt_pct(mean([r["semantic_retained_float"] for r in selected])),
                "Avg Token Reduction": fmt_pct(mean([r["token_reduction_float"] for r in selected])),
                "Avg Fallback": fmt_pct(mean([r["fallback_rate_float"] for r in selected])),
            }
        )
    return rows


def tacer_vs_adaptive_table(all_llama_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = defaultdict(dict)
    for row in all_llama_rows:
        grouped[(row["retriever"], row["dataset"])][row["method"]] = row
    rows = []
    for (retriever, dataset), by_method in sorted(grouped.items()):
        if "Adaptive-k" not in by_method or "TACER" not in by_method:
            continue
        adaptive = by_method["Adaptive-k"]
        tacer = by_method["TACER"]
        rows.append(
            {
                "Retriever": retriever,
                "Dataset": dataset,
                "Adaptive-k F1": fmt(adaptive["answer_f1_float"]),
                "TACER F1": fmt(tacer["answer_f1_float"]),
                "TACER F1 Delta": f"{tacer['answer_f1_float'] - adaptive['answer_f1_float']:+.3f}",
                "Adaptive-k Red.": fmt_pct(adaptive["token_reduction_float"]),
                "TACER Red.": fmt_pct(tacer["token_reduction_float"]),
                "TACER Red. Delta": f"{tacer['token_reduction_float'] - adaptive['token_reduction_float']:+.1f}%",
            }
        )
    return rows


def svg_pareto(path: Path, rows: list[dict[str, str]], title: str) -> None:
    width, height = 900, 580
    left, right, top, bottom = 90, 30, 70, 95
    x_min, x_max = 20, 90
    y_min, y_max = 65, 112

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def y(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="15">Token reduction vs Fixed Top-10 (%)</text>',
        f'<text transform="translate(24 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">F1 retained vs Fixed Top-10 (%)</text>',
    ]
    for tick in range(20, 91, 10):
        parts.append(f'<line x1="{x(tick):.1f}" y1="{height-bottom}" x2="{x(tick):.1f}" y2="{height-bottom+5}" stroke="#333"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="12">{tick}</text>')
    for tick in range(70, 111, 10):
        parts.append(f'<line x1="{left-5}" y1="{y(tick):.1f}" x2="{left}" y2="{y(tick):.1f}" stroke="#333"/>')
        parts.append(f'<text x="{left-10}" y="{y(tick)+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick}</text>')
        parts.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{width-right}" y2="{y(tick):.1f}" stroke="#e8e8e8"/>')

    selected = [row for row in rows if row["method"] in PLOT_METHODS]
    for row in selected:
        color = METHOD_COLORS[row["method"]]
        px = x(row["token_reduction_float"])
        py = y(row["f1_retained_float"])
        shape = DATASET_SHAPES.get(row["dataset"], "circle")
        if shape == "square":
            parts.append(f'<rect x="{px-5:.1f}" y="{py-5:.1f}" width="10" height="10" fill="{color}" opacity="0.82"/>')
        elif shape == "diamond":
            parts.append(f'<polygon points="{px:.1f},{py-7:.1f} {px+7:.1f},{py:.1f} {px:.1f},{py+7:.1f} {px-7:.1f},{py:.1f}" fill="{color}" opacity="0.82"/>')
        elif shape == "triangle":
            parts.append(f'<polygon points="{px:.1f},{py-7:.1f} {px+7:.1f},{py+6:.1f} {px-7:.1f},{py+6:.1f}" fill="{color}" opacity="0.82"/>')
        elif shape == "circle-open":
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="white" stroke="{color}" stroke-width="2" opacity="0.95"/>')
        else:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" opacity="0.82"/>')

    legend_x, legend_y = 615, 75
    parts.append(f'<rect x="{legend_x-15}" y="{legend_y-25}" width="260" height="155" fill="white" stroke="#ddd"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y}" font-family="Arial" font-size="13" font-weight="700">Method color</text>')
    for i, method in enumerate(PLOT_METHODS):
        yy = legend_y + 20 + i * 20
        parts.append(f'<circle cx="{legend_x+6}" cy="{yy-4}" r="5" fill="{METHOD_COLORS[method]}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{yy}" font-family="Arial" font-size="12">{method}</text>')
    parts.append(f'<text x="{legend_x}" y="{legend_y+135}" font-family="Arial" font-size="12">Shapes encode datasets.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_notes(path: Path) -> None:
    text = """# Report Summary Notes

## Main Pattern

TACER is best framed as the most stable quality-efficiency policy, not as the winner of every cell.
Adaptive-k is a strong score-only baseline, but it can under-select context on tasks where evidence is distributed.
Safe Adaptive Context is the conservative quality-preserving sibling of TACER.

## Key Results To Discuss

- HotpotQA is the clearest failure case for score-only Adaptive-k. Even with dense + cross-encoder retrieval, Adaptive-k retains only 79.1% F1, while TACER retains 97.2%.
- SciFact is the clearest TACER compression case. Under dense + cross-encoder retrieval, TACER retains 102.9% F1 and 100.9% semantic similarity while reducing tokens by 82.1%.
- ASQA shows convergence under strong retrieval. With dense + cross-encoder retrieval, all methods are close, and TACER retains 99.8% F1 with 54.4% token reduction.
- BioASQ shows near-lossless aggressive compression. TACER gives high token savings with small quality loss; Safe Adaptive is the safer quality-preserving point.
- MS MARCO shows that strong reranking plus Fixed Top-3 can be hard to beat, which is an important limitation and makes the paper more credible.

## Suggested Claim

Adaptive-k answers how many documents to keep. TACER chooses a context policy: compress aggressively when evidence is concentrated, and route to safer broader context when the task is likely to need distributed evidence.

## Suggested Limitation

TACER does not dominate every dataset. On short-answer web QA with very strong retrieval, small fixed-k baselines can be as good or better. The contribution is robustness across task types rather than universal superiority.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    llama_tfidf = read_rows(LLAMA_TFIDF, "Llama-70B", "TF-IDF")
    mistral_tfidf = read_rows(MISTRAL_TFIDF, "Mistral", "TF-IDF")
    tfidf_ce = read_rows(TFIDF_CE, "Llama-70B", "TF-IDF + CE")
    dense_ce = read_rows(DENSE_CE, "Llama-70B", "Dense + CE")
    all_llama = llama_tfidf + tfidf_ce + dense_ce

    artifacts = {
        "main_llama_tfidf_core": main_summary_table(llama_tfidf),
        "model_comparison_tfidf": model_comparison_table(llama_tfidf, mistral_tfidf),
        "retriever_robustness_core": retriever_summary_table(all_llama),
        "method_averages_by_retriever": aggregate_table(all_llama),
        "tacer_vs_adaptive_by_retriever": tacer_vs_adaptive_table(all_llama),
    }
    for name, rows in artifacts.items():
        write_csv(OUT / f"{name}.csv", rows)
        write_markdown(OUT / f"{name}.md", rows)

    svg_pareto(OUT / "pareto_llama_tfidf.svg", llama_tfidf, "Llama-70B TF-IDF: Quality Retention vs Token Reduction")
    svg_pareto(OUT / "pareto_llama_tfidf_cross_encoder.svg", tfidf_ce, "Llama-70B TF-IDF + Cross-Encoder")
    svg_pareto(OUT / "pareto_llama_dense_cross_encoder.svg", dense_ce, "Llama-70B Dense + Cross-Encoder")
    write_notes(OUT / "report_summary_notes.md")

    print(f"Wrote paper summary artifacts to {OUT}")
    for path in sorted(OUT.iterdir()):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
