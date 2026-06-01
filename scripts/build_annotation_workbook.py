#!/usr/bin/env python3
"""
Build an Excel annotation workbook with one tab per dataset.

Each dataset tab contains the worst N and best N examples according to an
evaluation metric, joined back to the dataset queries and selected documents.
The script uses only the Python standard library so it does not add project
dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


csv.field_size_limit(sys.maxsize)

HEADERS = [
    "candidate_id",
    "selection_reason",
    "dataset",
    "mode",
    "method_name",
    "query_id",
    "query_text",
    "reference_answer",
    "relevant_doc_ids",
    "selected_doc_ids",
    "selected_document_text",
    "model_answer",
    "answer_f1",
    "answer_coverage",
    "semantic_similarity",
    "ndcg_at_10",
    "mrr_at_10",
    "docs_used",
    "total_tokens",
    "annotator_1_label",
    "annotator_1_evidence_relevance",
    "annotator_1_evidence_sufficiency",
    "annotator_1_answer_faithfulness",
    "annotator_1_correctness",
    "annotator_1_confidence",
    "annotator_1_notes",
    "annotator_2_label",
    "annotator_2_evidence_relevance",
    "annotator_2_evidence_sufficiency",
    "annotator_2_answer_faithfulness",
    "annotator_2_correctness",
    "annotator_2_confidence",
    "annotator_2_notes",
    "adjudicated_label",
    "final_notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl_by_id(path: Path, id_field: str) -> dict[str, dict[str, object]]:
    rows = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                rows[str(row[id_field])] = row
    return rows


def as_float(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_doc_ids(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(doc_id) for doc_id in parsed]


def short_text(text: object, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def sort_key(row: dict[str, str], metric: str) -> tuple[float, str, str]:
    return (as_float(row, metric), row.get("query_id", ""), row.get("mode", ""))


def add_unique(
    selected: list[tuple[str, dict[str, str]]],
    seen: set[tuple[str, str]],
    reason: str,
    rows: list[dict[str, str]],
    limit: int,
) -> None:
    count = 0
    for row in rows:
        key = (row.get("mode", ""), row.get("query_id", ""))
        if key in seen:
            continue
        selected.append((reason, row))
        seen.add(key)
        count += 1
        if count >= limit:
            break


def select_best_worst(
    rows: list[dict[str, str]],
    metric: str,
    best_count: int,
    worst_count: int,
) -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    seen: set[tuple[str, str]] = set()
    sorted_rows = sorted(rows, key=lambda row: sort_key(row, metric))
    add_unique(selected, seen, f"worst_{metric}", sorted_rows, worst_count)
    add_unique(selected, seen, f"best_{metric}", reversed(sorted_rows), best_count)
    return selected


def build_rows(
    dataset_name: str,
    eval_csv: Path,
    queries_path: Path,
    documents_path: Path,
    mode: str,
    metric: str,
    best_count: int,
    worst_count: int,
    max_doc_chars: int,
) -> list[dict[str, object]]:
    rows = read_csv(eval_csv)
    if mode != "all":
        rows = [row for row in rows if row.get("mode") == mode]
    if not rows:
        raise SystemExit(f"No rows found for dataset={dataset_name!r}, mode={mode!r}, eval_csv={eval_csv}")

    queries_by_id = read_jsonl_by_id(queries_path, "query_id")
    docs_by_id = read_jsonl_by_id(documents_path, "doc_id")
    selected = select_best_worst(rows, metric, best_count, worst_count)

    candidates = []
    for index, (reason, row) in enumerate(selected, start=1):
        query_id = str(row.get("query_id", ""))
        query = queries_by_id.get(query_id, {})
        selected_doc_ids = parse_doc_ids(row.get("selected_doc_ids", ""))
        selected_docs = []
        for rank, doc_id in enumerate(selected_doc_ids, start=1):
            doc = docs_by_id.get(doc_id, {})
            selected_docs.append(f"[{rank}] {doc_id}: {short_text(doc.get('text', ''), max_doc_chars)}")

        candidate = {header: "" for header in HEADERS}
        candidate.update(
            {
                "candidate_id": f"{dataset_name.upper()}-{index:03d}",
                "selection_reason": reason,
                "dataset": dataset_name,
                "mode": row.get("mode", ""),
                "method_name": row.get("method_name", ""),
                "query_id": query_id,
                "query_text": query.get("text", ""),
                "reference_answer": query.get("reference_answer", ""),
                "relevant_doc_ids": ", ".join(str(doc_id) for doc_id in query.get("relevant_doc_ids", [])),
                "selected_doc_ids": ", ".join(selected_doc_ids),
                "selected_document_text": "\n\n".join(selected_docs),
                "model_answer": row.get("answer", ""),
                "answer_f1": row.get("answer_f1", ""),
                "answer_coverage": row.get("answer_coverage", ""),
                "semantic_similarity": row.get("semantic_similarity", ""),
                "ndcg_at_10": row.get("ndcg_at_10", ""),
                "mrr_at_10": row.get("mrr_at_10", ""),
                "docs_used": row.get("docs_used", ""),
                "total_tokens": row.get("total_tokens", ""),
            }
        )
        candidates.append(candidate)
    return candidates


def parse_dataset_spec(spec: str) -> tuple[str, Path, Path, Path]:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Dataset spec must be name:eval_csv:queries_jsonl:documents_jsonl"
        )
    name, eval_csv, queries, documents = parts
    return name, Path(eval_csv), Path(queries), Path(documents)


def sheet_name(name: str, used: set[str]) -> str:
    clean = re.sub(r"[\[\]:*?/\\]", "_", name).strip() or "Dataset"
    clean = clean[:31]
    candidate = clean
    suffix = 2
    while candidate in used:
        tail = f"_{suffix}"
        candidate = clean[: 31 - len(tail)] + tail
        suffix += 1
    used.add(candidate)
    return candidate


def column_label(index: int) -> str:
    label = ""
    while index:
        index, rem = divmod(index - 1, 26)
        label = chr(65 + rem) + label
    return label


def cell_xml(row_idx: int, col_idx: int, value: object) -> str:
    ref = f"{column_label(col_idx)}{row_idx}"
    text = "" if value is None else str(value)
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def worksheet_xml(rows: list[dict[str, object]]) -> str:
    table = [HEADERS] + [[row.get(header, "") for header in HEADERS] for row in rows]
    sheet_rows = []
    for row_idx, row in enumerate(table, start=1):
        cells = "".join(cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')

    cols = []
    for idx, header in enumerate(HEADERS, start=1):
        width = 18
        if header in {"query_text", "reference_answer", "selected_document_text", "model_answer"}:
            width = 55
        elif header.endswith("_notes") or header == "final_notes":
            width = 32
        cols.append(f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<cols>{''.join(cols)}</cols>"
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        '<autoFilter ref="A1:AI1"/>'
        "</worksheet>"
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(sheets)}</sheets>"
        "</workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    rels.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(rels)}"
        "</Relationships>"
    )


def content_types_xml(sheet_count: int) -> str:
    worksheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        f"{worksheets}</Types>"
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )


def core_xml() -> str:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>RAG annotation candidates</dc:title>"
        "<dc:creator>adaptive-rag-token-efficiency</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml(sheet_count: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Python</Application>"
        f"<Worksheets>{sheet_count}</Worksheets>"
        "</Properties>"
    )


def write_xlsx(path: Path, sheets: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    normalized = [(sheet_name(name, used_names), rows) for name, rows in sheets.items()]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(normalized)))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("xl/workbook.xml", workbook_xml([name for name, _rows in normalized]))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(normalized)))
        archive.writestr("xl/styles.xml", styles_xml())
        archive.writestr("docProps/core.xml", core_xml())
        archive.writestr("docProps/app.xml", app_xml(len(normalized)))
        for idx, (_name, rows) in enumerate(normalized, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one annotation workbook tab per dataset.")
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_spec,
        required=True,
        metavar="NAME:EVAL_CSV:QUERIES_JSONL:DOCUMENTS_JSONL",
        help="Dataset definition. Repeat this argument once per dataset.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/annotation_best_worst_by_dataset.xlsx"))
    parser.add_argument("--mode", default="answer_aware_fallback", help='Evaluation mode. Use "all" for all modes.')
    parser.add_argument(
        "--metric",
        default="answer_f1",
        choices=["answer_f1", "answer_coverage", "semantic_similarity", "ndcg_at_10", "mrr_at_10"],
    )
    parser.add_argument("--best-count", type=int, default=50)
    parser.add_argument("--worst-count", type=int, default=50)
    parser.add_argument("--max-doc-chars", type=int, default=900)
    args = parser.parse_args()

    sheets = {}
    for dataset_name, eval_csv, queries, documents in args.dataset:
        rows = build_rows(
            dataset_name=dataset_name,
            eval_csv=eval_csv,
            queries_path=queries,
            documents_path=documents,
            mode=args.mode,
            metric=args.metric,
            best_count=args.best_count,
            worst_count=args.worst_count,
            max_doc_chars=args.max_doc_chars,
        )
        sheets[dataset_name] = rows
        reasons = {}
        for row in rows:
            reasons[row["selection_reason"]] = reasons.get(row["selection_reason"], 0) + 1
        print(f"{dataset_name}: {len(rows)} rows {reasons}")

    write_xlsx(args.output, sheets)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
