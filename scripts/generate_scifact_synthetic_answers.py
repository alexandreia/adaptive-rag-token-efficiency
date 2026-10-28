#!/usr/bin/env python3
"""Generate QA-style synthetic reference answers for SciFact claims.

SciFact provides gold evidence documents, but not short QA-style reference
answers. This script creates those references with an OpenAI-compatible chat
model while preserving the project's standard JSONL query format.

It is resumable: if an output file or existing gold file already contains a
query_id, that answer is reused and the model is only called for missing rows.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_map(path: Path) -> dict[str, str]:
    return {str(row["doc_id"]): row["text"] for row in read_jsonl(path)}


def gold_answer_map(paths: list[Path]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for path in paths:
        for row in read_jsonl(path):
            answer = str(row.get("reference_answer", "")).strip()
            if answer:
                answers[str(row["query_id"])] = answer
    return answers


def evidence_text(query: dict, docs: dict[str, str], max_chars_per_doc: int) -> str:
    chunks = []
    for doc_id in query.get("relevant_doc_ids", []):
        text = docs.get(str(doc_id), "")
        if text:
            chunks.append(f"[doc {doc_id}] {text[:max_chars_per_doc]}")
    return "\n\n".join(chunks) if chunks else "No gold evidence text was available."


def build_prompt(query: dict, docs: dict[str, str], max_chars_per_doc: int) -> str:
    return f"""Create one short reference answer for evaluating a RAG system on SciFact.

The input is a scientific claim and the gold evidence document text. Decide whether the evidence supports, contradicts, or is insufficient for the claim.

Rules:
- Write one concise natural-language reference answer.
- Match the existing SciFact reference style: usually 4 to 20 words, and never more than 30 words.
- If the evidence supports the claim, state the supported claim directly.
- If the evidence contradicts the claim, begin with "The evidence contradicts the claim" and briefly say why.
- If the evidence is insufficient, write exactly: "The evidence is insufficient."
- Do not add citations, bullet points, or extra explanation.

Claim:
{query["text"]}

Gold evidence:
{evidence_text(query, docs, max_chars_per_doc)}
"""


def call_chat(
    prompt: str,
    *,
    model: str,
    api_url: str,
    api_key: str | None,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
) -> str:
    try:
        import requests
    except ImportError as error:
        raise SystemExit("Install requests first: python -m pip install requests") from error

    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    body = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You create concise reference answers for factual RAG evaluation.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"LLM API request failed: {response.status_code} {response.text}")
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def call_berget_cli(prompt: str, *, command: str, timeout: int) -> str:
    argv = shlex.split(command) + [prompt]
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Berget CLI request failed: {result.stderr.strip()}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    answer_lines = [line for line in lines if not line.lower().startswith("using api key")]
    return "\n".join(answer_lines).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, default=Path("data/scifact/documents.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/scifact/queries_all.jsonl"))
    parser.add_argument(
        "--existing-gold",
        type=Path,
        action="append",
        default=[Path("data/scifact/queries_150_seed0_llm_gold_v2.jsonl")],
        help="Existing synthetic-gold files to reuse. Can be passed more than once.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/scifact_250/queries_250_llm_gold.jsonl"))
    parser.add_argument("--target-count", type=int, default=250)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--api-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--no-api-key", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-chars-per-doc", type=int, default=2200)
    parser.add_argument("--dry-run", action="store_true", help="Write only reused rows; do not call the model.")
    parser.add_argument("--debug-auth", action="store_true", help="Print masked API-key diagnostics.")
    parser.add_argument("--use-berget-cli", action="store_true", help="Call Berget through its CLI instead of HTTP.")
    parser.add_argument("--berget-cli-command", default="npx berget chat run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_api_key and "localhost" not in args.api_url and "127.0.0.1" not in args.api_url:
        raise SystemExit("--no-api-key is only for local providers such as Ollama, not OpenRouter.")
    if not args.no_api_key and not os.environ.get(args.api_key_env):
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env} in this terminal, or pass "
            "--api-key-env with the variable name you are using."
        )
    if args.debug_auth:
        key = os.environ.get(args.api_key_env, "")
        cleaned = key.strip()
        print(f"api_url={args.api_url}")
        print(f"api_key_env={args.api_key_env}")
        print(f"key_present={bool(cleaned)}")
        print(f"key_length={len(cleaned)}")
        print(f"key_starts_with={cleaned[:10]!r}")
        print(f"contains_bearer={'bearer' in cleaned.lower()}")

    docs = document_map(args.documents)
    queries = read_jsonl(args.queries)[: args.target_count]

    reused_answers = gold_answer_map([*args.existing_gold, args.output])
    output_rows: list[dict] = []
    new_calls = 0
    reused_count = 0
    skipped_count = 0

    api_key = None if args.no_api_key else os.environ.get(args.api_key_env, "").strip()

    for index, query in enumerate(queries, start=1):
        query_id = str(query["query_id"])
        row = {
            "query_id": query_id,
            "text": query["text"],
            "relevant_doc_ids": query.get("relevant_doc_ids", []),
        }

        if query_id in reused_answers:
            row["reference_answer"] = reused_answers[query_id]
            output_rows.append(row)
            reused_count += 1
            continue

        if args.dry_run or (args.max_new is not None and new_calls >= args.max_new):
            skipped_count += 1
            continue

        prompt = build_prompt(query, docs, args.max_chars_per_doc)
        if args.use_berget_cli:
            answer = call_berget_cli(
                prompt,
                command=args.berget_cli_command,
                timeout=args.request_timeout_seconds,
            )
        else:
            answer = call_chat(
                prompt,
                model=args.model,
                api_url=args.api_url,
                api_key=api_key,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.request_timeout_seconds,
            )
        row["reference_answer"] = answer
        output_rows.append(row)
        new_calls += 1

        write_jsonl(args.output, output_rows)
        print(f"[{index}/{len(queries)}] generated {query_id}: {answer}")

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")
    print(f"Reused existing answers: {reused_count}")
    print(f"Generated new answers: {new_calls}")
    print(f"Skipped missing answers: {skipped_count}")


if __name__ == "__main__":
    main()
