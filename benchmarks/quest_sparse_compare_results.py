"""Compare Quest sparse decode JSONL benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source"] = path
                rows.append(row)
    return rows


def label(row: dict[str, Any]) -> str:
    fused_budget = row.get("fused_page_budget")
    budget = fused_budget if fused_budget else row.get("page_budget")
    return f"{row.get('case')}[budget={budget}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", help="Benchmark JSONL files to compare")
    args = parser.parse_args()

    rows = load_rows(args.jsonl)
    if not rows:
        raise SystemExit("No benchmark rows found.")

    baseline = rows[0]
    baseline_tpot = baseline.get("tpot_ms")
    baseline_text = baseline.get("generated_text")
    baseline_tokens = baseline.get("generated_token_ids")

    print("case\tprompt_tokens\tcompletion_tokens\ttpot_ms\tspeedup_vs_first\ttext_match\ttoken_match")
    for row in rows:
        tpot = row.get("tpot_ms")
        speedup = None
        if baseline_tpot and tpot:
            speedup = round(float(baseline_tpot) / float(tpot), 3)
        text_match = (
            baseline_text == row.get("generated_text")
            if baseline_text is not None and row.get("generated_text") is not None
            else None
        )
        token_match = (
            baseline_tokens == row.get("generated_token_ids")
            if baseline_tokens is not None and row.get("generated_token_ids") is not None
            else None
        )
        print(
            "\t".join(
                str(x)
                for x in (
                    label(row),
                    row.get("prompt_tokens"),
                    row.get("completion_tokens"),
                    tpot,
                    speedup,
                    text_match,
                    token_match,
                )
            )
        )


if __name__ == "__main__":
    main()
