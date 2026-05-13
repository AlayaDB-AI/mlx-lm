"""Score a small LongBench prediction directory without optional rouge deps."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: list[str], ground_truth: list[str]) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    return f1_score(
        normalize_answer(prediction).split(),
        normalize_answer(ground_truth).split(),
    )


def retrieval_score(prediction: str, ground_truth: str) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    target = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1 for number in numbers if number == target) / len(numbers)


def score_row(dataset: str, row: dict[str, Any]) -> float:
    pred = row.get("pred", "")
    answers = row.get("answers", [])
    if not answers:
        return 0.0
    if dataset == "passage_retrieval_en":
        scorer = retrieval_score
    else:
        scorer = qa_f1_score
    return max(scorer(pred, answer) for answer in answers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir")
    args = parser.parse_args()

    pred_dir = Path(args.prediction_dir)
    rows_out = []
    for path in sorted(pred_dir.glob("*.jsonl")):
        dataset = path.stem
        scores = []
        tpot_values = []
        ttft_values = []
        latencies = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scores.append(score_row(dataset, row))
            if row.get("tpot") is not None:
                tpot_values.append(float(row["tpot"]) * 1000.0)
            if row.get("ttft") is not None:
                ttft_values.append(float(row["ttft"]))
            if row.get("latency") is not None:
                latencies.append(float(row["latency"]))
        rows_out.append(
            {
                "dataset": dataset,
                "samples": len(scores),
                "score": round(100.0 * sum(scores) / len(scores), 2) if scores else None,
                "avg_tpot_ms": round(sum(tpot_values) / len(tpot_values), 3) if tpot_values else None,
                "avg_ttft_s": round(sum(ttft_values) / len(ttft_values), 3) if ttft_values else None,
                "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            }
        )
    print(json.dumps(rows_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
