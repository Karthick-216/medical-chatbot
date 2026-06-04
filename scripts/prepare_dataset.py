"""Prepare a compact medical QA dataset for instruction fine-tuning.

The script downloads a public Hugging Face dataset, normalizes question/answer
columns, adds the safety-oriented instruction template used by the app, and writes
JSONL files that can be consumed by `scripts/fine_tune.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

DEFAULT_DATASET = "medalpaca/medical_meadow_medqa"
DEFAULT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Prepare medical QA fine-tuning data.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset ID.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to load.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--max-rows", type=int, default=5000, help="Maximum rows to keep.")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Validation fraction.")
    return parser.parse_args()


def first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    """Return the first non-empty value among candidate columns."""

    for key in candidates:
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def to_instruction_record(row: dict[str, Any]) -> dict[str, str] | None:
    """Convert a heterogeneous medical QA row into instruction-tuning text."""

    question = first_present(row, ("instruction", "question", "input", "prompt"))
    context = first_present(row, ("context", "background", "passage"))
    answer = first_present(row, ("output", "answer", "response", "completion"))

    if not question or not answer:
        return None

    prompt = (
        "You are a careful medical support assistant. Provide general education, "
        "avoid diagnosis, and recommend professional care for urgent or unclear cases.\n\n"
        f"Question: {question}"
    )
    if context:
        prompt = f"{prompt}\n\nContext: {context}"

    return {"prompt": prompt, "response": answer}


def normalize_dataset(dataset: Dataset, max_rows: int) -> Dataset:
    """Clean records and keep a deterministic subset for small-scale fine-tuning."""

    records = []
    for row in dataset:
        record = to_instruction_record(row)
        if record:
            records.append(record)
        if len(records) >= max_rows:
            break
    return Dataset.from_list(records)


def write_jsonl(dataset: Dataset, path: Path) -> None:
    """Write a Hugging Face dataset split to JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in dataset:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Download, clean, split, and write fine-tuning data."""

    args = parse_args()
    raw_dataset = load_dataset(args.dataset, split=args.split)
    cleaned = normalize_dataset(raw_dataset, max_rows=args.max_rows)
    split = cleaned.train_test_split(test_size=args.validation_size, seed=42)

    output_dir = Path(args.output_dir)
    write_jsonl(split["train"], output_dir / "train.jsonl")
    write_jsonl(split["test"], output_dir / "validation.jsonl")
    print(f"Wrote {len(split['train'])} train and {len(split['test'])} validation records.")


if __name__ == "__main__":
    main()
