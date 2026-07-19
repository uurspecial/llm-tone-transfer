from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev

from sklearn.model_selection import StratifiedKFold

from common import DEFAULT_CLASSIFIER_LABELS, expand_path, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified 5-fold train/test splits from a single train.jsonl file."
    )
    parser.add_argument(
        "--input-file",
        default="unsloth_vscode/data/train.jsonl",
        help="Source json/jsonl file used for fold splitting.",
    )
    parser.add_argument(
        "--output-dir",
        default="unsloth_vscode/data/stratified_5_fold",
        help="Directory to write fold_{k}/train.jsonl and fold_{k}/test.jsonl.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {path}")

    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("JSON input must be a list of records.")
        return parsed

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_label_value(record: dict) -> object | None:
    for key in ("label", "labels", "score", "output"):
        if key in record and record[key] not in (None, ""):
            return record[key]
    lowered = {str(k).lower().strip(): v for k, v in record.items()}
    for key in ("label", "labels", "score", "output"):
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def class_ratio_map(labels: list[int], num_classes: int) -> dict[str, float]:
    total = len(labels)
    counts = [0] * num_classes
    for value in labels:
        counts[value] += 1
    return {str(i + 1): (counts[i] / total if total else 0.0) for i in range(num_classes)}


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2.")

    input_path = expand_path(args.input_file)
    output_dir = expand_path(args.output_dir)
    rows = load_rows(input_path)
    if len(rows) < args.n_splits:
        raise ValueError(
            f"Input rows ({len(rows)}) must be >= n_splits ({args.n_splits})."
        )

    label_names = DEFAULT_CLASSIFIER_LABELS
    num_classes = len(label_names)
    normalized_labels: list[int] = []
    for idx, row in enumerate(rows, start=1):
        raw_label = extract_label_value(row)
        if raw_label is None:
            raise ValueError(f"Row {idx} missing label field (label/labels/score/output).")
        normalized_labels.append(normalize_label(raw_label, label_names))

    class_counts = [0] * num_classes
    for value in normalized_labels:
        class_counts[value] += 1
    if min(class_counts) < args.n_splits:
        raise ValueError(
            f"Each class needs at least {args.n_splits} rows for stratified {args.n_splits}-fold. "
            f"Current counts={class_counts}."
        )

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_indices = list(range(len(rows)))

    fold_summaries: list[dict] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    per_class_train_ratios: dict[str, list[float]] = {str(i + 1): [] for i in range(num_classes)}
    per_class_test_ratios: dict[str, list[float]] = {str(i + 1): [] for i in range(num_classes)}

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(all_indices, normalized_labels)):
        train_rows = [rows[i] for i in train_idx]
        test_rows = [rows[i] for i in test_idx]

        fold_dir = output_dir / f"fold_{fold_id}"
        write_jsonl(fold_dir / "train.jsonl", train_rows)
        write_jsonl(fold_dir / "test.jsonl", test_rows)

        train_labels = [normalized_labels[i] for i in train_idx]
        test_labels = [normalized_labels[i] for i in test_idx]
        train_ratio = class_ratio_map(train_labels, num_classes)
        test_ratio = class_ratio_map(test_labels, num_classes)

        for class_key in train_ratio:
            per_class_train_ratios[class_key].append(train_ratio[class_key])
            per_class_test_ratios[class_key].append(test_ratio[class_key])

        train_sizes.append(len(train_rows))
        test_sizes.append(len(test_rows))
        fold_summaries.append(
            {
                "fold": fold_id,
                "train_size": len(train_rows),
                "test_size": len(test_rows),
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "train_path": str(fold_dir / "train.jsonl"),
                "test_path": str(fold_dir / "test.jsonl"),
            }
        )

    ratio_std = {
        "train": {
            k: (pstdev(v) if len(v) > 1 else 0.0) for k, v in per_class_train_ratios.items()
        },
        "test": {
            k: (pstdev(v) if len(v) > 1 else 0.0) for k, v in per_class_test_ratios.items()
        },
    }
    summary = {
        "input_file": str(input_path),
        "n_rows": len(rows),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "class_names": {str(i + 1): label_names[i] for i in range(num_classes)},
        "class_counts": {str(i + 1): class_counts[i] for i in range(num_classes)},
        "folds": fold_summaries,
        "std": {
            "train_size_std": pstdev(train_sizes) if len(train_sizes) > 1 else 0.0,
            "test_size_std": pstdev(test_sizes) if len(test_sizes) > 1 else 0.0,
            "class_ratio_std": ratio_std,
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved stratified {args.n_splits}-fold splits to: {output_dir}")
    print(f"Summary: {summary_path}")
    print("Standard deviation (test class ratio):")
    for class_id, value in summary["std"]["class_ratio_std"]["test"].items():
        print(f"  class {class_id}: {value:.6f}")


if __name__ == "__main__":
    main()
