from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-fold training by invoking train_classifier.py fold-by-fold and summarize metrics."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run train_classifier.py.")
    parser.add_argument("--script-path", default="unsloth_vscode/train_classifier.py", help="Path to train_classifier.py.")
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--train-file", default="unsloth_vscode/data/train.jsonl")
    parser.add_argument("--test-file", default=None, help="Optional extra file merged before CV.")
    parser.add_argument("--lora-output-dir", default="classifier_lora_cv")
    parser.add_argument("--output-dir", default="classifier_outputs")
    parser.add_argument("--metrics-file", default="classifier_cv_metrics.json")
    parser.add_argument("--summary-file", default="classifier_cv_summary.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run_fold(args: argparse.Namespace, fold: int) -> None:
    cmd = [
        args.python,
        args.script_path,
        "--base-model",
        args.base_model,
        "--train-file",
        args.train_file,
        "--lora-output-dir",
        args.lora_output_dir,
        "--output-dir",
        args.output_dir,
        "--metrics-file",
        args.metrics_file,
        "--n-splits",
        str(args.n_splits),
        "--fold-index",
        str(fold),
        "--seed",
        str(args.seed),
    ]
    if args.test_file:
        cmd.extend(["--test-file", args.test_file])
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    else:
        cmd.append("--no-load-in-4bit")

    print(f"\n=== Running fold_{fold} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_summary(metrics_file: Path, summary_file: Path) -> dict:
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    folds = payload.get("fold_metrics", [])
    if not folds:
        raise ValueError(f"No fold_metrics found in {metrics_file}")

    folds = sorted(folds, key=lambda x: x["fold"])
    acc = [float(item["eval_accuracy"]) for item in folds]
    macro_f1 = [float(item["eval_macro_f1"]) for item in folds]
    loss = [float(item["eval_loss"]) for item in folds]

    summary = {
        "n_folds_recorded": len(folds),
        "fold_metrics": folds,
        "mean_eval_accuracy": mean(acc),
        "std_eval_accuracy": pstdev(acc) if len(acc) > 1 else 0.0,
        "mean_eval_macro_f1": mean(macro_f1),
        "std_eval_macro_f1": pstdev(macro_f1) if len(macro_f1) > 1 else 0.0,
        "mean_eval_loss": mean(loss),
        "std_eval_loss": pstdev(loss) if len(loss) > 1 else 0.0,
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_file).expanduser().resolve()
    summary_path = Path(args.summary_file).expanduser().resolve()

    for fold in range(args.n_splits):
        run_fold(args, fold)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found after training: {metrics_path}")

    summary = build_summary(metrics_path, summary_path)
    print("\n=== 5-fold Summary ===")
    print(f"mean_acc      : {summary['mean_eval_accuracy']:.6f}")
    print(f"std_acc       : {summary['std_eval_accuracy']:.6f}")
    print(f"mean_macro_f1 : {summary['mean_eval_macro_f1']:.6f}")
    print(f"std_macro_f1  : {summary['std_eval_macro_f1']:.6f}")
    print(f"mean_loss     : {summary['mean_eval_loss']:.6f}")
    print(f"std_loss      : {summary['std_eval_loss']:.6f}")
    print(f"Saved summary : {summary_path}")


if __name__ == "__main__":
    main()
