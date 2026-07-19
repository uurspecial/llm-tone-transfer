from __future__ import annotations

import argparse
import inspect
import json
from statistics import mean, pstdev

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CLASSIFIER_LABELS,
    DEFAULT_TRAIN_FILE,
    build_classification_text,
    expand_path,
    normalize_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 4-class sentence classifier with LoRA.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Primary dataset path (json/jsonl).")
    parser.add_argument("--test-file", default=None, help="Optional second dataset path to merge before CV.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF model id or local base model path.")
    parser.add_argument("--lora-output-dir", default="classifier_lora", help="Where to save LoRA weights.")
    parser.add_argument("--output-dir", default="classifier_outputs", help="Checkpoint output directory.")
    parser.add_argument("--metrics-file", default="classifier_cv_metrics.json", help="Cross-validation metrics JSON path.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-splits", type=int, default=5, help="Number of stratified folds.")
    parser.add_argument("--fold-index", type=int, default=None, help="Run only one fold (0-based).")
    parser.add_argument("--split-only", action=argparse.BooleanOptionalAction, default=False, help="Only print split info.")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_raw_dataset(data_file: str):
    data_path = expand_path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    return load_dataset("json", data_files=str(data_path), split="train")


def extract_label_value(record: dict) -> object | None:
    candidate_keys = ("label", "labels", "score", "output")
    for key in candidate_keys:
        if key in record and record[key] not in (None, ""):
            return record[key]

    lowered = {str(key).lower().strip(): value for key, value in record.items()}
    for key in candidate_keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def sanitize_dataset(dataset, label_names: list[str]):
    cleaned_rows = []
    dropped = []

    for index, record in enumerate(dataset):
        try:
            text = build_classification_text(record)
            raw_label = extract_label_value(record)
            if raw_label is None:
                raise ValueError("missing label key/value")
            label_id = normalize_label(raw_label, label_names)
            cleaned_rows.append({"text": text, "label": label_id})
        except Exception as error:
            dropped.append((index, str(error), record))

    if dropped:
        print(f"Dropped {len(dropped)} invalid rows during preprocessing.")
        for index, reason, record in dropped[:3]:
            print(f"  - row {index}: {reason}; record={record}")

    if not cleaned_rows:
        raise ValueError("No valid rows remain after preprocessing.")

    return Dataset.from_list(cleaned_rows)


def import_fast_sequence_classification():
    try:
        from unsloth import FastSequenceClassification
    except ImportError:
        from unsloth import FastModel as FastSequenceClassification
    return FastSequenceClassification


def import_training_dependencies():
    import torch
    from peft import TaskType
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    return torch, TaskType, DataCollatorWithPadding, Trainer, TrainingArguments


def resolve_seqcls_base_model(model_name: str) -> str:
    lowered = model_name.lower()
    if "bnb-4bit" not in lowered:
        return model_name

    known_map = {
        "unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit": "unsloth/Meta-Llama-3.1-8B",
        "unsloth/meta-llama-3.1-8b-bnb-4bit": "unsloth/Meta-Llama-3.1-8B",
    }
    if lowered in known_map:
        fallback = known_map[lowered]
        print(
            "Detected pre-quantized bnb base model for sequence classification. "
            f"Switching to {fallback} to avoid Byte initialization errors."
        )
        return fallback

    fallback = model_name.replace("-bnb-4bit", "")
    if fallback != model_name:
        print(
            "Detected pre-quantized bnb base model for sequence classification. "
            f"Switching to {fallback} to avoid Byte initialization errors."
        )
    return fallback


def load_model_and_tokenizer(args: argparse.Namespace, label_names: list[str]):
    FastSequenceClassification = import_fast_sequence_classification()
    _, TaskType, _, _, _ = import_training_dependencies()
    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}

    base_model = resolve_seqcls_base_model(args.base_model)
    effective_load_in_4bit = args.load_in_4bit
    try:
        model, tokenizer = FastSequenceClassification.from_pretrained(
            model_name=base_model,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=effective_load_in_4bit,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
        )
    except Exception as error:
        message = str(error)
        byte_init_error = (
            "normal_kernel_cuda" in message and "not implemented for 'Byte'" in message
        )
        if not (effective_load_in_4bit and byte_init_error):
            raise
        print(
            "Detected Byte initialization error in 4-bit sequence classification load. "
            "Retrying with load_in_4bit=False for classifier-head initialization."
        )
        effective_load_in_4bit = False
        model, tokenizer = FastSequenceClassification.from_pretrained(
            model_name=base_model,
            max_seq_length=args.max_length,
            dtype=None,
            load_in_4bit=effective_load_in_4bit,
            num_labels=len(label_names),
            id2label=id2label,
            label2id=label2id,
        )

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = FastSequenceClassification.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )
    return model, tokenizer


def tokenize_dataset(dataset, tokenizer, max_length: int, label_names: list[str]):
    def preprocess(example):
        tokenized = tokenizer(example["text"], truncation=True, max_length=max_length)
        tokenized["labels"] = int(example["label"])
        return tokenized

    return dataset.map(preprocess, remove_columns=dataset.column_names)


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def run_fold(
    fold_id: int,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    all_dataset,
    args: argparse.Namespace,
    label_names: list[str],
):
    torch, _, DataCollatorWithPadding, Trainer, TrainingArguments = import_training_dependencies()
    model, tokenizer = load_model_and_tokenizer(args, label_names)
    train_raw = all_dataset.select(train_indices.tolist())
    eval_raw = all_dataset.select(eval_indices.tolist())
    train_dataset = tokenize_dataset(train_raw, tokenizer, args.max_length, label_names)
    eval_dataset = tokenize_dataset(eval_raw, tokenizer, args.max_length, label_names)

    fold_output_dir = expand_path(args.output_dir) / f"fold_{fold_id}"
    fold_lora_dir = expand_path(args.lora_output_dir) / f"fold_{fold_id}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    fold_lora_dir.mkdir(parents=True, exist_ok=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=args.seed + fold_id,
        remove_unused_columns=False,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    trainer_init_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(str(fold_lora_dir))
    tokenizer.save_pretrained(str(fold_lora_dir))
    print(
        f"Fold {fold_id}: train={len(train_indices)} eval={len(eval_indices)} "
        f"acc={metrics.get('eval_accuracy')} macro_f1={metrics.get('eval_macro_f1')}"
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be >= 2.")
    label_names = DEFAULT_CLASSIFIER_LABELS

    merged_dataset = load_raw_dataset(args.train_file)
    if args.test_file:
        extra_dataset = load_raw_dataset(args.test_file)
        train_rows = len(merged_dataset)
        extra_rows = len(extra_dataset)
        merged_dataset = concatenate_datasets([merged_dataset, extra_dataset])
        print(
            f"Merged rows: {len(merged_dataset)} "
            f"(train={train_rows}, extra={extra_rows})."
        )
    cleaned_dataset = sanitize_dataset(merged_dataset, label_names)
    print(f"Rows after cleaning: {len(cleaned_dataset)}")
    labels = np.array(cleaned_dataset["label"])
    if len(cleaned_dataset) < args.n_splits:
        raise ValueError(f"Dataset rows ({len(cleaned_dataset)}) must be >= n_splits ({args.n_splits}).")
    class_counts = np.bincount(labels, minlength=len(label_names))
    if class_counts.min() < args.n_splits:
        raise ValueError(
            f"Each label needs at least {args.n_splits} rows for stratified {args.n_splits}-fold. "
            f"Current counts={class_counts.tolist()}."
        )

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_indices = np.arange(len(cleaned_dataset))
    fold_metrics: list[dict[str, float | int]] = []

    for fold_id, (train_idx, eval_idx) in enumerate(splitter.split(all_indices, labels)):
        if args.fold_index is not None and fold_id != args.fold_index:
            continue
        if args.split_only:
            print(f"Fold {fold_id}: train={len(train_idx)} eval={len(eval_idx)}")
            continue
        metrics = run_fold(fold_id, train_idx, eval_idx, cleaned_dataset, args, label_names)
        fold_metrics.append(
            {
                "fold": fold_id,
                "eval_accuracy": float(metrics.get("eval_accuracy", 0.0)),
                "eval_macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
                "eval_loss": float(metrics.get("eval_loss", 0.0)),
            }
        )

    if args.split_only:
        return
    if not fold_metrics:
        raise ValueError("No fold was executed. Check --fold-index and --n-splits.")

    metrics_path = expand_path(args.metrics_file)
    existing_fold_metrics: list[dict[str, float | int]] = []
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                existing_payload = json.load(file)
            loaded = existing_payload.get("fold_metrics", [])
            if isinstance(loaded, list):
                existing_fold_metrics = loaded
        except Exception as error:
            print(f"Warning: failed to read existing metrics file {metrics_path}: {error}")

    by_fold: dict[int, dict[str, float | int]] = {}
    for entry in existing_fold_metrics:
        fold_value = entry.get("fold")
        if isinstance(fold_value, int):
            by_fold[fold_value] = entry
    for entry in fold_metrics:
        by_fold[int(entry["fold"])] = entry

    merged_fold_metrics = [by_fold[index] for index in sorted(by_fold.keys())]
    accuracy_values = [entry["eval_accuracy"] for entry in merged_fold_metrics]
    f1_values = [entry["eval_macro_f1"] for entry in merged_fold_metrics]
    summary = {
        "n_splits": args.n_splits,
        "train_file": str(expand_path(args.train_file)),
        "test_file": str(expand_path(args.test_file)) if args.test_file else None,
        "rows": len(cleaned_dataset),
        "fold_metrics": merged_fold_metrics,
        "completed_folds": [entry["fold"] for entry in merged_fold_metrics],
        "mean_eval_accuracy": mean(accuracy_values),
        "std_eval_accuracy": pstdev(accuracy_values) if len(accuracy_values) > 1 else 0.0,
        "mean_eval_macro_f1": mean(f1_values),
        "std_eval_macro_f1": pstdev(f1_values) if len(f1_values) > 1 else 0.0,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(
        f"CV done. mean_acc={summary['mean_eval_accuracy']:.4f}, "
        f"mean_macro_f1={summary['mean_eval_macro_f1']:.4f}"
    )
    print(f"Saved CV metrics: {metrics_path}")


if __name__ == "__main__":
    main()
