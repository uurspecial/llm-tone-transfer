from __future__ import annotations

import argparse
import json

try:
    from unsloth import FastSequenceClassification
except ImportError:
    from unsloth import FastModel as FastSequenceClassification

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import (
    DEFAULT_BASE_MODEL,
    DEFAULT_CLASSIFIER_LABELS,
    DEFAULT_LORA_DIR,
    DEFAULT_TEST_FILE,
    build_classification_text,
    expand_path,
    normalize_label,
    read_base_model_from_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a 4-class sentence classifier LoRA.")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Path to test json.")
    parser.add_argument("--lora-dir", default=DEFAULT_LORA_DIR, help="Path to classifier LoRA folder.")
    parser.add_argument("--base-model", default=None, help="Override base model path or HF id.")
    parser.add_argument("--results-file", default="classifier_test_results_notestfile.csv", help="CSV output path.")
    parser.add_argument("--report-file", default="classifier_metrics_notestfile.json", help="JSON metrics output path.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sentence", default=None, help="Single sentence inference mode.")
    return parser.parse_args()


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


def load_model_and_tokenizer(args: argparse.Namespace):
    lora_dir = expand_path(args.lora_dir)
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    base_model = args.base_model or read_base_model_from_adapter(lora_dir) or DEFAULT_BASE_MODEL
    label_names = DEFAULT_CLASSIFIER_LABELS
    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in id2label.items()}

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
    model = PeftModel.from_pretrained(model, str(lora_dir), is_trainable=False)
    if hasattr(model, "for_inference"):
        model.for_inference()
    model.eval()
    return model, tokenizer, label_names


def predict_texts(model, tokenizer, texts: list[str], max_length: int) -> list[int]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
    return logits.argmax(dim=-1).detach().cpu().tolist()


def run_single_sentence(args: argparse.Namespace, model, tokenizer, label_names: list[str]) -> None:
    prediction_id = predict_texts(model, tokenizer, [args.sentence], args.max_length)[0]
    print(label_names[prediction_id])


def run_dataset_eval(args: argparse.Namespace, model, tokenizer, label_names: list[str]) -> None:
    test_path = expand_path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    dataset = load_dataset("json", data_files=str(test_path), split="train")
    record_ids = []
    texts = []
    true_labels = []
    dropped = []
    for index, record in enumerate(dataset):
        try:
            text = build_classification_text(record)
            raw_label = extract_label_value(record)
            if raw_label is None:
                raise ValueError("missing label key/value")
            label_id = normalize_label(raw_label, label_names)
            sample_id = record.get("id", index + 1)
            texts.append(text)
            true_labels.append(label_id)
            record_ids.append(sample_id)
        except Exception as error:
            dropped.append((index, str(error), record))

    if dropped:
        print(f"Dropped {len(dropped)} invalid rows during evaluation preprocessing.")
        for index, reason, record in dropped[:3]:
            print(f"  - row {index}: {reason}; record={record}")
    if not texts:
        raise ValueError("No valid rows remain for evaluation.")

    predictions: list[int] = []
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        predictions.extend(predict_texts(model, tokenizer, batch_texts, args.max_length))

    results = pd.DataFrame(
        {
            "id": record_ids,
            "text": texts,
            # Match dataset labels (1..4) in output CSV
            "gold_label_id": [index + 1 for index in true_labels],
            "pred_label_id": [index + 1 for index in predictions],
        }
    )
    results["is_correct"] = results["gold_label_id"] == results["pred_label_id"]

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    label_ids = list(range(len(label_names)))
    matrix = confusion_matrix(true_labels, predictions, labels=label_ids)
    report = classification_report(
        true_labels,
        predictions,
        labels=label_ids,
        target_names=label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    results_path = expand_path(args.results_file)
    report_path = expand_path(args.report_file)
    results.to_csv(results_path, index=False)
    report_payload = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": label_names,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report_payload, file, ensure_ascii=False, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved results: {results_path}")
    print(f"Saved metrics: {report_path}")


def main() -> None:
    args = parse_args()
    model, tokenizer, label_names = load_model_and_tokenizer(args)

    if args.sentence:
        run_single_sentence(args, model, tokenizer, label_names)
    else:
        run_dataset_eval(args, model, tokenizer, label_names)


if __name__ == "__main__":
    main()
