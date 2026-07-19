from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"
DEFAULT_LORA_DIR = "~/Downloads/lora"
DEFAULT_TRAIN_FILE = "data/train.json"
DEFAULT_TEST_FILE = "data/test.json"
DEFAULT_CLASSIFIER_LABELS = ["1-溫和", "2-中性", "3-不滿", "4-酸"]

REQUIRED_COLUMNS = {"instruction", "input", "output"}

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def format_prompt(instruction: Any, input_text: Any, output_text: Any = "") -> str:
    instruction_text = "" if instruction is None else str(instruction)
    input_value = "" if input_text is None else str(input_text)
    output_value = "" if output_text is None else str(output_text)
    return ALPACA_PROMPT.format(instruction_text, input_value, output_value)


def read_base_model_from_adapter(lora_dir: str | Path) -> str | None:
    adapter_config = expand_path(lora_dir) / "adapter_config.json"
    if not adapter_config.exists():
        return None

    with adapter_config.open("r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("base_model_name_or_path")


def validate_columns(column_names: list[str] | tuple[str, ...], dataset_name: str) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(column_names))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"{dataset_name} is missing required column(s): {joined}. "
            "Expected JSON records with instruction, input, and output fields."
        )


def find_first_present(column_names: list[str] | tuple[str, ...], candidates: list[str]) -> str | None:
    existing = set(column_names)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def build_classification_text(record: dict[str, Any]) -> str:
    if "sentence" in record and record["sentence"] not in (None, ""):
        return str(record["sentence"])
    if "text" in record and record["text"] not in (None, ""):
        return str(record["text"])
    if "input" in record and "instruction" in record:
        instruction_text = "" if record["instruction"] is None else str(record["instruction"]).strip()
        input_text = "" if record["input"] is None else str(record["input"]).strip()
        if instruction_text and input_text:
            return f"{instruction_text}\n\n{input_text}"
        return instruction_text or input_text
    if "input" in record and record["input"] not in (None, ""):
        return str(record["input"])
    raise ValueError("Could not find text content. Expected sentence, text, or instruction/input columns.")


def normalize_label(raw_label: Any, label_names: list[str]) -> int:
    label_to_id = {name: index for index, name in enumerate(label_names)}
    normalized_to_id = {
        "1": 0,
        "1-溫和": 0,
        "溫和": 0,
        "mild": 0,
        "soft": 0,
        "2": 1,
        "2-中性": 1,
        "中性": 1,
        "neutral": 1,
        "3": 2,
        "3-不滿": 2,
        "不滿": 2,
        "dissatisfied": 2,
        "negative": 2,
        "4": 3,
        "4-酸": 3,
        "酸": 3,
        "sarcastic": 3,
        "snarky": 3,
    }

    if isinstance(raw_label, int):
        # Prefer dataset-style 1..4 labels first to avoid accidental 1-step offset.
        if raw_label in (1, 2, 3, 4):
            return raw_label - 1
        if raw_label in (0, 1, 2, 3):
            return raw_label
    if isinstance(raw_label, float) and raw_label.is_integer():
        return normalize_label(int(raw_label), label_names)

    text = str(raw_label).strip()
    if text in label_to_id:
        return label_to_id[text]
    if text in normalized_to_id:
        return normalized_to_id[text]

    raise ValueError(
        f"Unsupported label value: {raw_label!r}. "
        f"Expected one of 1-4 or names like {', '.join(label_names)}."
    )
