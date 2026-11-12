
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """

    # 1. Check that all required fields are present.
    required_fields = ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"]
    if not all(field in example for field in required_fields):
        return False
    
    # 2. Check that prompt, teacher_lyrics, and teacher_notes are non-empty strings.
    for field in ["prompt", "teacher_lyrics", "teacher_notes"]:
        if not isinstance(example[field], str) or not example[field].strip():
            return False
        
    # 3. Check that quality_score can be converted to float.
    try:
        float(example["quality_score"])
    except (ValueError, TypeError):
        return False
    
    # 4. Return True only if all checks pass.
    return True


def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""

    # 1. Validate the example.
    is_valid = validate_teacher_example(example)

    # 2. Raise ValueError if invalid.
    if not is_valid:
        raise ValueError("Invalid teacher example.")

    # 3. Convert quality_score to float.
    quality_score = float(example["quality_score"])

    # 4. Put any extra keys into metadata.
    required_fields = {
        "prompt", "teacher_lyrics", "teacher_notes", "quality_score"
    }
    metadata = {
        key: value for key, value in example.items()
        if key not in required_fields
    }

    # 5. Return a TeacherExample.
    return TeacherExample(
        prompt=example["prompt"],
        teacher_lyrics=example["teacher_lyrics"],
        teacher_notes=example["teacher_notes"],
        quality_score=quality_score,
        metadata=metadata
    )


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    # 1. Strip teacher_lyrics and teacher_notes.
    teacher_lyrics = teacher_lyrics.strip()
    teacher_notes = teacher_notes.strip()

    # 2. If include_thinking=True, return notes inside <think>...</think> tags,
    if include_thinking:
        response = f"<think>\n{teacher_notes}\n</think>\n\n{teacher_lyrics}"

    #    followed by a blank line and the lyrics.

    # 3. If include_thinking=False, return only the lyrics.
    else:
        response = teacher_lyrics

    # 4. Avoid extra leading/trailing whitespace.
    return response.strip()

def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""

    # 1. Strip the prompt.
    prompt = prompt.strip()
    # 2. Return a consistent format such as:
    #    "User: Write a parody...\\nAssistant:"
    return f"User: {prompt}\nAssistant:"


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""

    # 1. Use format_training_prompt(prompt).
    formatted_prompt = format_training_prompt(prompt)

    # 2. Append target_text after the assistant marker.
    full_text = f"{formatted_prompt}\n{target_text}"

    # 3. Return the full training string.
    return full_text


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""

    # 1. Call tokenizer.encode(text).
    token_ids = tokenizer.encode(text)

    # 2. Validate that the result is a list of integers.
    if not isinstance(token_ids, list) or any(type(token_id) is not int for token_id in token_ids):
        raise ValueError("Tokenizer must return a list of integers.")

    # 3. Return the token IDs.
    return token_ids

def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """

    # 1. Copy input_ids into labels.
    labels = input_ids.copy()

    # 2. If mask_prompt=True, replace labels before prompt_length with ignore_index.
    if mask_prompt:
        labels[:prompt_length] = [ignore_index] * len(labels[:prompt_length])

    # 3. Return labels.
    return labels

def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""

    # 1. Format the teacher response.
    target_text = format_teacher_response(
        example.teacher_lyrics, example.teacher_notes, include_thinking=include_thinking
    )

    # 2. Format the prompt.
    formatted_prompt = format_training_prompt(example.prompt)

    # 3. Build the full text.
    full_text = build_full_text(example.prompt, target_text)

    # 4. Tokenize the prompt and full text.
    prompt_ids = tokenize_text(tokenizer, formatted_prompt)
    input_ids = tokenize_text(tokenizer, full_text)
    prompt_length = len(prompt_ids)

    # 5. Create labels.
    labels = create_labels(input_ids, prompt_length, mask_prompt=mask_prompt)

    # 6. Return a DistillationRecord.
    return DistillationRecord(
        prompt=example.prompt,
        target_text=target_text,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=example.metadata.copy(),
    )

def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""

    # 1. Keep examples with quality_score >= min_quality_score.
    quality_examples = [
            example for example in examples if example.quality_score >= min_quality_score
        ]
    
    # 2. Keep examples with at least min_lyric_lines non-empty lyric lines.
    filtered = [
        example for example in quality_examples
        if len([line for line in example.teacher_lyrics.replace("\\n", "\n").splitlines() if line.strip()])
        >= min_lyric_lines
    ]

    # 3. Return the filtered list.
    return filtered


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""

    # 1. Validate validation_fraction is between 0 and 1.
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    # 2. Shuffle a copy of records using seed.
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)

    # 3. Split into train and validation records.
    validation_count = int(len(shuffled) * validation_fraction)
    validation_records = shuffled[:validation_count]
    train_records = shuffled[validation_count:]

    # 4. Return both lists.
    return train_records, validation_records

def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""

    # 1. Convert the dataclass to a dictionary.
    data = asdict(record)

    # 2. Return the dictionary.
    return data


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""

    # 1. Open path for writing with UTF-8 encoding.
    with Path(path).open("w", encoding="utf-8") as file:

    # 2. Write one JSON object per line.
        for record in records:

    # 3. Use record_to_json_dict for each record.
                file.write(json.dumps(record_to_json_dict(record), ensure_ascii=False) + "\n")

def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""

    # 1. Open path with UTF-8 encoding.
    with Path(path).open("r", encoding="utf-8") as file:

    # 2. Read each non-empty line.
        lines = [line for line in file if line.strip()]

    # 3. Parse JSON and append to a list.
        records = [json.loads(line) for line in lines]

    # 4. Return the list.
    return records

def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""

    # 1. Handle empty records.
    if not records:
        return "No records available."

    # 2. Include number of records.
    count = len(records)

    # 3. Include average input length.
    average_input_length = sum(len(record.input_ids) for record in records) / count

    # 4. Include average prompt length.
    average_prompt_length = sum(record.prompt_length for record in records) / count

    # 5. Include average target length.
    average_target_length = sum(
        len(record.input_ids) - record.prompt_length for record in records
    ) / count
    return (
        f"Number of records: {count}\n"
        f"Average input length: {average_input_length:.2f}\n"
        f"Average prompt length: {average_prompt_length:.2f}\n"
        f"Average target length: {average_target_length:.2f}"
    )
