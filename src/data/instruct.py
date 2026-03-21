"""Story 1.3 : Génération d'Instructions (format ShareGPT)."""

import json
import random
from pathlib import Path
from typing import Any

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("data_instruct")


def validate_code_blocks(instruction: dict[str, Any]) -> bool:
    for turn in instruction.get("conversations", []):
        if "```" in turn.get("value", ""):
            return True
    return False


def generate_instruction(
    record: dict[str, Any], templates: list[str]
) -> dict[str, Any]:
    template = random.choice(templates)
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"{template}\n\n```python\n{record['source']}\n```",
            },
            {
                "from": "gpt",
                "value": f"Here is the analysis of the function `{record['name']}`:\n\n```python\n{record['source']}\n```",
            },
        ]
    }


def convert_split(
    input_path: Path, output_path: Path, templates: list[str]
) -> int:
    if not input_path.exists():
        logger.warning(f"File not found: {input_path}")
        return 0

    records: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    instructions = [generate_instruction(r, templates) for r in records]

    valid = 0
    invalid = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for inst in instructions:
            if validate_code_blocks(inst):
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                valid += 1
            else:
                invalid += 1

    if invalid > 0:
        logger.warning(f"Dropped {invalid} entries without code blocks")
    logger.info(f"Generated {valid} valid instructions -> {output_path}")
    return valid


def run(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    output_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    templates = get_nested(
        config, "data", "instruction_templates",
        default=["Explain the following Python function:"],
    )

    for split_name in ["train", "validation", "test"]:
        input_path = output_dir / f"{split_name}.jsonl"
        output_path = output_dir / f"{split_name}_sharegpt.jsonl"
        convert_split(input_path, output_path, templates)
