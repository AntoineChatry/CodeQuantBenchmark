"""Story 1.2 : Nettoyage et Segmentation."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("data_clean")


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for record in records:
        h = compute_hash(record["source"])
        if h not in seen:
            seen.add(h)
            unique.append(record)
    removed = len(records) - len(unique)
    logger.info(f"Deduplication: {removed} duplicates removed, {len(unique)} remaining")
    return unique


def split_dataset(
    records: list[dict[str, Any]], ratios: list[float], seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_split(records: list[dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_report(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {"count": 0, "avg_lines": 0, "min_lines": 0, "max_lines": 0}
        lines = [r["num_lines"] for r in records]
        return {
            "count": len(records),
            "avg_lines": round(sum(lines) / len(lines), 1),
            "min_lines": min(lines),
            "max_lines": max(lines),
        }

    report = {
        "total": len(train) + len(val) + len(test),
        "train": stats(train),
        "validation": stats(val),
        "test": stats(test),
    }
    report_path = output_dir / "data_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Data report saved to {report_path}")


def run(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    output_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    ratios = get_nested(config, "data", "split_ratios", default=[0.8, 0.1, 0.1])

    raw_path = output_dir / "raw_dataset.jsonl"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found: {raw_path}. Run extract first.")
        return

    records: list[dict[str, Any]] = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records = deduplicate(records)
    train, val, test = split_dataset(records, ratios)

    save_split(train, output_dir / "train.jsonl")
    save_split(val, output_dir / "validation.jsonl")
    save_split(test, output_dir / "test.jsonl")

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    generate_report(train, val, test, output_dir)
