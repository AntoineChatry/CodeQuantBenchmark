"""Story 1.1 : Collecte et Extraction de Code via AST."""

import ast
import json
from pathlib import Path
from typing import Any

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("data_extract")


def extract_functions_from_file(file_path: Path) -> list[dict[str, Any]]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, ValueError) as e:
        logger.warning(f"Cannot parse {file_path}: {e}")
        return []

    functions: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                func_source = ast.get_source_segment(source, node)
                if func_source is None:
                    continue
                functions.append({
                    "name": node.name,
                    "source": func_source,
                    "file": str(file_path),
                    "lineno": node.lineno,
                    "num_lines": len(func_source.splitlines()),
                })
            except Exception as e:
                logger.warning(f"Error extracting {node.name} from {file_path}: {e}")
    return functions


def extract_from_directory(config: dict[str, Any]) -> list[dict[str, Any]]:
    source_dir = Path(get_nested(config, "data", "source_dir", default="./repos"))
    extensions = get_nested(config, "data", "extensions", default=[".py"])
    min_lines = get_nested(config, "data", "min_lines", default=3)
    max_lines = get_nested(config, "data", "max_lines", default=500)

    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return []

    python_exts = {".py"}
    ts_exts = [e for e in extensions if e not in python_exts]
    py_exts = [e for e in extensions if e in python_exts]

    all_functions: list[dict[str, Any]] = []

    for ext in py_exts:
        for file_path in source_dir.rglob(f"*{ext}"):
            funcs = extract_functions_from_file(file_path)
            for f in funcs:
                if min_lines <= f["num_lines"] <= max_lines:
                    all_functions.append(f)

    if ts_exts:
        from src.data.extract_treesitter import extract_from_directory as ts_extract
        ts_funcs = ts_extract(source_dir, ts_exts, min_lines, max_lines)
        all_functions.extend(ts_funcs)

    logger.info(f"Extracted {len(all_functions)} functions from {source_dir}")
    return all_functions


def save_raw_dataset(functions: list[dict[str, Any]], output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / "raw_dataset.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for func in functions:
            f.write(json.dumps(func, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(functions)} entries to {output_path}")
    return output_path


def run(config_path: str = "config.yaml") -> Path:
    config = load_config(config_path)
    functions = extract_from_directory(config)
    output_dir = get_nested(config, "data", "output_dir", default="./data")
    return save_raw_dataset(functions, output_dir)
