"""Extraction multi-langage de fonctions via tree-sitter.

Supporte Python, Rust, C, C++, Go, JavaScript, TypeScript, Java,
Ruby, PHP, Scala, Kotlin, Lua, Haskell, OCaml, Elixir, Bash.
"""

import warnings
from pathlib import Path
from typing import Any

from src.utils.logging import setup_logger

warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logger("data_extract_ts")

# Mapping extension -> (tree-sitter language name, AST node types for functions)
LANG_MAP: dict[str, tuple[str, list[str]]] = {
    ".py": ("python", ["function_definition"]),
    ".rs": ("rust", ["function_item"]),
    ".c": ("c", ["function_definition"]),
    ".h": ("c", ["function_definition"]),
    ".cpp": ("cpp", ["function_definition"]),
    ".hpp": ("cpp", ["function_definition"]),
    ".cc": ("cpp", ["function_definition"]),
    ".go": ("go", ["function_declaration", "method_declaration"]),
    ".js": ("javascript", ["function_declaration", "arrow_function"]),
    ".ts": ("typescript", ["function_declaration", "arrow_function"]),
    ".java": ("java", ["method_declaration"]),
    ".rb": ("ruby", ["method"]),
    ".php": ("php", ["function_definition", "method_declaration"]),
    ".scala": ("scala", ["function_definition"]),
    ".kt": ("kotlin", ["function_declaration"]),
    ".lua": ("lua", ["function_definition_statement"]),
    ".hs": ("haskell", ["function"]),
    ".ml": ("ocaml", ["let_binding"]),
    ".ex": ("elixir", ["call"]),
    ".sh": ("bash", ["function_definition"]),
}


def get_parser(lang_name: str) -> Any:
    from tree_sitter_languages import get_parser as ts_get_parser
    return ts_get_parser(lang_name)


def extract_functions_treesitter(
    file_path: Path, lang_name: str, node_types: list[str]
) -> list[dict[str, Any]]:
    try:
        source_bytes = file_path.read_bytes()
        source_text = source_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Cannot read {file_path}: {e}")
        return []

    try:
        parser = get_parser(lang_name)
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.warning(f"Cannot parse {file_path} as {lang_name}: {e}")
        return []

    functions: list[dict[str, Any]] = []
    _walk_tree(tree.root_node, source_text, str(file_path), node_types, functions)
    return functions


def _walk_tree(
    node: Any,
    source: str,
    file_path: str,
    node_types: list[str],
    results: list[dict[str, Any]],
) -> None:
    if node.type in node_types:
        start = node.start_point
        end = node.end_point
        func_text = source[node.start_byte:node.end_byte]
        num_lines = end[0] - start[0] + 1

        name = _extract_name(node)
        results.append({
            "name": name,
            "source": func_text,
            "file": file_path,
            "lineno": start[0] + 1,
            "num_lines": num_lines,
            "language": node.type,
        })

    for child in node.children:
        _walk_tree(child, source, file_path, node_types, results)


def _extract_name(node: Any) -> str:
    for child in node.children:
        if child.type in ("identifier", "name", "property_identifier"):
            return child.text.decode("utf-8", errors="ignore") if isinstance(child.text, bytes) else str(child.text)
    return "<anonymous>"


def extract_from_directory(
    source_dir: Path,
    extensions: list[str],
    min_lines: int = 3,
    max_lines: int = 500,
) -> list[dict[str, Any]]:
    all_functions: list[dict[str, Any]] = []

    for ext in extensions:
        if ext not in LANG_MAP:
            logger.warning(f"No tree-sitter grammar for extension '{ext}', skipping")
            continue

        lang_name, node_types = LANG_MAP[ext]

        try:
            get_parser(lang_name)
        except Exception:
            logger.warning(f"tree-sitter language '{lang_name}' not available, skipping {ext}")
            continue

        for file_path in source_dir.rglob(f"*{ext}"):
            funcs = extract_functions_treesitter(file_path, lang_name, node_types)
            for f in funcs:
                if min_lines <= f["num_lines"] <= max_lines:
                    all_functions.append(f)

    logger.info(f"Extracted {len(all_functions)} functions from {source_dir}")
    return all_functions
