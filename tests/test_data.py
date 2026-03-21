"""Tests unitaires pour le pipeline de données."""

import ast
import json
import tempfile
from pathlib import Path

from src.data.clean import compute_hash, deduplicate, split_dataset
from src.data.extract import extract_functions_from_file


def test_extract_functions():
    code = '''
def hello():
    return "world"

def add(a, b):
    return a + b
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name
    try:
        funcs = extract_functions_from_file(Path(tmp))
        assert len(funcs) == 2
        assert funcs[0]["name"] == "hello"
        assert funcs[1]["name"] == "add"
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_deduplicate():
    records = [
        {"source": "def a(): pass", "name": "a"},
        {"source": "def a(): pass", "name": "a"},
        {"source": "def b(): pass", "name": "b"},
    ]
    result = deduplicate(records)
    assert len(result) == 2


def test_split_dataset():
    records = [{"source": f"func_{i}", "num_lines": 5} for i in range(100)]
    train, val, test = split_dataset(records, [0.8, 0.1, 0.1])
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_compute_hash():
    h1 = compute_hash("hello")
    h2 = compute_hash("hello")
    h3 = compute_hash("world")
    assert h1 == h2
    assert h1 != h3
