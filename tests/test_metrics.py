"""Tests unitaires pour les métriques de benchmark."""

from src.benchmark.metrics import (
    extract_code_blocks,
    jaccard_similarity,
    ms_per_token,
    syntax_validity,
)


def test_jaccard_identical():
    assert jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert jaccard_similarity("hello world", "foo bar") == 0.0


def test_jaccard_partial():
    score = jaccard_similarity("hello world foo", "hello world bar")
    assert 0.0 < score < 1.0


def test_jaccard_empty():
    assert jaccard_similarity("", "") == 1.0


def test_syntax_valid():
    assert syntax_validity("def foo():\n    return 42") is True


def test_syntax_invalid():
    assert syntax_validity("def foo(\n    return") is False


def test_extract_code_blocks():
    text = "Some text\n```python\ndef foo():\n    pass\n```\nMore text"
    code = extract_code_blocks(text)
    assert "def foo():" in code


def test_extract_no_blocks():
    text = "just plain text"
    assert extract_code_blocks(text) == text


def test_ms_per_token():
    assert ms_per_token(1000.0, 100) == 10.0
    assert ms_per_token(0.0, 0) == 0.0


def test_syntax_validity_empty():
    assert syntax_validity("") is True
