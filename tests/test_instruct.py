"""Tests unitaires pour la validation des instructions."""

from src.data.instruct import validate_code_blocks


def test_validate_with_code():
    inst = {
        "conversations": [
            {"from": "human", "value": "Explain:\n```python\ndef foo(): pass\n```"},
            {"from": "gpt", "value": "```python\ndef foo(): pass\n```"},
        ]
    }
    assert validate_code_blocks(inst) is True


def test_validate_without_code():
    inst = {
        "conversations": [
            {"from": "human", "value": "Just text"},
            {"from": "gpt", "value": "More text"},
        ]
    }
    assert validate_code_blocks(inst) is False


def test_validate_empty():
    assert validate_code_blocks({}) is False
