"""Tests unitaires pour le chargement de configuration."""

import tempfile
from pathlib import Path

import yaml

from src.utils.config import get_nested, load_config


def test_load_config():
    data = {"project": {"name": "test"}, "data": {"extensions": [".py"]}}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(data, f)
        tmp = f.name
    try:
        config = load_config(tmp)
        assert config["project"]["name"] == "test"
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_get_nested():
    config = {"a": {"b": {"c": 42}}}
    assert get_nested(config, "a", "b", "c") == 42
    assert get_nested(config, "a", "x", default="nope") == "nope"
