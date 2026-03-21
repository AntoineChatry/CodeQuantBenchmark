"""Tests unitaires pour le module quantization."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.quantization.convert import find_convert_script


def test_find_convert_script_not_configured():
    config = {"quantization": {}}
    result = find_convert_script(config)
    assert result == ""


def test_find_convert_script_configured():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as f:
        f.write(b"# dummy")
        tmp = f.name
    try:
        config = {"quantization": {"convert_script_path": tmp}}
        result = find_convert_script(config)
        assert result == tmp
    finally:
        Path(tmp).unlink(missing_ok=True)
