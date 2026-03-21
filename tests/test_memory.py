"""Tests unitaires pour le module memory."""

import os
import tempfile
from pathlib import Path

from src.benchmark.memory import get_file_size, get_memory


def test_get_memory():
    snapshot = get_memory()
    assert snapshot.rss_bytes > 0
    assert snapshot.rss_mb > 0
    assert snapshot.peak_rss_bytes >= snapshot.rss_bytes


def test_get_file_size_existing():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b"x" * 1024)
        tmp = f.name
    try:
        info = get_file_size(tmp)
        assert info["file_size_bytes"] == 1024
        assert info["file_size_mb"] > 0.0
        assert info["file_size_gb"] >= 0.0
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_get_file_size_nonexistent():
    info = get_file_size("/nonexistent/file.gguf")
    assert info["file_size_bytes"] == 0
