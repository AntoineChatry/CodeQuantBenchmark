"""Test d'intégration synthétique end-to-end.

Crée un mini-repo Python synthétique, déroule le pipeline
data (extract → clean → instruct) puis benchmark avec un
moteur d'inférence mock, et vérifie que tous les artefacts
attendus sont produits avec le bon contenu.
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from src.benchmark.metrics import bleu_score, jaccard_similarity, syntax_validity
from src.inference.base import InferenceEngine, InferenceResult


# ─── Fixtures ────────────────────────────────────────────────────────────────

SYNTHETIC_PYTHON_CODE = '''
def add(a, b):
    """Add two numbers."""
    return a + b


def multiply(x, y):
    """Multiply two numbers."""
    return x * y


def fibonacci(n):
    """Return nth fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def factorial(n):
    """Return factorial of n."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def reverse_string(s):
    """Reverse a string."""
    return s[::-1]


def flatten(lst):
    """Flatten a nested list."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def binary_search(arr, target):
    """Binary search in sorted array."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def merge_sort(arr):
    """Sort array using merge sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
'''

SYNTHETIC_RUST_CODE = '''
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(x: i32, y: i32) -> i32 {
    x * y
}

fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    let mut a = 0u32;
    let mut b = 1u32;
    for _ in 2..=n {
        let temp = b;
        b = a + b;
        a = temp;
    }
    b
}
'''


class MockInferenceEngine(InferenceEngine):
    """Moteur d'inférence mock qui retourne du code Python valide."""

    def __init__(self) -> None:
        self.loaded = False
        self.model_path = ""

    def load(self, model_path: str) -> None:
        self.model_path = model_path
        self.loaded = True

    def generate(self, prompt: str, max_tokens: int = 512) -> InferenceResult:
        response = (
            "Here is the function:\n\n"
            "```python\n"
            "def example():\n"
            "    return 42\n"
            "```\n"
        )
        return InferenceResult(
            text=response,
            prompt_tokens=len(prompt.split()),
            completion_tokens=15,
            latency_ms=50.0,
        )

    def unload(self) -> None:
        self.loaded = False
        self.model_path = ""

    def get_metrics(self) -> dict[str, float]:
        return {"total_requests": 0, "total_latency_ms": 0.0, "avg_latency_ms": 0.0}


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    """Crée un workspace synthétique complet."""
    repos_dir = tmp_path / "repos" / "fake_project"
    repos_dir.mkdir(parents=True)

    (repos_dir / "utils.py").write_text(SYNTHETIC_PYTHON_CODE, encoding="utf-8")
    (repos_dir / "math_lib.rs").write_text(SYNTHETIC_RUST_CODE, encoding="utf-8")
    (repos_dir / "broken.py").write_text("def broken(\n  this won't parse", encoding="utf-8")

    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models_quantized"

    data_dir.mkdir()
    reports_dir.mkdir()
    models_dir.mkdir()

    # Fake GGUF files (juste pour les paths)
    for fmt in ["Q4_K_M", "Q8_0"]:
        (models_dir / f"model-{fmt}.gguf").write_bytes(b"\x00" * 1024)

    config = {
        "project": {"name": "TestBenchmark", "domain": "python_test"},
        "data": {
            "source_dir": str(repos_dir.parent),
            "output_dir": str(data_dir),
            "extensions": [".py"],
            "split_ratios": [0.8, 0.1, 0.1],
            "min_lines": 3,
            "max_lines": 500,
            "instruction_templates": [
                "Explain the following Python function:",
                "Refactor this Python function:",
            ],
        },
        "model": {"base_name": "test-model"},
        "training": {"output_dir": str(tmp_path / "models_trained")},
        "quantization": {
            "output_dir": str(models_dir),
            "formats": ["Q4_K_M", "Q8_0"],
        },
        "benchmark": {
            "test_file": str(data_dir / "test.jsonl"),
            "results_dir": str(reports_dir),
            "metrics": ["syntax_validity", "jaccard_similarity", "bleu", "latency"],
            "inference": {
                "api_base": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "max_tokens": 64,
                "temperature": 0.1,
            },
        },
        "reporting": {"output_dir": str(reports_dir)},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return {
        "tmp": tmp_path,
        "config_path": config_path,
        "repos": repos_dir.parent,
        "data": data_dir,
        "reports": reports_dir,
        "models": models_dir,
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestDataPipeline:
    """Test complet Epic 1 : extract → clean → instruct."""

    def test_extract(self, workspace: dict[str, Path]) -> None:
        from src.data.extract import run
        path = run(str(workspace["config_path"]))
        assert path.exists()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 5  # Au moins 5 fonctions extraites

        first = json.loads(lines[0])
        assert "name" in first
        assert "source" in first
        assert "num_lines" in first

    def test_clean(self, workspace: dict[str, Path]) -> None:
        from src.data.clean import run as clean_run
        from src.data.extract import run as extract_run

        extract_run(str(workspace["config_path"]))
        clean_run(str(workspace["config_path"]))

        data_dir = workspace["data"]
        assert (data_dir / "train.jsonl").exists()
        assert (data_dir / "validation.jsonl").exists()
        assert (data_dir / "test.jsonl").exists()
        assert (data_dir / "data_report.json").exists()

        with open(data_dir / "data_report.json", "r") as f:
            report = json.load(f)
        assert report["total"] > 0
        assert report["train"]["count"] > 0

    def test_instruct(self, workspace: dict[str, Path]) -> None:
        from src.data.clean import run as clean_run
        from src.data.extract import run as extract_run
        from src.data.instruct import run as instruct_run

        extract_run(str(workspace["config_path"]))
        clean_run(str(workspace["config_path"]))
        instruct_run(str(workspace["config_path"]))

        data_dir = workspace["data"]
        sharegpt = data_dir / "train_sharegpt.jsonl"
        assert sharegpt.exists()

        lines = sharegpt.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0

        sample = json.loads(lines[0])
        assert "conversations" in sample
        assert any("```" in turn["value"] for turn in sample["conversations"])

    def test_no_data_leak(self, workspace: dict[str, Path]) -> None:
        """Vérifie qu'il n'y a pas de fuite entre train et test."""
        from src.data.clean import run as clean_run
        from src.data.extract import run as extract_run

        extract_run(str(workspace["config_path"]))
        clean_run(str(workspace["config_path"]))

        data_dir = workspace["data"]

        def load_sources(path: Path) -> set[str]:
            sources = set()
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                record = json.loads(line)
                sources.add(record["source"])
            return sources

        train_sources = load_sources(data_dir / "train.jsonl")
        test_sources = load_sources(data_dir / "test.jsonl")
        assert train_sources.isdisjoint(test_sources), "Data leak: train/test overlap"


class TestBenchmarkWithMock:
    """Test Epic 3 avec moteur d'inférence mock."""

    def _setup_data(self, workspace: dict[str, Path]) -> None:
        from src.data.clean import run as clean_run
        from src.data.extract import run as extract_run
        extract_run(str(workspace["config_path"]))
        clean_run(str(workspace["config_path"]))

    def test_evaluate_model(self, workspace: dict[str, Path]) -> None:
        self._setup_data(workspace)
        from src.benchmark.engine import evaluate_model, load_test_set

        test_data = load_test_set(str(workspace["data"] / "test.jsonl"))
        assert len(test_data) > 0

        engine = MockInferenceEngine()
        results = evaluate_model(
            engine, "fake-model.gguf", test_data, max_tokens=64, model_label="Q4_K_M"
        )

        assert len(results) == len(test_data)
        for r in results:
            assert "jaccard" in r
            assert "bleu" in r
            assert "syntax_valid" in r
            assert "latency_ms" in r
            assert "ms_per_token" in r
            assert "model_file_size_mb" in r
            assert "rss_mb" in r
            assert r["model"] == "Q4_K_M"

    def test_syntax_validity_on_mock(self, workspace: dict[str, Path]) -> None:
        self._setup_data(workspace)
        from src.benchmark.engine import evaluate_model, load_test_set

        test_data = load_test_set(str(workspace["data"] / "test.jsonl"))
        engine = MockInferenceEngine()
        results = evaluate_model(engine, "fake.gguf", test_data, model_label="mock")

        valid_count = sum(1 for r in results if r["syntax_valid"])
        assert valid_count == len(results), "Mock returns valid Python, all should pass"

    def test_csv_output(self, workspace: dict[str, Path]) -> None:
        self._setup_data(workspace)
        from src.benchmark.engine import evaluate_model, load_test_set

        test_data = load_test_set(str(workspace["data"] / "test.jsonl"))
        engine = MockInferenceEngine()
        results = evaluate_model(engine, "fake.gguf", test_data, model_label="Q8_0")

        csv_path = workspace["reports"] / "benchmark_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(results)
        assert "jaccard" in loaded.columns
        assert "bleu" in loaded.columns
        assert "syntax_valid" in loaded.columns
        assert "model_file_size_mb" in loaded.columns


class TestReporting:
    """Test Epic 4 : génération des rapports."""

    def _create_fake_results(self, reports_dir: Path) -> None:
        results = []
        for fmt in ["Q4_K_M", "Q8_0"]:
            for i in range(5):
                results.append({
                    "model": fmt,
                    "sample_idx": i,
                    "jaccard": 0.7 if fmt == "Q8_0" else 0.5,
                    "bleu": 0.6 if fmt == "Q8_0" else 0.4,
                    "syntax_valid": True,
                    "latency_ms": 100.0 if fmt == "Q8_0" else 80.0,
                    "ms_per_token": 5.0 if fmt == "Q8_0" else 4.0,
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "model_file_size_mb": 4000 if fmt == "Q8_0" else 2000,
                    "rss_mb": 500.0,
                    "peak_rss_mb": 600.0,
                    "rss_delta_mb": 10.0,
                })

        df = pd.DataFrame(results)
        df.to_csv(reports_dir / "benchmark_results.csv", index=False)

    def test_plots(self, workspace: dict[str, Path]) -> None:
        from src.reporting.plots import run

        self._create_fake_results(workspace["reports"])
        run(str(workspace["config_path"]))

        assets_dir = workspace["reports"] / "assets"
        assert (assets_dir / "quality_vs_quantization.png").exists()
        assert (assets_dir / "latency_vs_quantization.png").exists()
        assert (assets_dir / "size_vs_quality.png").exists()

    def test_readme_generation(self, workspace: dict[str, Path]) -> None:
        from src.reporting.readme import run

        self._create_fake_results(workspace["reports"])

        import os
        original_cwd = os.getcwd()
        os.chdir(workspace["tmp"])
        try:
            path = run(str(workspace["config_path"]))
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "TestBenchmark" in content
            assert "Methodology" in content
            assert "Reproduce" in content
            assert "MIT" in content
        finally:
            os.chdir(original_cwd)


class TestTreeSitter:
    """Test extraction multi-langage via tree-sitter."""

    def test_python_extraction(self, workspace: dict[str, Path]) -> None:
        from src.data.extract_treesitter import extract_from_directory
        funcs = extract_from_directory(
            workspace["repos"], [".py"], min_lines=3, max_lines=500
        )
        assert len(funcs) >= 5
        names = [f["name"] for f in funcs]
        assert "add" in names
        assert "fibonacci" in names

    def test_rust_extraction(self, workspace: dict[str, Path]) -> None:
        from src.data.extract_treesitter import extract_from_directory
        funcs = extract_from_directory(
            workspace["repos"], [".rs"], min_lines=3, max_lines=500
        )
        assert len(funcs) >= 2
        names = [f["name"] for f in funcs]
        assert "add" in names
        assert "fibonacci" in names

    def test_multi_lang(self, workspace: dict[str, Path]) -> None:
        from src.data.extract_treesitter import extract_from_directory
        funcs = extract_from_directory(
            workspace["repos"], [".py", ".rs"], min_lines=3, max_lines=500
        )
        py_count = sum(1 for f in funcs if f["file"].endswith(".py"))
        rs_count = sum(1 for f in funcs if f["file"].endswith(".rs"))
        assert py_count >= 5
        assert rs_count >= 2

    def test_unsupported_extension(self, workspace: dict[str, Path]) -> None:
        from src.data.extract_treesitter import extract_from_directory
        funcs = extract_from_directory(
            workspace["repos"], [".xyz"], min_lines=3, max_lines=500
        )
        assert funcs == []


class TestEndToEnd:
    """Test pipeline complet : data → benchmark (mock) → report."""

    def test_full_pipeline_synthetic(self, workspace: dict[str, Path]) -> None:
        import os

        from src.benchmark.engine import evaluate_model, load_test_set
        from src.data.clean import run as clean_run
        from src.data.extract import run as extract_run
        from src.data.instruct import run as instruct_run

        # 1. Data pipeline
        extract_run(str(workspace["config_path"]))
        clean_run(str(workspace["config_path"]))
        instruct_run(str(workspace["config_path"]))

        data_dir = workspace["data"]
        assert (data_dir / "raw_dataset.jsonl").exists()
        assert (data_dir / "train.jsonl").exists()
        assert (data_dir / "test.jsonl").exists()
        assert (data_dir / "train_sharegpt.jsonl").exists()
        assert (data_dir / "data_report.json").exists()

        # 2. Benchmark with mock engine
        test_data = load_test_set(str(data_dir / "test.jsonl"))
        assert len(test_data) > 0

        all_results = []
        for fmt in ["Q4_K_M", "Q8_0"]:
            engine = MockInferenceEngine()
            results = evaluate_model(
                engine, f"model-{fmt}.gguf", test_data,
                max_tokens=64, model_label=fmt,
            )
            all_results.extend(results)

        reports_dir = workspace["reports"]
        df = pd.DataFrame(all_results)
        df.to_csv(reports_dir / "benchmark_results.csv", index=False)

        with open(reports_dir / "benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # 3. Reports
        from src.reporting.plots import run as plot_run
        plot_run(str(workspace["config_path"]))

        assets = reports_dir / "assets"
        assert (assets / "quality_vs_quantization.png").exists()
        assert (assets / "latency_vs_quantization.png").exists()
        assert (assets / "size_vs_quality.png").exists()

        # 4. README
        from src.reporting.readme import run as readme_run
        original_cwd = os.getcwd()
        os.chdir(workspace["tmp"])
        try:
            readme_path = readme_run(str(workspace["config_path"]))
            assert readme_path.exists()
        finally:
            os.chdir(original_cwd)

        # 5. Verify results integrity
        loaded_df = pd.read_csv(reports_dir / "benchmark_results.csv")
        assert len(loaded_df) == len(all_results)
        assert set(loaded_df["model"].unique()) == {"Q4_K_M", "Q8_0"}
        assert all(loaded_df["syntax_valid"])
        assert all(loaded_df["latency_ms"] > 0)
