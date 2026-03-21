"""Génère un environnement synthétique complet pour tester le pipeline sans données réelles.

Usage:
    python synthetic.py [--output ./synthetic_workspace]

Crée un dossier avec :
  - repos/ contenant du code Python et Rust synthétique
  - des faux fichiers GGUF
  - un config.yaml pointant vers ces données
  - un moteur d'inférence mock intégré

Ensuite on peut lancer les commandes normales du pipeline dessus.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml
from rich.console import Console

from src.benchmark.engine import evaluate_model, load_test_set
from src.inference.base import InferenceEngine, InferenceResult

app = typer.Typer(help="Synthetic workspace generator for testing")
console = Console()

PYTHON_CODE = '''
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

RUST_CODE = '''
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
    def __init__(self) -> None:
        self.loaded = False
        self.model_path = ""

    def load(self, model_path: str) -> None:
        self.model_path = model_path
        self.loaded = True

    def generate(self, prompt: str, max_tokens: int = 512) -> InferenceResult:
        return InferenceResult(
            text="```python\ndef example():\n    return 42\n```\n",
            prompt_tokens=len(prompt.split()),
            completion_tokens=15,
            latency_ms=50.0,
        )

    def unload(self) -> None:
        self.loaded = False

    def get_metrics(self) -> dict[str, float]:
        return {"total_requests": 0, "total_latency_ms": 0.0, "avg_latency_ms": 0.0}


def _build_config(output: Path) -> dict[str, Any]:
    return {
        "project": {"name": "SyntheticBenchmark", "domain": "python_synthetic"},
        "data": {
            "source_dir": str(output / "repos"),
            "output_dir": str(output / "data"),
            "extensions": [".py", ".rs"],
            "split_ratios": [0.8, 0.1, 0.1],
            "min_lines": 3,
            "max_lines": 500,
            "instruction_templates": [
                "Explain the following function:",
                "Refactor this function:",
            ],
        },
        "model": {"base_name": "synthetic-test-model"},
        "training": {"output_dir": str(output / "models_trained")},
        "quantization": {
            "output_dir": str(output / "models_quantized"),
            "formats": ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0"],
        },
        "benchmark": {
            "test_file": str(output / "data" / "test.jsonl"),
            "results_dir": str(output / "reports"),
            "metrics": ["syntax_validity", "jaccard_similarity", "bleu", "latency"],
            "inference": {
                "api_base": "http://localhost:8080/v1",
                "api_key": "not-needed",
                "max_tokens": 64,
                "temperature": 0.1,
            },
        },
        "reporting": {"output_dir": str(output / "reports")},
    }


@app.command()
def fakedata(
    output: str = typer.Argument("./synthetic_workspace", help="Output directory"),
) -> None:
    """Crée un workspace synthétique avec du code Python + Rust."""
    out = Path(output)
    if out.exists():
        shutil.rmtree(out)

    repos = out / "repos" / "fake_project"
    repos.mkdir(parents=True)
    (repos / "utils.py").write_text(PYTHON_CODE, encoding="utf-8")
    (repos / "math_lib.rs").write_text(RUST_CODE, encoding="utf-8")

    models = out / "models_quantized"
    models.mkdir(parents=True)
    for fmt in ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0"]:
        size = {"Q2_K": 512, "Q4_K_M": 1024, "Q6_K": 2048, "Q8_0": 4096}[fmt]
        (models / f"model-{fmt}.gguf").write_bytes(b"\x00" * size)

    config = _build_config(out)
    config_path = out / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[green]Workspace created: {out}[/green]")
    console.print(f"  repos:  {repos}")
    console.print(f"  config: {config_path}")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print(f"  python main.py data     --config {config_path}")
    console.print(f"  python synthetic.py benchmark {out}")
    console.print(f"  python main.py report   --config {config_path}")
    console.print(f"  python main.py readme   --config {config_path}")


@app.command()
def benchmark(
    workspace: str = typer.Argument("./synthetic_workspace", help="Workspace directory"),
) -> None:
    """Lance le benchmark avec le moteur mock sur les données synthétiques."""
    ws = Path(workspace)
    config_path = ws / "config.yaml"
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}. Run 'fakedata' first.[/red]")
        raise typer.Exit(code=1)

    test_file = ws / "data" / "test.jsonl"
    if not test_file.exists():
        console.print("[red]Test data not found. Run 'python main.py data --config ...' first.[/red]")
        raise typer.Exit(code=1)

    test_data = load_test_set(str(test_file))
    console.print(f"Loaded {len(test_data)} test samples")

    formats = ["Q2_K", "Q4_K_M", "Q6_K", "Q8_0"]
    all_results: list[dict[str, Any]] = []

    for fmt in formats:
        model_path = str(ws / "models_quantized" / f"model-{fmt}.gguf")
        console.print(f"  Benchmarking [cyan]{fmt}[/cyan]...")
        engine = MockInferenceEngine()
        results = evaluate_model(
            engine, model_path, test_data, max_tokens=64, model_label=fmt,
        )
        all_results.extend(results)

    reports = ws / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    csv_path = reports / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    json_path = reports / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Results saved:[/green]")
    console.print(f"  CSV:  {csv_path}")
    console.print(f"  JSON: {json_path}")

    console.print(f"\n[bold]Summary:[/bold]")
    summary = df.groupby("model").agg(
        jaccard=("jaccard", "mean"),
        bleu=("bleu", "mean"),
        syntax=("syntax_valid", "mean"),
        latency=("latency_ms", "mean"),
    )
    console.print(summary.to_string())


@app.command()
def full(
    output: str = typer.Argument("./synthetic_workspace", help="Output directory"),
) -> None:
    """Pipeline synthetique complet : fakedata > data > benchmark mock > report > readme."""
    import os

    out = Path(output).resolve()
    config_path = out / "config.yaml"

    console.print("[bold cyan]== 1. FAKEDATA ==[/bold cyan]")
    fakedata(output)

    console.print("\n[bold cyan]== 2. DATA PIPELINE ==[/bold cyan]")
    from src.data.clean import run as clean_run
    from src.data.extract import run as extract_run
    from src.data.instruct import run as instruct_run
    extract_run(str(config_path))
    clean_run(str(config_path))
    instruct_run(str(config_path))

    console.print("\n[bold cyan]== 3. BENCHMARK (mock) ==[/bold cyan]")
    benchmark(output)

    console.print("\n[bold cyan]== 4. REPORTS ==[/bold cyan]")
    from src.reporting.plots import run as plot_run
    plot_run(str(config_path))

    console.print("\n[bold cyan]== 5. README ==[/bold cyan]")
    from src.reporting.readme import run as readme_run
    original_cwd = os.getcwd()
    os.chdir(out)
    try:
        readme_run(str(config_path))
    finally:
        os.chdir(original_cwd)

    console.print(f"\n[green bold]Synthetic pipeline complete![/green bold]")
    console.print(f"  Workspace: {out}")
    console.print(f"  Reports:   {out / 'reports'}")
    console.print(f"  Plots:     {out / 'reports' / 'assets'}")
    console.print(f"  README:    {out / 'README.md'}")


if __name__ == "__main__":
    app()
