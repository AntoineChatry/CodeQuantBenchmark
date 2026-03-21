"""CLI principal — point d'entrée unique du pipeline."""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="codequant", help="CodeQuantBenchmark Pipeline CLI")
console = Console()


@app.command()
def clone(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Cloner des dépôts GitHub pour la collecte de données."""
    from src.data.clone import run
    count = run(config)
    console.print(f"[green]Cloned {count} repos.[/green]")


@app.command()
def extract(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 1.1 : Extraire les fonctions Python via AST."""
    from src.data.extract import run
    path = run(config)
    console.print(f"[green]Extraction done -> {path}[/green]")


@app.command()
def clean(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 1.2 : Dédupliquer et segmenter train/val/test."""
    from src.data.clean import run
    run(config)
    console.print("[green]Cleaning & splitting done[/green]")


@app.command()
def instruct(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 1.3 : Générer les paires instruction/réponse ShareGPT."""
    from src.data.instruct import run
    run(config)
    console.print("[green]Instruction generation done[/green]")


@app.command()
def data(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Pipeline complet Epic 1 : extract -> clean -> instruct."""
    from src.data.clean import run as clean_run
    from src.data.extract import run as extract_run
    from src.data.instruct import run as instruct_run

    console.print("[bold]Step 1/3: Extraction...[/bold]")
    extract_run(config)
    console.print("[bold]Step 2/3: Cleaning...[/bold]")
    clean_run(config)
    console.print("[bold]Step 3/3: Instruction generation...[/bold]")
    instruct_run(config)
    console.print("[green bold]Data pipeline complete.[/green bold]")


@app.command()
def train(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 2.1-2.2 : Entraîner le modèle (QLoRA si peft disponible)."""
    from src.training.train import run
    run(config)
    console.print("[green]Training done.[/green]")


@app.command()
def validate(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 2.3 : Validation post-training (gate avant quantification)."""
    from src.training.validate import run
    passed = run(config)
    if passed:
        console.print("[green]Validation PASSED — safe to quantize.[/green]")
    else:
        console.print("[red bold]Validation FAILED — do NOT quantize.[/red bold]")
        raise typer.Exit(code=1)


@app.command(name="train-remote")
def train_remote(
    config: str = typer.Option("config.yaml", help="Path to config file"),
    status: bool = typer.Option(False, help="Check status of Kaggle kernel"),
):
    """Push data sur HF Hub + génère un notebook pour Kaggle/Colab.

    \b
    (default)  : Push data + génère notebook_training.ipynb
    --status   : Check le statut du kernel Kaggle (optionnel)
    """
    from src.training.train_remote import (
        check_status as _check_status,
        run as remote_run,
    )
    from src.utils.config import load_config as _load_config

    if status:
        cfg = _load_config(config)
        _check_status(cfg)
        return

    nb_path = remote_run(config)
    console.print(f"\n[green]Notebook generated: {nb_path}[/green]")
    console.print("[yellow]Upload it to Kaggle/Colab, set GPU T4, add HF_TOKEN in Secrets, Run.[/yellow]")


@app.command()
def quantize(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 3.1 : Convertir et quantifier en GGUF (Q2_K -> Q8_0)."""
    from src.quantization.convert import run
    results = run(config)
    for r in results:
        status = "[green]OK[/green]" if r.get("valid") else "[red]FAIL[/red]"
        console.print(f"  {r.get('format', '?')}: {r.get('file_size_mb', '?')} MB {status}")
    console.print(f"[green]Quantization done: {len(results)} formats.[/green]")


@app.command()
def benchmark(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 3.2-3.3 : Lancer le benchmark sur les modèles quantifiés."""
    from src.benchmark.engine import run
    path = run(config)
    console.print(f"[green]Benchmark done -> {path}[/green]")


@app.command()
def report(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 4.1 : Générer les graphiques à partir des résultats."""
    from src.reporting.plots import run
    run(config)
    console.print("[green]Reports generated.[/green]")


@app.command()
def readme(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Epic 4.2 : Générer le README scientifique."""
    from src.reporting.readme import run
    path = run(config)
    console.print(f"[green]README generated -> {path}[/green]")


@app.command()
def release(
    config: str = typer.Option("config.yaml", help="Path to config file"),
    version: str = typer.Option("v1.0.0-benchmark", help="Version tag"),
):
    """Epic 4.3 : Créer un tag Git et une GitHub Release."""
    from src.reporting.release import run
    success = run(config, version=version)
    if success:
        console.print(f"[green]Release {version} created.[/green]")
    else:
        console.print("[red]Release failed.[/red]")
        raise typer.Exit(code=1)


@app.command()
def pipeline(config: str = typer.Option("config.yaml", help="Path to config file")):
    """Pipeline complet : data -> train -> validate -> quantize -> benchmark -> report."""
    from src.benchmark.engine import run as bench_run
    from src.data.clean import run as clean_run
    from src.data.extract import run as extract_run
    from src.data.instruct import run as instruct_run
    from src.quantization.convert import run as quant_run
    from src.reporting.plots import run as report_run
    from src.training.train import run as train_run
    from src.training.validate import run as validate_run

    console.print("[bold cyan]== 1. DATA PIPELINE ==[/bold cyan]")
    extract_run(config)
    clean_run(config)
    instruct_run(config)

    console.print("[bold cyan]== 2. TRAINING ==[/bold cyan]")
    train_run(config)

    console.print("[bold cyan]== 3. VALIDATION (GATE) ==[/bold cyan]")
    gate_passed = validate_run(config)
    if not gate_passed:
        console.print("[red bold]GATE FAILED — pipeline stopped.[/red bold]")
        raise typer.Exit(code=1)

    console.print("[bold cyan]== 4. QUANTIZATION ==[/bold cyan]")
    quant_run(config)

    console.print("[bold cyan]== 5. BENCHMARK ==[/bold cyan]")
    bench_run(config)

    console.print("[bold cyan]== 6. REPORTING ==[/bold cyan]")
    report_run(config)

    console.print("[green bold]Full pipeline complete.[/green bold]")


if __name__ == "__main__":
    app()
