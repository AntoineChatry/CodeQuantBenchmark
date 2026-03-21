"""Story 4.1 : Génération de rapports visuels."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("reporting")


def load_results(results_path: str) -> pd.DataFrame:
    return pd.read_csv(results_path)


def plot_quality_vs_quantization(df: pd.DataFrame, output_dir: Path) -> None:
    agg = df.groupby("model").agg(
        jaccard_mean=("jaccard", "mean"),
        syntax_rate=("syntax_valid", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(agg))
    ax1.bar(x, agg["jaccard_mean"], width=0.4, label="Jaccard (mean)", align="center")
    ax1.set_ylabel("Jaccard Similarity")
    ax1.set_xlabel("Quantization Format")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(agg["model"], rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(list(x), agg["syntax_rate"], "r-o", label="Syntax Validity Rate")
    ax2.set_ylabel("Syntax Validity Rate")
    ax2.set_ylim(0, 1.05)

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95))
    fig.suptitle("Code Quality vs Quantization Level")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_vs_quantization.png", dpi=150)
    plt.close(fig)
    logger.info("Saved quality_vs_quantization.png")


def plot_latency_vs_quantization(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="latency_ms", ax=ax)
    ax.set_title("Latency Distribution by Quantization Level")
    ax.set_xlabel("Quantization Format")
    ax.set_ylabel("Latency (ms)")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_vs_quantization.png", dpi=150)
    plt.close(fig)
    logger.info("Saved latency_vs_quantization.png")


def plot_size_vs_quality(df: pd.DataFrame, output_dir: Path) -> None:
    agg = df.groupby("model").agg(
        jaccard_mean=("jaccard", "mean"),
        ms_per_token_mean=("ms_per_token", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        agg["ms_per_token_mean"],
        agg["jaccard_mean"],
        s=100,
        c=range(len(agg)),
        cmap="viridis",
    )
    for i, row in agg.iterrows():
        ax.annotate(row["model"], (row["ms_per_token_mean"], row["jaccard_mean"]),
                     textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("ms/token (avg)")
    ax.set_ylabel("Jaccard Similarity (avg)")
    ax.set_title("Performance vs Quality Tradeoff")
    fig.tight_layout()
    fig.savefig(output_dir / "size_vs_quality.png", dpi=150)
    plt.close(fig)
    logger.info("Saved size_vs_quality.png")


def run(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    results_dir = Path(get_nested(config, "reporting", "output_dir", default="./reports"))
    results_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = results_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "benchmark_results.csv"
    if not csv_path.exists():
        logger.error(f"Results not found: {csv_path}. Run benchmark first.")
        return

    df = load_results(str(csv_path))
    plot_quality_vs_quantization(df, assets_dir)
    plot_latency_vs_quantization(df, assets_dir)
    plot_size_vs_quality(df, assets_dir)
    logger.info("All plots generated.")
