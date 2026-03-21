"""Story 4.2 : Génération du README scientifique."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("readme")

TEMPLATE = """# {project_name}

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Benchmark](https://img.shields.io/badge/Benchmark-v{version}-blue.svg)

## Hypothesis

Quantizing a code-specialized LLM (fine-tuned via QLoRA on **{domain}** data) \
from FP16 down to aggressive quantization levels (Q2_K) degrades code generation \
quality measurably, but mid-range formats (Q4_K_M, Q6_K) preserve most quality \
while significantly reducing model size and inference latency.

## Methodology

1. **Data Collection**: Extracted Python functions via AST from open-source repositories.
2. **Fine-Tuning**: QLoRA fine-tuning of `{base_model}` on {train_count} training samples.
3. **Quantization**: Converted to GGUF and quantized to: {quant_formats}.
4. **Evaluation**: Measured on {test_count} held-out test samples using:
   - **Jaccard Similarity** — token overlap with reference code
   - **BLEU Score** — n-gram precision with brevity penalty
   - **Syntax Validity** — `py_compile` pass rate
   - **Latency** — ms/token via llama.cpp server
   - **Memory** — RSS footprint during inference

## Results

### Summary Table

{results_table}

### Key Findings

{findings}

### Visualizations

| Quality vs Quantization | Latency vs Quantization | Performance vs Quality |
|:-:|:-:|:-:|
| ![Quality](reports/assets/quality_vs_quantization.png) | ![Latency](reports/assets/latency_vs_quantization.png) | ![Tradeoff](reports/assets/size_vs_quality.png) |

## How to Reproduce

```bash
# 1. Clone and setup
git clone <repo_url>
cd {project_name}

# 2. Run full pipeline
python main.py pipeline --config config.yaml

# Or run steps individually:
python main.py clone      # Clone GitHub repos
python main.py data       # Extract + Clean + Instruct
python main.py train      # Fine-tune with QLoRA
python main.py validate   # Validation gate
python main.py quantize   # Convert to GGUF formats
python main.py benchmark  # Run evaluation
python main.py report     # Generate plots
```

## Models

Fine-tuned adapters and quantized GGUF models are available on HuggingFace:
- Base: [{base_model}](https://huggingface.co/{base_model})
{hf_links}

## Configuration

All parameters are externalized in `config.yaml`. See the file for full documentation.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Generated on {date} by CodeQuantBenchmark pipeline.*
"""


def build_results_table(df: pd.DataFrame) -> str:
    agg = df.groupby("model").agg(
        file_size_mb=("model_file_size_mb", "first"),
        jaccard_mean=("jaccard", "mean"),
        bleu_mean=("bleu", "mean") if "bleu" in df.columns else ("jaccard", "mean"),
        syntax_rate=("syntax_valid", "mean"),
        latency_mean=("latency_ms", "mean"),
        ms_token_mean=("ms_per_token", "mean"),
        rss_mean=("rss_mb", "mean") if "rss_mb" in df.columns else ("latency_ms", "mean"),
    ).reset_index()

    header = "| Format | Size (MB) | Jaccard | BLEU | Syntax % | Latency (ms) | ms/tok | RSS (MB) |"
    sep = "|--------|-----------|---------|------|----------|--------------|--------|----------|"
    rows = [header, sep]

    for _, r in agg.iterrows():
        rows.append(
            f"| {r['model']} | {r['file_size_mb']:.1f} | {r['jaccard_mean']:.3f} | "
            f"{r['bleu_mean']:.3f} | {r['syntax_rate']:.0%} | "
            f"{r['latency_mean']:.1f} | {r['ms_token_mean']:.1f} | {r['rss_mean']:.1f} |"
        )
    return "\n".join(rows)


def build_findings(df: pd.DataFrame) -> str:
    agg = df.groupby("model").agg(
        jaccard_mean=("jaccard", "mean"),
        syntax_rate=("syntax_valid", "mean"),
    ).reset_index()

    if agg.empty:
        return "- No results available yet."

    best = agg.loc[agg["jaccard_mean"].idxmax()]
    worst = agg.loc[agg["jaccard_mean"].idxmin()]

    lines = [
        f"- **Best quality**: {best['model']} (Jaccard={best['jaccard_mean']:.3f}, Syntax={best['syntax_rate']:.0%})",
        f"- **Lowest quality**: {worst['model']} (Jaccard={worst['jaccard_mean']:.3f}, Syntax={worst['syntax_rate']:.0%})",
    ]
    if len(agg) > 1:
        drop = best["jaccard_mean"] - worst["jaccard_mean"]
        lines.append(f"- **Quality drop** from best to worst quantization: {drop:.3f} Jaccard points")

    return "\n".join(lines)


def run(config_path: str = "config.yaml") -> Path:
    config = load_config(config_path)
    project_name = get_nested(config, "project", "name", default="CodeQuantBenchmark")
    domain = get_nested(config, "project", "domain", default="python")
    base_model = get_nested(config, "model", "base_name", default="")
    formats = get_nested(config, "quantization", "formats", default=[])
    hf_repo = get_nested(config, "training", "hf_repo", default="")
    results_dir = Path(get_nested(config, "reporting", "output_dir", default="./reports"))

    csv_path = results_dir / "benchmark_results.csv"
    results_table = "*(Run the benchmark first to populate this table)*"
    findings = "- *(Run the benchmark first)*"
    test_count = 0
    train_count = 0

    data_report = Path(get_nested(config, "data", "output_dir", default="./data")) / "data_report.json"
    if data_report.exists():
        with open(data_report, "r", encoding="utf-8") as f:
            dr = json.load(f)
        train_count = dr.get("train", {}).get("count", 0)
        test_count = dr.get("test", {}).get("count", 0)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        results_table = build_results_table(df)
        findings = build_findings(df)
        if test_count == 0:
            test_count = df["sample_idx"].nunique()

    hf_links = ""
    if hf_repo:
        hf_links = f"- Fine-tuned: [{hf_repo}](https://huggingface.co/{hf_repo})"

    readme_content = TEMPLATE.format(
        project_name=project_name,
        version="1.0.0",
        domain=domain,
        base_model=base_model,
        train_count=train_count,
        test_count=test_count,
        quant_formats=", ".join(formats),
        results_table=results_table,
        findings=findings,
        hf_links=hf_links,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )

    readme_path = Path("README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"README.md generated ({len(readme_content)} chars)")
    return readme_path
