# CodeQuantBenchmark

Automated pipeline to evaluate the impact of quantization on code-specialized LLMs.

## Architecture

```
├── config.yaml                    # Single configuration file (model, data, quantization, benchmark)
├── main.py                        # Main CLI (typer)
├── synthetic.py                   # Synthetic workspace generator for testing
├── src/
│   ├── data/
│   │   ├── clone.py               # GitHub repo cloning
│   │   ├── extract.py             # Python AST extraction (stdlib)
│   │   ├── extract_treesitter.py  # Multi-language extraction (tree-sitter)
│   │   ├── clean.py               # MD5 deduplication + train/val/test split
│   │   └── instruct.py            # ShareGPT generation (instruction/response)
│   ├── training/
│   │   ├── train.py               # Fine-tuning with auto-detection GPU/CPU (Unsloth or Trainer)
│   │   ├── train_remote.py        # Push data to HF Hub + Kaggle/Colab notebook generation
│   │   └── validate.py            # Post-training validation (gate)
│   ├── quantization/
│   │   └── convert.py             # GGUF conversion + quantization (Q2_K → Q8_0)
│   ├── inference/
│   │   ├── base.py                # Abstract InferenceEngine interface
│   │   └── openai_compat.py       # OpenAI-compatible API implementation (llama.cpp server)
│   ├── benchmark/
│   │   ├── engine.py              # Multi-model benchmark orchestration
│   │   ├── metrics.py             # Jaccard, BLEU, syntax validity, ms/token
│   │   └── memory.py              # RSS/peak memory measurement + file size
│   ├── reporting/
│   │   ├── plots.py               # matplotlib/seaborn charts
│   │   ├── readme.py              # Scientific README generation
│   │   └── release.py             # Git tag + GitHub Release
│   └── utils/
│       ├── config.py              # YAML loading
│       └── logging.py             # Structured JSON logs
├── tests/                         # 47 tests (unit + integration)
├── pyproject.toml
└── LICENSE                        # MIT
```

## Step-by-step guide (for beginners)

The pipeline transforms raw source code into a quantized and benchmarked model. Here is the order to follow:

```
1. COLLECTION        2. PREPARATION       3. TRAINING              4. QUANTIZATION      5. BENCHMARK
   clone                extract              train (local/GPU)       quantize             benchmark
   (GitHub repos)       clean                or                                           report
                         instruct            train-remote                                 readme
                         [= data]            (push data to HF Hub
                                              + Kaggle/Colab notebook)
                                            validate
```

### Quick start

```bash
# 1. Prepare data (clone repos + extraction + cleaning + ShareGPT format)
python main.py data

# 2a. Train locally (CPU = debug, GPU = if available)
python main.py train

# 2b. OR train on Kaggle/Colab GPU (recommended)
python main.py train-remote
# -> Pushes data to HF Hub + generates notebook_training.ipynb
# -> Upload the notebook to Kaggle/Colab, set GPU T4, Run

# 3. Verify the model hasn't forgotten how to code (gate)
python main.py validate

# 4. Convert to GGUF and quantize (Q2_K, Q4_K_M, Q6_K, Q8_0)
python main.py quantize

# 5. Benchmark + report
python main.py benchmark
python main.py report

# Or all at once (local only):
python main.py pipeline
```

## Commands

All commands accept `--config <path>` (default: `config.yaml`).

### Full pipeline

```bash
python main.py pipeline            # All at once: data -> train -> validate -> quantize -> benchmark -> report
```

### Individual steps

| Command | Description |
|---------|-------------|
| `python main.py clone` | Clone GitHub repos (config: `data.github`) |
| `python main.py extract` | Function extraction via AST (Python) / tree-sitter (Rust, Go, JS...) |
| `python main.py clean` | Deduplication + train/val/test split (80/10/10) |
| `python main.py instruct` | Generate instruction/response pairs in ShareGPT format |
| `python main.py data` | The 3 steps above chained together |
| `python main.py train` | Local fine-tuning (auto-detects GPU/Unsloth or CPU/Trainer) |
| `python main.py train-remote` | Push data to HF Hub + generate notebook for Kaggle/Colab |
| `python main.py validate` | Post-training validation — gate before quantization |
| `python main.py quantize` | GGUF conversion + multi-format quantization |
| `python main.py benchmark` | Benchmark: quality, validity, latency, memory |
| `python main.py report` | Generate PNG charts |
| `python main.py readme` | Generate scientific README |
| `python main.py release` | Git tag + GitHub Release with artifacts |

### Remote training (Kaggle / Colab)

GPU training is semi-automated: data is pushed to HuggingFace Hub, and a ready-to-use `.ipynb` notebook is generated locally. You just need to upload it to Kaggle or Colab.

```bash
# 1. Configure in config.yaml:
#    training.hf_dataset: "username/benchmark-training-data"
#    training.hf_repo: "username/model-lora"        (optional, to push LoRA)
#    kaggle.hf_repo_gguf: "username/model-gguf"     (optional, to push GGUF)

# 2. Push data + generate notebook
python main.py train-remote

# 3. Upload notebook_training.ipynb to Kaggle or Colab
#    - Kaggle: New Notebook > File > Import Notebook
#    - Colab:  File > Upload Notebook

# 4. Configure the environment
#    - GPU: T4 (Kaggle: Settings > Accelerator / Colab: Runtime > Change runtime)
#    - HF_TOKEN: Kaggle (Add-ons > Secrets) / Colab (Secrets 🔑)

# 5. Run All

# (Optional) Check the Kaggle kernel status:
python main.py train-remote --status
```

The notebook loads data directly from HuggingFace Hub — no need to attach a Kaggle dataset.

Prerequisites: `huggingface_hub` (already installed), HF token configured (`huggingface-cli login` or `HF_TOKEN` env var).

### Synthetic test (without real data)

```bash
python synthetic.py fakedata                    # Create a workspace with synthetic data
python main.py data --config synthetic_workspace/config.yaml  # Run data pipeline on it
python synthetic.py benchmark synthetic_workspace             # Benchmark with mock engine
python main.py report --config synthetic_workspace/config.yaml # Charts

# Or all at once:
python synthetic.py full                        # Full synthetic pipeline
```

## Supported languages (AST extraction)

Python (stdlib `ast`), Rust, C, C++, Go, JavaScript, TypeScript, Java, Ruby, PHP, Scala, Kotlin, Lua, Haskell, OCaml, Elixir, Bash (via tree-sitter).

Configurable in `config.yaml`:
```yaml
data:
  extensions: [".py", ".rs", ".go", ".js"]
```

## Benchmark metrics

| Metric | Description |
|--------|-------------|
| **Jaccard** | Token-level similarity vs reference |
| **BLEU** | N-gram precision (1-4) with brevity penalty |
| **Syntax Validity** | `py_compile` pass rate on generated code |
| **Latency** | ms/token measured via API |
| **Memory** | RSS, peak RSS, memory delta per inference |
| **File Size** | GGUF file size |

## Tests

```bash
python -m pytest tests/ -v                      # All tests (47)
python -m pytest tests/test_integration.py -v    # Integration only
python -m pytest tests/test_metrics.py -v        # Metrics only
```

## Prerequisites

- Python 3.12+
- micromamba env `llamacpp` (see `pyproject.toml` for dependencies)
- For real benchmarking: a llama.cpp server running with a GGUF model
- For local QLoRA training: `peft` + `bitsandbytes` (GPU required)
- For remote training: `huggingface_hub` + HF token (`huggingface-cli login`)

## License

MIT
