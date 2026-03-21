"""Story 3.2/3.3 : BenchmarkEngine — orchestre l'évaluation."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.benchmark.memory import get_file_size, get_memory, get_memory_of_pid
from src.benchmark.metrics import (
    bleu_score,
    extract_code_blocks,
    jaccard_similarity,
    ms_per_token,
    syntax_validity,
)
from src.inference.base import InferenceEngine
from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("benchmark")


def _start_server(
    server_path: str, model_path: str, n_ctx: int = 2048, port: int = 8080
) -> subprocess.Popen:
    """Start llama-server and wait until it's ready."""
    cmd = [
        server_path,
        "-m", model_path,
        "-c", str(n_ctx),
        "--port", str(port),
    ]
    logger.info(f"Starting llama-server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready (up to 120s for large models on CPU)
    health_url = f"http://localhost:{port}/health"
    for _ in range(120):
        time.sleep(1)
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                logger.info("llama-server is ready")
                return proc
        except requests.ConnectionError:
            pass

    proc.kill()
    raise TimeoutError(f"llama-server failed to start within 120s for {model_path}")


def _stop_server(proc: subprocess.Popen) -> None:
    """Stop llama-server."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    logger.info("llama-server stopped")


def load_test_set(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_model(
    engine: InferenceEngine,
    model_path: str,
    test_data: list[dict[str, Any]],
    max_tokens: int = 512,
    model_label: str = "",
    server_pid: int = 0,
) -> list[dict[str, Any]]:
    file_info = get_file_size(model_path)
    _get_mem = (lambda: get_memory_of_pid(server_pid)) if server_pid else get_memory
    mem_before = _get_mem()
    engine.load(model_path)
    mem_after_load = _get_mem()

    results: list[dict[str, Any]] = []

    for i, sample in enumerate(test_data):
        prompt = sample.get("source", sample.get("prompt", ""))
        expected = sample.get("expected", prompt)

        try:
            mem_pre = _get_mem()
            result = engine.generate(prompt, max_tokens=max_tokens)
            mem_post = _get_mem()
            generated_code = extract_code_blocks(result.text)

            results.append({
                "model": model_label or model_path,
                "sample_idx": i,
                "jaccard": round(jaccard_similarity(expected, generated_code), 4),
                "bleu": bleu_score(expected, generated_code),
                "syntax_valid": syntax_validity(generated_code),
                "latency_ms": round(result.latency_ms, 2),
                "ms_per_token": ms_per_token(
                    result.latency_ms, result.completion_tokens
                ),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "model_file_size_mb": file_info["file_size_mb"],
                "rss_mb": mem_post.rss_mb,
                "peak_rss_mb": mem_post.peak_rss_mb,
                "rss_delta_mb": round(mem_post.rss_mb - mem_pre.rss_mb, 2),
            })
        except Exception as e:
            logger.error(f"Error on sample {i} with {model_label}: {e}")
            results.append({
                "model": model_label or model_path,
                "sample_idx": i,
                "jaccard": 0.0,
                "bleu": 0.0,
                "syntax_valid": False,
                "latency_ms": 0.0,
                "ms_per_token": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "model_file_size_mb": file_info["file_size_mb"],
                "rss_mb": 0.0,
                "peak_rss_mb": 0.0,
                "rss_delta_mb": 0.0,
                "error": str(e),
            })

    engine.unload()

    load_rss_delta = round(mem_after_load.rss_mb - mem_before.rss_mb, 2)
    logger.info(
        f"Model {model_label}: file={file_info['file_size_mb']}MB, "
        f"load_rss_delta={load_rss_delta}MB, peak={mem_after_load.peak_rss_mb}MB"
    )

    return results


def run(config_path: str = "config.yaml") -> Path:
    config = load_config(config_path)
    test_file = get_nested(config, "benchmark", "test_file", default="./data/test.jsonl")
    results_dir = Path(get_nested(config, "benchmark", "results_dir", default="./reports"))
    results_dir.mkdir(parents=True, exist_ok=True)

    quant_dir = Path(get_nested(config, "quantization", "output_dir", default="./models_quantized"))
    formats = get_nested(config, "quantization", "formats", default=[])
    server_path = get_nested(config, "benchmark", "inference", "llama_server_path", default="")
    api_base = get_nested(config, "benchmark", "inference", "api_base", default="http://localhost:8080/v1")
    api_key = get_nested(config, "benchmark", "inference", "api_key", default="not-needed")
    n_ctx = get_nested(config, "benchmark", "inference", "n_ctx", default=2048)
    max_tokens = get_nested(config, "benchmark", "inference", "max_tokens", default=512)
    temperature = get_nested(config, "benchmark", "inference", "temperature", default=0.1)
    n_samples = get_nested(config, "benchmark", "n_samples", default=0)

    test_data = load_test_set(test_file)
    if n_samples > 0:
        test_data = test_data[:n_samples]
    logger.info(f"Loaded {len(test_data)} test samples from {test_file}")

    if not server_path or not Path(server_path).exists():
        raise FileNotFoundError(
            f"llama-server not found: {server_path}. "
            "Set benchmark.inference.llama_server_path in config.yaml"
        )

    from src.inference.openai_compat import OpenAICompatEngine

    all_results: list[dict[str, Any]] = []

    for fmt in formats:
        model_files = list(quant_dir.glob(f"*{fmt}*.gguf"))
        if not model_files:
            logger.warning(f"No GGUF file found for format {fmt} in {quant_dir}")
            continue

        model_path = str(model_files[0])
        logger.info(f"Benchmarking {fmt}: {model_path}")

        # Start llama-server for this model
        proc = _start_server(server_path, model_path, n_ctx=n_ctx)
        try:
            engine = OpenAICompatEngine(
                api_base=api_base, api_key=api_key, temperature=temperature
            )
            results = evaluate_model(
                engine, model_path, test_data, max_tokens=max_tokens,
                model_label=fmt, server_pid=proc.pid,
            )
            all_results.extend(results)
        finally:
            _stop_server(proc)

    output_path = results_dir / "benchmark_results.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    json_path = results_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return output_path
