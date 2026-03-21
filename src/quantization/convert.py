"""Story 3.1 : Pipeline de conversion et quantification GGUF.

Utilise llama.cpp CLI (llama-quantize) via subprocess pour quantifier,
et le package gguf Python pour vérifier l'intégrité des fichiers générés.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from gguf import GGUFReader

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("quantization")


def _llama_cpp_dir(config: dict[str, Any]) -> Path | None:
    d = get_nested(config, "quantization", "llama_cpp_dir", default="")
    if d and Path(d).is_dir():
        return Path(d)
    return None


def find_llama_quantize(config: dict[str, Any]) -> str:
    configured = get_nested(config, "quantization", "llama_quantize_path", default="")
    if configured and Path(configured).exists():
        return configured

    # Auto-detect from llama_cpp_dir build output
    base = _llama_cpp_dir(config)
    if base:
        for sub in [
            "build/bin/Release/llama-quantize.exe",
            "build/bin/llama-quantize.exe",
            "build/bin/llama-quantize",
            "build/Release/llama-quantize.exe",
        ]:
            p = base / sub
            if p.exists():
                return str(p)

    for candidate in ["llama-quantize", "llama-quantize.exe"]:
        found = shutil.which(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "llama-quantize not found. Build llama.cpp first:\n"
        "  cd F:\\PythonWorkspace\\QuantizeLearn\\llama.cpp\n"
        "  cmake -B build && cmake --build build --config Release -t llama-quantize"
    )


def find_convert_script(config: dict[str, Any]) -> str:
    configured = get_nested(config, "quantization", "convert_script_path", default="")
    if configured and Path(configured).exists():
        return configured

    # Auto-detect from llama_cpp_dir
    base = _llama_cpp_dir(config)
    if base:
        script = base / "convert_hf_to_gguf.py"
        if script.exists():
            return str(script)

    return ""


def convert_to_f16_gguf(
    input_dir: str, output_path: str, convert_script: str
) -> bool:
    if not convert_script:
        logger.error(
            "No convert script configured. Set 'quantization.convert_script_path' "
            "in config.yaml (e.g. path to convert_hf_to_gguf.py from llama.cpp)."
        )
        return False

    cmd = ["python", convert_script, input_dir, "--outfile", output_path, "--outtype", "f16"]
    logger.info(f"Converting to F16 GGUF: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        return False

    logger.info(f"F16 GGUF created: {output_path}")
    return True


def quantize_gguf(
    llama_quantize: str, input_gguf: str, output_gguf: str, quant_type: str
) -> bool:
    cmd = [llama_quantize, input_gguf, output_gguf, quant_type]
    logger.info(f"Quantizing {quant_type}: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Quantization {quant_type} failed: {result.stderr}")
        return False

    logger.info(f"Quantized {quant_type} -> {output_gguf}")
    return True


def verify_gguf(path: str) -> dict[str, Any]:
    try:
        reader = GGUFReader(path)
        file_size = os.path.getsize(path)

        arch_field = reader.get_field("general.architecture")
        architecture = ""
        if arch_field is not None:
            parts = arch_field.parts
            if len(parts) > arch_field.data[0]:
                raw = parts[arch_field.data[0]]
                architecture = raw.tobytes().decode("utf-8", errors="replace")

        name_field = reader.get_field("general.name")
        model_name = ""
        if name_field is not None:
            parts = name_field.parts
            if len(parts) > name_field.data[0]:
                raw = parts[name_field.data[0]]
                model_name = raw.tobytes().decode("utf-8", errors="replace")

        return {
            "valid": True,
            "path": path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "architecture": architecture,
            "model_name": model_name,
            "alignment": reader.alignment,
        }
    except Exception as e:
        logger.error(f"GGUF verification failed for {path}: {e}")
        return {"valid": False, "path": path, "error": str(e)}


def run(config_path: str = "config.yaml") -> list[dict[str, Any]]:
    config = load_config(config_path)
    input_dir = Path(get_nested(config, "quantization", "input_dir", default="./models_merged"))
    output_dir = Path(get_nested(config, "quantization", "output_dir", default="./models_quantized"))
    formats = get_nested(config, "quantization", "formats", default=["Q4_K_M", "Q8_0"])

    output_dir.mkdir(parents=True, exist_ok=True)

    f16_gguf = output_dir / "model-f16.gguf"
    if not f16_gguf.exists():
        convert_script = find_convert_script(config)
        if not convert_to_f16_gguf(str(input_dir), str(f16_gguf), convert_script):
            return []
    else:
        logger.info(f"F16 GGUF already exists: {f16_gguf}")

    llama_quantize = find_llama_quantize(config)
    results: list[dict[str, Any]] = []

    for fmt in formats:
        out_path = output_dir / f"model-{fmt}.gguf"

        if out_path.exists():
            logger.info(f"Already exists, skipping: {out_path}")
        else:
            success = quantize_gguf(llama_quantize, str(f16_gguf), str(out_path), fmt)
            if not success:
                results.append({"format": fmt, "valid": False, "error": "quantization_failed"})
                continue

        info = verify_gguf(str(out_path))
        info["format"] = fmt
        results.append(info)
        logger.info(
            f"  {fmt}: {info.get('file_size_mb', '?')} MB, valid={info.get('valid')}"
        )

    return results
