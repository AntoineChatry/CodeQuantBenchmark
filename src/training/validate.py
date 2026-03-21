"""Story 2.3 : Validation Post-Training.

Teste le modèle fine-tuné sur N exemples de validation pour détecter
le catastrophic forgetting avant de passer à la quantification.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.benchmark.metrics import extract_code_blocks, syntax_validity
from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("validation")


@dataclass
class ValidationResult:
    sample_idx: int
    prompt: str
    generated: str
    has_code: bool
    syntax_valid: bool


def _pull_from_hub(hf_repo: str, local_path: Path, config: dict[str, Any]) -> None:
    """Telecharge un modele LoRA depuis HF Hub vers un dossier local."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    hf_token = get_nested(config, "training", "hf_token", default="") or None
    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {hf_repo} -> {local_path}")
    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(local_path),
        token=hf_token,
    )
    logger.info(f"Model downloaded to {local_path}")

    # Fix unsloth tokenizer_class that transformers doesn't recognize
    tok_config = local_path / "tokenizer_config.json"
    if tok_config.exists():
        import json as _json

        data = _json.loads(tok_config.read_text(encoding="utf-8"))
        if data.get("tokenizer_class") in ("TokenizersBackend",):
            data.pop("tokenizer_class")
            tok_config.write_text(
                _json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Patched tokenizer_config.json (removed unsloth tokenizer_class)")


def load_validation_samples(
    val_path: str, n_samples: int = 10
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_prompt(sample: dict[str, Any]) -> str:
    if "conversations" in sample:
        for turn in sample["conversations"]:
            if turn.get("from") == "human":
                return turn["value"]
    return sample.get("source", sample.get("prompt", sample.get("text", "")))


def validate_model(
    model_path: str,
    val_path: str,
    n_samples: int = 5,
    max_new_tokens: int = 64,
) -> tuple[list[ValidationResult], bool]:
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    samples = load_validation_samples(val_path, n_samples)
    logger.info(f"Validating on {len(samples)} samples (max_new_tokens={max_new_tokens})")

    results: list[ValidationResult] = []

    for i, sample in enumerate(samples):
        prompt = extract_prompt(sample)

        # Apply chat template so the model sees the same format as training
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)

        logger.info(f"  Sample {i}: generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        code = extract_code_blocks(generated)
        has_code = "```" in generated or any(
            kw in generated for kw in ["def ", "class ", "import ", "return "]
        )
        is_valid = syntax_validity(code) if has_code else False

        results.append(ValidationResult(
            sample_idx=i,
            prompt=prompt[:100],
            generated=generated[:200],
            has_code=has_code,
            syntax_valid=is_valid,
        ))
        logger.info(f"  Sample {i}: has_code={has_code}, syntax_valid={is_valid}")

    code_rate = sum(1 for r in results if r.has_code) / len(results) if results else 0
    valid_rate = sum(1 for r in results if r.syntax_valid) / len(results) if results else 0
    logger.info(f"Code presence rate: {code_rate:.0%}, Syntax valid rate: {valid_rate:.0%}")

    gate_threshold = 0.5
    gate_passed = code_rate >= gate_threshold

    if not gate_passed:
        logger.warning(
            f"GATE FAILED: code presence rate {code_rate:.0%} < {gate_threshold:.0%}. "
            "Possible catastrophic forgetting. Do NOT proceed to quantification."
        )
    else:
        logger.info("GATE PASSED: model produces code. Safe to proceed.")

    return results, gate_passed


def run(config_path: str = "config.yaml") -> bool:
    config = load_config(config_path)
    data_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    model_dir = Path(get_nested(config, "training", "output_dir", default="./models_trained"))
    model_path = model_dir / "final"
    n_samples = get_nested(config, "validation", "n_samples", default=10)

    val_file = data_dir / "validation_sharegpt.jsonl"
    if not val_file.exists():
        val_file = data_dir / "validation.jsonl"

    if not model_path.exists():
        # Prefer merged model (full fp16) over LoRA adapter for validation
        hf_repo = (
            get_nested(config, "training", "hf_repo_merged", default="")
            or get_nested(config, "training", "hf_repo", default="")
        )
        if hf_repo:
            logger.info(f"Local model not found. Pulling from HF Hub: {hf_repo}")
            _pull_from_hub(hf_repo, model_path, config)
        else:
            logger.error(
                f"Trained model not found: {model_path}. "
                "Set training.hf_repo_merged in config.yaml to pull from HF Hub."
            )
            return False
    if not val_file.exists():
        logger.error(f"Validation file not found: {val_file}")
        return False

    results, gate_passed = validate_model(
        str(model_path), str(val_file), n_samples=n_samples
    )

    report_dir = Path(get_nested(config, "benchmark", "results_dir", default="./reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "gate_passed": gate_passed,
                "n_samples": len(results),
                "results": [dataclasses.asdict(r) for r in results],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Validation report saved to {report_path}")
    return gate_passed
