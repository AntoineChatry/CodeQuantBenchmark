"""Story 2.1 & 2.2 : Configuration et entraînement QLoRA/Fine-Tuning.

Script all-in-one qui détecte automatiquement l'environnement :
  - GPU + unsloth  → FastLanguageModel + SFTTrainer + 4bit + packing (Colab/Kaggle)
  - GPU + peft     → AutoModel + Trainer + LoRA (GPU sans unsloth)
  - CPU            → AutoModel + Trainer + fp32 (debug/validation locale)

Supporte le chargement de données depuis des fichiers locaux ou HuggingFace Hub.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("training")

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_runtime() -> dict[str, Any]:
    """Détecte l'environnement d'exécution : GPU, unsloth, peft."""
    info: dict[str, Any] = {
        "has_cuda": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": None,
        "gpu_memory_gb": 0.0,
        "has_unsloth": False,
        "has_peft": False,
        "has_bitsandbytes": False,
        "mode": "cpu",  # cpu | gpu_unsloth | gpu_peft | gpu_full
    }

    if info["has_cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_mem / (1024 ** 3), 1)

    for pkg, key in [("unsloth", "has_unsloth"), ("peft", "has_peft"), ("bitsandbytes", "has_bitsandbytes")]:
        try:
            __import__(pkg)
            info[key] = True
        except ImportError:
            pass

    if info["has_cuda"] and info["has_unsloth"]:
        info["mode"] = "gpu_unsloth"
    elif info["has_cuda"] and info["has_peft"]:
        info["mode"] = "gpu_peft"
    elif info["has_cuda"]:
        info["mode"] = "gpu_full"
    else:
        info["mode"] = "cpu"

    return info


# ---------------------------------------------------------------------------
# Training logger (CSV)
# ---------------------------------------------------------------------------

class TrainingLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self._file = open(log_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "epoch", "loss", "learning_rate"])

    def log(self, step: int, epoch: float, loss: float, lr: float) -> None:
        self._writer.writerow([step, round(epoch, 4), round(loss, 6), lr])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class LogCallback:
    """Callback compatible transformers pour logger les métriques."""

    def __init__(self, training_logger: TrainingLogger):
        self.training_logger = training_logger

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        if logs and "loss" in logs:
            self.training_logger.log(
                step=state.global_step,
                epoch=state.epoch or 0.0,
                loss=logs["loss"],
                lr=logs.get("learning_rate", 0.0),
            )


# ---------------------------------------------------------------------------
# GPU path: Unsloth + SFTTrainer (like the reference notebook)
# ---------------------------------------------------------------------------

def _run_unsloth(config: dict[str, Any], runtime: dict[str, Any]) -> Path:
    """Entraînement via Unsloth + SFTTrainer — reproduit le notebook de référence."""
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    from unsloth.chat_templates import get_chat_template  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    from datasets import load_dataset as hf_load_dataset  # type: ignore[import-not-found]

    model_name = get_nested(config, "model", "base_name", default="")
    max_seq_length = get_nested(config, "training", "max_length", default=2048)
    training_cfg = get_nested(config, "training", default={})
    output_dir = Path(training_cfg.get("output_dir", "./models_trained"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    logger.info(f"[unsloth] Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # --- LoRA (aligned with notebook) ---
    lora_r = get_nested(config, "model", "lora_r", default=32)
    lora_alpha = get_nested(config, "model", "lora_alpha", default=32)
    target_modules = get_nested(
        config, "model", "target_modules",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.get("seed", 3407),
        use_rslora=False,
        loftq_config=None,
    )
    logger.info(f"[unsloth] LoRA: r={lora_r}, alpha={lora_alpha}, targets={target_modules}")

    # --- Chat template (ChatML, ShareGPT mapping — aligned with Kaggle notebook) ---
    chat_tmpl = get_nested(config, "training", "chat_template", default="chatml")
    tmpl_kwargs: dict[str, Any] = {
        "chat_template": chat_tmpl,
        "mapping": {"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    }
    if chat_tmpl == "chatml":
        tmpl_kwargs["map_eos_token"] = True
    tokenizer = get_chat_template(tokenizer, **tmpl_kwargs)

    def formatting_prompts_func(examples: dict[str, Any]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    # --- Dataset ---
    data_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    hf_dataset_name = get_nested(config, "training", "hf_dataset", default="")

    if hf_dataset_name:
        logger.info(f"[unsloth] Loading dataset from HF Hub: {hf_dataset_name}")
        train_ds = hf_load_dataset(hf_dataset_name, split="train")
        val_ds = hf_load_dataset(hf_dataset_name, split="validation") if _hf_split_exists(hf_dataset_name, "validation") else None
    else:
        train_file, val_file = _resolve_data_files(data_dir)
        logger.info(f"[unsloth] Loading train: {train_file}")
        train_ds = hf_load_dataset("json", data_files=str(train_file), split="train")
        if val_file and val_file.exists():
            logger.info(f"[unsloth] Loading val: {val_file}")
            val_ds = hf_load_dataset("json", data_files=str(val_file), split="train")
        else:
            val_ds = None

    train_ds = train_ds.map(formatting_prompts_func, batched=True)
    if val_ds is not None:
        val_ds = val_ds.map(formatting_prompts_func, batched=True)

    num_examples = len(train_ds)
    epochs = training_cfg.get("epochs", 3)
    if epochs == "auto":
        if num_examples < 3000:
            epochs = 3
        elif num_examples < 10000:
            epochs = 2
        else:
            epochs = 1
    logger.info(f"[unsloth] {num_examples} examples, {epochs} epochs")

    # --- Training logger ---
    training_logger = TrainingLogger(str(output_dir / "training_log.csv"))

    # --- SFTTrainer (aligned with Kaggle notebook) ---
    max_steps = training_cfg.get("max_steps", -1)
    sft_config_kwargs: dict[str, Any] = {
        "per_device_train_batch_size": training_cfg.get("batch_size", 2),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation", 4),
        "warmup_steps": training_cfg.get("warmup_steps", 5),
        "learning_rate": training_cfg.get("learning_rate", 2e-5),
        "logging_steps": training_cfg.get("logging_steps", 25),
        "optim": "adamw_8bit",
        "weight_decay": training_cfg.get("weight_decay", 0.001),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "linear"),
        "seed": training_cfg.get("seed", 3407),
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "report_to": "none",
    }
    if max_steps > 0:
        sft_config_kwargs["max_steps"] = max_steps
    else:
        sft_config_kwargs["num_train_epochs"] = epochs

    packing = training_cfg.get("packing", False)

    sft_kwargs: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": train_ds,
        "dataset_text_field": "text",
        "max_seq_length": max_seq_length,
        "packing": packing,
        "args": SFTConfig(**sft_config_kwargs),
    }
    if val_ds is not None:
        sft_kwargs["eval_dataset"] = val_ds
        sft_kwargs["args"].eval_strategy = "epoch"

    trainer = SFTTrainer(**sft_kwargs)
    trainer.add_callback(LogCallback(training_logger))

    logger.info("[unsloth] Starting training...")
    trainer.train()
    training_logger.close()

    # --- Save ---
    logger.info("[unsloth] Saving LoRA adapter...")
    model.save_pretrained(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    hf_repo = get_nested(config, "training", "hf_repo", default="")
    if hf_repo:
        hf_token = get_nested(config, "training", "hf_token", default="")
        logger.info(f"[unsloth] Pushing to HF Hub: {hf_repo}")
        push_kwargs: dict[str, Any] = {}
        if hf_token:
            push_kwargs["token"] = hf_token
        model.push_to_hub(hf_repo, **push_kwargs)
        tokenizer.push_to_hub(hf_repo, **push_kwargs)

    _cleanup_cache(output_dir)
    logger.info(f"[unsloth] Training complete. Output: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CPU / GPU-without-unsloth path: transformers Trainer
# ---------------------------------------------------------------------------

class CodeDataset(Dataset):  # type: ignore[type-arg]
    """Dataset qui charge du JSONL ShareGPT et tokenize via apply_chat_template ou fallback."""

    def __init__(self, file_path: str, tokenizer: Any, max_length: int = 2048):
        self.samples: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if "conversations" in sample:
            try:
                # Utilise apply_chat_template si le tokenizer le supporte (Mistral, etc.)
                text = self.tokenizer.apply_chat_template(
                    sample["conversations"], tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                # Fallback: formatage ShareGPT basique
                parts = []
                for turn in sample["conversations"]:
                    role = turn.get("from", "")
                    value = turn.get("value", "")
                    parts.append(f"<|{role}|>\n{value}")
                text = "\n".join(parts)
        else:
            text = sample.get("source", sample.get("text", ""))

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }


def _setup_model_cpu_or_gpu(
    config: dict[str, Any], runtime: dict[str, Any],
) -> tuple[Any, Any]:
    """Charge modèle + tokenizer pour le mode CPU ou GPU sans unsloth."""
    model_name = get_nested(config, "model", "base_name", default="")
    if not model_name:
        raise ValueError("model.base_name must be set in config.yaml")

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model: {model_name} (mode={runtime['mode']})")
    load_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if runtime["has_cuda"]:
        load_kwargs["torch_dtype"] = torch.float16
        if runtime["has_bitsandbytes"]:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
    else:
        load_kwargs["torch_dtype"] = torch.float32
        # Ignore quantization_config if bitsandbytes is not available
        try:
            from transformers import AutoConfig
            auto_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            if getattr(auto_cfg, "quantization_config", None):
                logger.warning(
                    "Model has quantization_config but running on CPU. Loading in fp32."
                )
                load_kwargs["quantization_config"] = None
                load_kwargs["ignore_mismatched_sizes"] = True
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # LoRA via peft (if available)
    if runtime["has_peft"]:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

        lora_r = get_nested(config, "model", "lora_r", default=32)
        lora_alpha = get_nested(config, "model", "lora_alpha", default=32)
        target_modules = get_nested(
            config, "model", "target_modules",
            default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA applied: r={lora_r}, alpha={lora_alpha}, targets={target_modules}")
        model.print_trainable_parameters()
    else:
        logger.warning(
            "peft not available — training will update ALL parameters. "
            "Install peft for LoRA support."
        )

    return model, tokenizer


def _build_training_args(
    config: dict[str, Any], output_dir: str, runtime: dict[str, Any],
) -> TrainingArguments:
    training_cfg = get_nested(config, "training", default={})
    use_fp16 = runtime["has_cuda"] and not runtime.get("has_bf16", False)
    use_bf16 = runtime["has_cuda"] and runtime.get("has_bf16", False)

    optim = "adamw_torch"
    if runtime["has_cuda"] and runtime["has_bitsandbytes"]:
        optim = "adamw_8bit"

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("epochs", 3),
        per_device_train_batch_size=training_cfg.get("batch_size", 2),
        per_device_eval_batch_size=training_cfg.get("batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation", 4),
        learning_rate=training_cfg.get("learning_rate", 2e-4),
        warmup_steps=training_cfg.get("warmup_steps", 10),
        logging_steps=training_cfg.get("logging_steps", 10),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        seed=training_cfg.get("seed", 3407),
        optim=optim,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
    )


def _run_trainer(config: dict[str, Any], runtime: dict[str, Any]) -> Path:
    """Entraînement via transformers Trainer (CPU ou GPU sans unsloth)."""
    data_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    output_dir = Path(get_nested(config, "training", "output_dir", default="./models_trained"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _setup_model_cpu_or_gpu(config, runtime)
    max_length = get_nested(config, "training", "max_length", default=2048)

    train_file, val_file = _resolve_data_files(data_dir)

    logger.info(f"Loading train: {train_file}")
    train_dataset = CodeDataset(str(train_file), tokenizer, max_length)
    val_dataset = None
    if val_file and val_file.exists():
        logger.info(f"Loading val: {val_file}")
        val_dataset = CodeDataset(str(val_file), tokenizer, max_length)

    training_args = _build_training_args(config, str(output_dir), runtime)
    if val_dataset is None:
        training_args.eval_strategy = "no"

    training_logger = TrainingLogger(str(output_dir / "training_log.csv"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.add_callback(LogCallback(training_logger))

    logger.info(f"Starting training (mode={runtime['mode']})...")
    trainer.train()
    training_logger.close()

    logger.info("Saving model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    hf_repo = get_nested(config, "training", "hf_repo", default="")
    if hf_repo:
        logger.info(f"Pushing to HuggingFace Hub: {hf_repo}")
        trainer.push_to_hub(repo_id=hf_repo)

    _cleanup_cache(output_dir)
    logger.info(f"Training complete. Output: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_data_files(data_dir: Path) -> tuple[Path, Path | None]:
    """Trouve les fichiers train/val, préférant le format ShareGPT."""
    train_file = data_dir / "train_sharegpt.jsonl"
    val_file = data_dir / "validation_sharegpt.jsonl"
    if not train_file.exists():
        train_file = data_dir / "train.jsonl"
    if not val_file.exists():
        val_file = data_dir / "validation.jsonl"
    return train_file, val_file if val_file.exists() else None


def _hf_split_exists(dataset_name: str, split: str) -> bool:
    """Vérifie si un split existe dans un dataset HF Hub."""
    try:
        from datasets import get_dataset_split_names  # type: ignore[import-not-found]
        return split in get_dataset_split_names(dataset_name)
    except Exception:
        return False


def _cleanup_cache(output_dir: Path) -> None:
    for cache_dir in output_dir.glob("checkpoint-*"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            logger.info(f"Cleaned checkpoint cache: {cache_dir}")

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
        cache_mb = cache_size / (1024 * 1024)
        if cache_mb > 5000:
            logger.warning(f"HF cache is {cache_mb:.0f}MB. Consider cleaning: {hf_cache}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: str = "config.yaml") -> Path:
    config = load_config(config_path)

    runtime = detect_runtime()
    logger.info(
        f"Runtime: mode={runtime['mode']}, device={runtime['device']}, "
        f"gpu={runtime['gpu_name'] or 'none'}, "
        f"vram={runtime['gpu_memory_gb']}GB, "
        f"unsloth={runtime['has_unsloth']}, peft={runtime['has_peft']}, "
        f"bnb={runtime['has_bitsandbytes']}"
    )

    if runtime["mode"] == "gpu_unsloth":
        return _run_unsloth(config, runtime)
    else:
        return _run_trainer(config, runtime)
