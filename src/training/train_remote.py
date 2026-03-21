"""Pipeline d'entraînement distant : push data sur HF Hub + génère un notebook.

Approche simple et fiable :
  1. Push les fichiers JSONL sur HuggingFace Hub comme dataset
  2. Génère un notebook .ipynb auto-contenu (charge data depuis HF Hub)
  3. L'utilisateur upload le notebook sur Kaggle/Colab, configure le GPU, Run

Usage :
  python main.py train-remote            # Push data + génère notebook
  python main.py train-remote --status   # Check statut Kaggle (si kernel existe)
"""

import json
import shutil
from pathlib import Path
from typing import Any

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("train_remote")


# ---------------------------------------------------------------------------
# Push dataset to HuggingFace Hub
# ---------------------------------------------------------------------------

def push_dataset_to_hf(config: dict[str, Any]) -> str:
    """Push les fichiers JSONL sur HF Hub. Retourne le dataset ID."""
    from huggingface_hub import HfApi, create_repo  # type: ignore[import-untyped]

    data_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    hf_dataset = get_nested(config, "training", "hf_dataset", default="")

    if not hf_dataset:
        raise ValueError(
            "training.hf_dataset must be set in config.yaml "
            "(e.g. 'antoinech7/benchmark-training-data')"
        )

    # Fichiers à uploader
    files_to_upload = []
    for name in ["train_sharegpt.jsonl", "train.jsonl", "validation_sharegpt.jsonl", "validation.jsonl"]:
        src = data_dir / name
        if src.exists():
            files_to_upload.append(src)

    for jsonl in data_dir.glob("*merged*.jsonl"):
        files_to_upload.append(jsonl)

    if not files_to_upload:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    # Créer le repo si nécessaire
    api = HfApi()
    try:
        create_repo(hf_dataset, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Upload
    logger.info(f"Pushing {len(files_to_upload)} files to HF Hub: {hf_dataset}")
    for f in files_to_upload:
        logger.info(f"  Uploading {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=hf_dataset,
            repo_type="dataset",
        )

    logger.info(f"Dataset ready: https://huggingface.co/datasets/{hf_dataset}")
    return hf_dataset


# ---------------------------------------------------------------------------
# Generate notebook
# ---------------------------------------------------------------------------

def generate_notebook(config: dict[str, Any]) -> Path:
    """Génère un notebook .ipynb prêt à exécuter sur Kaggle/Colab."""
    model_cfg = get_nested(config, "model", default={})
    training_cfg = get_nested(config, "training", default={})
    kaggle_cfg = get_nested(config, "kaggle", default={})

    hf_dataset = get_nested(config, "training", "hf_dataset", default="")
    hf_repo_lora = get_nested(config, "training", "hf_repo", default="")
    hf_repo_merged = get_nested(config, "training", "hf_repo_merged", default="")
    hf_repo_gguf = kaggle_cfg.get("hf_repo_gguf", "")
    gguf_quant = kaggle_cfg.get("gguf_quant", "q4_k_m")

    model_name = model_cfg.get("base_name", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    lora_r = model_cfg.get("lora_r", 16)
    lora_alpha = model_cfg.get("lora_alpha", 16)
    target_modules = json.dumps(model_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    max_seq_length = training_cfg.get("max_length", 2048)
    batch_size = training_cfg.get("batch_size", 2)
    grad_accum = training_cfg.get("gradient_accumulation", 4)
    warmup = training_cfg.get("warmup_steps", 5)
    lr = training_cfg.get("learning_rate", "2e-5")
    # Keep original string format (e.g. "2e-5" not "2e-05")
    lr = str(lr) if not isinstance(lr, str) else lr
    logging_steps = training_cfg.get("logging_steps", 25)
    weight_decay = training_cfg.get("weight_decay", 0.001)
    lr_scheduler = training_cfg.get("lr_scheduler_type", "linear")
    seed = training_cfg.get("seed", 3407)
    epochs = training_cfg.get("epochs", 2)
    max_steps = training_cfg.get("max_steps", -1)
    packing = training_cfg.get("packing", False)
    chat_template = training_cfg.get("chat_template", "chatml")

    # Construire les cellules du notebook
    cells = []

    def add_md(source: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})

    def add_code(source: str) -> None:
        cells.append({
            "cell_type": "code", "metadata": {}, "source": _lines(source),
            "execution_count": None, "outputs": [],
        })

    # --- Cells (structure alignée sur le notebook de référence) ---
    add_md("# CodeQuantBenchmark — Training\n\nAuto-generated notebook. GPU T4 recommended.")

    # Cell 1: Installation (%%capture pour masquer l'output)
    add_code(
        "%%capture\n"
        "import os\n"
        "\n"
        "!pip install pip3-autoremove\n"
        "!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n"
        "!pip install unsloth\n"
        "!pip install transformers==4.56.2\n"
        "!pip install --no-deps trl==0.22.2"
    )

    # Cell 2: HF Login
    add_code(
        "# HF Login (uses Kaggle Secrets or Colab userdata)\n"
        "import os\n"
        "hf_token = None\n"
        "try:\n"
        "    from kaggle_secrets import UserSecretsClient\n"
        "    hf_token = UserSecretsClient().get_secret('HF_TOKEN')\n"
        "except Exception:\n"
        "    pass\n"
        "if not hf_token:\n"
        "    try:\n"
        "        from google.colab import userdata\n"
        "        hf_token = userdata.get('HF_TOKEN')\n"
        "    except Exception:\n"
        "        pass\n"
        "if not hf_token:\n"
        "    hf_token = os.environ.get('HF_TOKEN', '')\n"
        "\n"
        "if hf_token:\n"
        "    from huggingface_hub import login\n"
        "    login(token=hf_token, add_to_git_credential=False)\n"
        "    print('Logged in to HuggingFace Hub')\n"
        "else:\n"
        "    print('WARNING: No HF_TOKEN found. Push to Hub will fail.')"
    )

    # Cell 3: Model + LoRA
    add_md("## Model + LoRA")

    add_code(
        "import unsloth\n"
        "from unsloth import FastLanguageModel\n"
        "import torch\n"
        "\n"
        f"model, tokenizer = FastLanguageModel.from_pretrained(\n"
        f"    model_name='{model_name}',\n"
        f"    max_seq_length={max_seq_length},\n"
        f"    dtype=None,\n"
        f"    load_in_4bit=True,\n"
        f")\n"
        f"\n"
        f"model = FastLanguageModel.get_peft_model(\n"
        f"    model,\n"
        f"    r={lora_r},\n"
        f"    target_modules={target_modules},\n"
        f"    lora_alpha={lora_alpha},\n"
        f"    lora_dropout=0,\n"
        f"    bias='none',\n"
        f"    use_gradient_checkpointing='unsloth',\n"
        f"    random_state={seed},\n"
        f"    use_rslora=False,\n"
        f"    loftq_config=None,\n"
        f")"
    )

    # Cell 4: Chat template + formatting + Dataset loading (single cell like reference)
    map_eos_line = "    map_eos_token = True,\n" if chat_template == "chatml" else ""

    if hf_dataset:
        dataset_load = (
            f"dataset = load_dataset('{hf_dataset}', "
            f"data_files='train_sharegpt.jsonl', split='train', token=hf_token)\n"
        )
    else:
        dataset_load = (
            "# No hf_dataset configured -- set training.hf_dataset in config.yaml\n"
            "raise ValueError('training.hf_dataset not set in config.yaml')\n"
        )

    add_code(
        "from unsloth.chat_templates import get_chat_template\n"
        "\n"
        f"tokenizer = get_chat_template(\n"
        f"    tokenizer,\n"
        f"    chat_template = \"{chat_template}\",\n"
        f"    mapping = {{\"role\" : \"from\", \"content\" : \"value\", \"user\" : \"human\", \"assistant\" : \"gpt\"}},\n"
        f"{map_eos_line}"
        f")\n"
        "\n"
        "def formatting_prompts_func(examples):\n"
        "    convos = examples[\"conversations\"]\n"
        "    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]\n"
        "    return { \"text\" : texts, }\n"
        "\n"
        "from datasets import load_dataset\n"
        f"{dataset_load}"
        f"print(f'{{len(dataset)}} examples loaded')\n"
        "dataset = dataset.map(formatting_prompts_func, batched = True,)"
    )

    # Cell 5: Training
    add_md("## Training")

    steps_or_epochs = f"    max_steps={max_steps}," if max_steps > 0 else f"    num_train_epochs={epochs},"

    add_code(
        "from trl import SFTConfig, SFTTrainer\n"
        "\n"
        "trainer = SFTTrainer(\n"
        "    model = model,\n"
        "    tokenizer = tokenizer,\n"
        "    train_dataset = dataset,\n"
        "    dataset_text_field = \"text\",\n"
        f"    max_seq_length = {max_seq_length},\n"
        f"    packing = {packing},\n"
        "    args = SFTConfig(\n"
        f"        per_device_train_batch_size = {batch_size},\n"
        f"        gradient_accumulation_steps = {grad_accum},\n"
        f"        warmup_steps = {warmup},\n"
        f"    {steps_or_epochs}\n"
        f"        learning_rate = {lr},\n"
        f"        logging_steps = {logging_steps},\n"
        f"        optim = \"adamw_8bit\",\n"
        f"        weight_decay = {weight_decay},\n"
        f"        lr_scheduler_type = \"{lr_scheduler}\",\n"
        f"        seed = {seed},\n"
        "        output_dir = \"outputs\",\n"
        "        save_strategy = \"epoch\",\n"
        "        report_to = \"none\",\n"
        "    ),\n"
        ")\n"
        "\n"
        "gpu_stats = torch.cuda.get_device_properties(0)\n"
        "start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n"
        "print(f\"GPU = {gpu_stats.name}. Max memory = {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)} GB.\")\n"
        "print(f\"{start_gpu_memory} GB reserved before training.\")\n"
        "\n"
        "trainer_stats = trainer.train()\n"
        "\n"
        "used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n"
        "print(f\"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes\")\n"
        "print(f\"Peak memory: {used_memory} GB\")"
    )

    # Cell 6: Save LoRA
    add_md("## Save + Push")

    add_code(
        "model.save_pretrained('lora_model')\n"
        "tokenizer.save_pretrained('lora_model')"
    )

    if hf_repo_lora:
        add_code(
            f"print('Pushing LoRA to: {hf_repo_lora}')\n"
            f"model.push_to_hub('{hf_repo_lora}')\n"
            f"tokenizer.push_to_hub('{hf_repo_lora}')"
        )

    # Cell 7: Merge LoRA + Push merged fp16 model
    if hf_repo_merged:
        add_md("## Merge LoRA + Push merged model (fp16)")

        add_code(
            "import subprocess, gc\n"
            "\n"
            "# Free memory before merge\n"
            "subprocess.run(['rm', '-rf', 'outputs/'], check=False)\n"
            "subprocess.run(['pip', 'cache', 'purge'], check=False)\n"
            "gc.collect()\n"
            "torch.cuda.empty_cache()\n"
            "\n"
            f"print('Merging LoRA into base model and pushing to: {hf_repo_merged}')\n"
            f"model.save_pretrained_merged('{hf_repo_merged}', tokenizer, save_method='merged_16bit')\n"
            f"model.push_to_hub_merged('{hf_repo_merged}', tokenizer, save_method='merged_16bit')\n"
            f"print('Merged model pushed to: https://huggingface.co/{hf_repo_merged}')"
        )

    if hf_repo_gguf:
        add_code(
            "import subprocess\n"
            "subprocess.run(['rm', '-rf', 'outputs/'], check=False)\n"
            "subprocess.run(['pip', 'cache', 'purge'], check=False)\n"
            "\n"
            f"print('Converting + pushing GGUF to: {hf_repo_gguf}')\n"
            f"model.push_to_hub_gguf('{hf_repo_gguf}', tokenizer, quantization_method='{gguf_quant}')"
        )

    add_code("print('Done!')")

    # Build notebook JSON
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

    out_path = Path("notebook_training.ipynb")
    out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Notebook generated: {out_path}")
    return out_path


def _lines(text: str) -> list[str]:
    """Convertit un texte en liste de lignes (format notebook)."""
    raw_lines = text.split("\n")
    result = []
    for i, line in enumerate(raw_lines):
        if i < len(raw_lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result


# ---------------------------------------------------------------------------
# Kaggle status check (optional, si le kernel existe)
# ---------------------------------------------------------------------------

def check_status(config: dict[str, Any]) -> None:
    """Vérifie le statut du kernel Kaggle (optionnel)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-not-found]
    except ImportError:
        logger.error("kaggle package not installed")
        return

    api = KaggleApi()
    api.authenticate()
    username = api.config_values.get("username", "")
    kaggle_cfg = get_nested(config, "kaggle", default={})
    kernel_slug = kaggle_cfg.get("kernel_slug", "benchmark-training")
    full_slug = f"{username}/{kernel_slug}"

    status_response = api.kernels_status(full_slug)
    # Normalize enum
    raw = getattr(status_response, "status", str(status_response))
    s = str(raw)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    logger.info(f"Kernel {full_slug}: {s.lower()}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config_path: str = "config.yaml") -> str:
    """Push data sur HF Hub + génère le notebook."""
    config = load_config(config_path)

    # 1. Push data
    hf_dataset = push_dataset_to_hf(config)

    # 2. Generate notebook
    nb_path = generate_notebook(config)

    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Upload {nb_path} to Kaggle or Colab")
    logger.info("  2. Set GPU to T4 (Kaggle: Settings > Accelerator)")
    logger.info("  3. Add HF_TOKEN in Secrets (Kaggle: Add-ons > Secrets)")
    logger.info("  4. Run All")

    return str(nb_path)
