# CodeQuantBenchmark

Pipeline automatisé pour évaluer l'impact de la quantification sur des LLM spécialisés en code.

## Architecture

```
├── config.yaml                    # Configuration unique (modèle, data, quantif, benchmark)
├── main.py                        # CLI principal (typer)
├── synthetic.py                   # Générateur de workspace synthétique pour tester
├── src/
│   ├── data/
│   │   ├── clone.py               # Clone de repos GitHub
│   │   ├── extract.py             # Extraction AST Python (stdlib)
│   │   ├── extract_treesitter.py  # Extraction multi-langage (tree-sitter)
│   │   ├── clean.py               # Déduplication MD5 + split train/val/test
│   │   └── instruct.py            # Génération ShareGPT (instruction/réponse)
│   ├── training/
│   │   ├── train.py               # Fine-tuning auto-détection GPU/CPU (Unsloth ou Trainer)
│   │   ├── train_remote.py        # Push data HF Hub + génération notebook Kaggle/Colab
│   │   └── validate.py            # Validation post-training (gate)
│   ├── quantization/
│   │   └── convert.py             # Conversion GGUF + quantification (Q2_K → Q8_0)
│   ├── inference/
│   │   ├── base.py                # Interface abstraite InferenceEngine
│   │   └── openai_compat.py       # Implémentation API OpenAI-compatible (llama.cpp server)
│   ├── benchmark/
│   │   ├── engine.py              # Orchestration benchmark multi-modèles
│   │   ├── metrics.py             # Jaccard, BLEU, syntax validity, ms/token
│   │   └── memory.py              # Mesure RSS/peak mémoire + taille fichier
│   ├── reporting/
│   │   ├── plots.py               # Graphiques matplotlib/seaborn
│   │   ├── readme.py              # Génération README scientifique
│   │   └── release.py             # Git tag + GitHub Release
│   └── utils/
│       ├── config.py              # Chargement YAML
│       └── logging.py             # Logs JSON structurés
├── tests/                         # 47 tests (unitaires + intégration)
├── pyproject.toml
└── LICENSE                        # MIT
```

## Guide pas-a-pas (pour debutants)

Le pipeline transforme du code source brut en un modele quantifie et benchmark. Voici l'ordre a suivre :

```
1. COLLECTE         2. PREPARATION       3. TRAINING              4. QUANTIFICATION    5. BENCHMARK
   clone               extract              train (local/GPU)       quantize             benchmark
   (repos GitHub)      clean                ou                                           report
                        instruct            train-remote                                 readme
                        [= data]            (push data HF Hub
                                             + notebook Kaggle/Colab)
                                            validate
```

### Demarrage rapide

```bash
# 1. Preparer les donnees (clone repos + extraction + nettoyage + format ShareGPT)
python main.py data

# 2a. Entrainer en local (CPU = debug, GPU = si disponible)
python main.py train

# 2b. OU entrainer sur Kaggle/Colab GPU (recommande)
python main.py train-remote
# -> Push data sur HF Hub + génère notebook_training.ipynb
# -> Upload le notebook sur Kaggle/Colab, set GPU T4, Run

# 3. Verifier que le modele n'a pas oublie comment coder (gate)
python main.py validate

# 4. Convertir en GGUF et quantifier (Q2_K, Q4_K_M, Q6_K, Q8_0)
python main.py quantize

# 5. Benchmark + rapport
python main.py benchmark
python main.py report

# Ou tout d'un coup (local seulement) :
python main.py pipeline
```

## Commandes

Toutes les commandes acceptent `--config <path>` (defaut : `config.yaml`).

### Pipeline complet

```bash
python main.py pipeline            # Tout d'un coup : data -> train -> validate -> quantize -> benchmark -> report
```

### Etapes individuelles

| Commande | Description |
|----------|-------------|
| `python main.py clone` | Clone des repos GitHub (config: `data.github`) |
| `python main.py extract` | Extraction de fonctions via AST (Python) / tree-sitter (Rust, Go, JS...) |
| `python main.py clean` | Deduplication + split train/val/test (80/10/10) |
| `python main.py instruct` | Generation paires instruction/reponse ShareGPT |
| `python main.py data` | Les 3 etapes ci-dessus enchainees |
| `python main.py train` | Fine-tuning local (auto-detecte GPU/Unsloth ou CPU/Trainer) |
| `python main.py train-remote` | Push data HF Hub + génère notebook pour Kaggle/Colab |
| `python main.py validate` | Validation post-training — gate avant quantification |
| `python main.py quantize` | Conversion GGUF + quantification multi-formats |
| `python main.py benchmark` | Benchmark : qualite, validite, latence, memoire |
| `python main.py report` | Generation des graphiques PNG |
| `python main.py readme` | Generation du README scientifique |
| `python main.py release` | Git tag + GitHub Release avec artefacts |

### Training remote (Kaggle / Colab)

Le training GPU se fait en semi-auto : les données sont pushées sur HuggingFace Hub, et un notebook `.ipynb` prêt à l'emploi est généré localement. Il suffit de l'uploader sur Kaggle ou Colab.

```bash
# 1. Configurer dans config.yaml :
#    training.hf_dataset: "username/benchmark-training-data"
#    training.hf_repo: "username/model-lora"        (optionnel, pour push LoRA)
#    kaggle.hf_repo_gguf: "username/model-gguf"     (optionnel, pour push GGUF)

# 2. Push data + générer le notebook
python main.py train-remote

# 3. Upload notebook_training.ipynb sur Kaggle ou Colab
#    - Kaggle : New Notebook > File > Import Notebook
#    - Colab  : File > Upload Notebook

# 4. Configurer l'environnement
#    - GPU : T4 (Kaggle: Settings > Accelerator / Colab: Runtime > Change runtime)
#    - HF_TOKEN : Kaggle (Add-ons > Secrets) / Colab (Secrets 🔑)

# 5. Run All

# (Optionnel) Check le statut du kernel Kaggle :
python main.py train-remote --status
```

Le notebook charge les données directement depuis HuggingFace Hub — pas besoin d'attacher un dataset Kaggle.

Prérequis : `huggingface_hub` (déjà installé), token HF configuré (`huggingface-cli login` ou `HF_TOKEN` env var).

### Test synthétique (sans données réelles)

```bash
python synthetic.py fakedata                    # Crée un workspace avec données synthétiques
python main.py data --config synthetic_workspace/config.yaml  # Pipeline data dessus
python synthetic.py benchmark synthetic_workspace             # Benchmark avec moteur mock
python main.py report --config synthetic_workspace/config.yaml # Graphiques

# Ou tout d'un coup :
python synthetic.py full                        # Pipeline synthétique complet
```

## Langages supportés (extraction AST)

Python (stdlib `ast`), Rust, C, C++, Go, JavaScript, TypeScript, Java, Ruby, PHP, Scala, Kotlin, Lua, Haskell, OCaml, Elixir, Bash (via tree-sitter).

Configurable dans `config.yaml` :
```yaml
data:
  extensions: [".py", ".rs", ".go", ".js"]
```

## Métriques de benchmark

| Métrique | Description |
|----------|-------------|
| **Jaccard** | Similarité token-level vs référence |
| **BLEU** | Précision n-gram (1-4) avec brevity penalty |
| **Syntax Validity** | `py_compile` pass rate sur le code généré |
| **Latency** | ms/token mesuré via API |
| **Memory** | RSS, peak RSS, delta mémoire par inférence |
| **File Size** | Taille du fichier GGUF |

## Tests

```bash
python -m pytest tests/ -v                      # Tous les tests (47)
python -m pytest tests/test_integration.py -v    # Intégration seule
python -m pytest tests/test_metrics.py -v        # Métriques seules
```

## Prérequis

- Python 3.12+
- Env micromamba `llamacpp` (voir `pyproject.toml` pour les dépendances)
- Pour le benchmark réel : un serveur llama.cpp tournant avec un modèle GGUF
- Pour le training QLoRA local : `peft` + `bitsandbytes` (GPU requis)
- Pour le training remote : `huggingface_hub` + token HF (`huggingface-cli login`)

## Licence

MIT
