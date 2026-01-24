# FedEDA

## Setup

```bash
# Clone with submodules
git clone https://github.com/ju5tbyte/fededa.git
cd fededa
git submodule update --init

# Install dependencies
pip install -e .
```

## Quick Start

### Evaluate on MMCircuitEval

```bash
python scripts/evaluate_mmcircuiteval.py
```

Configuration: [configs/evaluation/mmcircuiteval.yaml](configs/evaluation/mmcircuiteval.yaml)

### Evaluate on ORD-QA

```bash
python scripts/evaluate_ordqa.py
```

Configuration: [configs/evaluation/ordqa.yaml](configs/evaluation/ordqa.yaml)

### Evaluate on EDA-Corpus

```bash
python scripts/evaluate_edacorpus.py
```

Configuration: [configs/evaluation/edacorpus.yaml](configs/evaluation/edacorpus.yaml)

## Project Structure

```
FedEDA/
├── configs/              # Hydra configurations
│   ├── evaluation/       # Evaluation benchmark configs
│   └── model/            # Model configurations
├── external/             # External dependencies (git submodules)
│   ├── MMCircuitEval/   # MMCircuitEval benchmark
│   ├── ORD-QA/          # ORD-QA benchmark
│   ├── EDA-Corpus/      # EDA-Corpus benchmark
│   └── UniEval/         # Unieval framework
├── scripts/              # Executable scripts
│   ├── evaluate_mmcircuiteval.py
│   ├── evaluate_ordqa.py
│   ├── evaluate_edacorpus.py
│   └── train.py
├── src/
│   ├── evaluation/       # Evaluation modules
│   │   ├── mmcircuiteval/  # MMCircuitEval runner
│   │   ├── ordqa/          # ORD-QA runner
│   │   ├── edacorpus/      # EDA-Corpus runner
│   │   └── custom_evaluator.py  # Custom evaluation framework (Unieval, Embedding Similarity, LLM-as-a-judge)
│   ├── models/           # Model implementations
│   │   ├── qwen_vl_model.py  # Qwen-VL wrapper
│   │   ├── qwen3_model.py    # Qwen-3 wrapper
│   │   └── builder.py        # Model registry
│   └── utils/            # Utilities
└── outputs/              # Checkpoints and logs
```

## Configuration

Uses Hydra for configuration management. Edit YAML files in [configs/](configs/).
