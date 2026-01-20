# FedEDA

## Setup

```bash
# Clone with submodules
git clone --recursive <repository-url>

# Install dependencies
pip install -e .
```

## Quick Start

### Evaluate on MMCircuitEval

```bash
python scripts/evaluate_mmcircuiteval.py
```

Configuration: [configs/evaluation/mmcircuiteval.yaml](configs/evaluation/mmcircuiteval.yaml)

## Project Structure

```
FedEDA/
├── configs/              # Hydra configurations
├── external/             # External dependencies (git submodules)
│   └── MMCircuitEval/   # Benchmark evaluation
├── scripts/              # Executable scripts
│   └── evaluate_mmcircuiteval.py
├── src/
│   ├── evaluation/       # Evaluation modules
│   ├── models/           # Model implementations (Qwen-VL)
│   └── utils/            # Utilities
└── outputs/              # Checkpoints and logs
```

## Models

- **Qwen-VL**: Vision-Language Model via [src/models/qwen_vl_model.py](src/models/qwen_vl_model.py)

## Configuration

Uses Hydra for configuration management. Edit YAML files in [configs/](configs/).
