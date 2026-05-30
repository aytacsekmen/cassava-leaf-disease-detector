# Cassava Leaf Disease Detector

A deep learning pipeline for classifying diseases in cassava leaf images. The model uses transfer learning with multiple architectures (EfficientNetV2, MobileViT, SwinV2) and is built on top of PyTorch Lightning, with full experiment tracking, data versioning, and reproducible training.

## What This Project Does

Cassava is one of the most widely grown crops in Africa, and it is highly susceptible to several viral diseases that can destroy entire harvests. Identifying these diseases early from leaf images allows farmers to take action before it is too late.

This project trains multiple neural network architectures to classify cassava leaf images into five categories:

| Class | Disease                              |
| ----- | ------------------------------------ |
| 0     | Cassava Bacterial Blight (CBB)       |
| 1     | Cassava Brown Streak Disease (CBSD)  |
| 2     | Cassava Green Mottle (CGM)           |
| 3     | Cassava Mosaic Disease (CMD)         |
| 4     | Healthy                              |

The dataset comes from the [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) competition and contains roughly 21,000 labeled images. Data is downloaded from a Google Drive mirror for zero-friction access (no Kaggle credentials needed).

### Models

Three architectures are trained concurrently via SLURM array jobs and compared:

| Model             | timm Name                  | Features | Type                       |
| ----------------- | -------------------------- | -------- | -------------------------- |
| EfficientNetV2-B0 | `tf_efficientnetv2_b0`     | 1280     | Lightweight CNN            |
| MobileViT-S       | `mobilevit_s`              | 640      | Edge-optimized hybrid ViT  |
| SwinV2-Tiny       | `swinv2_tiny_window8_256`  | 768      | Shifted-window Transformer |

All models use pretrained ImageNet-1K weights via the `timm` library. A custom classification head with dropout is attached for the 5-class task. Training uses:

- AdamW optimizer with cosine annealing learning rate schedule
- Mixed-precision (FP16) training for faster GPU throughput
- Early stopping with high patience (15 epochs)
- Model checkpointing that keeps only the top 3 best weights by validation loss
- Per-model checkpoint and plot directories to prevent overwrites during concurrent sweeps

### Tools and Infrastructure

- **PyTorch Lightning** for training loop and callback management
- **Hydra** for hierarchical YAML configuration (Compose API with config groups)
- **MLflow** for experiment tracking and metric logging
- **DVC** for data and model versioning with local storage
- **SLURM** array jobs for concurrent multi-model training
- **ruff** for linting and formatting
- **pytest** for unit testing

---

## Project Structure

```
cassava-leaf-disease-detector/
├── commands.py                    # CLI entry point (fire + hydra)
├── pyproject.toml                 # Dependencies and tool config
├── .pre-commit-config.yaml        # Code quality hooks
├── configs/
│   ├── config.yaml                # Single Hydra entry point with defaults
│   ├── model/                     # Model config group
│   │   ├── default.yaml           #   Default backbone (tf_efficientnet_b3)
│   │   ├── efficientnetv2_b0.yaml #   EfficientNetV2-B0
│   │   ├── mobilevit_s.yaml       #   MobileViT-S
│   │   └── swinv2_tiny.yaml       #   SwinV2-Tiny
│   ├── preprocessing/             # Preprocessing config group
│   │   └── preprocessing.yaml     #   Image size, splits, normalization
│   ├── training/                  # Training config group
│   │   └── training.yaml          #   Batch size, LR, epochs, callbacks
│   └── mlflow/                    # MLflow config group
│       └── mlflow.yaml            #   Tracking server config
├── cassava_detector/              # Python package
│   ├── data/
│   │   ├── download.py            # Google Drive dataset download
│   │   ├── preprocess.py          # Train/val/test splitting
│   │   └── datamodule.py          # Lightning DataModule
│   ├── model/
│   │   └── classifier.py          # Lightning classifier module
│   ├── training/
│   │   └── trainer.py             # Training orchestration
│   ├── inference/
│   │   └── predict.py             # Single-image inference
│   └── utils/
│       └── plotting.py            # Training & comparison plot generation
├── scripts/
│   ├── train_sweep.sbatch         # SLURM array job for multi-model sweep
│   ├── download_and_preprocess.sbatch  # SLURM job for data prep + DVC tracking
│   └── generate_comparison.sh     # Post-sweep comparison plot generation
├── tests/
│   ├── test_datamodule.py
│   └── test_metrics.py
├── data/                          # Raw and processed data (gitignored, DVC-tracked)
├── models/                        # Saved checkpoints (gitignored)
└── plots/                         # Generated training plots (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- (Optional) An MLflow server running at `http://127.0.0.1:8080`

### 1. Install uv

If you do not have `uv` installed yet:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/your-username/cassava-leaf-disease-detector.git
cd cassava-leaf-disease-detector

uv sync --all-extras
```

This creates a virtual environment and installs all dependencies from `uv.lock`.

### 3. Install pre-commit hooks

```bash
uv run pre-commit install
```

To check that everything passes:

```bash
uv run pre-commit run -a
```

### 4. Initialize DVC

```bash
uv run dvc init
```

The DVC config at `.dvc/config` is already set up with two local storage remotes (one for data, one for models).

---

## Usage

All commands go through the single `commands.py` entry point.

### Download the dataset

```bash
uv run python commands.py download
```

This downloads the Cassava Leaf Disease dataset from Google Drive into `data/raw/`. No API credentials are needed.

After downloading, track data with DVC:

```bash
uv run dvc add data/raw
```

### Preprocess (split data)

```bash
uv run python commands.py preprocess
```

Creates stratified train/validation/test CSV splits in `data/processed/`. Default split: 75% train, 15% validation, 10% test.

Track processed data with DVC:

```bash
uv run dvc add data/processed
```

### Train a single model

```bash
uv run python commands.py train
```

This starts training with the default model config. To select a specific model architecture using Hydra config groups:

```bash
# Use MobileViT-S
uv run python commands.py train --overrides="['model=mobilevit_s']"

# Use SwinV2-Tiny
uv run python commands.py train --overrides="['model=swinv2_tiny']"

# Use EfficientNetV2-B0
uv run python commands.py train --overrides="['model=efficientnetv2_b0']"
```

To override individual training parameters:

```bash
uv run python commands.py train --overrides="['training.batch_size=64', 'training.max_epochs=100']"
```

### Train all models (SLURM sweep)

On a SLURM cluster, run all 3 architectures concurrently:

```bash
# Step 1: Download, preprocess, and DVC-track data
mkdir -p logs
sbatch scripts/download_and_preprocess.sbatch

# Step 2: After data is ready, launch the sweep
sbatch scripts/train_sweep.sbatch
```

This submits a SLURM array job with 3 tasks (one per architecture). Each task selects a different Hydra config group and trains on a separate GPU.

### Generate comparison plots

After all sweep jobs complete:

```bash
uv run python commands.py compare
```

Or use the convenience script:

```bash
bash scripts/generate_comparison.sh
```

This creates comparative plots in `plots/comparison/`:

- `comparison_val_loss.png` — Validation loss curves for all models
- `comparison_val_accuracy.png` — Validation accuracy curves
- `comparison_val_f1.png` — Validation F1 score curves
- `comparison_summary.png` — Bar chart of best metrics per model

### Run inference

```bash
uv run python commands.py infer --checkpoint_path=models/mobilevit_s/best.ckpt --image_path=path/to/leaf.jpg
```

### Run tests

```bash
uv run pytest
```

### Type checking

```bash
uv run mypy cassava_detector/
```

---

## Configuration

All parameters live in YAML files under `configs/`. There are no magic numbers in the code.

### Hierarchical Config Groups

The project uses Hydra's hierarchical config group system with a single entry point:

```yaml
# configs/config.yaml
defaults:
  - preprocessing: preprocessing
  - model: default
  - training: training
  - mlflow: mlflow
```

To swap the model architecture for a sweep, override the config group:

```bash
--overrides="['model=mobilevit_s']"
```

### Config Groups

| Group            | Files                                                          | What it controls                                                 |
| ---------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| `preprocessing`  | `preprocessing/preprocessing.yaml`                             | Image size, train/val/test split ratios, normalization stats     |
| `model`          | `model/default.yaml`, `model/efficientnetv2_b0.yaml`, etc.     | Backbone architecture, number of classes, dropout rate            |
| `training`       | `training/training.yaml`                                       | Batch size, learning rate, epochs, early stopping, checkpointing |
| `mlflow`         | `mlflow/mlflow.yaml`                                           | MLflow tracking URI and experiment name                          |

---

## Multi-Model Sweep

The sweep trains three architectures concurrently using SLURM array jobs with Hydra config group selection:

```
Array Index 0 → model=efficientnetv2_b0  (tf_efficientnetv2_b0)
Array Index 1 → model=mobilevit_s        (mobilevit_s)
Array Index 2 → model=swinv2_tiny        (swinv2_tiny_window8_256)
```

Each model's outputs are isolated:

- **Checkpoints**: `models/<backbone_name>/`
- **Per-run plots**: `plots/<backbone_name>/`
- **Comparison plots**: `plots/comparison/`

---

## Experiment Tracking

The pipeline logs the following to MLflow:

- **Metrics**: `train_loss`, `val_loss`, `val_accuracy`, `val_f1`, `test_loss`, `test_accuracy`, `test_f1`
- **Hyperparameters**: All values from the Hydra configs (including `backbone`)
- **Git commit ID**: Automatically captured at training time

To start an MLflow server locally:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

For cluster-wide access (needed for SLURM sweep jobs running on different nodes):

```bash
uv run mlflow server --host 0.0.0.0 --port 8080
```

Then open `http://127.0.0.1:8080` in your browser to view experiments.

---

## Data Versioning (DVC)

Data and model artifacts are tracked with DVC, not Git. After downloading and preprocessing:

```bash
uv run dvc add data/raw
uv run dvc add data/processed
git add data/raw.dvc data/processed.dvc data/.gitignore
git commit -m "Track data artifacts with DVC"
uv run dvc push
```

The `.dvc/config` file specifies two storage remotes: one for data and one for models.