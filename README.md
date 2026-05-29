# Cassava Leaf Disease Detector

A deep learning pipeline for classifying diseases in cassava leaf images. The model uses transfer learning with EfficientNet and is built on top of PyTorch Lightning, with full experiment tracking, data versioning, and reproducible training.

## What This Project Does

Cassava is one of the most widely grown crops in Africa, and it is highly susceptible to several viral diseases that can destroy entire harvests. Identifying these diseases early from leaf images allows farmers to take action before it is too late.

This project trains a convolutional neural network to classify cassava leaf images into five categories:

| Class | Disease |
| ----- | -------------------------------------------- |
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

The dataset comes from the [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) competition and contains roughly 21,000 labeled images.

### Model

The classifier is built on an EfficientNet-B3 backbone (via the `timm` library), pretrained on ImageNet. A custom classification head with dropout is attached for the 5-class task. Training uses:

- AdamW optimizer with cosine annealing learning rate schedule
- Mixed-precision (FP16) training for faster GPU throughput
- Early stopping with high patience (15 epochs) to allow for longer training and avoid cutting off improvements too soon
- Model checkpointing that keeps only the top 3 best weights by validation loss

### Tools and Infrastructure

- **PyTorch Lightning** for training loop and callback management
- **Hydra** for hierarchical YAML configuration
- **MLflow** for experiment tracking and metric logging
- **DVC** for data and model versioning with local storage
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
│   ├── preprocessing.yaml         # Image size, splits, normalization
│   ├── model.yaml                 # Backbone, classes, dropout
│   ├── training.yaml              # Batch size, LR, epochs, callbacks
│   └── mlflow.yaml                # Tracking server config
├── cassava_detector/              # Python package
│   ├── data/
│   │   ├── download.py            # Kaggle dataset download
│   │   ├── preprocess.py          # Train/val/test splitting
│   │   └── datamodule.py          # Lightning DataModule
│   ├── model/
│   │   └── classifier.py          # Lightning classifier module
│   ├── training/
│   │   └── trainer.py             # Training orchestration
│   ├── inference/
│   │   └── predict.py             # Single-image inference
│   └── utils/
│       └── plotting.py            # Training plot generation
├── tests/
│   ├── test_datamodule.py
│   └── test_metrics.py
├── data/                          # Raw and processed data (gitignored)
├── models/                        # Saved checkpoints (gitignored)
└── plots/                         # Generated training plots (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.10 or later
- A Kaggle account with API credentials configured at `~/.kaggle/kaggle.json`
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

This pulls the Cassava Leaf Disease dataset from Kaggle into `data/raw/`.

### Preprocess (split data)

```bash
uv run python commands.py preprocess
```

Creates stratified train/validation/test CSV splits in `data/processed/`. Default split: 75% train, 15% validation, 10% test.

### Train the model

```bash
uv run python commands.py train
```

This starts training with all settings from the YAML configs. To override specific values:

```bash
uv run python commands.py train --overrides="['training.batch_size=64', 'training.max_epochs=100']"
```

Training logs metrics to MLflow (make sure the server is running) and saves the top 3 checkpoints to `models/`. After training finishes, loss and metric plots are saved to `plots/`.

### Run inference

```bash
uv run python commands.py infer --checkpoint_path=models/best.ckpt --image_path=path/to/leaf.jpg
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

All parameters live in YAML files under `configs/`. There are no magic numbers in the code. Key settings:

| File | What it controls |
| ---------------------- | ---------------------------------------------------------------- |
| `preprocessing.yaml` | Image size, train/val/test split ratios, normalization stats |
| `model.yaml` | Backbone architecture, number of classes, dropout rate |
| `training.yaml` | Batch size, learning rate, epochs, early stopping, checkpointing |
| `mlflow.yaml` | MLflow tracking URI and experiment name |

---

## Experiment Tracking

The pipeline logs the following to MLflow:

- **Metrics**: `train_loss`, `val_loss`, `val_accuracy`, `val_f1`, `test_loss`, `test_accuracy`, `test_f1`
- **Hyperparameters**: All values from the Hydra configs
- **Git commit ID**: Automatically captured at training time

To start an MLflow server locally:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080` in your browser to view experiments.