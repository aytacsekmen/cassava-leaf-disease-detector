"""Training orchestration for the cassava leaf disease classifier."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from cassava_detector.data.datamodule import CassavaDataModule
from cassava_detector.model.classifier import CassavaClassifier
from cassava_detector.utils.plotting import generate_plots

logger = logging.getLogger(__name__)


def _get_git_commit_id() -> str:
    """Retrieve the current Git commit hash.

    Returns:
        The short commit hash, or ``"unknown"`` if Git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Could not determine Git commit ID.")
        return "unknown"


def train(config: DictConfig) -> None:
    """Run the full training pipeline.

    Creates the data module, model, callbacks, and logger, then fits the
    model using PyTorch Lightning's Trainer.  After training completes,
    generates training plots and saves them to the ``plots/`` directory.

    Args:
        config: Merged Hydra configuration containing ``preprocessing``,
            ``model``, ``training``, and ``mlflow`` sections.
    """
    preprocessing_config = config.preprocessing
    model_config = config.model
    training_config = config.training
    mlflow_config = config.mlflow

    pl.seed_everything(preprocessing_config.random_seed, workers=True)

    image_dir = str(Path(preprocessing_config.raw_data_dir) / "train_images")

    data_module = CassavaDataModule(
        processed_data_dir=preprocessing_config.processed_data_dir,
        image_dir=image_dir,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        image_size=preprocessing_config.image_size,
        mean=list(preprocessing_config.mean),
        std=list(preprocessing_config.std),
    )

    model = CassavaClassifier(
        backbone=model_config.backbone,
        num_classes=model_config.num_classes,
        pretrained=model_config.pretrained,
        dropout_rate=model_config.dropout_rate,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        scheduler_t_max=training_config.scheduler_t_max,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",
        filename="cassava-{epoch:02d}-{val_loss:.4f}",
        monitor=training_config.checkpoint_monitor,
        mode=training_config.checkpoint_mode,
        save_top_k=training_config.checkpoint_top_k,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor=training_config.checkpoint_monitor,
        mode=training_config.checkpoint_mode,
        patience=training_config.early_stopping_patience,
        verbose=True,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_config.experiment_name,
        tracking_uri=mlflow_config.tracking_uri,
        log_model=False,
    )

    git_commit = _get_git_commit_id()
    hyperparams: dict[str, Any] = {
        "backbone": model_config.backbone,
        "num_classes": model_config.num_classes,
        "pretrained": model_config.pretrained,
        "dropout_rate": model_config.dropout_rate,
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "max_epochs": training_config.max_epochs,
        "image_size": preprocessing_config.image_size,
        "precision": training_config.precision,
        "early_stopping_patience": training_config.early_stopping_patience,
        "checkpoint_top_k": training_config.checkpoint_top_k,
        "gradient_clip_val": training_config.gradient_clip_val,
        "scheduler_t_max": training_config.scheduler_t_max,
        "git_commit": git_commit,
    }
    mlflow_logger.log_hyperparams(hyperparams)

    logger.info("Git commit: %s", git_commit)
    logger.info("Starting training with config: %s", hyperparams)

    trainer = pl.Trainer(
        max_epochs=training_config.max_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=mlflow_logger,
        deterministic=True,
    )

    trainer.fit(model, datamodule=data_module)

    logger.info(
        "Training complete. Best model: %s (val_loss=%.4f)",
        checkpoint_callback.best_model_path,
        checkpoint_callback.best_model_score or float("nan"),
    )

    try:
        generate_plots(
            tracking_uri=mlflow_config.tracking_uri,
            experiment_name=mlflow_config.experiment_name,
            output_dir="plots",
        )
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    if checkpoint_callback.best_model_path:
        trainer.test(model, datamodule=data_module, ckpt_path="best")
