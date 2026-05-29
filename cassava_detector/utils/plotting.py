"""Generate training plots from MLflow experiment metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def generate_plots(
    tracking_uri: str,
    experiment_name: str,
    output_dir: str = "plots",
) -> None:
    """Generate and save training metric plots from MLflow logs.

    Queries the most recent MLflow run for the given experiment and
    creates two plots:

    1. Training and validation loss curves over epochs.
    2. Validation accuracy and F1 score curves over epochs.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of the MLflow experiment.
        output_dir: Directory to save the generated plot images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment '%s' not found. Skipping plot generation.", experiment_name)
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        logger.warning("No runs found for experiment '%s'.", experiment_name)
        return

    run_id = runs.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    _plot_loss_curves(client, run_id, output_path)
    _plot_metric_curves(client, run_id, output_path)

    logger.info("Training plots saved to %s", output_path)


def _plot_loss_curves(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    output_path: Path,
) -> None:
    """Plot training and validation loss curves.

    Args:
        client: MLflow tracking client instance.
        run_id: ID of the MLflow run to plot.
        output_path: Directory to save the plot.
    """
    train_loss_history = client.get_metric_history(run_id, "train_loss_epoch")
    val_loss_history = client.get_metric_history(run_id, "val_loss")

    if not train_loss_history or not val_loss_history:
        logger.warning("Insufficient loss data for plot. Skipping loss curves.")
        return

    train_epochs = [metric.step for metric in train_loss_history]
    train_losses = [metric.value for metric in train_loss_history]
    val_epochs = [metric.step for metric in val_loss_history]
    val_losses = [metric.value for metric in val_loss_history]

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(train_epochs, train_losses, label="Train Loss", linewidth=2)
    axis.plot(val_epochs, val_losses, label="Val Loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.legend()
    axis.grid(True, alpha=0.3)

    loss_plot_path = output_path / "loss_curves.png"
    fig.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Loss curves saved to %s", loss_plot_path)


def _plot_metric_curves(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    output_path: Path,
) -> None:
    """Plot validation accuracy and F1 score curves.

    Args:
        client: MLflow tracking client instance.
        run_id: ID of the MLflow run to plot.
        output_path: Directory to save the plot.
    """
    accuracy_history = client.get_metric_history(run_id, "val_accuracy")
    f1_history = client.get_metric_history(run_id, "val_f1")

    if not accuracy_history or not f1_history:
        logger.warning("Insufficient metric data for plot. Skipping metric curves.")
        return

    accuracy_epochs = [metric.step for metric in accuracy_history]
    accuracy_values = [metric.value for metric in accuracy_history]
    f1_epochs = [metric.step for metric in f1_history]
    f1_values = [metric.value for metric in f1_history]

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(accuracy_epochs, accuracy_values, label="Val Accuracy", linewidth=2)
    axis.plot(f1_epochs, f1_values, label="Val F1 Score", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Score")
    axis.set_title("Validation Accuracy and F1 Score")
    axis.legend()
    axis.grid(True, alpha=0.3)
    axis.set_ylim(0.0, 1.0)

    metric_plot_path = output_path / "metric_curves.png"
    fig.savefig(metric_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Metric curves saved to %s", metric_plot_path)
