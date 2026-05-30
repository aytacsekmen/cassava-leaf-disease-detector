"""Generate training plots from MLflow experiment metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

COMPARISON_COLORS = [
    "#2196F3",
    "#FF5722",
    "#4CAF50",
    "#9C27B0",
    "#FF9800",
    "#00BCD4",
]


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


def generate_comparison_plots(
    tracking_uri: str,
    experiment_name: str,
    output_dir: str = "plots/comparison",
) -> None:
    """Generate comparative plots across all model architectures.

    Queries all runs from the MLflow experiment, groups them by
    backbone architecture, and creates side-by-side comparison plots
    for loss, accuracy, and F1 score.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Name of the MLflow experiment.
        output_dir: Directory to save the comparison plot images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning("Experiment '%s' not found. Skipping comparison plots.", experiment_name)
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if runs.empty:
        logger.warning("No runs found for experiment '%s'.", experiment_name)
        return

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    backbone_runs = _group_runs_by_backbone(runs)
    if len(backbone_runs) < 2:
        logger.warning(
            "Found %d backbone(s). Need at least 2 for comparison plots.",
            len(backbone_runs),
        )
        return

    logger.info("Generating comparison plots for backbones: %s", list(backbone_runs.keys()))

    _plot_comparison_loss(client, backbone_runs, output_path)
    _plot_comparison_accuracy(client, backbone_runs, output_path)
    _plot_comparison_f1(client, backbone_runs, output_path)
    _plot_summary_bar_chart(runs, backbone_runs, output_path)

    logger.info("Comparison plots saved to %s", output_path)


def _group_runs_by_backbone(runs: "pd.DataFrame") -> dict[str, str]:
    """Group MLflow runs by their backbone hyperparameter.

    For each backbone, keeps only the most recent run (first in the
    time-sorted DataFrame).

    Args:
        runs: DataFrame of MLflow runs sorted by start_time DESC.

    Returns:
        Dictionary mapping backbone name to run_id.
    """
    backbone_runs: dict[str, str] = {}
    for _, run in runs.iterrows():
        backbone = run.get("params.backbone", None)
        if backbone and backbone not in backbone_runs:
            backbone_runs[backbone] = run["run_id"]
    return backbone_runs


def _plot_comparison_loss(
    client: mlflow.tracking.MlflowClient,
    backbone_runs: dict[str, str],
    output_path: Path,
) -> None:
    """Plot validation loss curves for all backbones on one chart.

    Args:
        client: MLflow tracking client instance.
        backbone_runs: Mapping of backbone name to run_id.
        output_path: Directory to save the plot.
    """
    fig, axis = plt.subplots(figsize=(12, 7))

    for color_idx, (backbone, run_id) in enumerate(backbone_runs.items()):
        val_loss_history = client.get_metric_history(run_id, "val_loss")
        if not val_loss_history:
            logger.warning("No val_loss data for %s. Skipping.", backbone)
            continue
        epochs = [metric.step for metric in val_loss_history]
        values = [metric.value for metric in val_loss_history]
        color = COMPARISON_COLORS[color_idx % len(COMPARISON_COLORS)]
        axis.plot(epochs, values, label=backbone, linewidth=2, color=color)

    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Validation Loss", fontsize=12)
    axis.set_title("Model Comparison — Validation Loss", fontsize=14, fontweight="bold")
    axis.legend(fontsize=10)
    axis.grid(True, alpha=0.3)

    plot_path = output_path / "comparison_val_loss.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison loss plot saved to %s", plot_path)


def _plot_comparison_accuracy(
    client: mlflow.tracking.MlflowClient,
    backbone_runs: dict[str, str],
    output_path: Path,
) -> None:
    """Plot validation accuracy curves for all backbones on one chart.

    Args:
        client: MLflow tracking client instance.
        backbone_runs: Mapping of backbone name to run_id.
        output_path: Directory to save the plot.
    """
    fig, axis = plt.subplots(figsize=(12, 7))

    for color_idx, (backbone, run_id) in enumerate(backbone_runs.items()):
        accuracy_history = client.get_metric_history(run_id, "val_accuracy")
        if not accuracy_history:
            logger.warning("No val_accuracy data for %s. Skipping.", backbone)
            continue
        epochs = [metric.step for metric in accuracy_history]
        values = [metric.value for metric in accuracy_history]
        color = COMPARISON_COLORS[color_idx % len(COMPARISON_COLORS)]
        axis.plot(epochs, values, label=backbone, linewidth=2, color=color)

    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Validation Accuracy", fontsize=12)
    axis.set_title("Model Comparison — Validation Accuracy", fontsize=14, fontweight="bold")
    axis.legend(fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.set_ylim(0.0, 1.0)

    plot_path = output_path / "comparison_val_accuracy.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison accuracy plot saved to %s", plot_path)


def _plot_comparison_f1(
    client: mlflow.tracking.MlflowClient,
    backbone_runs: dict[str, str],
    output_path: Path,
) -> None:
    """Plot validation F1 score curves for all backbones on one chart.

    Args:
        client: MLflow tracking client instance.
        backbone_runs: Mapping of backbone name to run_id.
        output_path: Directory to save the plot.
    """
    fig, axis = plt.subplots(figsize=(12, 7))

    for color_idx, (backbone, run_id) in enumerate(backbone_runs.items()):
        f1_history = client.get_metric_history(run_id, "val_f1")
        if not f1_history:
            logger.warning("No val_f1 data for %s. Skipping.", backbone)
            continue
        epochs = [metric.step for metric in f1_history]
        values = [metric.value for metric in f1_history]
        color = COMPARISON_COLORS[color_idx % len(COMPARISON_COLORS)]
        axis.plot(epochs, values, label=backbone, linewidth=2, color=color)

    axis.set_xlabel("Epoch", fontsize=12)
    axis.set_ylabel("Validation F1 Score", fontsize=12)
    axis.set_title("Model Comparison — Validation F1 Score", fontsize=14, fontweight="bold")
    axis.legend(fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.set_ylim(0.0, 1.0)

    plot_path = output_path / "comparison_val_f1.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparison F1 plot saved to %s", plot_path)


def _plot_summary_bar_chart(
    runs: "pd.DataFrame",
    backbone_runs: dict[str, str],
    output_path: Path,
) -> None:
    """Plot a summary bar chart comparing final metrics across models.

    Shows the best validation loss, accuracy, and F1 score for each
    backbone as grouped bars.

    Args:
        runs: DataFrame of all MLflow runs.
        backbone_runs: Mapping of backbone name to run_id.
        output_path: Directory to save the plot.
    """
    backbones = list(backbone_runs.keys())
    metric_names = ["val_loss", "val_accuracy", "val_f1"]
    metric_display = ["Best Val Loss", "Best Val Accuracy", "Best Val F1"]

    metric_values: dict[str, list[float]] = {name: [] for name in metric_names}

    for backbone in backbones:
        run_id = backbone_runs[backbone]
        run_row = runs[runs["run_id"] == run_id].iloc[0]
        for metric_name in metric_names:
            col_name = f"metrics.{metric_name}"
            value = run_row.get(col_name, float("nan"))
            metric_values[metric_name].append(float(value))

    num_groups = len(backbones)
    num_metrics = len(metric_names)
    bar_positions = np.arange(num_groups)
    bar_width = 0.25

    fig, axis = plt.subplots(figsize=(14, 7))

    for metric_idx, (metric_name, display_name) in enumerate(
        zip(metric_names, metric_display)
    ):
        offset = (metric_idx - num_metrics / 2 + 0.5) * bar_width
        values = metric_values[metric_name]
        color = COMPARISON_COLORS[metric_idx % len(COMPARISON_COLORS)]
        bars = axis.bar(
            bar_positions + offset,
            values,
            bar_width,
            label=display_name,
            color=color,
            alpha=0.85,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    axis.set_xlabel("Model Architecture", fontsize=12)
    axis.set_ylabel("Metric Value", fontsize=12)
    axis.set_title(
        "Model Comparison — Summary Metrics",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xticks(bar_positions)
    axis.set_xticklabels(backbones, fontsize=10)
    axis.legend(fontsize=10)
    axis.grid(True, alpha=0.3, axis="y")

    plot_path = output_path / "comparison_summary.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary bar chart saved to %s", plot_path)
