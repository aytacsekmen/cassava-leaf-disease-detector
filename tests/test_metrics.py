"""Tests for metric calculations used in the classifier."""

from __future__ import annotations

import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

NUM_CLASSES = 5


def test_accuracy_perfect_predictions() -> None:
    """Test accuracy is 1.0 when all predictions are correct."""
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
    predictions = torch.tensor([0, 1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3, 4])

    result = accuracy(predictions, targets)
    assert result.item() == pytest.approx(1.0)


def test_accuracy_all_wrong() -> None:
    """Test accuracy is 0.0 when no predictions are correct."""
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
    predictions = torch.tensor([1, 2, 3, 4, 0])
    targets = torch.tensor([0, 1, 2, 3, 4])

    result = accuracy(predictions, targets)
    assert result.item() == pytest.approx(0.0)


def test_f1_score_perfect_predictions() -> None:
    """Test F1 score is 1.0 when all predictions are correct."""
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
    predictions = torch.tensor([0, 1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3, 4])

    result = f1_metric(predictions, targets)
    assert result.item() == pytest.approx(1.0)


def test_f1_score_partial_correctness() -> None:
    """Test F1 score is between 0 and 1 for partial predictions."""
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
    predictions = torch.tensor([0, 1, 2, 0, 0])
    targets = torch.tensor([0, 1, 2, 3, 4])

    result = f1_metric(predictions, targets)
    assert 0.0 < result.item() < 1.0


def test_accuracy_single_class_batch() -> None:
    """Test accuracy with a batch containing only one class."""
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
    predictions = torch.tensor([2, 2, 2, 2])
    targets = torch.tensor([2, 2, 2, 2])

    result = accuracy(predictions, targets)
    assert result.item() > 0.0


def test_metrics_handle_large_batch() -> None:
    """Test that metrics work correctly with a larger batch size."""
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")
    f1_metric = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")

    torch.manual_seed(42)
    batch_size = 256
    predictions = torch.randint(0, NUM_CLASSES, (batch_size,))
    targets = torch.randint(0, NUM_CLASSES, (batch_size,))

    accuracy_result = accuracy(predictions, targets)
    f1_result = f1_metric(predictions, targets)

    assert 0.0 <= accuracy_result.item() <= 1.0
    assert 0.0 <= f1_result.item() <= 1.0


def test_accuracy_accumulation_across_batches() -> None:
    """Test that accuracy accumulates correctly across multiple batches."""
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro")

    batch_one_predictions = torch.tensor([0, 1, 2])
    batch_one_targets = torch.tensor([0, 1, 2])
    accuracy.update(batch_one_predictions, batch_one_targets)

    batch_two_predictions = torch.tensor([3, 4, 0])
    batch_two_targets = torch.tensor([3, 4, 0])
    accuracy.update(batch_two_predictions, batch_two_targets)

    result = accuracy.compute()
    assert result.item() == pytest.approx(1.0)
