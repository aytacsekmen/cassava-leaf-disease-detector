"""Tests for the CassavaDataModule and CassavaDataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from cassava_detector.data.datamodule import (
    CassavaDataModule,
    CassavaDataset,
    collate_skip_none,
)


@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset structure for testing.

    Returns:
        Path to the temporary directory containing CSV files and images.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    for idx in range(10):
        image = Image.new("RGB", (64, 64), color=(idx * 25, 100, 200))
        image.save(image_dir / f"img_{idx:04d}.jpg")

    train_data = {
        "image_id": [f"img_{idx:04d}.jpg" for idx in range(6)],
        "label": [0, 1, 2, 3, 4, 0],
    }
    val_data = {
        "image_id": [f"img_{idx:04d}.jpg" for idx in range(6, 8)],
        "label": [1, 2],
    }
    test_data = {
        "image_id": [f"img_{idx:04d}.jpg" for idx in range(8, 10)],
        "label": [3, 4],
    }

    pd.DataFrame(train_data).to_csv(processed_dir / "train.csv", index=False)
    pd.DataFrame(val_data).to_csv(processed_dir / "val.csv", index=False)
    pd.DataFrame(test_data).to_csv(processed_dir / "test.csv", index=False)

    return tmp_path


def test_dataset_length(sample_data_dir: Path) -> None:
    """Test that the dataset reports the correct number of samples."""
    dataset = CassavaDataset(
        csv_path=sample_data_dir / "processed" / "train.csv",
        image_dir=sample_data_dir / "images",
    )
    assert len(dataset) == 6


def test_dataset_returns_tensor_and_label(sample_data_dir: Path) -> None:
    """Test that each sample is a (tensor, int) tuple."""
    dataset = CassavaDataset(
        csv_path=sample_data_dir / "processed" / "train.csv",
        image_dir=sample_data_dir / "images",
    )
    sample = dataset[0]
    assert sample is not None
    image_tensor, label = sample
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape[0] == 3
    assert isinstance(label, int)


def test_dataset_handles_corrupted_image(sample_data_dir: Path) -> None:
    """Test that a corrupted image returns None instead of crashing."""
    corrupted_path = sample_data_dir / "images" / "img_0000.jpg"
    corrupted_path.write_bytes(b"not a valid image")

    dataset = CassavaDataset(
        csv_path=sample_data_dir / "processed" / "train.csv",
        image_dir=sample_data_dir / "images",
    )
    result = dataset[0]
    assert result is None


def test_collate_skip_none_filters_none() -> None:
    """Test that the collate function filters out None samples."""
    valid_sample_a = (torch.randn(3, 32, 32), 0)
    valid_sample_b = (torch.randn(3, 32, 32), 1)
    batch = [valid_sample_a, None, valid_sample_b, None]

    images, labels = collate_skip_none(batch)
    assert images.shape[0] == 2
    assert labels.shape[0] == 2


def test_collate_skip_none_raises_on_all_none() -> None:
    """Test that the collate function raises when all samples are None."""
    with pytest.raises(RuntimeError, match="All samples in this batch"):
        collate_skip_none([None, None, None])


def test_train_val_no_overlap(sample_data_dir: Path) -> None:
    """Test that train and validation splits have no overlapping samples."""
    train_df = pd.read_csv(sample_data_dir / "processed" / "train.csv")
    val_df = pd.read_csv(sample_data_dir / "processed" / "val.csv")

    train_ids = set(train_df["image_id"].tolist())
    val_ids = set(val_df["image_id"].tolist())

    assert train_ids.isdisjoint(val_ids), "Train and validation sets overlap!"


def test_datamodule_setup(sample_data_dir: Path) -> None:
    """Test that the DataModule sets up datasets correctly."""
    data_module = CassavaDataModule(
        processed_data_dir=str(sample_data_dir / "processed"),
        image_dir=str(sample_data_dir / "images"),
        batch_size=2,
        num_workers=0,
        image_size=64,
    )
    data_module.setup(stage="fit")

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None
    assert len(data_module.train_dataset) == 6
    assert len(data_module.val_dataset) == 2


def test_datamodule_train_dataloader(sample_data_dir: Path) -> None:
    """Test that the training DataLoader yields batches."""
    data_module = CassavaDataModule(
        processed_data_dir=str(sample_data_dir / "processed"),
        image_dir=str(sample_data_dir / "images"),
        batch_size=3,
        num_workers=0,
        image_size=64,
    )
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    assert images.shape[0] <= 3
    assert images.shape[1] == 3
    assert labels.shape[0] == images.shape[0]
