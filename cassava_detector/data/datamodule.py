"""PyTorch Lightning DataModule for the Cassava Leaf Disease dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class CassavaDataset(Dataset):
    """Dataset that loads cassava leaf images and their disease labels.

    Handles corrupted or missing images gracefully by returning ``None``
    for bad samples.  Use :func:`collate_skip_none` as the DataLoader
    collate function to filter these out automatically.

    Args:
        csv_path: Path to a CSV file with ``image_id`` and ``label`` columns.
        image_dir: Directory containing the image files.
        transform: Optional torchvision transforms to apply.
    """

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        try:
            dataframe = pd.read_csv(csv_path)
            self.image_ids: list[str] = dataframe["image_id"].tolist()
            self.labels: list[int] = dataframe["label"].tolist()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read CSV at {csv_path}: {exc}"
            ) from exc

        logger.info(
            "CassavaDataset initialized with %d samples from %s",
            len(self.image_ids),
            csv_path,
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Optional[tuple[torch.Tensor, int]]:
        """Load a single image and its label.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_tensor, label) or ``None`` if the image
            could not be loaded.
        """
        image_id = self.image_ids[index]
        label = self.labels[index]
        image_path = self.image_dir / image_id

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            logger.warning(
                "Skipping corrupted image %s (index %d): %s",
                image_path,
                index,
                exc,
            )
            return None

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, label


def collate_skip_none(
    batch: list[Optional[tuple[torch.Tensor, int]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function that filters out ``None`` samples.

    This prevents a single corrupted image from crashing the entire
    training run.

    Args:
        batch: List of (image, label) tuples, possibly containing None.

    Returns:
        Tuple of stacked image tensors and label tensors.

    Raises:
        RuntimeError: If all samples in the batch are None.
    """
    filtered = [sample for sample in batch if sample is not None]
    if len(filtered) == 0:
        raise RuntimeError("All samples in this batch are corrupted or missing.")

    if len(filtered) < len(batch):
        skipped_count = len(batch) - len(filtered)
        logger.warning("Skipped %d corrupted samples in batch.", skipped_count)

    images, labels = zip(*filtered)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class CassavaDataModule(pl.LightningDataModule):
    """Lightning DataModule that manages train/val/test data loaders.

    All normalization uses the provided mean and std values (typically
    computed from the training set or using ImageNet statistics).
    This ensures no data leakage from validation/test sets.

    Args:
        processed_data_dir: Directory containing train.csv, val.csv, test.csv.
        image_dir: Directory containing the actual image files.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        image_size: Target size for resizing images.
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.
    """

    def __init__(
        self,
        processed_data_dir: str,
        image_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 384,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.processed_data_dir = Path(processed_data_dir)
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.train_dataset: Optional[CassavaDataset] = None
        self.val_dataset: Optional[CassavaDataset] = None
        self.test_dataset: Optional[CassavaDataset] = None

    def _train_transforms(self) -> transforms.Compose:
        """Build augmentation pipeline for training data."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def _eval_transforms(self) -> transforms.Compose:
        """Build transform pipeline for validation and test data."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        """Create dataset instances for each split.

        Args:
            stage: Either ``'fit'``, ``'validate'``, ``'test'``, or ``None``.
        """
        if stage in ("fit", None):
            self.train_dataset = CassavaDataset(
                csv_path=self.processed_data_dir / "train.csv",
                image_dir=self.image_dir,
                transform=self._train_transforms(),
            )
            self.val_dataset = CassavaDataset(
                csv_path=self.processed_data_dir / "val.csv",
                image_dir=self.image_dir,
                transform=self._eval_transforms(),
            )

        if stage in ("test", None):
            self.test_dataset = CassavaDataset(
                csv_path=self.processed_data_dir / "test.csv",
                image_dir=self.image_dir,
                transform=self._eval_transforms(),
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        assert self.train_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_skip_none,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        assert self.val_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_skip_none,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        assert self.test_dataset is not None, "Call setup('test') first."
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_skip_none,
        )
