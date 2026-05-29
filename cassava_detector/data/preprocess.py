"""Preprocess raw data into train/validation/test splits."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

CLASS_LABELS: dict[int, str] = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}


def preprocess(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    val_split: float = 0.15,
    test_split: float = 0.10,
    random_seed: int = 42,
) -> Path:
    """Split raw data into stratified train/validation/test sets.

    Reads the ``train.csv`` file from the raw data directory, performs
    a two-stage stratified split, and writes the resulting CSVs to the
    processed data directory.

    Args:
        raw_data_dir: Path to the directory with raw dataset files.
        processed_data_dir: Destination directory for split CSV files.
        val_split: Fraction of data reserved for validation.
        test_split: Fraction of data reserved for testing.
        random_seed: Random seed for reproducible splitting.

    Returns:
        Path to the processed data directory.

    Raises:
        FileNotFoundError: If ``train.csv`` is missing from raw data.
        ValueError: If split fractions are invalid.
    """
    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_data_dir)

    labels_csv = raw_path / "train.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"Expected train.csv at {labels_csv}. "
            f"Run the download step first."
        )

    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0"
        )

    processed_path.mkdir(parents=True, exist_ok=True)

    logger.info("Reading labels from %s", labels_csv)
    labels_df = pd.read_csv(labels_csv)
    logger.info("Total samples: %d", len(labels_df))
    logger.info("Class distribution:\n%s", labels_df["label"].value_counts().to_string())

    holdout_fraction = val_split + test_split
    train_df, holdout_df = train_test_split(
        labels_df,
        test_size=holdout_fraction,
        stratify=labels_df["label"],
        random_state=random_seed,
    )

    relative_test_fraction = test_split / holdout_fraction
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_fraction,
        stratify=holdout_df["label"],
        random_state=random_seed,
    )

    train_csv_path = processed_path / "train.csv"
    val_csv_path = processed_path / "val.csv"
    test_csv_path = processed_path / "test.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    logger.info("Train samples: %d -> %s", len(train_df), train_csv_path)
    logger.info("Validation samples: %d -> %s", len(val_df), val_csv_path)
    logger.info("Test samples: %d -> %s", len(test_df), test_csv_path)

    return processed_path
