"""Download the Cassava Leaf Disease dataset from Kaggle."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)

KAGGLE_DATASET_SLUG = "cassava-leaf-disease-classification"
KAGGLE_COMPETITION = "cassava-leaf-disease-classification"


def download_data(raw_data_dir: str = "data/raw") -> Path:
    """Download the Cassava Leaf Disease dataset from Kaggle.

    Uses ``kagglehub`` to fetch the competition data and copies it into
    the local ``raw_data_dir`` directory.  If the data already exists
    locally the download is skipped.

    Args:
        raw_data_dir: Destination directory for the raw dataset files.

    Returns:
        Path to the directory containing the downloaded data.

    Raises:
        RuntimeError: If the download fails or no files are found.
    """
    raw_path = Path(raw_data_dir)

    if raw_path.exists() and any(raw_path.iterdir()):
        logger.info("Raw data already exists at %s, skipping download.", raw_path)
        return raw_path

    raw_path.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Downloading dataset from Kaggle competition: %s", KAGGLE_COMPETITION)
        downloaded_path = kagglehub.competition_download(KAGGLE_COMPETITION)
        downloaded_path = Path(downloaded_path)
        logger.info("Dataset downloaded to temporary path: %s", downloaded_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset from Kaggle. "
            f"Make sure your Kaggle API credentials are configured. Error: {exc}"
        ) from exc

    try:
        for item in downloaded_path.iterdir():
            destination = raw_path / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)
        logger.info("Dataset files copied to %s", raw_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to copy downloaded files to {raw_path}. Error: {exc}"
        ) from exc

    file_count = sum(1 for _ in raw_path.rglob("*") if _.is_file())
    if file_count == 0:
        raise RuntimeError(f"No files found in {raw_path} after download.")

    logger.info("Download complete. %d files available in %s", file_count, raw_path)
    return raw_path
