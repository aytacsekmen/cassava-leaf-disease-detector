"""Download the Cassava Leaf Disease dataset from Google Drive."""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import gdown

logger = logging.getLogger(__name__)

GDRIVE_FILE_ID = "1WujIaxqsQFuKeQxwR6jxPMvGQsuEDBwQ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
DOWNLOADED_ZIP_NAME = "cassava_leaf_disease.zip"


def download_data(raw_data_dir: str = "data/raw") -> Path:
    """Download the Cassava Leaf Disease dataset from Google Drive.

    Uses ``gdown`` to fetch a zip archive from Google Drive and extracts
    it into the local ``raw_data_dir`` directory.  If the data already
    exists locally the download is skipped.

    Args:
        raw_data_dir: Destination directory for the raw dataset files.

    Returns:
        Path to the directory containing the downloaded data.

    Raises:
        RuntimeError: If the download or extraction fails.
    """
    raw_path = Path(raw_data_dir)

    if raw_path.exists() and any(raw_path.iterdir()):
        logger.info("Raw data already exists at %s, skipping download.", raw_path)
        return raw_path

    raw_path.mkdir(parents=True, exist_ok=True)

    zip_path = raw_path / DOWNLOADED_ZIP_NAME

    try:
        logger.info("Downloading dataset from Google Drive (file ID: %s)", GDRIVE_FILE_ID)
        gdown.download(GDRIVE_URL, str(zip_path), quiet=False)
        logger.info("Downloaded zip archive to %s", zip_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download dataset from Google Drive. Error: {exc}"
        ) from exc

    if not zip_path.exists():
        raise RuntimeError(
            f"Download completed but zip file not found at {zip_path}. "
            f"The Google Drive link may be invalid or require access permissions."
        )

    try:
        logger.info("Extracting zip archive to %s", raw_path)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_path)
        logger.info("Extraction complete.")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Downloaded file is not a valid zip archive: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to extract zip archive: {exc}"
        ) from exc

    zip_path.unlink()
    logger.info("Removed zip archive: %s", zip_path)

    _flatten_nested_directory(raw_path)

    file_count = sum(1 for item in raw_path.rglob("*") if item.is_file())
    if file_count == 0:
        raise RuntimeError(f"No files found in {raw_path} after extraction.")

    logger.info("Download complete. %d files available in %s", file_count, raw_path)
    return raw_path


def _flatten_nested_directory(target_dir: Path) -> None:
    """Flatten a single nested subdirectory into its parent.

    If the zip extraction created a single top-level directory, move
    its contents up one level to keep the expected directory layout.

    Args:
        target_dir: Directory to check and potentially flatten.
    """
    children = list(target_dir.iterdir())
    if len(children) == 1 and children[0].is_dir():
        nested_dir = children[0]
        logger.info(
            "Flattening nested directory %s into %s",
            nested_dir.name,
            target_dir,
        )
        for item in nested_dir.iterdir():
            destination = target_dir / item.name
            shutil.move(str(item), str(destination))
        nested_dir.rmdir()
