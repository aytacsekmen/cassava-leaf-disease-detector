"""Single CLI entry point for the cassava leaf disease detection pipeline.

Uses ``fire`` for command dispatch and ``hydra`` (Compose API) for
configuration management.  All pipeline stages are accessible as
subcommands: ``download``, ``preprocess``, ``train``, and ``infer``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from cassava_detector.data.download import download_data
from cassava_detector.data.preprocess import preprocess as run_preprocess
from cassava_detector.inference.predict import infer as run_infer
from cassava_detector.training.trainer import train as run_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = str(PROJECT_ROOT / "configs")


def _load_config(overrides: Optional[list[str]] = None) -> DictConfig:
    """Load and merge all Hydra configuration files.

    Uses the Hydra Compose API to load configs from the ``configs/``
    directory without the decorator-based approach.

    Args:
        overrides: Optional list of Hydra-style overrides
            (e.g., ``["training.batch_size=64"]``).

    Returns:
        Merged OmegaConf DictConfig with all configuration sections.
    """
    resolved_overrides = overrides or []

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        preprocessing_cfg = compose(
            config_name="preprocessing",
            overrides=[
                override
                for override in resolved_overrides
                if override.startswith("preprocessing.")
            ],
        )
        model_cfg = compose(
            config_name="model",
            overrides=[
                override
                for override in resolved_overrides
                if override.startswith("model.")
            ],
        )
        training_cfg = compose(
            config_name="training",
            overrides=[
                override
                for override in resolved_overrides
                if override.startswith("training.")
            ],
        )
        mlflow_cfg = compose(
            config_name="mlflow",
            overrides=[
                override
                for override in resolved_overrides
                if override.startswith("mlflow.")
            ],
        )

    merged = OmegaConf.create({
        "preprocessing": OmegaConf.to_container(preprocessing_cfg),
        "model": OmegaConf.to_container(model_cfg),
        "training": OmegaConf.to_container(training_cfg),
        "mlflow": OmegaConf.to_container(mlflow_cfg),
    })

    return merged


class Commands:
    """CLI commands for the cassava leaf disease detection pipeline.

    Each method corresponds to a pipeline stage.  Configuration is
    automatically loaded from YAML files in the ``configs/`` directory.
    """

    def download(self, **kwargs: Any) -> None:
        """Download the Cassava Leaf Disease dataset from Kaggle.

        Keyword Args:
            Any Hydra-style overrides (e.g., raw_data_dir="data/raw").
        """
        config = _load_config()
        raw_data_dir = kwargs.get(
            "raw_data_dir", config.preprocessing.raw_data_dir
        )
        result = download_data(raw_data_dir=raw_data_dir)
        logger.info("Data downloaded to: %s", result)

    def preprocess(self, **kwargs: Any) -> None:
        """Split raw data into train/validation/test sets.

        Keyword Args:
            Any Hydra-style overrides for preprocessing config values.
        """
        config = _load_config()
        preprocessing_cfg = config.preprocessing

        result = run_preprocess(
            raw_data_dir=kwargs.get("raw_data_dir", preprocessing_cfg.raw_data_dir),
            processed_data_dir=kwargs.get(
                "processed_data_dir", preprocessing_cfg.processed_data_dir
            ),
            val_split=kwargs.get("val_split", preprocessing_cfg.val_split),
            test_split=kwargs.get("test_split", preprocessing_cfg.test_split),
            random_seed=kwargs.get("random_seed", preprocessing_cfg.random_seed),
        )
        logger.info("Preprocessing complete. Output at: %s", result)

    def train(self, **kwargs: Any) -> None:
        """Run the full training pipeline.

        Keyword Args:
            overrides: List of Hydra-style config overrides.
        """
        overrides = kwargs.get("overrides", [])
        if isinstance(overrides, str):
            overrides = [overrides]
        config = _load_config(overrides=overrides)
        logger.info("Resolved config:\n%s", OmegaConf.to_yaml(config))
        run_train(config)

    def infer(
        self,
        checkpoint_path: str,
        image_path: str,
        **kwargs: Any,
    ) -> None:
        """Run inference on a single image.

        Args:
            checkpoint_path: Path to the trained model checkpoint.
            image_path: Path to the image to classify.

        Keyword Args:
            image_size: Override for image resize dimension.
        """
        config = _load_config()
        result = run_infer(
            checkpoint_path=checkpoint_path,
            image_path=image_path,
            image_size=kwargs.get("image_size", config.preprocessing.image_size),
            mean=list(config.preprocessing.mean),
            std=list(config.preprocessing.std),
        )
        logger.info("Inference result: %s", result)


def main() -> None:
    """Dispatch CLI commands via fire."""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
