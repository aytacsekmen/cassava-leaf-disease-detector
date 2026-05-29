"""Inference logic for predicting cassava leaf diseases from images."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from cassava_detector.data.preprocess import CLASS_LABELS
from cassava_detector.model.classifier import CassavaClassifier

logger = logging.getLogger(__name__)


def infer(
    checkpoint_path: str,
    image_path: str,
    image_size: int = 384,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> dict[str, str | float]:
    """Run inference on a single image using a trained checkpoint.

    Args:
        checkpoint_path: Path to the ``.ckpt`` model checkpoint file.
        image_path: Path to the input image.
        image_size: Size to resize the image to before inference.
        mean: Channel-wise mean for normalization.
        std: Channel-wise standard deviation for normalization.

    Returns:
        Dictionary containing:
            - ``predicted_class``: Integer class label.
            - ``predicted_label``: Human-readable disease name.
            - ``confidence``: Prediction confidence score.

    Raises:
        FileNotFoundError: If the checkpoint or image file does not exist.
    """
    checkpoint_file = Path(checkpoint_path)
    image_file = Path(image_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    norm_mean = mean or [0.485, 0.456, 0.406]
    norm_std = std or [0.229, 0.224, 0.225]

    inference_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    logger.info("Loading model from %s", checkpoint_file)
    model = CassavaClassifier.load_from_checkpoint(str(checkpoint_file))
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    try:
        image = Image.open(image_file).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to open image {image_file}: {exc}") from exc

    input_tensor = inference_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_index = predicted_class.item()
    class_label = CLASS_LABELS.get(class_index, f"Unknown ({class_index})")
    confidence_score = confidence.item()

    logger.info(
        "Prediction: %s (class %d) with confidence %.2f%%",
        class_label,
        class_index,
        confidence_score * 100,
    )

    return {
        "predicted_class": class_index,
        "predicted_label": class_label,
        "confidence": round(confidence_score, 4),
    }
