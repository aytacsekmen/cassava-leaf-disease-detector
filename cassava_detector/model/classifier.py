"""Cassava leaf disease classifier using a pretrained backbone from timm."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import timm


class CassavaClassifier(pl.LightningModule):
    """PyTorch Lightning module for cassava leaf disease classification.

    Uses a pretrained backbone from the ``timm`` library with a custom
    classification head.  Logs training loss, validation loss, validation
    accuracy, and validation F1 score to the configured logger.

    Args:
        backbone: Name of the timm model to use as feature extractor.
        num_classes: Number of target disease classes.
        pretrained: Whether to load pretrained ImageNet weights.
        dropout_rate: Dropout probability before the final linear layer.
        learning_rate: Initial learning rate for the optimizer.
        weight_decay: L2 regularization coefficient.
        scheduler_t_max: T_max parameter for CosineAnnealingLR scheduler.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnet_b3",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_t_max: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_t_max = scheduler_t_max

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            images: Batch of images with shape ``(B, C, H, W)``.

        Returns:
            Logits tensor with shape ``(B, num_classes)``.
        """
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.

        Returns:
            Scalar training loss.
        """
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Run one validation step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.
        """
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        self.val_accuracy.update(predictions, labels)
        self.val_f1.update(predictions, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Run one test step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the current batch.
        """
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        self.test_accuracy.update(predictions, labels)
        self.test_f1.update(predictions, labels)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)

    def configure_optimizers(self) -> dict:
        """Set up the AdamW optimizer with cosine annealing LR schedule.

        Returns:
            Dictionary with optimizer and LR scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.scheduler_t_max,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
