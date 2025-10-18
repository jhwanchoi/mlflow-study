"""Vision model wrapper with support for multiple architectures."""

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from src.config import get_settings

logger = logging.getLogger(__name__)


class VisionModel(nn.Module):
    """Vision model wrapper supporting multiple architectures."""

    def __init__(
        self,
        model_name: Literal["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"],
        num_classes: int,
        pretrained: bool = True,
    ):
        """Initialize vision model.

        Args:
            model_name: Name of the model architecture
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        logger.info(
            f"Initializing {model_name} with {num_classes} classes (pretrained={pretrained})"
        )

        # Load base model
        if model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v3_small(weights=weights)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)

        elif model_name == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v3_large(weights=weights)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)

        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for newly added layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if not self.pretrained or m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)  # type: ignore[no-any-return]

    def get_num_parameters(self) -> dict[str, int]:
        """Get number of model parameters.

        Returns:
            Dictionary with total, trainable, and frozen parameters
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone layers for fine-tuning."""
        if self.model_name.startswith("mobilenet"):
            # Freeze all layers except classifier
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            logger.info("Frozen MobileNet backbone, only classifier is trainable")

        elif self.model_name == "resnet18":
            # Freeze all layers except fc
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
            logger.info("Frozen ResNet backbone, only fc layer is trainable")

        params = self.get_num_parameters()
        logger.info(f"Parameters after freezing: {params}")

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfrozen all model parameters")

        params = self.get_num_parameters()
        logger.info(f"Parameters after unfreezing: {params}")


def create_model() -> VisionModel:
    """Create model from settings.

    Returns:
        Initialized VisionModel instance
    """
    settings = get_settings()

    model = VisionModel(
        model_name=settings.model_name,
        num_classes=settings.num_classes,
        pretrained=settings.pretrained,
    )

    params = model.get_num_parameters()
    logger.info(
        f"Created {settings.model_name}: "
        f"Total={params['total']:,}, Trainable={params['trainable']:,}, Frozen={params['frozen']:,}"
    )

    return model
