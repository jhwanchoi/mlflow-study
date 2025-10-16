"""Configuration management using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLflow tracking server URI",
    )
    mlflow_s3_endpoint_url: str = Field(
        default="http://localhost:9000",
        description="S3-compatible endpoint for MinIO",
    )
    aws_access_key_id: str = Field(
        default="minio",
        description="AWS access key for MinIO",
    )
    aws_secret_access_key: str = Field(
        default="minio123",
        description="AWS secret key for MinIO",
    )
    experiment_name: str = Field(
        default="vision-model-training",
        description="MLflow experiment name",
    )

    # Model Configuration
    model_name: Literal["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"] = Field(
        default="mobilenet_v3_small",
        description="Model architecture to use",
    )
    num_classes: int = Field(
        default=10,
        description="Number of output classes",
        gt=0,
    )
    pretrained: bool = Field(
        default=True,
        description="Use pretrained weights",
    )

    # Training Configuration
    batch_size: int = Field(
        default=64,
        description="Training batch size",
        gt=0,
    )
    learning_rate: float = Field(
        default=0.001,
        description="Learning rate",
        gt=0.0,
    )
    epochs: int = Field(
        default=20,
        description="Number of training epochs",
        gt=0,
    )
    num_workers: int = Field(
        default=4,
        description="Number of data loading workers",
        ge=0,
    )
    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay for optimizer",
        ge=0.0,
    )
    momentum: float = Field(
        default=0.9,
        description="Momentum for SGD optimizer",
        ge=0.0,
        le=1.0,
    )

    # Device Configuration
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="mps",
        description="Device to use for training (mps for M2 Mac)",
    )

    # Data Configuration
    data_dir: str = Field(
        default="./data",
        description="Directory for dataset storage",
    )
    dataset: Literal["CIFAR10", "CIFAR100", "FashionMNIST"] = Field(
        default="CIFAR10",
        description="Dataset to use",
    )
    train_val_split: float = Field(
        default=0.9,
        description="Train/validation split ratio",
        gt=0.0,
        lt=1.0,
    )

    # Augmentation Configuration
    use_augmentation: bool = Field(
        default=True,
        description="Use data augmentation during training",
    )
    random_crop: bool = Field(
        default=True,
        description="Apply random crop augmentation",
    )
    random_horizontal_flip: bool = Field(
        default=True,
        description="Apply random horizontal flip",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_interval: int = Field(
        default=10,
        description="Log metrics every N batches",
        gt=0,
    )

    # Checkpoint Configuration
    save_best_only: bool = Field(
        default=True,
        description="Save only the best model checkpoint",
    )
    early_stopping_patience: int = Field(
        default=5,
        description="Early stopping patience (epochs)",
        gt=0,
    )

    @property
    def image_size(self) -> tuple[int, int]:
        """Get image size based on dataset."""
        if self.dataset in ["CIFAR10", "CIFAR100"]:
            return (32, 32)
        elif self.dataset == "FashionMNIST":
            return (28, 28)
        return (224, 224)

    @property
    def dataset_mean(self) -> tuple[float, float, float]:
        """Get dataset normalization mean."""
        if self.dataset in ["CIFAR10", "CIFAR100"]:
            return (0.4914, 0.4822, 0.4465)
        return (0.5, 0.5, 0.5)

    @property
    def dataset_std(self) -> tuple[float, float, float]:
        """Get dataset normalization standard deviation."""
        if self.dataset in ["CIFAR10", "CIFAR100"]:
            return (0.2470, 0.2435, 0.2616)
        return (0.5, 0.5, 0.5)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
