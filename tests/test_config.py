"""Tests for configuration management."""

import pytest
from pydantic import ValidationError

from src.config import get_settings


class TestSettings:
    """Test Settings configuration."""

    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default settings initialization."""
        # Clear environment to test true defaults
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.mlflow_tracking_uri == "http://localhost:5001"
        assert settings.model_name == "mobilenet_v3_small"
        assert settings.num_classes == 10
        assert settings.batch_size == 64
        assert settings.learning_rate == 0.001
        assert settings.epochs == 20
        assert settings.device == "mps"
        assert settings.dataset == "CIFAR10"

    def test_custom_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test custom settings from environment."""
        monkeypatch.setenv("MODEL_NAME", "resnet18")
        monkeypatch.setenv("BATCH_SIZE", "128")
        monkeypatch.setenv("LEARNING_RATE", "0.01")
        monkeypatch.setenv("EPOCHS", "50")
        monkeypatch.setenv("DEVICE", "cuda")

        # Clear cache to reload settings
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.model_name == "resnet18"
        assert settings.batch_size == 128
        assert settings.learning_rate == 0.01
        assert settings.epochs == 50
        assert settings.device == "cuda"

    def test_invalid_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test validation of invalid model name."""
        monkeypatch.setenv("MODEL_NAME", "invalid_model")

        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()

    def test_invalid_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test validation of invalid batch size."""
        monkeypatch.setenv("BATCH_SIZE", "0")

        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()

    def test_invalid_learning_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test validation of invalid learning rate."""
        monkeypatch.setenv("LEARNING_RATE", "-0.001")

        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()

    def test_image_size_cifar10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test image size for CIFAR-10."""
        monkeypatch.setenv("DATASET", "CIFAR10")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.image_size == (32, 32)

    def test_image_size_fashion_mnist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test image size for Fashion-MNIST."""
        monkeypatch.setenv("DATASET", "FashionMNIST")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.image_size == (28, 28)

    def test_dataset_mean_std(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test dataset normalization parameters."""
        monkeypatch.setenv("DATASET", "CIFAR10")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.dataset_mean == (0.4914, 0.4822, 0.4465)
        assert settings.dataset_std == (0.2470, 0.2435, 0.2616)

    def test_settings_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_valid_device_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test all valid device options."""
        for device in ["cpu", "cuda", "mps"]:
            monkeypatch.setenv("DEVICE", device)
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.device == device

    def test_valid_dataset_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test all valid dataset options."""
        for dataset in ["CIFAR10", "CIFAR100", "FashionMNIST"]:
            monkeypatch.setenv("DATASET", dataset)
            get_settings.cache_clear()
            settings = get_settings()
            assert settings.dataset == dataset

    def test_augmentation_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test augmentation configuration."""
        monkeypatch.setenv("USE_AUGMENTATION", "true")
        monkeypatch.setenv("RANDOM_CROP", "true")
        monkeypatch.setenv("RANDOM_HORIZONTAL_FLIP", "true")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.use_augmentation is True
        assert settings.random_crop is True
        assert settings.random_horizontal_flip is True

    def test_train_val_split_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test train/val split ratio validation."""
        # Valid split
        monkeypatch.setenv("TRAIN_VAL_SPLIT", "0.8")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.train_val_split == 0.8

        # Invalid split (>= 1.0)
        monkeypatch.setenv("TRAIN_VAL_SPLIT", "1.0")
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()

        # Invalid split (<= 0.0)
        monkeypatch.setenv("TRAIN_VAL_SPLIT", "0.0")
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()
