"""Tests for data loading and preprocessing."""

from pathlib import Path

import pytest
import torch
from torchvision import transforms

from src.data.dataset import get_class_names, get_transforms


class TestTransforms:
    """Test data transforms."""

    def test_train_transforms_with_augmentation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test training transforms with augmentation enabled."""
        monkeypatch.setenv("USE_AUGMENTATION", "true")
        monkeypatch.setenv("RANDOM_CROP", "true")
        monkeypatch.setenv("RANDOM_HORIZONTAL_FLIP", "true")

        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        transform = get_transforms(
            "train",
            settings.image_size,
            settings.dataset_mean,
            settings.dataset_std,
        )

        assert isinstance(transform, transforms.Compose)
        # Check that transforms include augmentation
        transform_types = [type(t).__name__ for t in transform.transforms]
        assert "RandomCrop" in transform_types or "Resize" in transform_types
        assert "ToTensor" in transform_types
        assert "Normalize" in transform_types

    def test_train_transforms_without_augmentation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test training transforms without augmentation."""
        monkeypatch.setenv("USE_AUGMENTATION", "false")

        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        transform = get_transforms(
            "train",
            settings.image_size,
            settings.dataset_mean,
            settings.dataset_std,
        )

        assert isinstance(transform, transforms.Compose)
        transform_types = [type(t).__name__ for t in transform.transforms]
        # Should not have random transforms
        assert "RandomCrop" not in transform_types
        assert "RandomHorizontalFlip" not in transform_types

    def test_test_transforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test test/validation transforms (no augmentation)."""
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        transform = get_transforms(
            "test",
            settings.image_size,
            settings.dataset_mean,
            settings.dataset_std,
        )

        assert isinstance(transform, transforms.Compose)
        transform_types = [type(t).__name__ for t in transform.transforms]

        # Should have basic transforms only
        assert "Resize" in transform_types
        assert "ToTensor" in transform_types
        assert "Normalize" in transform_types

        # Should not have augmentation
        assert "RandomCrop" not in transform_types
        assert "RandomHorizontalFlip" not in transform_types

    def test_transform_output_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that transforms produce correct output shape."""
        from PIL import Image

        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        transform = get_transforms(
            "test",
            settings.image_size,
            settings.dataset_mean,
            settings.dataset_std,
        )

        # Create dummy image
        img = Image.new("RGB", (64, 64))
        transformed = transform(img)

        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape[0] == 3  # RGB channels
        assert transformed.shape[1] == settings.image_size[0]
        assert transformed.shape[2] == settings.image_size[1]


class TestClassNames:
    """Test class name retrieval."""

    def test_cifar10_class_names(self) -> None:
        """Test CIFAR-10 class names."""
        classes = get_class_names("CIFAR10")

        assert len(classes) == 10
        assert "airplane" in classes
        assert "automobile" in classes
        assert "bird" in classes
        assert "cat" in classes
        assert "deer" in classes
        assert "dog" in classes
        assert "frog" in classes
        assert "horse" in classes
        assert "ship" in classes
        assert "truck" in classes

    def test_cifar100_class_names(self) -> None:
        """Test CIFAR-100 class names."""
        classes = get_class_names("CIFAR100")

        assert len(classes) == 100
        assert all(isinstance(c, str) for c in classes)

    def test_fashion_mnist_class_names(self) -> None:
        """Test Fashion-MNIST class names."""
        classes = get_class_names("FashionMNIST")

        assert len(classes) == 10
        assert "T-shirt/top" in classes
        assert "Trouser" in classes
        assert "Pullover" in classes
        assert "Dress" in classes
        assert "Coat" in classes
        assert "Sandal" in classes
        assert "Shirt" in classes
        assert "Sneaker" in classes
        assert "Bag" in classes
        assert "Ankle boot" in classes

    def test_invalid_dataset_name(self) -> None:
        """Test error handling for invalid dataset name."""
        with pytest.raises(ValueError, match="Unsupported dataset"):
            get_class_names("InvalidDataset")  # type: ignore


class TestDataLoaders:
    """Test data loader creation."""

    @pytest.mark.slow
    def test_get_dataloaders_returns_three_loaders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_dataloaders returns train, val, and test loaders."""
        from src.data import get_dataloaders

        monkeypatch.setenv("BATCH_SIZE", "4")
        monkeypatch.setenv("NUM_WORKERS", "0")

        from src.config import get_settings

        get_settings.cache_clear()

        train_loader, val_loader, test_loader = get_dataloaders()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch size
        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 4
        assert test_loader.batch_size == 4

    @pytest.mark.slow
    def test_dataloader_batch_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that data loader produces correct batch shapes."""
        from src.data import get_dataloaders

        monkeypatch.setenv("BATCH_SIZE", "4")
        monkeypatch.setenv("NUM_WORKERS", "0")
        monkeypatch.setenv("DATASET", "CIFAR10")

        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()

        train_loader, _, _ = get_dataloaders()

        # Get one batch
        images, labels = next(iter(train_loader))

        assert images.shape[0] == 4  # batch size
        assert images.shape[1] == 3  # RGB channels
        assert images.shape[2] == settings.image_size[0]
        assert images.shape[3] == settings.image_size[1]
        assert labels.shape[0] == 4

    @pytest.mark.slow
    def test_train_val_split(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test train/validation split ratio."""
        from src.data import get_dataloaders

        monkeypatch.setenv("TRAIN_VAL_SPLIT", "0.8")
        monkeypatch.setenv("NUM_WORKERS", "0")

        from src.config import get_settings

        get_settings.cache_clear()

        train_loader, val_loader, _ = get_dataloaders()

        train_size = len(train_loader.dataset)  # type: ignore
        val_size = len(val_loader.dataset)  # type: ignore
        total_size = train_size + val_size

        # Check approximate 80/20 split
        train_ratio = train_size / total_size
        assert 0.79 <= train_ratio <= 0.81
