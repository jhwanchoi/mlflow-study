"""Dataset loading and preprocessing module."""

import logging
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_transforms(
    mode: Literal["train", "test"],
    image_size: Tuple[int, int],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> transforms.Compose:
    """Get data transforms for training or testing.

    Args:
        mode: 'train' or 'test' mode
        image_size: Target image size (height, width)
        mean: Normalization mean values
        std: Normalization standard deviation values

    Returns:
        Composed transforms
    """
    settings = get_settings()

    if mode == "train" and settings.use_augmentation:
        transform_list = [
            transforms.Resize(image_size),
        ]

        if settings.random_crop:
            transform_list.append(transforms.RandomCrop(image_size, padding=4))

        if settings.random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

    return transforms.Compose(transform_list)


def get_dataset(
    dataset_name: Literal["CIFAR10", "CIFAR100", "FashionMNIST"],
    data_dir: Path,
    train: bool,
    transform: transforms.Compose,
) -> Dataset:
    """Get PyTorch dataset.

    Args:
        dataset_name: Name of the dataset
        data_dir: Directory to store/load dataset
        train: Whether to load training or test set
        transform: Data transforms to apply

    Returns:
        PyTorch Dataset instance
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "CIFAR10":
        return datasets.CIFAR10(  # type: ignore[no-any-return]
            root=str(data_dir), train=train, download=True, transform=transform
        )
    elif dataset_name == "CIFAR100":
        return datasets.CIFAR100(  # type: ignore[no-any-return]
            root=str(data_dir), train=train, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        return datasets.FashionMNIST(  # type: ignore[no-any-return]
            root=str(data_dir), train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    settings = get_settings()

    # Get image size and normalization parameters
    image_size = settings.image_size
    mean = settings.dataset_mean
    std = settings.dataset_std

    logger.info(f"Loading {settings.dataset} dataset from {settings.data_dir}")
    logger.info(f"Image size: {image_size}, Mean: {mean}, Std: {std}")

    # Get transforms
    train_transform = get_transforms("train", image_size, mean, std)
    test_transform = get_transforms("test", image_size, mean, std)

    # Load datasets
    data_dir = Path(settings.data_dir)
    train_val_dataset = get_dataset(
        settings.dataset, data_dir, train=True, transform=train_transform
    )
    test_dataset = get_dataset(settings.dataset, data_dir, train=False, transform=test_transform)

    # Split train into train and validation
    dataset_len = len(train_val_dataset)  # type: ignore[arg-type]
    train_size = int(settings.train_val_split * dataset_len)
    val_size = dataset_len - train_size

    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, "  # type: ignore[arg-type]
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"  # type: ignore[arg-type]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_class_names(dataset_name: Literal["CIFAR10", "CIFAR100", "FashionMNIST"]) -> list[str]:
    """Get class names for the dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of class names
    """
    if dataset_name == "CIFAR10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    elif dataset_name == "CIFAR100":
        return [f"class_{i}" for i in range(100)]
    elif dataset_name == "FashionMNIST":
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
