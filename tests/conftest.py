"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch


@pytest.fixture(scope="session")
def test_device() -> torch.device:
    """Get test device (CPU for consistent testing)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create temporary MLflow tracking URI."""
    tracking_dir = tmp_path_factory.mktemp("mlflow")
    return f"file://{tracking_dir}"


@pytest.fixture
def set_test_env(
    monkeypatch: pytest.MonkeyPatch, temp_data_dir: Path, mlflow_tracking_uri: str
) -> None:
    """Set test environment variables (not auto-used, call explicitly when needed)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_tracking_uri)
    monkeypatch.setenv("DATA_DIR", str(temp_data_dir))
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("BATCH_SIZE", "4")
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("NUM_WORKERS", "0")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")


@pytest.fixture
def sample_image_batch(test_device: torch.device) -> torch.Tensor:
    """Create sample image batch for testing."""
    return torch.randn(4, 3, 32, 32, device=test_device)


@pytest.fixture
def sample_labels(test_device: torch.device) -> torch.Tensor:
    """Create sample labels for testing."""
    return torch.randint(0, 10, (4,), device=test_device)


@pytest.fixture
def mock_cifar10_data(temp_data_dir: Path) -> Path:
    """Create mock CIFAR-10 data directory structure."""
    cifar_dir = temp_data_dir / "cifar-10-batches-py"
    cifar_dir.mkdir(parents=True, exist_ok=True)
    return cifar_dir
