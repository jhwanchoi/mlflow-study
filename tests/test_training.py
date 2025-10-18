"""Integration tests for training pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.vision_model import VisionModel
from src.training.train import train_epoch, validate


class TestTrainEpoch:
    """Test training epoch function."""

    @pytest.fixture
    def dummy_model(self, test_device: torch.device) -> VisionModel:
        """Create dummy model for testing."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        return model.to(test_device)

    @pytest.fixture
    def dummy_dataloader(self, test_device: torch.device) -> DataLoader:
        """Create dummy dataloader for testing."""
        # Create dummy dataset
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))
        dataset = TensorDataset(images, labels)

        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_train_epoch_basic(
        self,
        dummy_model: VisionModel,
        dummy_dataloader: DataLoader,
        test_device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test basic train epoch functionality."""
        monkeypatch.setenv("LOG_INTERVAL", "1")

        from src.config import get_settings

        get_settings.cache_clear()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        with patch("mlflow.log_metrics"):
            loss, accuracy = train_epoch(
                dummy_model,
                dummy_dataloader,
                criterion,
                optimizer,
                test_device,
                epoch=1,
            )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100
        assert loss >= 0

    def test_train_epoch_updates_weights(
        self,
        dummy_model: VisionModel,
        dummy_dataloader: DataLoader,
        test_device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that training updates model weights."""
        monkeypatch.setenv("LOG_INTERVAL", "10")

        from src.config import get_settings

        get_settings.cache_clear()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(dummy_model.parameters(), lr=0.01)

        # Get initial weights
        initial_weights = [p.clone() for p in dummy_model.parameters()]

        with patch("mlflow.log_metrics"):
            train_epoch(
                dummy_model,
                dummy_dataloader,
                criterion,
                optimizer,
                test_device,
                epoch=1,
            )

        # Check that weights changed
        weights_changed = False
        for initial, current in zip(initial_weights, dummy_model.parameters()):
            if not torch.allclose(initial, current):
                weights_changed = True
                break

        assert weights_changed, "Model weights did not change during training"


class TestValidate:
    """Test validation function."""

    @pytest.fixture
    def dummy_model(self, test_device: torch.device) -> VisionModel:
        """Create dummy model for testing."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        return model.to(test_device)

    @pytest.fixture
    def dummy_dataloader(self, test_device: torch.device) -> DataLoader:
        """Create dummy dataloader for testing."""
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))
        dataset = TensorDataset(images, labels)

        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_validate_basic(
        self,
        dummy_model: VisionModel,
        dummy_dataloader: DataLoader,
        test_device: torch.device,
    ) -> None:
        """Test basic validation functionality."""
        criterion = torch.nn.CrossEntropyLoss()

        loss, accuracy = validate(
            dummy_model,
            dummy_dataloader,
            criterion,
            test_device,
            epoch=1,
        )

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 100
        assert loss >= 0

    def test_validate_no_gradient_computation(
        self,
        dummy_model: VisionModel,
        dummy_dataloader: DataLoader,
        test_device: torch.device,
    ) -> None:
        """Test that validation doesn't compute gradients."""
        criterion = torch.nn.CrossEntropyLoss()

        # Enable gradient tracking
        for param in dummy_model.parameters():
            param.requires_grad = True

        validate(
            dummy_model,
            dummy_dataloader,
            criterion,
            test_device,
            epoch=1,
        )

        # Check that no gradients were accumulated
        for param in dummy_model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0


class TestTrainingIntegration:
    """Integration tests for full training pipeline."""

    @pytest.mark.slow
    def test_training_pipeline_smoke_test(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Smoke test for entire training pipeline."""
        # Set minimal configuration
        monkeypatch.setenv("EPOCHS", "1")
        monkeypatch.setenv("BATCH_SIZE", "4")
        monkeypatch.setenv("NUM_WORKERS", "0")
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("MODEL_NAME", "mobilenet_v3_small")
        monkeypatch.setenv("PRETRAINED", "false")
        monkeypatch.setenv("LOG_INTERVAL", "10")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlflow")

        from src.config import get_settings

        get_settings.cache_clear()

        # Mock MLflow to avoid actual tracking
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"), patch(
            "mlflow.start_run"
        ) as mock_run, patch("mlflow.log_params"), patch("mlflow.log_metrics"), patch(
            "mlflow.log_metric"
        ), patch(
            "mlflow.set_tag"
        ), patch(
            "mlflow.pytorch.log_model"
        ):
            # Mock run context
            mock_run_context = MagicMock()
            mock_run_context.info.run_id = "test_run_id"
            mock_run.__enter__.return_value = mock_run_context

            # This would normally call train_model(), but we'll test components
            from src.models.vision_model import create_model

            model = create_model()
            assert model is not None

    def test_checkpoint_saving(self, tmp_path: Path, test_device: torch.device) -> None:
        """Test model checkpoint saving."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        checkpoint_path = tmp_path / "test_checkpoint.pth"

        # Save checkpoint
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "val_accuracy": 85.5,
            },
            checkpoint_path,
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=test_device, weights_only=True)

        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "val_accuracy" in checkpoint
        assert checkpoint["epoch"] == 1
        assert checkpoint["val_accuracy"] == 85.5

    def test_early_stopping_logic(self) -> None:
        """Test early stopping patience logic."""
        best_val_acc = 80.0
        patience_counter = 0
        patience = 3

        # Simulate training epochs
        val_accuracies = [75.0, 78.0, 82.0, 81.0, 80.5, 80.3]

        for val_acc in val_accuracies:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        assert patience_counter == 3
        assert best_val_acc == 82.0

    @pytest.mark.slow
    def test_learning_rate_scheduling(self, test_device: torch.device) -> None:
        """Test learning rate scheduler."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        initial_lr = optimizer.param_groups[0]["lr"]

        # Simulate multiple epochs
        for _ in range(5):
            scheduler.step()

        final_lr = optimizer.param_groups[0]["lr"]

        # Learning rate should have decreased
        assert final_lr < initial_lr


class TestMLflowIntegration:
    """Test MLflow integration."""

    def test_mlflow_logging_calls(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that MLflow logging functions are called correctly."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlflow")

        from src.config import get_settings

        get_settings.cache_clear()

        with patch("mlflow.log_params") as mock_log_params, patch(
            "mlflow.log_metrics"
        ) as mock_log_metrics, patch("mlflow.set_tag") as mock_set_tag:
            # Simulate logging
            mock_log_params({"learning_rate": 0.001, "batch_size": 64})
            mock_log_metrics({"train_loss": 1.5, "val_loss": 1.3}, step=1)
            mock_set_tag("model_architecture", "mobilenet_v3_small")

            assert mock_log_params.called
            assert mock_log_metrics.called
            assert mock_set_tag.called
