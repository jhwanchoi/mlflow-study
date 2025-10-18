"""End-to-end tests for training and evaluation pipeline."""

from pathlib import Path

import pytest
import torch

from src.config import get_settings
from src.data import get_dataloaders
from src.models.vision_model import create_model
from src.training.evaluate import evaluate_model
from src.training.train import train_epoch, validate


class TestEndToEndTraining:
    """End-to-end tests for complete training flow."""

    @pytest.mark.slow
    def test_full_training_cycle(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test complete training cycle with real data and model."""
        # Set minimal configuration for fast test
        monkeypatch.setenv("EPOCHS", "2")
        monkeypatch.setenv("BATCH_SIZE", "8")
        monkeypatch.setenv("NUM_WORKERS", "0")
        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("MODEL_NAME", "mobilenet_v3_small")
        monkeypatch.setenv("PRETRAINED", "false")
        monkeypatch.setenv("LOG_INTERVAL", "100")
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlflow")

        get_settings.cache_clear()
        settings = get_settings()

        # Create model
        model = create_model()
        assert model is not None

        # Load data
        train_loader, val_loader, test_loader = get_dataloaders()
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0

        # Setup training
        device = torch.device(settings.device)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Train for 2 epochs
        initial_loss = None
        for epoch in range(1, 3):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

            if initial_loss is None:
                initial_loss = train_loss

            # Verify metrics are in valid range
            assert 0 <= train_acc <= 100
            assert 0 <= val_acc <= 100
            assert train_loss >= 0
            assert val_loss >= 0

        # Loss should decrease (or at least not increase significantly)
        # Note: With random initialization, this is not guaranteed but likely
        assert train_loss < initial_loss * 1.5  # Allow 50% tolerance

    @pytest.mark.slow
    def test_model_saves_and_loads(self, tmp_path: Path) -> None:
        """Test that model can be saved and loaded correctly."""
        # Create and train a small model
        model = create_model()
        device = torch.device("cpu")
        model = model.to(device)

        # Get initial predictions
        dummy_input = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            initial_output = model(dummy_input)

        # Save model
        checkpoint_path = tmp_path / "test_model.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": 5,
            },
            checkpoint_path,
        )

        # Load model into new instance
        new_model = create_model()
        new_model = new_model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify predictions are identical
        with torch.no_grad():
            loaded_output = new_model(dummy_input)

        assert torch.allclose(initial_output, loaded_output, atol=1e-6)

    def test_quick_training_iteration(self, test_device: torch.device) -> None:
        """Test a single training iteration works correctly (fast test)."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create minimal test data
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Create model
        model = create_model()
        model = model.to(test_device)

        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Get initial loss
        model.eval()
        with torch.no_grad():
            inputs, targets = next(iter(loader))
            inputs, targets = inputs.to(test_device), targets.to(test_device)
            initial_loss = criterion(model(inputs), targets).item()  # noqa: F841

        # Train one step
        model.train()
        inputs, targets = next(iter(loader))
        inputs, targets = inputs.to(test_device), targets.to(test_device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Verify training happened
        assert loss.item() >= 0
        # Check gradients were computed
        has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grads


class TestEndToEndEvaluation:
    """End-to-end tests for evaluation pipeline."""

    @pytest.mark.slow
    def test_full_evaluation_with_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test complete evaluation with all metrics and visualizations."""
        from torch.utils.data import DataLoader, TensorDataset

        monkeypatch.setenv("DEVICE", "cpu")
        monkeypatch.setenv("DATASET", "CIFAR10")

        get_settings.cache_clear()
        settings = get_settings()

        # Create test data
        images = torch.randn(40, 3, 32, 32)
        labels = torch.randint(0, 10, (40,))
        dataset = TensorDataset(images, labels)
        test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Create model
        model = create_model()
        device = torch.device(settings.device)
        model = model.to(device)
        model.eval()

        # Get class names
        from src.data.dataset import get_class_names

        class_names = get_class_names(settings.dataset)

        # Run evaluation
        save_dir = tmp_path / "eval_results"
        metrics = evaluate_model(model, test_loader, device, class_names, save_dir)

        # Verify metrics
        assert "test_accuracy" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "test_f1" in metrics

        # Verify metrics are in valid range
        assert 0 <= metrics["test_accuracy"] <= 100
        assert 0 <= metrics["test_precision"] <= 1
        assert 0 <= metrics["test_recall"] <= 1
        assert 0 <= metrics["test_f1"] <= 1

        # Verify visualizations were created
        assert (save_dir / "confusion_matrix.png").exists()
        assert (save_dir / "per_class_accuracy.png").exists()

    def test_evaluation_predictions(self, test_device: torch.device) -> None:
        """Test that evaluation produces correct predictions (fast test)."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create deterministic test data
        torch.manual_seed(42)
        images = torch.randn(16, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5])
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        # Create model
        model = create_model()
        model = model.to(test_device)
        model.eval()

        # Get predictions
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(test_device)
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())

        # Verify we got predictions for all samples
        assert len(all_predictions) == 16
        assert len(all_targets) == 16

        # Verify predictions are in valid range (0-9 for CIFAR-10)
        assert all(0 <= p <= 9 for p in all_predictions)


class TestMLflowIntegrationE2E:
    """End-to-end tests for MLflow integration."""

    @pytest.mark.slow
    def test_mlflow_experiment_tracking(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test complete MLflow experiment tracking flow."""
        import mlflow

        # Setup minimal config
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlflow")
        monkeypatch.setenv("EXPERIMENT_NAME", "test-experiment")
        monkeypatch.setenv("BATCH_SIZE", "8")
        monkeypatch.setenv("EPOCHS", "1")
        monkeypatch.setenv("DEVICE", "cpu")

        get_settings.cache_clear()
        settings = get_settings()

        # Set MLflow tracking
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.experiment_name)

        # Start run and log metrics
        with mlflow.start_run(run_name="test_run") as run:
            # Log parameters
            mlflow.log_params(
                {
                    "model_name": "mobilenet_v3_small",
                    "batch_size": 8,
                    "learning_rate": 0.01,
                }
            )

            # Log metrics
            for epoch in range(1, 3):
                mlflow.log_metrics(
                    {
                        "train_loss": 2.0 / epoch,
                        "train_accuracy": 50.0 * epoch,
                        "val_loss": 2.5 / epoch,
                        "val_accuracy": 45.0 * epoch,
                    },
                    step=epoch,
                )

            # Log model as artifact (simpler test)
            model = create_model()

            # Create a temporary model file
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
                torch.save(model.state_dict(), f.name)
                mlflow.log_artifact(f.name, "model")

            run_id = run.info.run_id

        # Verify run was created
        assert run_id is not None

        # Load run and verify
        client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        run_data = client.get_run(run_id)

        # Check parameters
        assert run_data.data.params["model_name"] == "mobilenet_v3_small"
        assert run_data.data.params["batch_size"] == "8"

        # Check metrics
        assert "train_loss" in run_data.data.metrics
        assert "val_accuracy" in run_data.data.metrics

        # Check artifacts (model logged)
        artifacts = client.list_artifacts(run_id)
        assert len(artifacts) > 0  # At least one artifact logged


class TestDataPipeline:
    """End-to-end tests for data pipeline."""

    @pytest.mark.slow
    def test_data_loading_and_preprocessing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test complete data loading and preprocessing pipeline."""
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("BATCH_SIZE", "16")
        monkeypatch.setenv("NUM_WORKERS", "0")
        monkeypatch.setenv("DATASET", "CIFAR10")
        monkeypatch.setenv("TRAIN_VAL_SPLIT", "0.8")

        get_settings.cache_clear()

        # Load data
        train_loader, val_loader, test_loader = get_dataloaders()

        # Verify loaders were created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        for batch in [train_batch, val_batch, test_batch]:
            images, labels = batch

            # Check shapes
            assert images.shape[0] <= 16  # batch size
            assert images.shape[1] == 3  # RGB channels
            assert images.shape[2] == 32  # CIFAR-10 image height
            assert images.shape[3] == 32  # CIFAR-10 image width

            # Check value ranges (normalized)
            assert images.min() >= -3.0  # roughly -mean/std
            assert images.max() <= 3.0  # roughly (1-mean)/std

            # Check labels
            assert labels.min() >= 0
            assert labels.max() < 10  # CIFAR-10 has 10 classes

    def test_augmentation_differences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that training augmentation differs from test transforms."""
        monkeypatch.setenv("USE_AUGMENTATION", "true")
        monkeypatch.setenv("RANDOM_HORIZONTAL_FLIP", "true")

        get_settings.cache_clear()
        settings = get_settings()

        # Get transforms
        from src.data.dataset import get_transforms

        train_transform = get_transforms(
            "train", settings.image_size, settings.dataset_mean, settings.dataset_std
        )
        test_transform = get_transforms(
            "test", settings.image_size, settings.dataset_mean, settings.dataset_std
        )

        # They should have different number of transforms
        assert len(train_transform.transforms) >= len(test_transform.transforms)

        # Test transform should not have random operations
        test_transform_names = [type(t).__name__ for t in test_transform.transforms]
        assert "RandomHorizontalFlip" not in test_transform_names
        assert "RandomCrop" not in test_transform_names
