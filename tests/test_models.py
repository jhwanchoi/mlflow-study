"""Tests for model architecture."""

import pytest
import torch

from src.models.vision_model import VisionModel, create_model


class TestVisionModel:
    """Test VisionModel class."""

    def test_mobilenet_v3_small_initialization(self, test_device: torch.device) -> None:
        """Test MobileNetV3-Small initialization."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        assert model is not None
        assert model.model_name == "mobilenet_v3_small"
        assert model.num_classes == 10

    def test_mobilenet_v3_large_initialization(self, test_device: torch.device) -> None:
        """Test MobileNetV3-Large initialization."""
        model = VisionModel(
            model_name="mobilenet_v3_large",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        assert model is not None
        assert model.model_name == "mobilenet_v3_large"

    def test_resnet18_initialization(self, test_device: torch.device) -> None:
        """Test ResNet18 initialization."""
        model = VisionModel(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        assert model is not None
        assert model.model_name == "resnet18"

    def test_invalid_model_name(self) -> None:
        """Test error handling for invalid model name."""
        with pytest.raises(ValueError, match="Unsupported model"):
            VisionModel(
                model_name="invalid_model",  # type: ignore
                num_classes=10,
                pretrained=False,
            )

    def test_forward_pass(
        self, test_device: torch.device, sample_image_batch: torch.Tensor
    ) -> None:
        """Test forward pass through model."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)
        model.eval()

        with torch.no_grad():
            output = model(sample_image_batch)

        assert output.shape == (4, 10)  # batch_size, num_classes
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_output_range(
        self, test_device: torch.device, sample_image_batch: torch.Tensor
    ) -> None:
        """Test that model outputs are in reasonable range."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)
        model.eval()

        with torch.no_grad():
            output = model(sample_image_batch)

        # Logits should be in reasonable range
        assert output.min() > -100
        assert output.max() < 100

    def test_get_num_parameters(self, test_device: torch.device) -> None:
        """Test parameter counting."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        params = model.get_num_parameters()

        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params

        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["frozen"] == 0
        assert params["total"] == params["trainable"] + params["frozen"]

    def test_freeze_backbone_mobilenet(self, test_device: torch.device) -> None:
        """Test freezing MobileNet backbone."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        params_before = model.get_num_parameters()
        model.freeze_backbone()
        params_after = model.get_num_parameters()

        # After freezing, should have fewer trainable parameters
        assert params_after["trainable"] < params_before["trainable"]
        assert params_after["frozen"] > params_before["frozen"]
        assert params_after["total"] == params_before["total"]

    def test_freeze_backbone_resnet(self, test_device: torch.device) -> None:
        """Test freezing ResNet backbone."""
        model = VisionModel(
            model_name="resnet18",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        params_before = model.get_num_parameters()
        model.freeze_backbone()
        params_after = model.get_num_parameters()

        assert params_after["trainable"] < params_before["trainable"]
        assert params_after["frozen"] > 0

    def test_unfreeze_all(self, test_device: torch.device) -> None:
        """Test unfreezing all parameters."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)

        # Freeze then unfreeze
        model.freeze_backbone()
        model.unfreeze_all()
        params = model.get_num_parameters()

        # All parameters should be trainable
        assert params["frozen"] == 0
        assert params["trainable"] == params["total"]

    def test_different_num_classes(self, test_device: torch.device) -> None:
        """Test model with different number of classes."""
        for num_classes in [10, 100, 1000]:
            model = VisionModel(
                model_name="mobilenet_v3_small",
                num_classes=num_classes,
                pretrained=False,
            )
            model = model.to(test_device)

            # Create sample input
            x = torch.randn(2, 3, 32, 32, device=test_device)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (2, num_classes)

    def test_gradient_flow(
        self,
        test_device: torch.device,
        sample_image_batch: torch.Tensor,
        sample_labels: torch.Tensor,
    ) -> None:
        """Test that gradients flow through the model."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Forward pass
        output = model(sample_image_batch)
        loss = criterion(output, sample_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"


class TestCreateModel:
    """Test create_model factory function."""

    def test_create_model_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating model from settings."""
        monkeypatch.setenv("MODEL_NAME", "mobilenet_v3_small")
        monkeypatch.setenv("NUM_CLASSES", "10")
        monkeypatch.setenv("PRETRAINED", "false")

        from src.config import get_settings

        get_settings.cache_clear()

        model = create_model()

        assert model is not None
        assert model.model_name == "mobilenet_v3_small"
        assert model.num_classes == 10

    def test_create_different_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating different model architectures."""
        from src.config import get_settings

        for model_name in ["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"]:
            monkeypatch.setenv("MODEL_NAME", model_name)
            monkeypatch.setenv("PRETRAINED", "false")

            get_settings.cache_clear()
            model = create_model()

            assert model.model_name == model_name


class TestModelCompatibility:
    """Test model compatibility with different configurations."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, test_device: torch.device, batch_size: int) -> None:
        """Test model with different batch sizes."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)
        model.eval()

        x = torch.randn(batch_size, 3, 32, 32, device=test_device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 10)

    @pytest.mark.parametrize("image_size", [28, 32, 64, 224])
    def test_different_image_sizes(self, test_device: torch.device, image_size: int) -> None:
        """Test model with different image sizes."""
        model = VisionModel(
            model_name="mobilenet_v3_small",
            num_classes=10,
            pretrained=False,
        )
        model = model.to(test_device)
        model.eval()

        x = torch.randn(2, 3, image_size, image_size, device=test_device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10)
