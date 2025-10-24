"""
BentoML service for MLflow vision models (BentoML 1.2+ API).

This service provides REST API endpoints for image classification using models
trained with MLflow.
"""

import os

import bentoml
import torch
from PIL import Image as PILImage
from torchvision import transforms

from .model_loader import get_model_metadata, load_mlflow_model


# Model configuration from environment variables
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
MODEL_STAGE = os.getenv("MODEL_STAGE")  # DEPRECATED: e.g., "Production", "Staging"
MODEL_ALIAS = os.getenv("MODEL_ALIAS")  # RECOMMENDED: e.g., "champion", "challenger"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10 class names
CIFAR10_CLASSES = [
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


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class VisionClassifier:
    """Vision model classifier service."""

    def __init__(self) -> None:
        """Initialize the model from MLflow."""
        import os

        # Set AWS/MinIO environment variables for boto3
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
            "AWS_SECRET_ACCESS_KEY", "minio123"
        )
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
        )
        os.environ["AWS_S3_ENDPOINT_URL"] = os.getenv(
            "AWS_S3_ENDPOINT_URL", "http://localhost:9000"
        )

        # Load model from MLflow
        if MODEL_RUN_ID:
            # Option 1: Load from specific run ID (for development/debugging)
            self.model = load_mlflow_model(
                run_id=MODEL_RUN_ID, tracking_uri=MLFLOW_TRACKING_URI
            )
            self.metadata = get_model_metadata(run_id=MODEL_RUN_ID)

        elif MODEL_NAME and MODEL_ALIAS:
            # Option 2: Load from Model Registry by alias (RECOMMENDED for production)
            self.model = load_mlflow_model(
                model_name=MODEL_NAME,
                model_alias=MODEL_ALIAS,
                tracking_uri=MLFLOW_TRACKING_URI,
            )
            self.metadata = get_model_metadata(
                model_name=MODEL_NAME, model_alias=MODEL_ALIAS
            )

        elif MODEL_NAME and MODEL_STAGE:
            # Option 3: Load from Model Registry by stage (DEPRECATED)
            self.model = load_mlflow_model(
                model_name=MODEL_NAME,
                model_stage=MODEL_STAGE,
                tracking_uri=MLFLOW_TRACKING_URI,
            )
            self.metadata = get_model_metadata(
                model_name=MODEL_NAME, model_stage=MODEL_STAGE
            )

        elif MODEL_NAME:
            # Option 4: Load from Model Registry by version
            version = MODEL_VERSION if MODEL_VERSION != "latest" else None
            if not version:
                from .model_loader import get_latest_model_version

                version = get_latest_model_version(MODEL_NAME, MLFLOW_TRACKING_URI)

            self.model = load_mlflow_model(
                model_name=MODEL_NAME,
                model_version=version,
                tracking_uri=MLFLOW_TRACKING_URI,
            )
            self.metadata = get_model_metadata(
                model_name=MODEL_NAME, model_version=version
            )

        else:
            raise ValueError(
                "Either MODEL_RUN_ID or MODEL_NAME must be set in environment"
            )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),  # CIFAR-10 size
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @bentoml.api
    def predict_image(
        self,
        image: PILImage.Image,
    ) -> dict:
        """
        Predict class for a single image.

        Args:
            image: PIL Image

        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        # Get predicted class
        predicted_idx = int(torch.argmax(probs))
        predicted_class = CIFAR10_CLASSES[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Create probabilities dict
        class_probs = {
            CIFAR10_CLASSES[i]: float(probs[i]) for i in range(len(CIFAR10_CLASSES))
        }

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_idx,
            "confidence": confidence,
            "probabilities": class_probs,
        }

    @bentoml.api
    def predict_batch(
        self,
        images: list[PILImage.Image],
    ) -> list[dict]:
        """
        Predict classes for a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            List of prediction dictionaries
        """
        # Preprocess images
        img_tensors = [self.transform(img) for img in images]
        batch = torch.stack(img_tensors).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)

        # Process each prediction
        results = []
        for prob in probs:
            predicted_idx = int(torch.argmax(prob))
            predicted_class = CIFAR10_CLASSES[predicted_idx]
            confidence = float(prob[predicted_idx])

            class_probs = {
                CIFAR10_CLASSES[i]: float(prob[i]) for i in range(len(CIFAR10_CLASSES))
            }

            results.append(
                {
                    "predicted_class": predicted_class,
                    "predicted_index": predicted_idx,
                    "confidence": confidence,
                    "probabilities": class_probs,
                }
            )

        return results

    @bentoml.api
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_metadata": self.metadata,
            "device": str(self.device),
            "classes": CIFAR10_CLASSES,
            "num_classes": len(CIFAR10_CLASSES),
        }

    @bentoml.api
    def health(self) -> dict:
        """
        Health check endpoint.

        Returns:
            Dictionary with health status
        """
        return {"status": "healthy", "service": "mlflow_vision_classifier"}
