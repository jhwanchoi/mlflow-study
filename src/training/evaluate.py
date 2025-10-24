"""Model evaluation module with visualization."""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_settings
from src.data import get_dataloaders
from src.data.dataset import get_class_names

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    save_dir: Optional[Path] = None,
) -> dict[str, float]:
    """Evaluate model and generate metrics.

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        class_names: List of class names
        save_dir: Directory to save visualizations

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_predictions: list[Any] = []
    all_targets: list[Any] = []
    all_probs: list[Any] = []

    logger.info("Running evaluation...")

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # Convert to numpy arrays
    pred_array = np.array(all_predictions)
    target_array = np.array(all_targets)
    # Note: all_probs collected for future use (ROC curves, calibration plots, etc.)

    # Calculate accuracy
    accuracy = 100.0 * float(np.sum(pred_array == target_array)) / len(target_array)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(target_array, pred_array)

    # Generate classification report
    report = classification_report(
        target_array,
        pred_array,
        target_names=class_names,
        output_dict=True,
    )

    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if class_name in report:
            per_class_metrics[f"precision_{class_name}"] = report[class_name]["precision"]
            per_class_metrics[f"recall_{class_name}"] = report[class_name]["recall"]
            per_class_metrics[f"f1_{class_name}"] = report[class_name]["f1-score"]

    # Save visualizations
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot confusion matrix
        plot_confusion_matrix(cm, class_names, save_dir / "confusion_matrix.png")

        # Plot per-class accuracy
        plot_per_class_metrics(cm, class_names, save_dir / "per_class_accuracy.png")

        logger.info(f"Saved evaluation visualizations to {save_dir}")

    # Compile metrics
    metrics = {
        "test_accuracy": accuracy,
        "test_precision": report["weighted avg"]["precision"],
        "test_recall": report["weighted avg"]["recall"],
        "test_f1": report["weighted avg"]["f1-score"],
    }
    metrics.update(per_class_metrics)

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Plot and save confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"label": "Accuracy"},
    )

    plt.title("Confusion Matrix (Normalized)", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {save_path}")


def plot_per_class_metrics(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
) -> None:
    """Plot per-class accuracy bar chart.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save plot
    """
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    plt.figure(figsize=(14, 6))

    bars = plt.bar(range(len(class_names)), per_class_acc, color="steelblue", alpha=0.8)

    # Color bars based on performance
    for i, bar in enumerate(bars):
        if per_class_acc[i] >= 80:
            bar.set_color("green")
        elif per_class_acc[i] >= 60:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Per-Class Accuracy", fontsize=16, pad=20)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, v in enumerate(per_class_acc):
        plt.text(i, v + 2, f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved per-class accuracy to {save_path}")


def evaluate_from_mlflow(run_id: str) -> dict[str, float]:
    """Load model from MLflow and evaluate.

    Args:
        run_id: MLflow run ID

    Returns:
        Dictionary of evaluation metrics
    """
    import os

    settings = get_settings()

    # Setup
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set AWS/MinIO environment variables for boto3
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url
    os.environ["AWS_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Setup device
    device = torch.device(settings.device)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading test dataset...")
    _, _, test_loader = get_dataloaders()

    # Load model from MLflow
    logger.info(f"Loading model from MLflow run: {run_id}")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model = model.to(device)

    # Get class names
    class_names = get_class_names(settings.dataset)

    # Evaluate
    save_dir = Path("evaluation_results")
    metrics = evaluate_model(model, test_loader, device, class_names, save_dir)

    # Log metrics to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

        # Log artifacts
        if save_dir.exists():
            for artifact in save_dir.glob("*.png"):
                mlflow.log_artifact(str(artifact))

        # Log evaluation table for MLflow UI
        import pandas as pd

        # Create evaluation table with per-class metrics
        eval_data = []
        for class_name in class_names:
            class_metrics = {
                "class": class_name,
                "precision": metrics.get(f"precision_{class_name}", 0.0),
                "recall": metrics.get(f"recall_{class_name}", 0.0),
                "f1_score": metrics.get(f"f1_{class_name}", 0.0),
            }
            eval_data.append(class_metrics)

        eval_df = pd.DataFrame(eval_data)

        # Log as table artifact
        mlflow.log_table(data=eval_df, artifact_file="evaluation_table.json")

    logger.info("Evaluation complete!")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    logger.info(f"Test F1 Score: {metrics['test_f1']:.4f}")

    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_id = sys.argv[1]
        evaluate_from_mlflow(run_id)
    else:
        logger.error("Usage: python evaluate.py <mlflow_run_id>")
