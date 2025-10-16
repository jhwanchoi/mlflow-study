"""Training module with MLflow integration."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_settings
from src.data import get_dataloaders
from src.models.vision_model import create_model

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    settings = get_settings()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100.0 * correct / total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.2f}%"})

        # Log batch metrics at specified intervals
        if (batch_idx + 1) % settings.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metrics(
                {
                    "batch_train_loss": loss.item(),
                    "batch_train_accuracy": 100.0 * correct / total,
                },
                step=step,
            )

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            avg_loss = running_loss / (len(val_loader) if len(val_loader) > 0 else 1)
            accuracy = 100.0 * correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.2f}%"})

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def train_model() -> str:
    """Train model with MLflow tracking.

    Returns:
        MLflow run ID
    """
    settings = get_settings()

    # Setup logging
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    # Set S3 endpoint for artifact storage
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url

    # Setup device
    device = torch.device(settings.device)
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Create model
    logger.info("Creating model...")
    model = create_model()
    model = model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=settings.learning_rate,
        momentum=settings.momentum,
        weight_decay=settings.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)

    # Start MLflow run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{settings.model_name}_{settings.dataset}_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")
        logger.info(f"Run name: {run_name}")

        # Log parameters
        params = {
            "model_name": settings.model_name,
            "dataset": settings.dataset,
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "momentum": settings.momentum,
            "weight_decay": settings.weight_decay,
            "epochs": settings.epochs,
            "device": settings.device,
            "pretrained": settings.pretrained,
            "num_classes": settings.num_classes,
            "use_augmentation": settings.use_augmentation,
        }

        model_params = model.get_num_parameters()
        params.update(
            {
                f"model_{k}": v
                for k, v in model_params.items()
            }
        )

        mlflow.log_params(params)

        # Log model architecture
        mlflow.set_tag("model_architecture", settings.model_name)
        mlflow.set_tag("dataset", settings.dataset)

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, settings.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{settings.epochs}")

            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Log epoch metrics
            mlflow.log_metrics(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                if settings.save_best_only:
                    logger.info(f"New best validation accuracy: {val_acc:.2f}%")
                    # Save model checkpoint
                    checkpoint_path = Path("checkpoints")
                    checkpoint_path.mkdir(exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_accuracy": val_acc,
                        },
                        checkpoint_path / "best_model.pth",
                    )
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{settings.early_stopping_patience}")

            # Early stopping
            if patience_counter >= settings.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        # Log best metrics
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Log final model
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(
            model,
            "model",
            pip_requirements=[
                f"torch=={torch.__version__}",
                "torchvision",
            ],
        )

        # Log checkpoint if exists
        checkpoint_path = Path("checkpoints/best_model.pth")
        if checkpoint_path.exists():
            mlflow.log_artifact(str(checkpoint_path))

        logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"MLflow run ID: {run.info.run_id}")

        return run.info.run_id


if __name__ == "__main__":
    train_model()
