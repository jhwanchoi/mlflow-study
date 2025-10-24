"""
Ray Tune hyperparameter optimization with MLflow integration.

This module provides automated hyperparameter tuning using Ray Tune,
with automatic logging to MLflow for experiment tracking and comparison.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

# Optional: HyperOpt search algorithm
try:
    from ray.tune.search.hyperopt import HyperOptSearch
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    HyperOptSearch = None

from src.config import get_settings
from src.data import get_dataloaders
from src.models import create_model

logger = logging.getLogger(__name__)


def train_epoch_simple(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    global_step: int,
    log_interval: int = 10,
) -> tuple[float, float, int]:
    """Train for one epoch (simplified for tuning).

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        global_step: Global step counter across all epochs
        log_interval: Log metrics every N batches

    Returns:
        Tuple of (average_loss, accuracy, updated_global_step)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Get active MLflow run for batch-level logging
    active_run = mlflow.active_run()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        # Increment global step
        global_step += 1

        # Log batch-level metrics to MLflow at specified intervals
        if active_run and (batch_idx + 1) % log_interval == 0:
            batch_accuracy = 100.0 * correct / total
            mlflow.log_metrics(
                {
                    "batch_train_loss": loss.item(),
                    "batch_train_accuracy": batch_accuracy,
                },
                step=global_step,
            )

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, global_step


def validate_simple(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    global_step: int,
    log_interval: int = 10,
) -> tuple[float, float, int]:
    """Validate the model (simplified for tuning).

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        global_step: Global step counter
        log_interval: Log metrics every N batches

    Returns:
        Tuple of (average_loss, accuracy, updated_global_step)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Get active MLflow run for batch-level logging
    active_run = mlflow.active_run()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Increment global step
            global_step += 1

            # Log batch-level validation metrics at specified intervals
            if active_run and (batch_idx + 1) % log_interval == 0:
                batch_accuracy = 100.0 * correct / total
                mlflow.log_metrics(
                    {
                        "batch_val_loss": loss.item(),
                        "batch_val_accuracy": batch_accuracy,
                    },
                    step=global_step,
                )

    avg_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy, global_step


def trainable(config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable function.

    This function is called for each trial with different hyperparameters.
    It trains a model and reports metrics back to Ray Tune and MLflow.

    Args:
        config: Dictionary containing hyperparameters to tune
    """
    # Get base settings
    base_settings = get_settings()

    # Extract hyperparameters from config
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    momentum = config["momentum"]
    epochs = config.get("epochs", 10)

    # Setup device
    device = torch.device(base_settings.device)

    # Load data (using batch_size from settings)
    train_loader, val_loader, _ = get_dataloaders()

    # Create model (using model_name from settings)
    model = create_model()
    model = model.to(device)

    # Setup training with tuned hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup MLflow tracking for this trial
    # Note: We manually handle MLflow logging instead of using MLflowLoggerCallback
    # to have full control over batch-level metrics
    base_settings = get_settings()
    mlflow.set_tracking_uri(base_settings.mlflow_tracking_uri)

    # Set experiment
    experiment_name = f"{base_settings.experiment_name}-tuning"
    mlflow.set_experiment(experiment_name)

    # Get or create active MLflow run
    active_run = mlflow.active_run()
    if not active_run:
        # Create a new run for this trial
        trial_name = f"trial_{tune.get_context().get_trial_id()}"
        mlflow.start_run(run_name=trial_name)
        active_run = mlflow.active_run()

        # Log hyperparameters as MLflow params
        mlflow.log_params({
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "epochs": epochs,
            "batch_size": base_settings.batch_size,
            "model_name": base_settings.model_name,
            "dataset": base_settings.dataset,
        })

        # Log tags
        mlflow.set_tags({
            "framework": "ray-tune",
            "task": "hyperparameter-tuning",
            "trial_id": tune.get_context().get_trial_id(),
        })

    # Global step counter across all epochs for continuous step logging
    global_step = 0

    for epoch in range(1, epochs + 1):
        # Train with batch-level logging
        train_loss, train_acc, global_step = train_epoch_simple(
            model, train_loader, criterion, optimizer, device, global_step
        )

        # Validate with batch-level logging
        val_loss, val_acc, global_step = validate_simple(
            model, val_loader, criterion, device, global_step
        )

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Prepare metrics dictionary for both MLflow and Ray Tune
        epoch_metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr,
            "training_iteration": epoch,  # Ray Tune convention
        }

        # Log epoch-level metrics to MLflow
        # Using global_step ensures continuous step progression across epochs
        if active_run:
            mlflow_metrics = {
                "epoch_train_loss": train_loss,
                "epoch_train_accuracy": train_acc,
                "epoch_val_loss": val_loss,
                "epoch_val_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch": epoch,
            }
            # Also log Ray Tune convention metrics for compatibility
            mlflow_metrics.update(epoch_metrics)
            mlflow.log_metrics(mlflow_metrics, step=global_step)

        # Optional: Save checkpoint for recovery
        checkpoint_dir = Path("./checkpoints") / f"trial_{tune.get_context().get_trial_id()}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
            },
            checkpoint_path,
        )

        # Report metrics to Ray Tune with checkpoint
        # This allows Ray Tune to track progress and make scheduling decisions
        checkpoint = Checkpoint.from_directory(str(checkpoint_dir))
        tune.report(epoch_metrics, checkpoint=checkpoint)


def create_search_space(
    learning_rate_range: tuple[float, float] = (1e-4, 1e-2),
    weight_decay_range: tuple[float, float] = (1e-5, 1e-3),
    momentum_range: tuple[float, float] = (0.8, 0.99),
    epochs: int = 10,
) -> Dict[str, Any]:
    """
    Create Ray Tune search space configuration.

    Args:
        learning_rate_range: Min and max learning rate (log scale)
        weight_decay_range: Min and max weight decay (log scale)
        momentum_range: Min and max momentum
        epochs: Number of epochs per trial

    Returns:
        Dictionary defining the search space

    Note:
        Currently tuning only optimizer hyperparameters (learning_rate, weight_decay, momentum).
        To tune batch_size and model_name, refactor get_dataloaders() and create_model()
        to accept these as parameters.
    """
    search_space = {
        "learning_rate": tune.loguniform(learning_rate_range[0], learning_rate_range[1]),
        "weight_decay": tune.loguniform(weight_decay_range[0], weight_decay_range[1]),
        "momentum": tune.uniform(momentum_range[0], momentum_range[1]),
        "epochs": epochs,
    }

    return search_space


def tune_model(
    num_samples: int = 10,
    max_concurrent_trials: int = 2,
    search_space: Optional[Dict[str, Any]] = None,
    metric: str = "val_accuracy",
    mode: str = "max",
    scheduler: Optional[Any] = None,
    search_alg: Optional[Any] = "default",
) -> ray.tune.ResultGrid:
    """
    Run hyperparameter tuning with Ray Tune.

    Args:
        num_samples: Number of trials to run
        max_concurrent_trials: Maximum number of concurrent trials
        search_space: Custom search space (uses default if None)
        metric: Metric to optimize
        mode: Optimization mode ("max" or "min")
        scheduler: Custom scheduler (uses ASHA if None)
        search_alg: Search algorithm ("default" for HyperOpt, None for random/grid search, or custom)

    Returns:
        Ray Tune ResultGrid with all trial results
    """
    settings = get_settings()

    # Setup logging
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = f"{settings.experiment_name}-tuning"
    mlflow.set_experiment(experiment_name)

    # Set S3 endpoint for artifact storage
    os.environ["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.mlflow_s3_endpoint_url

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Use default search space if not provided
    if search_space is None:
        search_space = create_search_space()

    # Use ASHA scheduler if not provided
    # Note: Don't pass metric/mode here since they're passed to TuneConfig
    if scheduler is None:
        scheduler = ASHAScheduler(
            time_attr="epoch",
            max_t=search_space.get("epochs", 10),
            grace_period=3,  # Minimum epochs before early stopping
            reduction_factor=2,
        )

    # Use HyperOpt search algorithm if "default" is specified
    if search_alg == "default":
        if HYPEROPT_AVAILABLE:
            search_alg = HyperOptSearch(metric=metric, mode=mode)
            logger.info("Using HyperOpt search algorithm")
        else:
            logger.warning(
                "HyperOpt not available. Using default random search. "
                "Install hyperopt for better search: pip install hyperopt"
            )
            search_alg = None
    # If None, use default Ray Tune search (supports grid_search)

    logger.info(f"Starting Ray Tune with {num_samples} trials")
    logger.info(f"Optimizing {metric} ({mode})")
    logger.info(f"Search space: {search_space}")

    # Setup storage path (Ray requires absolute path)
    storage_path = Path("ray_results").absolute()
    storage_path.mkdir(exist_ok=True)

    # Note: We handle MLflow logging inside trainable function
    # to have full control over batch-level and epoch-level metrics
    # MLflowLoggerCallback is not used to avoid duplicate runs

    # Run tuning
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            metric=metric,
            mode=mode,
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(storage_path),
            callbacks=[],  # No callbacks - we handle MLflow manually
        ),
    )

    results = tuner.fit()

    # Print best hyperparameters
    best_result = results.get_best_result(metric=metric, mode=mode)

    logger.info("=" * 80)
    logger.info("Best trial config:")
    logger.info(best_result.config)
    logger.info(f"Best trial final {metric}: {best_result.metrics[metric]:.4f}")
    logger.info("=" * 80)

    # Log best config to MLflow
    with mlflow.start_run(run_name=f"best_trial_{experiment_name}"):
        mlflow.log_params(best_result.config)

        # Filter only numeric metrics (exclude checkpoint, etc.)
        numeric_metrics = {
            k: v for k, v in best_result.metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        mlflow.log_metrics(numeric_metrics)

        mlflow.set_tag("best_trial", True)
        mlflow.set_tag("tuning_experiment", experiment_name)

    return results


if __name__ == "__main__":
    # Example: Run hyperparameter tuning
    results = tune_model(
        num_samples=5,  # Start small for testing
        max_concurrent_trials=2,
    )

    print("\nTuning complete!")
    print(f"Total trials: {len(results)}")
