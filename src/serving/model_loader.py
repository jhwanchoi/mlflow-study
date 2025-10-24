"""
MLflow model loader for BentoML serving.

This module provides utilities to load PyTorch models from MLflow.
"""

import os
from typing import Any, Dict, Optional

import mlflow
import torch
from mlflow.tracking import MlflowClient


def load_mlflow_model(
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    model_stage: Optional[str] = None,
    model_alias: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Any:
    """
    Load a PyTorch model from MLflow.

    Args:
        run_id: MLflow run ID (loads from run if specified)
        model_name: Registered model name (loads from Model Registry if specified)
        model_version: Model version (required if model_name is specified without stage/alias)
        model_stage: Model stage (DEPRECATED - e.g., "Production", "Staging")
        model_alias: Model alias (RECOMMENDED - e.g., "champion", "challenger", "production")
        tracking_uri: MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env var)

    Returns:
        Loaded PyTorch model

    Raises:
        ValueError: If neither run_id nor (model_name, model_version/stage/alias) is provided
        RuntimeError: If model loading fails

    Examples:
        # Load from run ID
        model = load_mlflow_model(run_id="abc123")

        # Load from Model Registry by version
        model = load_mlflow_model(model_name="vision-model", model_version="1")

        # Load from Model Registry by alias (RECOMMENDED for production)
        model = load_mlflow_model(model_name="vision-model", model_alias="champion")

        # Load from Model Registry by stage (DEPRECATED but still supported)
        model = load_mlflow_model(model_name="vision-model", model_stage="Production")
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    try:
        if run_id:
            # Load model from run artifacts
            model_uri = f"runs:/{run_id}/model"
            # Force CPU loading to avoid MPS device issues in Docker
            model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            return model

        elif model_name and model_alias:
            # Load model from Model Registry by alias (RECOMMENDED)
            # Alias format: models://<model_name>@<alias>
            model_uri = f"models:/{model_name}@{model_alias}"
            # Force CPU loading to avoid MPS device issues in Docker
            model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            return model

        elif model_name and model_stage:
            # Load model from Model Registry by stage (DEPRECATED)
            # Stage format: models://<model_name>/<stage>
            model_uri = f"models:/{model_name}/{model_stage}"
            # Force CPU loading to avoid MPS device issues in Docker
            model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            return model

        elif model_name and model_version:
            # Load model from Model Registry by version
            model_uri = f"models:/{model_name}/{model_version}"
            # Force CPU loading to avoid MPS device issues in Docker
            model = mlflow.pytorch.load_model(model_uri, map_location="cpu")
            return model

        else:
            raise ValueError(
                "Either run_id or (model_name with model_version/model_stage/model_alias) must be provided"
            )

    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow: {e}") from e


def get_latest_model_version(
    model_name: str, tracking_uri: Optional[str] = None
) -> str:
    """
    Get the latest version of a registered model.

    Args:
        model_name: Registered model name
        tracking_uri: MLflow tracking URI

    Returns:
        Latest model version as string

    Raises:
        RuntimeError: If model not found or retrieval fails
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            raise RuntimeError(f"No versions found for model: {model_name}")

        # Get the latest version
        latest_version = max(versions, key=lambda v: int(v.version))
        return str(latest_version.version)

    except Exception as e:
        raise RuntimeError(
            f"Failed to get latest version for {model_name}: {e}"
        ) from e


def get_model_metadata(
    run_id: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    model_stage: Optional[str] = None,
    model_alias: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get metadata for a model from MLflow.

    Args:
        run_id: MLflow run ID
        model_name: Registered model name
        model_version: Model version
        model_stage: Model stage (DEPRECATED - e.g., "Production", "Staging")
        model_alias: Model alias (RECOMMENDED - e.g., "champion", "challenger")

    Returns:
        Dictionary containing model metadata (params, metrics, etc.)
    """
    client = MlflowClient()

    try:
        if run_id:
            run = client.get_run(run_id)
            return {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
                "artifact_uri": run.info.artifact_uri,
            }

        elif model_name and model_alias:
            # Get model version by alias (RECOMMENDED)
            version_info = client.get_model_version_by_alias(model_name, model_alias)
            # Get run info for additional metadata
            run = client.get_run(version_info.run_id)

            return {
                "name": version_info.name,
                "version": version_info.version,
                "alias": model_alias,
                "aliases": list(version_info.aliases),  # Convert protobuf repeated field to list
                "run_id": version_info.run_id,
                "status": version_info.status,
                "source": version_info.source,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }

        elif model_name and model_stage:
            # Get model version by stage (DEPRECATED)
            versions = client.get_latest_versions(model_name, stages=[model_stage])
            if not versions:
                raise ValueError(f"No model found with stage '{model_stage}' for {model_name}")

            version_info = versions[0]
            # Get run info for additional metadata
            run = client.get_run(version_info.run_id)

            return {
                "name": version_info.name,
                "version": version_info.version,
                "stage": model_stage,
                "run_id": version_info.run_id,
                "status": version_info.status,
                "source": version_info.source,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }

        elif model_name and model_version:
            version_info = client.get_model_version(model_name, model_version)
            # Get run info for additional metadata
            run = client.get_run(version_info.run_id)

            return {
                "name": version_info.name,
                "version": version_info.version,
                "run_id": version_info.run_id,
                "status": version_info.status,
                "source": version_info.source,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }

        else:
            raise ValueError(
                "Either run_id or (model_name with model_version/model_stage/model_alias) must be provided"
            )

    except Exception as e:
        raise RuntimeError(f"Failed to get model metadata: {e}") from e
