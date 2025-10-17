#!/usr/bin/env python3
"""Simple MLflow artifact upload test script."""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import mlflow

# Try to import torch and model, but make it optional
try:
    import torch
    import torch.nn as nn
    import mlflow.pytorch
    from src.models.vision_model import create_model
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  PyTorch not available: {e}")
    TORCH_AVAILABLE = False

def test_mlflow_connection():
    """Test basic MLflow connection."""
    print("🔄 Testing MLflow connection...")
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    
    try:
        # Test if we can reach the server
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ MLflow connection successful! Found {len(experiments)} experiments.")
        return True
    except Exception as e:
        print(f"❌ MLflow connection failed: {e}")
        return False

def test_simple_artifact_upload():
    """Test simple artifact upload."""
    print("🔄 Testing simple artifact upload...")
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test artifact from MLflow test script.")
            temp_file = f.name
        
        # Start MLflow run and log artifact
        with mlflow.start_run(run_name="artifact_test") as run:
            mlflow.log_artifact(temp_file, "test_artifacts")
            run_id = run.info.run_id
            print(f"✅ Simple artifact upload successful! Run ID: {run_id}")
        
        # Clean up
        os.unlink(temp_file)
        return True, run_id
        
    except Exception as e:
        print(f"❌ Simple artifact upload failed: {e}")
        return False, None

def test_pytorch_model_logging():
    """Test PyTorch model logging."""
    if not TORCH_AVAILABLE:
        print("⚠️  Skipping PyTorch model logging test - PyTorch not available")
        return False, None
    
    print("🔄 Testing PyTorch model logging...")
    
    try:
        # Create a simple model
        model = create_model()
        print(f"📊 Model created: {model.__class__.__name__}")
        
        # Start MLflow run and log model
        with mlflow.start_run(run_name="pytorch_model_test") as run:
            # Log some parameters
            mlflow.log_param("model_type", "mobilenet_v3_small")
            mlflow.log_param("test_mode", True)
            
            # Log a metric
            mlflow.log_metric("test_accuracy", 0.95)
            
            # Log the PyTorch model
            mlflow.pytorch.log_model(
                model,
                "model",
                pip_requirements=[
                    f"torch=={torch.__version__}",
                    "torchvision",
                ],
            )
            
            run_id = run.info.run_id
            print(f"✅ PyTorch model logging successful! Run ID: {run_id}")
            
        return True, run_id
        
    except Exception as e:
        print(f"❌ PyTorch model logging failed: {e}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_model_loading(run_id):
    """Test loading the logged model."""
    if not TORCH_AVAILABLE or run_id is None:
        print("⚠️  Skipping model loading test - PyTorch not available or no run_id")
        return False
        
    print(f"🔄 Testing model loading from run {run_id}...")
    
    try:
        # Load model from MLflow
        model_uri = f"runs:/{run_id}/model"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        
        print(f"✅ Model loading successful! Model type: {type(loaded_model)}")
        
        # Test inference
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = loaded_model(dummy_input)
            print(f"✅ Model inference successful! Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting MLflow artifact tests...")
    print("=" * 50)
    
    # Set environment variables for S3
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ["AWS_S3_ENDPOINT_URL"] = os.getenv("AWS_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")
    
    print(f"🔧 MLflow URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"🔧 S3 Endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
    
    # Set experiment
    mlflow.set_experiment("test-mlflow-artifacts")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Basic connection
    if test_mlflow_connection():
        tests_passed += 1
    
    # Test 2: Simple artifact upload
    artifact_success, artifact_run_id = test_simple_artifact_upload()
    if artifact_success:
        tests_passed += 1
    
    # Test 3: PyTorch model logging
    model_success, model_run_id = test_pytorch_model_logging()
    if model_success:
        tests_passed += 1
        
        # Test 4: Model loading
        if test_model_loading(model_run_id):
            tests_passed += 1
    
    print("=" * 50)
    print(f"🏁 Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! MLflow artifact storage is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)