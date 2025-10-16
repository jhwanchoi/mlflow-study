# Terraform Configuration for Local MLflow Infrastructure

## Overview
This Terraform configuration provisions a local MLflow infrastructure using Docker containers. It serves as a foundation that can be extended to Kubernetes environments.

## Components
- **PostgreSQL**: Backend store for MLflow metadata
- **MinIO**: S3-compatible artifact storage
- **MLflow Server**: Tracking server

## Usage

### Initialize Terraform
```bash
cd terraform/local
terraform init
```

### Plan Infrastructure
```bash
terraform plan
```

### Apply Infrastructure
```bash
terraform apply
```

### Destroy Infrastructure
```bash
terraform destroy
```

## Future Migration Path

This configuration is designed to be easily migrated to Kubernetes:

### Phase 1: Local Docker (Current)
```hcl
provider "docker" {
  host = "unix:///var/run/docker.sock"
}
```

### Phase 2: Kubernetes
```hcl
provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}
```

## Outputs
- `mlflow_tracking_uri`: http://localhost:5000
- `minio_console_url`: http://localhost:9001
- `postgres_connection_string`: Connection string for direct DB access
