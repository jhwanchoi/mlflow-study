terraform {
  required_version = ">= 1.6.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Docker Network
resource "docker_network" "mlflow_network" {
  name = "mlflow-network"
  driver = "bridge"
}

# PostgreSQL Volume
resource "docker_volume" "postgres_data" {
  name = "mlflow-postgres-data"
}

# MinIO Volume
resource "docker_volume" "minio_data" {
  name = "mlflow-minio-data"
}

# PostgreSQL Container
resource "docker_container" "postgres" {
  name  = "mlflow-postgres"
  image = docker_image.postgres.image_id

  env = [
    "POSTGRES_DB=mlflow",
    "POSTGRES_USER=mlflow",
    "POSTGRES_PASSWORD=mlflow"
  ]

  ports {
    internal = 5432
    external = 5432
  }

  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }

  networks_advanced {
    name = docker_network.mlflow_network.name
  }

  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U mlflow"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
}

resource "docker_image" "postgres" {
  name = "postgres:15-alpine"
}

# MinIO Container
resource "docker_container" "minio" {
  name    = "mlflow-minio"
  image   = docker_image.minio.image_id
  command = ["server", "/data", "--console-address", ":9001"]

  env = [
    "MINIO_ROOT_USER=minio",
    "MINIO_ROOT_PASSWORD=minio123"
  ]

  ports {
    internal = 9000
    external = 9000
  }

  ports {
    internal = 9001
    external = 9001
  }

  volumes {
    volume_name    = docker_volume.minio_data.name
    container_path = "/data"
  }

  networks_advanced {
    name = docker_network.mlflow_network.name
  }
}

resource "docker_image" "minio" {
  name = "minio/minio:latest"
}

# MLflow Container
resource "docker_container" "mlflow" {
  name    = "mlflow-server"
  image   = docker_image.mlflow.image_id
  command = [
    "mlflow", "server",
    "--backend-store-uri", "postgresql://mlflow:mlflow@mlflow-postgres:5432/mlflow",
    "--default-artifact-root", "s3://mlflow/artifacts",
    "--host", "0.0.0.0",
    "--port", "5000"
  ]

  env = [
    "AWS_ACCESS_KEY_ID=minio",
    "AWS_SECRET_ACCESS_KEY=minio123",
    "MLFLOW_S3_ENDPOINT_URL=http://mlflow-minio:9000"
  ]

  ports {
    internal = 5000
    external = 5000
  }

  networks_advanced {
    name = docker_network.mlflow_network.name
  }

  depends_on = [
    docker_container.postgres,
    docker_container.minio
  ]
}

resource "docker_image" "mlflow" {
  name = "ghcr.io/mlflow/mlflow:v2.10.2"
}
