output "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  value       = "http://localhost:5000"
}

output "minio_console_url" {
  description = "MinIO console URL"
  value       = "http://localhost:9001"
}

output "postgres_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://mlflow:mlflow@localhost:5432/mlflow"
  sensitive   = true
}
