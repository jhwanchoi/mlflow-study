# General Configuration
variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "mdpg-mlops"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod"
  }
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# EKS Configuration
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "mdpg-mlops-eks"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 2
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 5
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 2
}

# RDS Configuration
variable "rds_identifier" {
  description = "Identifier for the RDS instance"
  type        = string
  default     = "mdpg-mlops-mlflow-db"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.small"
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling in GB"
  type        = number
  default     = 100
}

variable "rds_database_name" {
  description = "Name of the database to create"
  type        = string
  default     = "mlflow"
}

variable "rds_username" {
  description = "Master username for the database"
  type        = string
  default     = "mlflow"
}

variable "rds_backup_retention_period" {
  description = "Number of days to retain backups"
  type        = number
  default     = 7
}

# S3 Configuration
variable "s3_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  type        = string
  default     = "mdpg-mlops-mlflow-artifacts"
}

variable "s3_cors_allowed_origins" {
  description = "Allowed origins for S3 CORS"
  type        = list(string)
  default     = ["https://mlflow.mdpg.ai"]
}

variable "s3_size_alarm_threshold" {
  description = "S3 bucket size threshold in bytes for CloudWatch alarm"
  type        = number
  default     = 107374182400  # 100GB
}

# MLflow Service Account Configuration
variable "mlflow_namespace" {
  description = "Kubernetes namespace for MLflow"
  type        = string
  default     = "mlflow"
}

variable "mlflow_service_account_name" {
  description = "Kubernetes service account name for MLflow"
  type        = string
  default     = "mlflow-sa"
}

# Ray Configuration
variable "ray_namespace" {
  description = "Kubernetes namespace for Ray cluster"
  type        = string
  default     = "ray"
}

variable "ray_service_account_name" {
  description = "Kubernetes service account name for Ray cluster"
  type        = string
  default     = "ray-sa"
}

# External DNS Configuration
variable "enable_external_dns" {
  description = "Enable External DNS for automatic DNS management"
  type        = bool
  default     = false
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
