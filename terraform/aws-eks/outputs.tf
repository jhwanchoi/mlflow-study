# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.vpc.public_subnets
}

# EKS Cluster Outputs
output "cluster_id" {
  description = "ID of the EKS cluster"
  value       = module.eks.cluster_id
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_oidc_provider_arn" {
  description = "ARN of the OIDC Provider for EKS"
  value       = module.eks.oidc_provider_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

# Node Group Outputs
output "node_security_group_id" {
  description = "Security group ID attached to the EKS nodes"
  value       = module.eks.node_security_group_id
}

# RDS Outputs
output "rds_endpoint" {
  description = "Connection endpoint for the RDS instance"
  value       = aws_db_instance.mlflow.endpoint
}

output "rds_address" {
  description = "Hostname of the RDS instance"
  value       = aws_db_instance.mlflow.address
}

output "rds_port" {
  description = "Port the RDS instance is listening on"
  value       = aws_db_instance.mlflow.port
}

output "rds_database_name" {
  description = "Name of the database"
  value       = aws_db_instance.mlflow.db_name
}

output "rds_username" {
  description = "Master username for the database"
  value       = var.rds_username
  sensitive   = true
}

output "rds_secret_arn" {
  description = "ARN of the Secrets Manager secret containing RDS credentials"
  value       = aws_secretsmanager_secret.rds_password.arn
}

# S3 Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "s3_bucket_region" {
  description = "Region of the S3 bucket"
  value       = aws_s3_bucket.mlflow_artifacts.region
}

# IAM Role Outputs
output "mlflow_iam_role_arn" {
  description = "ARN of the IAM role for MLflow service account"
  value       = aws_iam_role.mlflow_s3_access.arn
}

output "ray_iam_role_arn" {
  description = "ARN of the IAM role for Ray cluster service account"
  value       = aws_iam_role.ray_cluster.arn
}

output "aws_load_balancer_controller_role_arn" {
  description = "ARN of the IAM role for AWS Load Balancer Controller"
  value       = aws_iam_role.aws_load_balancer_controller.arn
}

output "cluster_autoscaler_role_arn" {
  description = "ARN of the IAM role for Cluster Autoscaler"
  value       = aws_iam_role.cluster_autoscaler.arn
}

# Security Group Outputs
output "rds_security_group_id" {
  description = "ID of the RDS security group"
  value       = aws_security_group.rds.id
}

output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

# Configuration Outputs (for Helm charts and Kubernetes manifests)
output "mlflow_backend_store_uri" {
  description = "Backend store URI for MLflow (PostgreSQL connection string)"
  value       = "postgresql://${var.rds_username}:PASSWORD_FROM_SECRETS_MANAGER@${aws_db_instance.mlflow.address}:${aws_db_instance.mlflow.port}/${var.rds_database_name}"
  sensitive   = true
}

output "mlflow_artifact_root" {
  description = "Artifact root for MLflow (S3 bucket URI)"
  value       = "s3://${aws_s3_bucket.mlflow_artifacts.id}/"
}

# Kubernetes Configuration Command
output "kubectl_config_command" {
  description = "Command to update kubeconfig for kubectl access"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# Summary Output
output "deployment_summary" {
  description = "Summary of deployed resources"
  value = {
    cluster_name       = module.eks.cluster_name
    cluster_endpoint   = module.eks.cluster_endpoint
    rds_endpoint       = aws_db_instance.mlflow.endpoint
    s3_bucket          = aws_s3_bucket.mlflow_artifacts.id
    region             = var.aws_region
    environment        = var.environment
    mlflow_namespace   = var.mlflow_namespace
    ray_namespace      = var.ray_namespace
  }
}

# Next Steps Instructions
output "next_steps" {
  description = "Commands to run after Terraform apply"
  value = <<-EOT
    # 1. Update kubeconfig
    aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}

    # 2. Verify cluster access
    kubectl get nodes

    # 3. Create namespaces
    kubectl create namespace ${var.mlflow_namespace}
    kubectl create namespace ${var.ray_namespace}

    # 4. Get RDS password from Secrets Manager
    aws secretsmanager get-secret-value --secret-id ${aws_secretsmanager_secret.rds_password.name} --query SecretString --output text

    # 5. Install AWS Load Balancer Controller
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
      -n kube-system \
      --set clusterName=${module.eks.cluster_name} \
      --set serviceAccount.create=true \
      --set serviceAccount.name=aws-load-balancer-controller \
      --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=${aws_iam_role.aws_load_balancer_controller.arn}

    # 6. Deploy MLflow using Helm (see docs/mlflow_remote_setup.md)
  EOT
}
