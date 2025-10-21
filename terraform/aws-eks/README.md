# MDPG MLOps Platform - AWS EKS Infrastructure

This directory contains Terraform configurations for deploying the MDPG MLOps platform on AWS EKS.

## Architecture Overview

The infrastructure includes:

- **Amazon EKS**: Managed Kubernetes cluster (v1.28+)
- **Amazon VPC**: Network isolation with public/private subnets across 3 AZs
- **Amazon RDS**: PostgreSQL database for MLflow metadata
- **Amazon S3**: Object storage for MLflow artifacts
- **IAM Roles (IRSA)**: Pod-level permissions for secure access
- **Security Groups**: Network access control for all resources

## Cost Estimate

**Development Environment**: ~$168/month
- EKS Control Plane: $73/month
- Worker Nodes (2x t3.medium): $60/month
- RDS (db.t3.small): $30/month
- S3: ~$5/month

**Production Environment**: ~$292/month (before optimization)

See [docs/cost_estimation.md](../../docs/cost_estimation.md) for detailed cost analysis.

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
   ```bash
   aws configure
   ```

2. **Terraform** (>= 1.0)
   ```bash
   brew install terraform  # macOS
   ```

3. **kubectl** for Kubernetes management
   ```bash
   brew install kubectl  # macOS
   ```

4. **Helm** for deploying applications
   ```bash
   brew install helm  # macOS
   ```

## Quick Start

### 1. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings
vim terraform.tfvars
```

Key variables to customize:
- `s3_bucket_name`: Must be globally unique
- `aws_region`: Your preferred AWS region
- `environment`: dev, staging, or prod

### 2. Initialize Terraform

```bash
terraform init
```

For remote state (recommended):
```bash
# First, create the S3 bucket and DynamoDB table manually or use bootstrap
terraform init \
  -backend-config="bucket=mdpg-terraform-state" \
  -backend-config="key=mlops/eks/terraform.tfstate" \
  -backend-config="region=us-west-2"
```

### 3. Review the Plan

```bash
terraform plan -out=tfplan
```

Review the resources that will be created:
- VPC with 6 subnets (3 public, 3 private)
- EKS cluster with 2-5 worker nodes
- RDS PostgreSQL instance
- S3 bucket with lifecycle policies
- IAM roles and policies
- Security groups

### 4. Apply the Configuration

```bash
terraform apply tfplan
```

This will take approximately 15-20 minutes.

### 5. Configure kubectl

```bash
aws eks update-kubeconfig --region us-west-2 --name mdpg-mlops-eks
kubectl get nodes
```

## Post-Deployment Steps

After Terraform completes, run these commands:

### 1. Create Kubernetes Namespaces

```bash
kubectl create namespace mlflow
kubectl create namespace ray
```

### 2. Install AWS Load Balancer Controller

```bash
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=mdpg-mlops-eks \
  --set serviceAccount.create=true \
  --set serviceAccount.name=aws-load-balancer-controller \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=$(terraform output -raw aws_load_balancer_controller_role_arn)
```

### 3. Get RDS Credentials

```bash
# Get the secret ARN from Terraform output
terraform output rds_secret_arn

# Retrieve the credentials
aws secretsmanager get-secret-value \
  --secret-id $(terraform output -raw rds_secret_arn) \
  --query SecretString \
  --output text | jq .
```

### 4. Deploy MLflow

See [docs/mlflow_remote_setup.md](../../docs/mlflow_remote_setup.md) for detailed instructions.

## File Structure

```
terraform/aws-eks/
├── main.tf                    # VPC, EKS cluster, security groups
├── rds.tf                     # PostgreSQL RDS instance
├── s3.tf                      # S3 bucket for artifacts
├── iam.tf                     # IAM roles and policies (IRSA)
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output values
├── terraform.tfvars.example   # Configuration template
└── README.md                  # This file
```

## Terraform Modules Used

- [terraform-aws-modules/vpc/aws](https://registry.terraform.io/modules/terraform-aws-modules/vpc/aws) (~> 5.0)
- [terraform-aws-modules/eks/aws](https://registry.terraform.io/modules/terraform-aws-modules/eks/aws) (~> 19.0)

## Important Outputs

After `terraform apply`, you can view outputs:

```bash
# All outputs
terraform output

# Specific output
terraform output cluster_endpoint
terraform output rds_endpoint
terraform output s3_bucket_name
terraform output mlflow_iam_role_arn

# Next steps instructions
terraform output -raw next_steps
```

## Resource Naming Convention

All resources follow the pattern: `mdpg-{service}-{resource-type}-{environment}`

Examples:
- EKS Cluster: `mdpg-mlops-eks`
- RDS Instance: `mdpg-mlops-mlflow-db`
- S3 Bucket: `mdpg-mlops-mlflow-artifacts`
- IAM Role: `mdpg-mlops-mlflow-s3-access`

## Security Features

- **Encryption**: All data encrypted at rest (RDS, S3, EBS)
- **Network Isolation**: Private subnets for data plane
- **IRSA**: Pod-level IAM permissions without node-level credentials
- **Security Groups**: Principle of least privilege
- **Secrets Management**: RDS credentials stored in AWS Secrets Manager
- **TLS**: Enforced for S3 access

## Scaling

### Manual Scaling

```bash
# Scale node group
aws eks update-nodegroup-config \
  --cluster-name mdpg-mlops-eks \
  --nodegroup-name general \
  --scaling-config desiredSize=4
```

### Auto Scaling

Install Cluster Autoscaler:
```bash
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName=mdpg-mlops-eks \
  --set awsRegion=us-west-2 \
  --set rbac.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=$(terraform output -raw cluster_autoscaler_role_arn)
```

## Monitoring

CloudWatch alarms are automatically created for:
- RDS CPU utilization (>80%)
- RDS storage space (<5GB)
- RDS database connections (>80)
- S3 bucket size (>100GB)

View alarms:
```bash
aws cloudwatch describe-alarms
```

## Troubleshooting

### EKS Cluster Access Issues

```bash
# Verify AWS credentials
aws sts get-caller-identity

# Update kubeconfig
aws eks update-kubeconfig --name mdpg-mlops-eks --region us-west-2

# Check cluster status
aws eks describe-cluster --name mdpg-mlops-eks
```

### Node Group Issues

```bash
# Check node group status
kubectl get nodes
kubectl describe nodes

# View node group in AWS
aws eks describe-nodegroup \
  --cluster-name mdpg-mlops-eks \
  --nodegroup-name general
```

### RDS Connection Issues

```bash
# Test connection from a pod
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h $(terraform output -raw rds_address) -U mlflow -d mlflow
```

## Cleanup

To destroy all resources:

```bash
# Review what will be destroyed
terraform plan -destroy

# Destroy all resources
terraform destroy
```

**Warning**: This will delete all data. Ensure you have backups before proceeding.

For production environments:
1. Export MLflow data from RDS
2. Download critical artifacts from S3
3. Document any manual configurations

## Next Steps

1. **Deploy MLflow**: Follow [docs/mlflow_remote_setup.md](../../docs/mlflow_remote_setup.md)
2. **Configure DNS**: Set up Route53 or use LoadBalancer DNS
3. **Set up SSL/TLS**: Use AWS Certificate Manager
4. **Deploy Ray Cluster**: Follow [docs/ray_tune_guide.md](../../docs/ray_tune_guide.md)
5. **Configure Monitoring**: Set up Prometheus/Grafana

## Support

For issues or questions:
- Check [docs/eks_infrastructure.md](../../docs/eks_infrastructure.md) for detailed documentation
- Review AWS EKS documentation: https://docs.aws.amazon.com/eks/
- Check Terraform module documentation: https://registry.terraform.io/

## License

Internal MDPG use only.
