# MDPG MLOps Platform - Deployment Scripts

Automated deployment scripts for setting up the complete MLOps platform on AWS EKS.

## Overview

These scripts automate the deployment of:
- AWS EKS infrastructure (VPC, EKS, RDS, S3)
- Kubernetes controllers (ALB Controller, Cluster Autoscaler)
- MLflow tracking server
- Verification and testing

## Quick Start

### Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.0
- kubectl
- Helm >= 3.8
- jq (JSON processor)

### Step-by-Step Deployment

```bash
# 1. Verify prerequisites
./scripts/setup/01-verify-prerequisites.sh

# 2. Deploy AWS infrastructure with Terraform (15-20 minutes)
./scripts/setup/02-deploy-terraform.sh

# 3. Configure kubectl and create namespaces
./scripts/setup/03-configure-kubectl.sh

# 4. Install Kubernetes controllers
./scripts/setup/04-install-controllers.sh

# 5. Deploy MLflow
./scripts/setup/05-deploy-mlflow.sh

# 6. Verify deployment
./scripts/setup/06-verify-deployment.sh
```

## Scripts Details

### 01-verify-prerequisites.sh

Checks for required tools and AWS permissions.

**Validates:**
- AWS CLI, Terraform, kubectl, Helm installation
- AWS credentials and permissions
- Project directory structure
- terraform.tfvars configuration

**Usage:**
```bash
./scripts/setup/01-verify-prerequisites.sh
```

**Exit codes:**
- 0: All prerequisites met
- 1: Missing prerequisites

### 02-deploy-terraform.sh

Deploys AWS infrastructure using Terraform.

**Creates:**
- VPC with 6 subnets across 3 AZs
- EKS cluster with 2-5 worker nodes
- RDS PostgreSQL for MLflow
- S3 bucket for artifacts
- IAM roles and security groups

**Usage:**
```bash
./scripts/setup/02-deploy-terraform.sh
```

**Duration:** ~15-20 minutes

**Output:** `.terraform-outputs.json` (contains all Terraform outputs)

**Cost:** ~$168/month (dev environment)

### 03-configure-kubectl.sh

Configures kubectl to access the EKS cluster.

**Actions:**
- Updates kubeconfig
- Verifies cluster access
- Creates namespaces (mlflow, ray, monitoring)

**Usage:**
```bash
./scripts/setup/03-configure-kubectl.sh
```

**Prerequisites:**
- 02-deploy-terraform.sh completed
- .terraform-outputs.json exists

### 04-install-controllers.sh

Installs required Kubernetes controllers.

**Installs:**
- AWS Load Balancer Controller (for ALB/NLB)
- Cluster Autoscaler (for node scaling)

**Usage:**
```bash
./scripts/setup/04-install-controllers.sh
```

**Prerequisites:**
- kubectl configured
- .terraform-outputs.json exists

### 05-deploy-mlflow.sh

Deploys MLflow tracking server using Helm.

**Actions:**
- Retrieves RDS credentials from Secrets Manager
- Creates Kubernetes secret
- Generates Helm values file
- Deploys MLflow with 2 replicas
- Configures ALB ingress

**Usage:**
```bash
./scripts/setup/05-deploy-mlflow.sh
```

**Prerequisites:**
- All previous scripts completed
- MLflow namespace created

**Output:** MLflow UI accessible via ALB

### 06-verify-deployment.sh

Verifies all components are running correctly.

**Checks:**
- EKS cluster connectivity
- Namespace existence
- Controller deployments
- MLflow pods and services
- ALB provisioning
- Health endpoints

**Usage:**
```bash
./scripts/setup/06-verify-deployment.sh
```

**Output:** Detailed status of all components

## Configuration

### Before Running

1. Copy and customize Terraform variables:
```bash
cd terraform/aws-eks
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars
```

2. Important variables to customize:
   - `s3_bucket_name`: Must be globally unique
   - `aws_region`: Your preferred AWS region
   - `environment`: dev, staging, or prod

### Cost Optimization

Development environment (~$168/month):
- t3.medium worker nodes
- db.t3.small RDS
- 2 node minimum

Production optimized (~$132/month):
- Use Spot instances
- Enable cluster autoscaler
- S3 lifecycle policies (already configured)

## Troubleshooting

### Script Fails

1. **Check prerequisites:**
```bash
./scripts/setup/01-verify-prerequisites.sh
```

2. **View detailed error output:**
- Scripts show colored output (green ✓, red ✗, yellow ⚠)
- Read error messages carefully

3. **Check AWS permissions:**
```bash
aws sts get-caller-identity
aws eks list-clusters --region us-west-2
```

### Terraform Errors

1. **State lock issues:**
```bash
cd terraform/aws-eks
terraform force-unlock <lock-id>
```

2. **Resource already exists:**
- Check AWS console for existing resources
- Use `terraform import` if needed

3. **Clean start:**
```bash
cd terraform/aws-eks
rm -rf .terraform terraform.tfstate*
terraform init
```

### Kubernetes Issues

1. **kubectl not working:**
```bash
aws eks update-kubeconfig --name mdpg-mlops-eks --region us-west-2
kubectl cluster-info
```

2. **Pods not starting:**
```bash
kubectl get pods -n mlflow
kubectl describe pod <pod-name> -n mlflow
kubectl logs <pod-name> -n mlflow
```

3. **ALB not provisioned:**
- Wait 2-3 minutes for ALB creation
- Check controller logs:
```bash
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

## Cleanup

To destroy all resources:

```bash
# 1. Delete Kubernetes resources
helm uninstall mlflow -n mlflow
helm uninstall aws-load-balancer-controller -n kube-system
helm uninstall cluster-autoscaler -n kube-system

# 2. Destroy AWS infrastructure
cd terraform/aws-eks
terraform destroy

# 3. Clean up local files
rm -f ../.terraform-outputs.json
```

**Warning:** This will delete all data. Backup before destroying!

## Manual Steps After Deployment

### 1. DNS Configuration

Point your domain to the ALB:

```bash
# Get ALB endpoint
kubectl get ingress mlflow -n mlflow

# Create Route53 record
aws route53 change-resource-record-sets \
  --hosted-zone-id YOUR_ZONE_ID \
  --change-batch file://dns-change.json
```

### 2. SSL/TLS Setup

1. Request certificate in ACM:
```bash
aws acm request-certificate \
  --domain-name mlflow.mdpg.ai \
  --validation-method DNS \
  --region us-west-2
```

2. Add certificate ARN to ingress:
```bash
helm upgrade mlflow ./charts/mlflow \
  -n mlflow \
  --set ingress.annotations."alb\.ingress\.kubernetes\.io/certificate-arn"="arn:aws:acm:..."
```

### 3. Enable Authentication

Update MLflow values:

```yaml
mlflow:
  authentication:
    enabled: true
    adminUsername: "admin"
    adminPassword: "secure-password"
```

Upgrade deployment:
```bash
helm upgrade mlflow ./charts/mlflow -n mlflow -f values-prod.yaml
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to EKS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Deploy infrastructure
        run: |
          ./scripts/setup/02-deploy-terraform.sh
          ./scripts/setup/03-configure-kubectl.sh
          ./scripts/setup/04-install-controllers.sh
          ./scripts/setup/05-deploy-mlflow.sh
          ./scripts/setup/06-verify-deployment.sh
```

## Monitoring

### View Logs

```bash
# MLflow logs
kubectl logs -n mlflow -l app.kubernetes.io/name=mlflow -f

# Controller logs
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller -f

# All events
kubectl get events -n mlflow --sort-by='.lastTimestamp'
```

### Metrics

```bash
# Resource usage
kubectl top pods -n mlflow
kubectl top nodes

# HPA status
kubectl get hpa -n mlflow

# Cluster info
kubectl cluster-info
kubectl get nodes
```

## Support

For issues:
- Check [terraform/aws-eks/README.md](../terraform/aws-eks/README.md)
- Review [charts/mlflow/README.md](../charts/mlflow/README.md)
- See [docs/eks_infrastructure.md](../docs/eks_infrastructure.md)

## License

Internal MDPG use only.
