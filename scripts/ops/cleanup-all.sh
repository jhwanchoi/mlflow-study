#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - Complete Cleanup"
echo "========================================="
echo ""

echo -e "${RED}⚠️  WARNING: This will DELETE ALL resources!${NC}"
echo ""
echo "This will remove:"
echo "  - Kubernetes deployments (MLflow, Ray, Controllers)"
echo "  - EKS cluster and worker nodes"
echo "  - RDS PostgreSQL database"
echo "  - S3 bucket (if empty)"
echo "  - VPC and networking"
echo "  - IAM roles and policies"
echo ""
echo -e "${YELLOW}This action CANNOT be undone!${NC}"
echo ""

read -p "Are you sure you want to proceed? Type 'DELETE' to confirm: " confirm

if [ "$confirm" != "DELETE" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup process..."
echo ""

# Step 1: Delete Kubernetes resources
echo -e "${BLUE}Step 1: Deleting Kubernetes resources${NC}"
echo ""

if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
    echo "Deleting Helm releases..."

    # Delete MLflow
    if helm list -n mlflow 2>/dev/null | grep -q mlflow; then
        echo "  Deleting MLflow..."
        helm uninstall mlflow -n mlflow || true
    fi

    # Delete Ray
    if helm list -n ray 2>/dev/null | grep -q ray; then
        echo "  Deleting Ray..."
        helm uninstall ray -n ray || true
    fi

    # Delete controllers
    echo "  Deleting AWS Load Balancer Controller..."
    helm uninstall aws-load-balancer-controller -n kube-system 2>/dev/null || true

    echo "  Deleting Cluster Autoscaler..."
    helm uninstall cluster-autoscaler -n kube-system 2>/dev/null || true

    echo ""
    echo "Waiting for LoadBalancers to be deleted (important for VPC cleanup)..."
    sleep 30

    # Delete namespaces
    echo "  Deleting namespaces..."
    kubectl delete namespace mlflow --ignore-not-found=true --timeout=60s || true
    kubectl delete namespace ray --ignore-not-found=true --timeout=60s || true
    kubectl delete namespace monitoring --ignore-not-found=true --timeout=60s || true

    echo -e "${GREEN}✓ Kubernetes resources deleted${NC}"
else
    echo "kubectl not configured or cluster not accessible, skipping..."
fi

echo ""

# Step 2: Destroy Terraform resources
echo -e "${BLUE}Step 2: Destroying Terraform infrastructure${NC}"
echo ""

cd terraform/aws-eks

if [ ! -f "terraform.tfstate" ] && [ ! -f ".terraform/terraform.tfstate" ]; then
    echo "No Terraform state found. Resources may not have been created."
    echo "Checking for remote state..."

    # Try to initialize to get remote state
    terraform init -upgrade 2>/dev/null || true
fi

echo "Planning destroy..."
terraform plan -destroy -out=destroy.tfplan

echo ""
echo -e "${YELLOW}Review the destroy plan above.${NC}"
echo ""
read -p "Proceed with destroying infrastructure? (yes/no): " terraform_confirm

if [ "$terraform_confirm" != "yes" ]; then
    echo "Terraform destroy cancelled."
    rm -f destroy.tfplan
    exit 0
fi

echo ""
echo "Destroying infrastructure (this may take 10-15 minutes)..."
terraform apply destroy.tfplan

rm -f destroy.tfplan

echo ""
echo -e "${GREEN}✓ Terraform infrastructure destroyed${NC}"

cd - > /dev/null

echo ""

# Step 3: Cleanup local files
echo -e "${BLUE}Step 3: Cleaning up local files${NC}"
echo ""

read -p "Delete local state files and outputs? (y/n): " cleanup_local

if [ "$cleanup_local" = "y" ]; then
    echo "  Removing .terraform-outputs.json..."
    rm -f .terraform-outputs.json

    echo "  Removing Terraform state backups..."
    rm -f terraform/aws-eks/terraform.tfstate.backup
    rm -f terraform/aws-eks/.terraform.tfstate.lock.info

    echo "  Removing temporary files..."
    rm -f /tmp/aws_verify.log
    rm -f /tmp/mlflow-values.yaml

    echo -e "${GREEN}✓ Local files cleaned${NC}"
else
    echo "Keeping local files"
fi

echo ""

# Step 4: Verify cleanup
echo -e "${BLUE}Step 4: Verifying cleanup${NC}"
echo ""

echo "Checking for remaining resources..."
echo ""

REGION=$(aws configure get region || echo "ap-northeast-2")

echo "EKS Clusters in $REGION:"
aws eks list-clusters --region "$REGION" 2>/dev/null || echo "  None found or unable to check"

echo ""
echo "RDS Instances in $REGION:"
aws rds describe-db-instances --region "$REGION" --query 'DBInstances[?starts_with(DBInstanceIdentifier, `mdpg`)].DBInstanceIdentifier' --output text 2>/dev/null || echo "  None found or unable to check"

echo ""
echo "S3 Buckets with 'mdpg' prefix:"
aws s3 ls 2>/dev/null | grep mdpg || echo "  None found"

echo ""

# Step 5: Summary
echo "========================================="
echo -e "${GREEN}Cleanup Complete!${NC}"
echo "========================================="
echo ""
echo "What was deleted:"
echo "  ✓ Kubernetes resources (MLflow, Ray, Controllers)"
echo "  ✓ EKS cluster and nodes"
echo "  ✓ RDS database"
echo "  ✓ VPC and networking"
echo "  ✓ IAM roles"
echo ""
echo "Note: S3 buckets may remain if they contain objects."
echo "To delete S3 bucket manually:"
echo "  aws s3 rm s3://YOUR-BUCKET-NAME --recursive"
echo "  aws s3 rb s3://YOUR-BUCKET-NAME"
echo ""
echo "Cost: You will no longer be charged for these resources."
echo ""
