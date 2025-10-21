#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - Terraform Deployment"
echo "========================================="
echo ""

cd terraform/aws-eks

# Check if terraform.tfvars exists
if [ ! -f "terraform.tfvars" ]; then
    echo -e "${RED}Error: terraform.tfvars not found${NC}"
    echo "Please copy terraform.tfvars.example to terraform.tfvars and customize it:"
    echo "  cp terraform.tfvars.example terraform.tfvars"
    exit 1
fi

echo -e "${BLUE}Step 1: Initialize Terraform${NC}"
echo ""
terraform init
echo ""

echo -e "${BLUE}Step 2: Validate Configuration${NC}"
echo ""
terraform validate
echo ""

echo -e "${BLUE}Step 3: Create Execution Plan${NC}"
echo ""
terraform plan -out=tfplan
echo ""

echo "========================================="
echo -e "${YELLOW}Review the plan above carefully${NC}"
echo "This will create:"
echo "  - VPC with 6 subnets across 3 AZs"
echo "  - EKS cluster with 2-5 worker nodes"
echo "  - RDS PostgreSQL instance"
echo "  - S3 bucket for artifacts"
echo "  - IAM roles and security groups"
echo ""
echo "Estimated cost: ~\$168/month (dev environment)"
echo "========================================="
echo ""

read -p "Do you want to apply this plan? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Deployment cancelled."
    rm -f tfplan
    exit 0
fi

echo ""
echo -e "${BLUE}Step 4: Apply Terraform Configuration${NC}"
echo ""
echo "This will take approximately 15-20 minutes..."
echo ""

terraform apply tfplan

echo ""
echo -e "${GREEN}✓ Terraform deployment completed!${NC}"
echo ""

echo "========================================="
echo "Deployment Summary:"
echo "========================================="
terraform output deployment_summary
echo ""

echo "Important outputs:"
echo "  - RDS Endpoint: $(terraform output -raw rds_endpoint)"
echo "  - S3 Bucket: $(terraform output -raw s3_bucket_name)"
echo "  - EKS Cluster: $(terraform output -raw cluster_name)"
echo ""

# Save outputs to a file for later use
terraform output -json > ../../.terraform-outputs.json
echo -e "${GREEN}✓ Outputs saved to .terraform-outputs.json${NC}"
echo ""

echo "Next steps:"
terraform output -raw next_steps
echo ""

cd - > /dev/null

echo "Run next script:"
echo "  ./scripts/setup/03-configure-kubectl.sh"
