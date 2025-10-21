#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "MDPG MLOps Platform - Prerequisites Check"
echo "========================================="
echo ""

# Track if any prerequisite is missing
MISSING_PREREQS=0

# Function to check command existence
check_command() {
    local cmd=$1
    local name=$2
    local install_hint=$3

    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $name is installed"
        if [ "$cmd" != "aws" ]; then
            echo "  Version: $($cmd --version 2>&1 | head -1)"
        else
            echo "  Version: $(aws --version 2>&1)"
        fi
    else
        echo -e "${RED}✗${NC} $name is NOT installed"
        echo "  Install: $install_hint"
        MISSING_PREREQS=1
    fi
}

echo "Checking required tools..."
echo ""

# Check AWS CLI
check_command "aws" "AWS CLI" "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"

# Check Terraform
check_command "terraform" "Terraform" "brew install terraform (macOS) or https://www.terraform.io/downloads"

# Check kubectl
check_command "kubectl" "kubectl" "brew install kubectl (macOS) or https://kubernetes.io/docs/tasks/tools/"

# Check Helm
check_command "helm" "Helm" "brew install helm (macOS) or https://helm.sh/docs/intro/install/"

# Check jq (optional but recommended)
check_command "jq" "jq (JSON processor)" "brew install jq (macOS)"

echo ""
echo "Checking AWS credentials..."
echo ""

# Check AWS credentials
if aws sts get-caller-identity &> /dev/null; then
    echo -e "${GREEN}✓${NC} AWS credentials are configured"
    echo "  Account: $(aws sts get-caller-identity --query Account --output text)"
    echo "  User/Role: $(aws sts get-caller-identity --query Arn --output text)"
    echo "  Region: $(aws configure get region || echo "default")"
else
    echo -e "${RED}✗${NC} AWS credentials are NOT configured"
    echo "  Run: aws configure"
    MISSING_PREREQS=1
fi

echo ""
echo "Checking AWS permissions..."
echo ""

# Check basic AWS permissions
echo -n "  EKS permissions... "
if aws eks list-clusters --region us-west-2 &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "    Error: Cannot list EKS clusters. Check IAM permissions."
    MISSING_PREREQS=1
fi

echo -n "  RDS permissions... "
if aws rds describe-db-instances --region us-west-2 &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "    Error: Cannot list RDS instances. Check IAM permissions."
    MISSING_PREREQS=1
fi

echo -n "  S3 permissions... "
if aws s3 ls &> /dev/null; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "    Error: Cannot list S3 buckets. Check IAM permissions."
    MISSING_PREREQS=1
fi

echo ""
echo "Checking directory structure..."
echo ""

# Check required directories
if [ -d "terraform/aws-eks" ]; then
    echo -e "${GREEN}✓${NC} terraform/aws-eks/ exists"
else
    echo -e "${RED}✗${NC} terraform/aws-eks/ does NOT exist"
    MISSING_PREREQS=1
fi

if [ -d "charts/mlflow" ]; then
    echo -e "${GREEN}✓${NC} charts/mlflow/ exists"
else
    echo -e "${RED}✗${NC} charts/mlflow/ does NOT exist"
    MISSING_PREREQS=1
fi

echo ""
echo "Checking Terraform configuration..."
echo ""

cd terraform/aws-eks

if [ -f "terraform.tfvars" ]; then
    echo -e "${GREEN}✓${NC} terraform.tfvars exists"

    # Check if required variables are set
    if grep -q "s3_bucket_name" terraform.tfvars && ! grep -q "s3_bucket_name.*=" terraform.tfvars | grep -q "mdpg-mlops-mlflow-artifacts"; then
        echo -e "${YELLOW}⚠${NC}  Warning: s3_bucket_name may need to be customized (must be globally unique)"
    fi
else
    echo -e "${YELLOW}⚠${NC}  terraform.tfvars does NOT exist"
    echo "  Recommendation: Copy terraform.tfvars.example to terraform.tfvars"
    echo "  Command: cp terraform.tfvars.example terraform.tfvars"
fi

cd - > /dev/null

echo ""
echo "========================================="

if [ $MISSING_PREREQS -eq 0 ]; then
    echo -e "${GREEN}✓ All prerequisites are met!${NC}"
    echo ""
    echo "You can proceed with:"
    echo "  ./scripts/setup/02-deploy-terraform.sh"
    exit 0
else
    echo -e "${RED}✗ Some prerequisites are missing${NC}"
    echo ""
    echo "Please install missing tools and configure AWS credentials."
    echo "Then run this script again."
    exit 1
fi
