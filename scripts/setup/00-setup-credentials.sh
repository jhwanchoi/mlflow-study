#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - AWS Credentials Setup"
echo "========================================="
echo ""

echo -e "${BLUE}This script will help you set up AWS credentials${NC}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Please install it first:"
    echo "  brew install awscli  # macOS"
    echo "  or visit: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

echo "Choose your setup method:"
echo "  1) AWS CLI configure (recommended)"
echo "  2) Use existing credentials"
echo "  3) Use AWS Profile"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Running AWS configure...${NC}"
        echo ""
        echo "You'll need:"
        echo "  - AWS Access Key ID"
        echo "  - AWS Secret Access Key"
        echo "  - Default region (e.g., ap-northeast-2 for Seoul, us-west-2 for Oregon)"
        echo ""
        echo "Note: If you have special characters (+, /, =) in your Secret Key,"
        echo "      the verification might fail, but the credentials will still be saved."
        echo ""
        aws configure
        ;;
    2)
        echo ""
        echo -e "${BLUE}Using existing credentials${NC}"
        if [ -f ~/.aws/credentials ]; then
            echo -e "${GREEN}✓ Credentials file found${NC}"
        else
            echo "No credentials file found at ~/.aws/credentials"
            echo "Please run: aws configure"
            exit 1
        fi
        ;;
    3)
        echo ""
        read -p "Enter AWS profile name: " profile_name
        export AWS_PROFILE=$profile_name
        echo "export AWS_PROFILE=$profile_name" >> ~/.bashrc
        echo "export AWS_PROFILE=$profile_name" >> ~/.zshrc 2>/dev/null || true
        echo -e "${GREEN}✓ Profile set to: $profile_name${NC}"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}Verifying AWS credentials...${NC}"
echo ""

# Use --no-cli-pager to avoid pager issues and set proper error handling
AWS_PAGER="" aws sts get-caller-identity 2>&1 | tee /tmp/aws_verify.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ AWS credentials are valid!${NC}"
    echo ""
    ACCOUNT=$(AWS_PAGER="" aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
    ARN=$(AWS_PAGER="" aws sts get-caller-identity --query Arn --output text 2>/dev/null || echo "unknown")
    REGION=$(aws configure get region 2>/dev/null || echo 'not set')

    echo "Account: $ACCOUNT"
    echo "User/Role: $ARN"
    echo "Region: $REGION"
else
    echo ""
    echo "⚠️  Credential verification had issues, but this might be okay."
    echo ""
    echo "If you successfully ran 'aws configure', the credentials should be saved."
    echo "Let's check the credentials file..."
    echo ""

    if [ -f ~/.aws/credentials ]; then
        echo -e "${GREEN}✓ Credentials file exists at ~/.aws/credentials${NC}"
        echo ""
        echo "Trying alternative verification method..."

        # Try with explicit profile
        if AWS_PAGER="" aws sts get-caller-identity --profile default 2>/dev/null; then
            echo -e "${GREEN}✓ Credentials work with default profile${NC}"
        else
            echo ""
            echo "The credentials file exists but verification is failing."
            echo "This can happen with special characters in the Secret Key."
            echo ""
            echo "Recommendation:"
            echo "  1. Continue with deployment (credentials might still work)"
            echo "  2. Or generate a new Access Key from AWS Console"
            echo ""
            read -p "Continue anyway? (y/n): " continue_choice
            if [ "$continue_choice" != "y" ]; then
                exit 1
            fi
        fi
    else
        echo "Failed to verify credentials. Please check your AWS Access Key and Secret Key."
        exit 1
    fi
fi

echo ""
echo "========================================="
echo -e "${BLUE}Setting up Terraform variables${NC}"
echo "========================================="
echo ""

cd terraform/aws-eks

if [ ! -f terraform.tfvars ]; then
    echo "Creating terraform.tfvars from template..."
    cp terraform.tfvars.example terraform.tfvars

    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: Please customize terraform.tfvars${NC}"
    echo ""
    echo "Required changes:"
    echo "  1. s3_bucket_name - Must be globally unique!"
    echo "     Current: mdpg-mlops-mlflow-artifacts"
    echo "     Suggestion: mdpg-mlops-mlflow-artifacts-$(date +%Y%m%d)"
    echo ""
    echo "  2. Review and adjust:"
    echo "     - aws_region"
    echo "     - node_instance_types (t3.medium or t3.small)"
    echo "     - rds_instance_class (db.t3.small or db.t3.micro)"
    echo ""

    # Generate unique bucket name
    UNIQUE_SUFFIX=$(date +%Y%m%d)-$(openssl rand -hex 3)
    BUCKET_NAME="mdpg-mlops-mlflow-artifacts-${UNIQUE_SUFFIX}"

    # Update bucket name in terraform.tfvars
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/mdpg-mlops-mlflow-artifacts/${BUCKET_NAME}/" terraform.tfvars
    else
        sed -i "s/mdpg-mlops-mlflow-artifacts/${BUCKET_NAME}/" terraform.tfvars
    fi

    echo -e "${GREEN}✓ Updated s3_bucket_name to: ${BUCKET_NAME}${NC}"
    echo ""

    read -p "Open terraform.tfvars for editing? (y/n): " edit_tfvars
    if [ "$edit_tfvars" = "y" ]; then
        ${EDITOR:-vim} terraform.tfvars
    fi
else
    echo -e "${GREEN}✓ terraform.tfvars already exists${NC}"
fi

cd - > /dev/null

echo ""
echo "========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Your AWS credentials are configured."
echo ""
echo "Next steps:"
echo "  1. Review terraform.tfvars:"
echo "     vim terraform/aws-eks/terraform.tfvars"
echo ""
echo "  2. Run the deployment scripts:"
echo "     ./scripts/setup/01-verify-prerequisites.sh"
echo "     ./scripts/setup/02-deploy-terraform.sh"
echo ""
echo "Estimated cost: ~\$168/month (dev environment)"
echo "Estimated deployment time: ~30 minutes"
echo ""
