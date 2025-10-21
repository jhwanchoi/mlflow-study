#!/bin/bash
set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "Terraform S3 Backend Bootstrap"
echo "========================================="
echo ""

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo "ap-northeast-2")

# Generate unique bucket name
BACKEND_BUCKET="mdpg-terraform-state-${ACCOUNT_ID}"
LOCK_TABLE="mdpg-terraform-locks"

echo -e "${BLUE}Configuration:${NC}"
echo "  AWS Account: $ACCOUNT_ID"
echo "  Region: $REGION"
echo "  Backend Bucket: $BACKEND_BUCKET"
echo "  Lock Table: $LOCK_TABLE"
echo ""

read -p "Create Terraform backend resources? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}Step 1: Creating S3 bucket for Terraform state${NC}"
echo ""

# Check if bucket exists
if aws s3 ls "s3://$BACKEND_BUCKET" 2>/dev/null; then
    echo -e "${GREEN}✓ Bucket already exists${NC}"
else
    # Create bucket
    if [ "$REGION" = "us-east-1" ]; then
        aws s3api create-bucket \
            --bucket "$BACKEND_BUCKET" \
            --region "$REGION"
    else
        aws s3api create-bucket \
            --bucket "$BACKEND_BUCKET" \
            --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi

    echo -e "${GREEN}✓ Bucket created${NC}"
fi

echo ""
echo -e "${BLUE}Step 2: Enabling bucket versioning${NC}"
echo ""

aws s3api put-bucket-versioning \
    --bucket "$BACKEND_BUCKET" \
    --versioning-configuration Status=Enabled

echo -e "${GREEN}✓ Versioning enabled${NC}"

echo ""
echo -e "${BLUE}Step 3: Enabling bucket encryption${NC}"
echo ""

aws s3api put-bucket-encryption \
    --bucket "$BACKEND_BUCKET" \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'

echo -e "${GREEN}✓ Encryption enabled${NC}"

echo ""
echo -e "${BLUE}Step 4: Blocking public access${NC}"
echo ""

aws s3api put-public-access-block \
    --bucket "$BACKEND_BUCKET" \
    --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo -e "${GREEN}✓ Public access blocked${NC}"

echo ""
echo -e "${BLUE}Step 5: Creating DynamoDB table for state locking${NC}"
echo ""

# Check if table exists
if aws dynamodb describe-table --table-name "$LOCK_TABLE" --region "$REGION" 2>/dev/null; then
    echo -e "${GREEN}✓ Table already exists${NC}"
else
    aws dynamodb create-table \
        --table-name "$LOCK_TABLE" \
        --attribute-definitions AttributeName=LockID,AttributeType=S \
        --key-schema AttributeName=LockID,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --region "$REGION"

    echo "Waiting for table to be active..."
    aws dynamodb wait table-exists --table-name "$LOCK_TABLE" --region "$REGION"

    echo -e "${GREEN}✓ Table created${NC}"
fi

echo ""
echo -e "${GREEN}✓ Terraform backend bootstrap complete!${NC}"
echo ""

echo "========================================="
echo "Backend Configuration:"
echo "========================================="
echo ""
echo "Update terraform/aws-eks/main.tf with:"
echo ""
cat <<EOF
  backend "s3" {
    bucket         = "$BACKEND_BUCKET"
    key            = "mlops/eks/terraform.tfstate"
    region         = "$REGION"
    encrypt        = true
    dynamodb_table = "$LOCK_TABLE"
  }
EOF
echo ""

echo "Or reinitialize with backend config:"
echo ""
echo "  cd terraform/aws-eks"
echo "  terraform init -reconfigure \\"
echo "    -backend-config=\"bucket=$BACKEND_BUCKET\" \\"
echo "    -backend-config=\"key=mlops/eks/terraform.tfstate\" \\"
echo "    -backend-config=\"region=$REGION\" \\"
echo "    -backend-config=\"encrypt=true\" \\"
echo "    -backend-config=\"dynamodb_table=$LOCK_TABLE\""
echo ""

echo "Cost: Very minimal (~\$0.50/month for DynamoDB, ~\$0.02/month for S3)"
echo ""
