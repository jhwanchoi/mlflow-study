#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - kubectl Configuration"
echo "========================================="
echo ""

# Check if Terraform outputs exist
if [ ! -f ".terraform-outputs.json" ]; then
    echo -e "${RED}Error: .terraform-outputs.json not found${NC}"
    echo "Please run 02-deploy-terraform.sh first"
    exit 1
fi

# Extract cluster name and region
CLUSTER_NAME=$(jq -r '.cluster_name.value' .terraform-outputs.json)
AWS_REGION=$(jq -r '.deployment_summary.value.region' .terraform-outputs.json)

echo -e "${BLUE}Configuring kubectl for cluster: $CLUSTER_NAME${NC}"
echo "Region: $AWS_REGION"
echo ""

# Update kubeconfig
aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"

echo ""
echo -e "${GREEN}✓ kubeconfig updated${NC}"
echo ""

echo "Verifying cluster access..."
echo ""

# Verify cluster access
kubectl cluster-info
echo ""

echo "Checking nodes..."
kubectl get nodes
echo ""

echo -e "${GREEN}✓ kubectl configured successfully!${NC}"
echo ""

echo "Creating namespaces..."
echo ""

# Create namespaces
kubectl create namespace mlflow --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace ray --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo -e "${GREEN}✓ Namespaces created${NC}"
echo "  - mlflow"
echo "  - ray"
echo "  - monitoring"
echo ""

echo "Current context:"
kubectl config current-context
echo ""

echo "Run next script:"
echo "  ./scripts/setup/04-install-controllers.sh"
