#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - Install Kubernetes Controllers"
echo "========================================="
echo ""

# Check prerequisites
if [ ! -f ".terraform-outputs.json" ]; then
    echo -e "${RED}Error: .terraform-outputs.json not found${NC}"
    exit 1
fi

CLUSTER_NAME=$(jq -r '.cluster_name.value' .terraform-outputs.json)
ALB_CONTROLLER_ROLE_ARN=$(jq -r '.aws_load_balancer_controller_role_arn.value' .terraform-outputs.json)
CLUSTER_AUTOSCALER_ROLE_ARN=$(jq -r '.cluster_autoscaler_role_arn.value' .terraform-outputs.json)
AWS_REGION=$(jq -r '.deployment_summary.value.region' .terraform-outputs.json)

echo "Cluster: $CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo ""

# Add Helm repositories
echo -e "${BLUE}Step 1: Adding Helm repositories${NC}"
echo ""

helm repo add eks https://aws.github.io/eks-charts
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm repo update

echo ""
echo -e "${GREEN}✓ Helm repositories added${NC}"
echo ""

# Install AWS Load Balancer Controller
echo -e "${BLUE}Step 2: Installing AWS Load Balancer Controller${NC}"
echo ""

# Check if already installed
if helm list -n kube-system | grep -q aws-load-balancer-controller; then
    echo -e "${YELLOW}AWS Load Balancer Controller already installed, upgrading...${NC}"
    helm upgrade aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=true \
        --set serviceAccount.name=aws-load-balancer-controller \
        --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$ALB_CONTROLLER_ROLE_ARN"
else
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=true \
        --set serviceAccount.name=aws-load-balancer-controller \
        --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$ALB_CONTROLLER_ROLE_ARN"
fi

echo ""
echo "Waiting for AWS Load Balancer Controller to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/aws-load-balancer-controller -n kube-system || true

echo ""
echo -e "${GREEN}✓ AWS Load Balancer Controller installed${NC}"
echo ""

# Install Cluster Autoscaler
echo -e "${BLUE}Step 3: Installing Cluster Autoscaler${NC}"
echo ""

if helm list -n kube-system | grep -q cluster-autoscaler; then
    echo -e "${YELLOW}Cluster Autoscaler already installed, upgrading...${NC}"
    helm upgrade cluster-autoscaler autoscaler/cluster-autoscaler \
        -n kube-system \
        --set autoDiscovery.clusterName="$CLUSTER_NAME" \
        --set awsRegion="$AWS_REGION" \
        --set rbac.serviceAccount.create=true \
        --set rbac.serviceAccount.name=cluster-autoscaler \
        --set rbac.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$CLUSTER_AUTOSCALER_ROLE_ARN"
else
    helm install cluster-autoscaler autoscaler/cluster-autoscaler \
        -n kube-system \
        --set autoDiscovery.clusterName="$CLUSTER_NAME" \
        --set awsRegion="$AWS_REGION" \
        --set rbac.serviceAccount.create=true \
        --set rbac.serviceAccount.name=cluster-autoscaler \
        --set rbac.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$CLUSTER_AUTOSCALER_ROLE_ARN"
fi

echo ""
echo -e "${GREEN}✓ Cluster Autoscaler installed${NC}"
echo ""

# Verify installations
echo -e "${BLUE}Step 4: Verifying installations${NC}"
echo ""

echo "Checking AWS Load Balancer Controller..."
kubectl get deployment aws-load-balancer-controller -n kube-system
echo ""

echo "Checking Cluster Autoscaler..."
kubectl get deployment cluster-autoscaler -n kube-system
echo ""

echo -e "${GREEN}✓ All controllers installed successfully!${NC}"
echo ""

echo "Installed components:"
echo "  ✓ AWS Load Balancer Controller (for ALB/NLB ingress)"
echo "  ✓ Cluster Autoscaler (for node auto-scaling)"
echo ""

echo "Run next script:"
echo "  ./scripts/setup/05-deploy-mlflow.sh"
