#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - Verify Deployment"
echo "========================================="
echo ""

# Function to check status
check_status() {
    local name=$1
    local command=$2

    echo -n "  $name... "
    if eval "$command" &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        return 1
    fi
}

# Check EKS cluster
echo -e "${BLUE}Checking EKS Cluster${NC}"
check_status "Cluster connection" "kubectl cluster-info"
check_status "Nodes ready" "kubectl get nodes | grep -q Ready"
echo ""

# Check namespaces
echo -e "${BLUE}Checking Namespaces${NC}"
check_status "mlflow namespace" "kubectl get namespace mlflow"
check_status "ray namespace" "kubectl get namespace ray"
check_status "monitoring namespace" "kubectl get namespace monitoring"
echo ""

# Check controllers
echo -e "${BLUE}Checking Controllers${NC}"
check_status "AWS Load Balancer Controller" "kubectl get deployment aws-load-balancer-controller -n kube-system"
check_status "Cluster Autoscaler" "kubectl get deployment cluster-autoscaler -n kube-system"
echo ""

# Check MLflow
echo -e "${BLUE}Checking MLflow Deployment${NC}"
check_status "MLflow deployment" "kubectl get deployment mlflow -n mlflow"
check_status "MLflow pods running" "kubectl get pods -n mlflow | grep mlflow | grep Running"
check_status "MLflow service" "kubectl get svc mlflow -n mlflow"
check_status "MLflow ingress" "kubectl get ingress mlflow -n mlflow"
echo ""

# Check HPA
echo -e "${BLUE}Checking Auto-scaling${NC}"
kubectl get hpa -n mlflow
echo ""

# Check pod logs
echo -e "${BLUE}Checking MLflow Logs (last 10 lines)${NC}"
echo "========================================="
MLFLOW_POD=$(kubectl get pods -n mlflow -l app.kubernetes.io/name=mlflow -o jsonpath='{.items[0].metadata.name}')
if [ -n "$MLFLOW_POD" ]; then
    kubectl logs -n mlflow "$MLFLOW_POD" --tail=10
else
    echo -e "${RED}No MLflow pods found${NC}"
fi
echo "========================================="
echo ""

# Test MLflow health endpoint
echo -e "${BLUE}Testing MLflow Health Endpoint${NC}"
echo ""

# Get service endpoint
MLFLOW_SVC=$(kubectl get svc mlflow -n mlflow -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

if [ -z "$MLFLOW_SVC" ]; then
    # Try cluster IP
    MLFLOW_IP=$(kubectl get svc mlflow -n mlflow -o jsonpath='{.spec.clusterIP}')
    echo "Testing internal endpoint: $MLFLOW_IP"

    kubectl run curl-test --image=curlimages/curl --rm -it --restart=Never -- \
        curl -s "http://$MLFLOW_IP/health" || true
else
    echo "External endpoint: $MLFLOW_SVC"
fi

echo ""

# Get ALB information
echo -e "${BLUE}ALB Information${NC}"
echo "========================================="
kubectl get ingress mlflow -n mlflow -o wide
echo ""

ALB_ENDPOINT=$(kubectl get ingress mlflow -n mlflow -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

if [ -n "$ALB_ENDPOINT" ]; then
    echo -e "${GREEN}✓ ALB is provisioned${NC}"
    echo "  Endpoint: http://$ALB_ENDPOINT"
    echo ""
    echo "Testing ALB endpoint..."
    curl -s -o /dev/null -w "  HTTP Status: %{http_code}\n" "http://$ALB_ENDPOINT/health" || echo "  Connection failed (ALB may still be initializing)"
else
    echo -e "${YELLOW}⚠ ALB endpoint not yet available${NC}"
    echo "  This is normal if deployment just completed"
    echo "  Wait a few minutes and check: kubectl get ingress -n mlflow"
fi

echo ""
echo "========================================="

# Summary
echo -e "${BLUE}Deployment Summary${NC}"
echo "========================================="
echo ""

if [ -f ".terraform-outputs.json" ]; then
    echo "Infrastructure:"
    echo "  - EKS Cluster: $(jq -r '.cluster_name.value' .terraform-outputs.json)"
    echo "  - RDS Endpoint: $(jq -r '.rds_address.value' .terraform-outputs.json)"
    echo "  - S3 Bucket: $(jq -r '.s3_bucket_name.value' .terraform-outputs.json)"
    echo ""
fi

echo "Kubernetes Resources:"
kubectl get all -n mlflow
echo ""

echo -e "${GREEN}✓ Deployment verification completed!${NC}"
echo ""

echo "========================================="
echo "Next Steps:"
echo "========================================="
echo ""
echo "1. Access MLflow UI:"
if [ -n "$ALB_ENDPOINT" ]; then
    echo "   http://$ALB_ENDPOINT"
else
    echo "   Wait for ALB to be provisioned, then run:"
    echo "   kubectl get ingress mlflow -n mlflow"
fi
echo ""
echo "2. Configure DNS (optional):"
echo "   Point mlflow.mdpg.ai to ALB endpoint"
echo ""
echo "3. Set up SSL/TLS (optional):"
echo "   - Request certificate in ACM"
echo "   - Add certificate ARN to ingress annotations"
echo ""
echo "4. Test MLflow from your local machine:"
echo "   export MLFLOW_TRACKING_URI=http://$ALB_ENDPOINT"
echo "   python -m mlflow experiments list"
echo ""
echo "5. Deploy Ray Cluster (for distributed training):"
echo "   See docs/ray_tune_guide.md"
echo ""
