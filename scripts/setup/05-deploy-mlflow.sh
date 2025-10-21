#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "MDPG MLOps Platform - Deploy MLflow"
echo "========================================="
echo ""

# Check prerequisites
if [ ! -f ".terraform-outputs.json" ]; then
    echo -e "${RED}Error: .terraform-outputs.json not found${NC}"
    exit 1
fi

# Extract values from Terraform outputs
RDS_ENDPOINT=$(jq -r '.rds_endpoint.value' .terraform-outputs.json)
RDS_ADDRESS=$(jq -r '.rds_address.value' .terraform-outputs.json)
RDS_PORT=$(jq -r '.rds_port.value' .terraform-outputs.json)
RDS_DATABASE=$(jq -r '.rds_database_name.value' .terraform-outputs.json)
RDS_USERNAME=$(jq -r '.rds_username.value' .terraform-outputs.json)
RDS_SECRET_ARN=$(jq -r '.rds_secret_arn.value' .terraform-outputs.json)
S3_BUCKET=$(jq -r '.s3_bucket_name.value' .terraform-outputs.json)
MLFLOW_IAM_ROLE=$(jq -r '.mlflow_iam_role_arn.value' .terraform-outputs.json)
AWS_REGION=$(jq -r '.deployment_summary.value.region' .terraform-outputs.json)

echo "Configuration:"
echo "  RDS Endpoint: $RDS_ADDRESS:$RDS_PORT"
echo "  S3 Bucket: $S3_BUCKET"
echo "  IAM Role: $MLFLOW_IAM_ROLE"
echo "  Region: $AWS_REGION"
echo ""

# Get RDS password from Secrets Manager
echo -e "${BLUE}Step 1: Retrieving RDS credentials${NC}"
echo ""

RDS_SECRET=$(aws secretsmanager get-secret-value \
    --secret-id "$RDS_SECRET_ARN" \
    --query SecretString \
    --output text)

RDS_PASSWORD=$(echo "$RDS_SECRET" | jq -r '.password')

if [ -z "$RDS_PASSWORD" ]; then
    echo -e "${RED}Error: Failed to retrieve RDS password${NC}"
    exit 1
fi

echo -e "${GREEN}✓ RDS credentials retrieved${NC}"
echo ""

# Create Kubernetes secret for database
echo -e "${BLUE}Step 2: Creating Kubernetes secret${NC}"
echo ""

kubectl create secret generic mlflow-db-secret \
    -n mlflow \
    --from-literal=username="$RDS_USERNAME" \
    --from-literal=password="$RDS_PASSWORD" \
    --from-literal=host="$RDS_ADDRESS" \
    --from-literal=port="$RDS_PORT" \
    --from-literal=database="$RDS_DATABASE" \
    --dry-run=client -o yaml | kubectl apply -f -

echo -e "${GREEN}✓ Secret created${NC}"
echo ""

# Create values file for Helm
echo -e "${BLUE}Step 3: Creating Helm values file${NC}"
echo ""

cat > /tmp/mlflow-values.yaml <<EOF
replicaCount: 2

image:
  repository: ghcr.io/mlflow/mlflow
  tag: "v2.8.1"

mlflow:
  backendStoreUri: "postgresql://${RDS_USERNAME}:${RDS_PASSWORD}@${RDS_ADDRESS}:${RDS_PORT}/${RDS_DATABASE}"
  artifactRoot: "s3://${S3_BUCKET}/"
  workers: 4

  authentication:
    enabled: false  # Enable if needed

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: "${MLFLOW_IAM_ROLE}"
  name: "mlflow-sa"

ingress:
  enabled: true
  className: "alb"
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
  hosts:
    - host: mlflow.mdpg.ai
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

awsRegion: "${AWS_REGION}"
EOF

echo -e "${GREEN}✓ Values file created: /tmp/mlflow-values.yaml${NC}"
echo ""

# Deploy MLflow with Helm
echo -e "${BLUE}Step 4: Deploying MLflow with Helm${NC}"
echo ""

if helm list -n mlflow | grep -q mlflow; then
    echo -e "${YELLOW}MLflow already installed, upgrading...${NC}"
    helm upgrade mlflow ./charts/mlflow \
        -n mlflow \
        -f /tmp/mlflow-values.yaml
else
    helm install mlflow ./charts/mlflow \
        -n mlflow \
        -f /tmp/mlflow-values.yaml
fi

echo ""
echo -e "${GREEN}✓ MLflow deployed${NC}"
echo ""

# Wait for deployment
echo "Waiting for MLflow pods to be ready..."
kubectl wait --for=condition=available --timeout=300s \
    deployment/mlflow -n mlflow || true

echo ""
echo -e "${GREEN}✓ MLflow deployment completed!${NC}"
echo ""

# Get service information
echo "========================================="
echo "MLflow Service Information:"
echo "========================================="
echo ""

echo "Pods:"
kubectl get pods -n mlflow
echo ""

echo "Service:"
kubectl get svc -n mlflow
echo ""

echo "Ingress:"
kubectl get ingress -n mlflow
echo ""

# Get ALB endpoint
ALB_ENDPOINT=$(kubectl get ingress mlflow -n mlflow -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")

echo "========================================="
echo "Access MLflow UI:"
echo "========================================="
echo ""
echo "  ALB Endpoint: http://$ALB_ENDPOINT"
echo "  Custom Domain: https://mlflow.mdpg.ai (after DNS setup)"
echo ""
echo "Note: ALB provisioning may take 2-3 minutes"
echo ""

echo "Clean up temporary files..."
rm -f /tmp/mlflow-values.yaml

echo "Run next script to verify:"
echo "  ./scripts/setup/06-verify-deployment.sh"
