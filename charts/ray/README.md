## Ray Cluster Helm Chart

Helm chart for deploying Ray Cluster on Kubernetes for distributed ML training and hyperparameter tuning.

## Installation

```bash
# Get IAM role from Terraform
cd terraform/aws-eks
RAY_IAM_ROLE=$(terraform output -raw ray_iam_role_arn)
cd -

# Create values file
cat > values-ray.yaml <<EOF
serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: "$RAY_IAM_ROLE"

mlflow:
  trackingUri: "https://mlflow.mdpg.ai"

awsRegion: "us-west-2"
EOF

# Install Ray
helm install ray ./charts/ray -n ray -f values-ray.yaml

# Verify
kubectl get pods -n ray
```

## Usage with MLflow

See [docs/ray_tune_guide.md](../../docs/ray_tune_guide.md) for complete examples.
