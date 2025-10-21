# MLflow Helm Chart

Helm chart for deploying MLflow tracking server on Kubernetes (AWS EKS).

## Prerequisites

- Kubernetes 1.23+
- Helm 3.8+
- AWS EKS cluster with IRSA configured
- RDS PostgreSQL database (deployed via Terraform)
- S3 bucket for artifacts (deployed via Terraform)
- AWS Load Balancer Controller installed in the cluster

## Installation

### 1. Get Configuration from Terraform

After deploying the infrastructure with Terraform, get the required values:

```bash
# Navigate to terraform directory
cd terraform/aws-eks

# Get RDS endpoint
terraform output -raw rds_endpoint

# Get S3 bucket name
terraform output -raw s3_bucket_name

# Get MLflow IAM role ARN
terraform output -raw mlflow_iam_role_arn

# Get RDS credentials from Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id $(terraform output -raw rds_secret_arn) \
  --query SecretString \
  --output text | jq .
```

### 2. Create values.yaml

Create a `values-prod.yaml` file with your configuration:

```yaml
replicaCount: 2

image:
  repository: ghcr.io/mlflow/mlflow
  tag: "v2.8.1"

mlflow:
  # From Terraform: terraform output -raw mlflow_backend_store_uri
  backendStoreUri: "postgresql://mlflow:PASSWORD@mdpg-mlops-mlflow-db.xxxxx.us-west-2.rds.amazonaws.com:5432/mlflow"

  # From Terraform: terraform output -raw mlflow_artifact_root
  artifactRoot: "s3://mdpg-mlops-mlflow-artifacts/"

  workers: 4

  authentication:
    enabled: true
    adminUsername: "admin"
    adminPassword: "your-secure-password"  # Change this!

serviceAccount:
  create: true
  annotations:
    # From Terraform: terraform output -raw mlflow_iam_role_arn
    eks.amazonaws.com/role-arn: "arn:aws:iam::ACCOUNT_ID:role/mdpg-mlops-mlflow-s3-access"
  name: "mlflow-sa"

ingress:
  enabled: true
  className: "alb"
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    # alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:us-west-2:ACCOUNT_ID:certificate/xxx"
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

awsRegion: "us-west-2"
```

### 3. Install the Chart

```bash
# Create namespace
kubectl create namespace mlflow

# Install MLflow
helm install mlflow ./charts/mlflow \
  -n mlflow \
  -f values-prod.yaml

# Check deployment status
kubectl get pods -n mlflow
kubectl get ingress -n mlflow
kubectl get svc -n mlflow
```

### 4. Verify Deployment

```bash
# Check pod logs
kubectl logs -n mlflow -l app.kubernetes.io/name=mlflow

# Check HPA status
kubectl get hpa -n mlflow

# Get ALB endpoint
kubectl get ingress -n mlflow -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}'
```

### 5. Access MLflow UI

```bash
# Get the ALB hostname
MLFLOW_URL=$(kubectl get ingress -n mlflow -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}')
echo "MLflow UI: http://$MLFLOW_URL"

# Or use your custom domain
echo "MLflow UI: https://mlflow.mdpg.ai"
```

## Configuration

### Key Configuration Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of MLflow replicas | `2` |
| `image.repository` | MLflow image repository | `ghcr.io/mlflow/mlflow` |
| `image.tag` | MLflow image tag | `v2.8.1` |
| `mlflow.backendStoreUri` | PostgreSQL connection string | `""` |
| `mlflow.artifactRoot` | S3 bucket URI | `""` |
| `mlflow.workers` | Number of gunicorn workers | `4` |
| `mlflow.authentication.enabled` | Enable authentication | `true` |
| `serviceAccount.annotations` | IRSA role annotation | `{}` |
| `ingress.enabled` | Enable ingress | `true` |
| `resources.limits.cpu` | CPU limit | `2000m` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Minimum replicas | `2` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |

### Database Secret (Alternative to inline credentials)

For better security, use Kubernetes secrets instead of inline credentials:

```bash
# Create secret from AWS Secrets Manager
kubectl create secret generic mlflow-db-secret \
  -n mlflow \
  --from-literal=username=mlflow \
  --from-literal=password=YOUR_PASSWORD \
  --from-literal=host=mdpg-mlops-mlflow-db.xxxxx.us-west-2.rds.amazonaws.com \
  --from-literal=port=5432 \
  --from-literal=database=mlflow
```

Then update `values.yaml`:

```yaml
mlflow:
  backendStoreUri: ""  # Leave empty

databaseSecret:
  enabled: true
  secretName: "mlflow-db-secret"
```

## Upgrading

```bash
# Update values
vim values-prod.yaml

# Upgrade release
helm upgrade mlflow ./charts/mlflow \
  -n mlflow \
  -f values-prod.yaml

# Rollback if needed
helm rollback mlflow -n mlflow
```

## Uninstalling

```bash
# Uninstall the release
helm uninstall mlflow -n mlflow

# Delete namespace
kubectl delete namespace mlflow
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n mlflow

# Check pod events
kubectl describe pod -n mlflow <pod-name>

# Check logs
kubectl logs -n mlflow <pod-name>
```

### Database Connection Issues

```bash
# Test database connection
kubectl run -it --rm debug --image=postgres:15 --restart=Never -n mlflow -- \
  psql -h mdpg-mlops-mlflow-db.xxxxx.us-west-2.rds.amazonaws.com -U mlflow -d mlflow

# Check if secret exists (if using databaseSecret)
kubectl get secret mlflow-db-secret -n mlflow -o yaml
```

### S3 Access Issues

```bash
# Check service account annotations
kubectl get sa mlflow-sa -n mlflow -o yaml

# Check pod environment
kubectl exec -n mlflow <pod-name> -- env | grep AWS

# Verify IRSA role
aws iam get-role --role-name mdpg-mlops-mlflow-s3-access
```

### Ingress Not Working

```bash
# Check ingress status
kubectl get ingress -n mlflow
kubectl describe ingress -n mlflow

# Check ALB controller logs
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller

# Verify security group rules
aws ec2 describe-security-groups --filters "Name=tag:kubernetes.io/cluster/mdpg-mlops-eks,Values=owned"
```

## Monitoring

### Health Checks

MLflow exposes a health endpoint at `/health`:

```bash
# From inside the cluster
kubectl run -it --rm curl --image=curlimages/curl --restart=Never -n mlflow -- \
  curl http://mlflow:80/health

# From outside (using ALB)
curl https://mlflow.mdpg.ai/health
```

### Metrics

Enable Prometheus metrics (optional):

```yaml
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
```

### Logs

```bash
# Stream logs from all pods
kubectl logs -n mlflow -l app.kubernetes.io/name=mlflow -f

# Stream logs from specific pod
kubectl logs -n mlflow <pod-name> -f
```

## Security Best Practices

1. **Use Secrets for Credentials**: Never hardcode credentials in `values.yaml`
2. **Enable TLS**: Configure ACM certificate in ingress annotations
3. **Enable Authentication**: Set `mlflow.authentication.enabled: true`
4. **Use IRSA**: Leverage pod-level IAM permissions
5. **Network Policies**: Restrict pod-to-pod communication
6. **Resource Limits**: Set appropriate CPU/memory limits
7. **Pod Security**: Use non-root user and read-only filesystem

## Advanced Configuration

### Custom Image

If you built a custom MLflow image:

```yaml
image:
  repository: ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/mlflow
  tag: "custom-2.8.1"
  pullPolicy: Always

imagePullSecrets:
  - name: ecr-registry-secret
```

### Multiple Environments

Create separate values files:

```bash
charts/mlflow/
├── values-dev.yaml
├── values-staging.yaml
└── values-prod.yaml
```

Deploy with:

```bash
helm install mlflow-dev ./charts/mlflow -n mlflow-dev -f values-dev.yaml
helm install mlflow-prod ./charts/mlflow -n mlflow-prod -f values-prod.yaml
```

### Custom Node Selector

Deploy to specific node groups:

```yaml
nodeSelector:
  workload-type: mlops

tolerations:
  - key: "dedicated"
    operator: "Equal"
    value: "mlops"
    effect: "NoSchedule"
```

## Support

For issues:
- Check [docs/mlflow_remote_setup.md](../../docs/mlflow_remote_setup.md)
- Review [docs/eks_infrastructure.md](../../docs/eks_infrastructure.md)
- MLflow documentation: https://www.mlflow.org/docs/latest/index.html

## License

Internal MDPG use only.
