# Operations Scripts

Scripts for managing and maintaining the deployed MLOps platform.

## cleanup-all.sh

Complete cleanup of all AWS resources deployed via Terraform.

### Usage

```bash
./scripts/ops/cleanup-all.sh
```

### What it does

1. **Deletes Kubernetes resources**
   - MLflow deployment
   - Ray cluster
   - AWS Load Balancer Controller
   - Cluster Autoscaler
   - Namespaces (mlflow, ray, monitoring)

2. **Destroys Terraform infrastructure**
   - EKS cluster
   - RDS PostgreSQL
   - S3 bucket (if empty)
   - VPC and networking
   - IAM roles and policies

3. **Cleans up local files** (optional)
   - .terraform-outputs.json
   - Terraform state backups
   - Temporary files

4. **Verifies cleanup**
   - Lists remaining EKS clusters
   - Lists remaining RDS instances
   - Lists S3 buckets with 'mdpg' prefix

### Safety Features

- Requires typing 'DELETE' to confirm
- Shows destroy plan before applying
- Waits for LoadBalancers to be deleted
- Verifies remaining resources

### Important Notes

⚠️ **This action cannot be undone!**

- All data in RDS will be lost
- All artifacts in S3 will remain (must be deleted manually if needed)
- All Kubernetes configurations will be lost

### Manual S3 Cleanup

If S3 bucket is not empty:

```bash
# List bucket contents
aws s3 ls s3://mdpg-mlops-mlflow-artifacts-XXX --recursive

# Delete all objects
aws s3 rm s3://mdpg-mlops-mlflow-artifacts-XXX --recursive

# Delete bucket
aws s3 rb s3://mdpg-mlops-mlflow-artifacts-XXX
```

### Estimated Time

- Kubernetes cleanup: 2-3 minutes
- Terraform destroy: 10-15 minutes
- **Total: ~15-20 minutes**

### Cost Impact

After cleanup, you will **stop being charged** for:
- EKS control plane (~$73/month)
- EC2 worker nodes (~$60/month)
- RDS database (~$30/month)
- S3 storage (minimal)

## Before Running

### Backup Important Data

If you want to keep your MLflow experiments and models:

```bash
# Export MLflow data
kubectl exec -n mlflow $(kubectl get pod -n mlflow -l app.kubernetes.io/name=mlflow -o jsonpath='{.items[0].metadata.name}') -- \
  mlflow experiments list

# Backup RDS database
RDS_ENDPOINT=$(cd terraform/aws-eks && terraform output -raw rds_endpoint)
pg_dump -h $RDS_ENDPOINT -U mlflow -d mlflow > mlflow_backup.sql

# Download artifacts from S3
S3_BUCKET=$(cd terraform/aws-eks && terraform output -raw s3_bucket_name)
aws s3 sync s3://$S3_BUCKET/ ./mlflow_artifacts_backup/
```

### Partial Cleanup

If you only want to delete specific components:

```bash
# Delete only MLflow
helm uninstall mlflow -n mlflow

# Delete only Ray
helm uninstall ray -n ray

# Scale down nodes (keep cluster but reduce cost)
cd terraform/aws-eks
terraform apply -var="node_group_desired_size=0"
```

## Troubleshooting

### "VPC has dependencies" error

Wait longer for LoadBalancers to be deleted:

```bash
# Check for remaining load balancers
aws elbv2 describe-load-balancers --region ap-northeast-2 | grep mdpg

# Force delete if needed
aws elbv2 delete-load-balancer --load-balancer-arn <arn>
```

### Terraform state lock

```bash
cd terraform/aws-eks
terraform force-unlock <lock-id>
```

### S3 bucket not empty

```bash
S3_BUCKET="mdpg-mlops-mlflow-artifacts-XXX"
aws s3 rm s3://$S3_BUCKET --recursive --region ap-northeast-2
```

## Support

For issues during cleanup:
- Check AWS Console for remaining resources
- Review CloudWatch logs
- Check terraform.tfstate for resource IDs
