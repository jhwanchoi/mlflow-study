# 배포 자동화 스크립트 가이드

**작성일**: 2025-10-21
**버전**: 1.0
**대상**: Phase 5.4 - 배포 자동화

---

## 개요

휴먼 에러를 최소화하기 위한 자동화 스크립트 모음입니다.

### 스크립트 구조

```
scripts/
├── setup/              # 초기 배포 스크립트
│   ├── 01-setup-aws.sh
│   ├── 02-deploy-eks.sh
│   ├── 03-deploy-mlflow.sh
│   ├── 04-setup-users.sh
│   ├── 05-test-connection.sh
│   └── 06-verify-all.sh
├── ops/                # 운영 스크립트
│   ├── backup-mlflow.sh
│   ├── restore-mlflow.sh
│   ├── scale-mlflow.sh
│   └── logs-mlflow.sh
└── dev/                # 개발 스크립트
    ├── local-to-remote.sh
    └── remote-to-local.sh
```

---

## Setup 스크립트

### 01-setup-aws.sh

```bash
#!/bin/bash
set -e

echo "🔧 AWS CLI 설정 확인"

# AWS CLI 설치 확인
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI가 설치되지 않았습니다"
    echo "설치: brew install awscli"
    exit 1
fi

# AWS 자격증명 확인
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials가 설정되지 않았습니다"
    echo "설정: aws configure"
    exit 1
fi

echo "✅ AWS 설정 완료"
aws sts get-caller-identity
```

### 02-deploy-eks.sh

```bash
#!/bin/bash
set -e

echo "🚀 EKS 클러스터 배포"

cd terraform/aws-eks

# Terraform 초기화
terraform init

# Plan 생성
terraform plan -out=tfplan

# 사용자 확인
read -p "Deploy EKS cluster? (yes/no): " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "배포 취소"
    exit 0
fi

# Apply
terraform apply tfplan

# kubeconfig 업데이트
echo "📝 kubeconfig 업데이트"
CLUSTER_NAME=$(terraform output -raw cluster_name)
aws eks update-kubeconfig --name $CLUSTER_NAME --region us-west-2

# 검증
echo "✅ EKS 배포 완료"
kubectl get nodes
```

### 03-deploy-mlflow.sh

```bash
#!/bin/bash
set -e

echo "🚀 MLflow 서버 배포"

# Terraform output 가져오기
cd terraform/aws-eks
IRSA_ROLE_ARN=$(terraform output -raw mlflow_irsa_role_arn)
RDS_URI=$(terraform output -raw rds_connection_string)
S3_BUCKET=$(terraform output -raw s3_bucket_name)
cd ../..

# values-production.yaml 업데이트
cat > charts/mlflow/values-production.yaml <<EOF
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: "${IRSA_ROLE_ARN}"

mlflow:
  backendStoreUri: "${RDS_URI}"
  artifactRoot: "s3://${S3_BUCKET}/"
  authentication:
    enabled: false  # 초기에는 비활성화

ingress:
  annotations:
    alb.ingress.kubernetes.io/certificate-arn: "${ACM_CERTIFICATE_ARN}"
EOF

# Namespace 생성
kubectl create namespace ml-platform --dry-run=client -o yaml | kubectl apply -f -

# Helm 배포
helm upgrade --install mlflow ./charts/mlflow \
  --namespace ml-platform \
  --values charts/mlflow/values-production.yaml \
  --wait \
  --timeout 10m

echo "✅ MLflow 배포 완료"
kubectl get pods -n ml-platform
```

### 04-setup-users.sh

```bash
#!/bin/bash
set -e

echo "👥 MLflow 사용자 계정 생성"

# Admin 계정 생성
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-admin-user \
    --username admin \
    --password "${ADMIN_PASSWORD:-ChangeMe123!}"

# 일반 사용자 계정
for user in mlops_engineer_1 mlops_engineer_2 ml_engineer_1; do
    echo "Creating user: $user"
    kubectl exec -it deployment/mlflow -n ml-platform -- \
      mlflow server create-user \
        --username $user \
        --password "SecurePassword123!"
done

# 인증 활성화
helm upgrade mlflow ./charts/mlflow \
  --namespace ml-platform \
  --set mlflow.authentication.enabled=true \
  --reuse-values \
  --wait

echo "✅ 사용자 계정 생성 완료"
```

### 05-test-connection.sh

```bash
#!/bin/bash
set -e

echo "🔍 MLflow 연결 테스트"

# Ingress URL 가져오기
MLFLOW_URL=$(kubectl get ingress mlflow -n ml-platform -o jsonpath='{.spec.rules[0].host}')
export MLFLOW_TRACKING_URI="https://${MLFLOW_URL}"
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="${ADMIN_PASSWORD:-ChangeMe123!}"

# Python 연결 테스트
python3 <<EOF
import mlflow

mlflow.set_experiment("deployment-test")
with mlflow.start_run():
    mlflow.log_param("deployment_script", "05-test-connection.sh")
    mlflow.log_metric("success", 1.0)

print("✅ MLflow 연결 성공!")
print(f"URL: ${MLFLOW_TRACKING_URI}")
EOF
```

### 06-verify-all.sh

```bash
#!/bin/bash
set -e

echo "🔍 전체 시스템 검증"

# EKS Nodes
echo "=== EKS Nodes ==="
kubectl get nodes

# MLflow Pods
echo "=== MLflow Pods ==="
kubectl get pods -n ml-platform

# MLflow Service
echo "=== MLflow Service ==="
kubectl get svc mlflow -n ml-platform

# Ingress
echo "=== Ingress ==="
kubectl get ingress mlflow -n ml-platform

# RDS 연결
echo "=== RDS Connection ==="
cd terraform/aws-eks
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
echo "RDS Endpoint: $RDS_ENDPOINT"

# S3 버킷
echo "=== S3 Bucket ==="
S3_BUCKET=$(terraform output -raw s3_bucket_name)
aws s3 ls s3://$S3_BUCKET/

echo "✅ 전체 검증 완료"
```

---

## Operations 스크립트

### ops/backup-mlflow.sh

```bash
#!/bin/bash
set -e

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/mlflow_${BACKUP_DATE}"

mkdir -p $BACKUP_DIR

echo "💾 MLflow 백업 시작: $BACKUP_DIR"

# 1. RDS 스냅샷 생성
cd terraform/aws-eks
DB_INSTANCE=$(terraform output -raw rds_endpoint | cut -d: -f1)
aws rds create-db-snapshot \
  --db-instance-identifier $DB_INSTANCE \
  --db-snapshot-identifier mlflow-backup-${BACKUP_DATE}

# 2. S3 버킷 동기화
S3_BUCKET=$(terraform output -raw s3_bucket_name)
aws s3 sync s3://$S3_BUCKET/ $BACKUP_DIR/s3/

echo "✅ 백업 완료: $BACKUP_DIR"
```

### ops/scale-mlflow.sh

```bash
#!/bin/bash
set -e

REPLICAS=${1:-3}

echo "📈 MLflow 스케일링: $REPLICAS replicas"

kubectl scale deployment mlflow -n ml-platform --replicas=$REPLICAS

kubectl rollout status deployment/mlflow -n ml-platform

echo "✅ 스케일링 완료"
kubectl get pods -n ml-platform
```

---

## 사용 방법

### 전체 배포 (최초 1회)

```bash
# 1. AWS 설정 확인
./scripts/setup/01-setup-aws.sh

# 2. EKS 클러스터 배포 (15-20분)
./scripts/setup/02-deploy-eks.sh

# 3. MLflow 서버 배포 (5분)
./scripts/setup/03-deploy-mlflow.sh

# 4. 사용자 계정 생성
ADMIN_PASSWORD=YourSecurePassword ./scripts/setup/04-setup-users.sh

# 5. 연결 테스트
./scripts/setup/05-test-connection.sh

# 6. 전체 검증
./scripts/setup/06-verify-all.sh
```

### 일일 운영

```bash
# 백업
./scripts/ops/backup-mlflow.sh

# 로그 확인
./scripts/ops/logs-mlflow.sh

# 스케일링
./scripts/ops/scale-mlflow.sh 5
```

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
