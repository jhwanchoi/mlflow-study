# AWS EKS 인프라 배포 가이드

**작성일**: 2025-10-21
**버전**: 1.0
**대상**: Phase 5 - AWS EKS 인프라 구축

---

## 목차

1. [개요](#개요)
2. [아키텍처](#아키텍처)
3. [사전 요구사항](#사전-요구사항)
4. [Terraform 프로젝트 구조](#terraform-프로젝트-구조)
5. [단계별 배포 가이드](#단계별-배포-가이드)
6. [검증 및 테스트](#검증-및-테스트)
7. [문제 해결](#문제-해결)
8. [비용 최적화](#비용-최적화)

---

## 개요

### 목표

로컬 Docker Compose 환경에서 **AWS EKS 기반 중앙화 MLOps 플랫폼**으로 마이그레이션하여:
- **멀티 유저 지원**: MLOps 엔지니어 2명, ML 엔지니어 1명 (확장 가능)
- **중앙화된 MLflow 서버**: 모든 실험 추적 및 모델 레지스트리 공유
- **확장 가능한 인프라**: Ray Tune, Airflow 등 추가 서비스 통합 준비

### 핵심 구성요소

- **AWS EKS**: Kubernetes 1.28+ 클러스터
- **RDS PostgreSQL**: MLflow 메타데이터 저장소 (Multi-AZ)
- **S3**: MLflow 아티팩트 저장소
- **ALB Ingress**: HTTPS 로드 밸런싱
- **IRSA**: IAM Roles for Service Accounts (보안)

### 예상 비용

**월 운영 비용** (24/7 운영 기준):
- EKS Control Plane: $73
- Worker Nodes (t3.medium × 2): $60
- RDS PostgreSQL (db.t3.small): $30
- S3 + 데이터 전송: ~$10
- ALB: $17
- **총계**: ~$190/월

**추가 비용** (사용량 기반):
- GPU 노드 (p3.2xlarge Spot): ~$0.90/시간
- 예상 GPU 사용 (20시간/월): ~$18

---

## 아키텍처

### 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│                       Client Environments                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ VSCode        │  │ Jupyter Hub  │  │ CI/CD Pipelines      │  │
│  │ (Local Dev)   │  │ (Notebooks)  │  │ (GitHub Actions)     │  │
│  └───────┬───────┘  └──────┬───────┘  └──────┬───────────────┘  │
│          │ MLflow Client            │      │                     │
└──────────┼──────────────────────────┼──────┼─────────────────────┘
           │                          │      │
           └──────────────────────────┼──────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                       AWS EKS Cluster                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ALB Ingress Controller                                    │  │
│  │  - HTTPS (ACM Certificate)                                 │  │
│  │  - mlflow.mdpg.ai                                      │  │
│  └─────────┬──────────────────────────────────────────────────┘  │
│            ▼                                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  MLflow Tracking Server                                    │  │
│  │  - Deployment (HPA: 2-5 pods)                              │  │
│  │  - MLflow Authentication                                   │  │
│  │  - IRSA (S3 Access)                                        │  │
│  └────────┬───────────────────────────────────┬────────────────┘  │
│           │                                   │                   │
│  ┌────────┴────────────┐         ┌───────────┴────────────────┐  │
│  │  Node Group (CPU)   │         │  Node Group (GPU) [Future] │  │
│  │  - t3.medium × 2    │         │  - p3.2xlarge (Spot)       │  │
│  │  - Auto-scaling     │         │  - Ray Workers             │  │
│  └─────────────────────┘         └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────────┐     ┌──────────────────────────────────┐
│  AWS RDS PostgreSQL     │     │  AWS S3 Bucket                   │
│  - db.t3.small          │     │  - mlflow-artifacts-prod         │
│  - Multi-AZ             │     │  - Versioning enabled            │
│  - Encrypted (AES-256)  │     │  - Lifecycle policies            │
│  - Automated Backups    │     │  - IRSA for pod access           │
└─────────────────────────┘     └──────────────────────────────────┘
```

### 네트워크 구성

```
VPC: 10.0.0.0/16
├── AZ us-west-2a
│   ├── Public Subnet: 10.0.1.0/24 (NAT Gateway, ALB)
│   └── Private Subnet: 10.0.11.0/24 (EKS Nodes, RDS)
├── AZ us-west-2b
│   ├── Public Subnet: 10.0.2.0/24
│   └── Private Subnet: 10.0.12.0/24
└── AZ us-west-2c
    ├── Public Subnet: 10.0.3.0/24
    └── Private Subnet: 10.0.13.0/24
```

---

## 사전 요구사항

### 1. AWS 계정 및 권한

```bash
# AWS CLI 설치 확인
aws --version
# aws-cli/2.x.x

# AWS 자격증명 설정
aws configure
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region name: us-west-2
# Default output format: json

# 권한 확인 (다음 서비스에 대한 권한 필요)
# - EKS (전체)
# - EC2 (VPC, Subnet, Security Group, NAT Gateway 등)
# - RDS (PostgreSQL 생성/관리)
# - S3 (버킷 생성/관리)
# - IAM (IRSA 역할 생성)
# - ACM (SSL 인증서)
```

### 2. 도구 설치

```bash
# Terraform 설치 (1.6.0+)
brew install terraform
terraform version

# kubectl 설치
brew install kubectl
kubectl version --client

# Helm 설치 (3.12.0+)
brew install helm
helm version

# eksctl 설치 (선택, EKS 관리 도구)
brew install eksctl
eksctl version
```

### 3. 도메인 및 SSL 인증서 (선택)

MLflow 서버에 HTTPS로 접속하려면:

1. **도메인 보유** (예: `example.com`)
2. **Route 53 Hosted Zone** 또는 외부 DNS 사용
3. **AWS Certificate Manager**에서 SSL 인증서 발급

```bash
# ACM 인증서 발급 (DNS 검증)
aws acm request-certificate \
  --domain-name mlflow.mdpg.ai \
  --validation-method DNS \
  --region us-west-2
```

---

## Terraform 프로젝트 구조

### 디렉토리 구조

```
terraform/
└── aws-eks/
    ├── main.tf           # EKS 클러스터, VPC, 서브넷
    ├── rds.tf            # RDS PostgreSQL
    ├── s3.tf             # S3 버킷
    ├── iam.tf            # IRSA 역할
    ├── variables.tf      # 입력 변수
    ├── outputs.tf        # 출력 값
    ├── terraform.tfvars  # 변수 값 (gitignore!)
    └── README.md         # 배포 가이드
```

### main.tf - EKS 클러스터

```hcl
# terraform/aws-eks/main.tf

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "mlops-terraform-state"
    key    = "eks/terraform.tfstate"
    region = "us-west-2"
    # DynamoDB table for state locking (optional but recommended)
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "MLOps Platform"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# VPC Module (AWS VPC Module 사용)
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs

  enable_nat_gateway = true
  single_nat_gateway = false  # Multi-AZ NAT (HA)
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Kubernetes tags (EKS 요구사항)
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# EKS Module
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint access
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # IRSA (IAM Roles for Service Accounts)
  enable_irsa = true

  # Managed Node Groups
  eks_managed_node_groups = {
    # CPU Worker Nodes
    cpu_workers = {
      name           = "cpu-workers"
      instance_types = ["t3.medium"]

      min_size     = 2
      max_size     = 5
      desired_size = 2

      # Use latest EKS optimized AMI
      ami_type = "AL2_x86_64"

      labels = {
        workload = "cpu"
      }

      tags = {
        NodeGroup = "cpu-workers"
      }
    }

    # GPU Worker Nodes (추후 Phase 6에서 활성화)
    # gpu_workers = {
    #   name           = "gpu-workers"
    #   instance_types = ["p3.2xlarge"]
    #   capacity_type  = "SPOT"  # 비용 절감
    #
    #   min_size     = 0
    #   max_size     = 5
    #   desired_size = 0
    #
    #   ami_type = "AL2_x86_64_GPU"
    #
    #   labels = {
    #     workload = "gpu"
    #   }
    #
    #   taints = [{
    #     key    = "nvidia.com/gpu"
    #     value  = "true"
    #     effect = "NO_SCHEDULE"
    #   }]
    # }
  }

  # Cluster Security Group Rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  # Node Security Group Rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
  }
}

# ALB Ingress Controller용 IAM Role
module "alb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "alb-ingress-controller"

  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
}

# EBS CSI Driver용 IAM Role (PVC 지원)
module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "ebs-csi-controller"

  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}
```

### rds.tf - PostgreSQL 데이터베이스

```hcl
# terraform/aws-eks/rds.tf

# RDS Subnet Group
resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-db-subnet"
  }
}

# RDS Security Group
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "PostgreSQL from EKS nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-rds-sg"
  }
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "mlflow" {
  identifier = "${var.project_name}-mlflow-db"

  # Engine
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.small"

  # Storage
  allocated_storage     = 20
  max_allocated_storage = 100  # Auto-scaling
  storage_type          = "gp3"
  storage_encrypted     = true

  # Database
  db_name  = "mlflow"
  username = var.db_username
  password = var.db_password  # Use AWS Secrets Manager in production!

  # Network
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # High Availability
  multi_az = true

  # Backups
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  # Deletion protection
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-mlflow-db-final-snapshot"

  # Monitoring
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  tags = {
    Name = "${var.project_name}-mlflow-db"
  }
}

# Secrets Manager (권장: 비밀번호 관리)
resource "aws_secretsmanager_secret" "db_password" {
  name = "${var.project_name}/mlflow/db-password"

  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = var.db_username
    password = var.db_password
    host     = aws_db_instance.mlflow.address
    port     = aws_db_instance.mlflow.port
    dbname   = aws_db_instance.mlflow.db_name
  })
}
```

### s3.tf - S3 아티팩트 저장소

```hcl
# terraform/aws-eks/s3.tf

# S3 Bucket for MLflow Artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_name}-mlflow-artifacts"

  tags = {
    Name = "${var.project_name}-mlflow-artifacts"
  }
}

# Bucket Versioning
resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Server-Side Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block Public Access
resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle Policy (오래된 버전 정리)
resource "aws_s3_bucket_lifecycle_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    id     = "delete-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
```

### iam.tf - IRSA 역할

```hcl
# terraform/aws-eks/iam.tf

# MLflow Pod용 IRSA Role (S3 접근)
module "mlflow_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.project_name}-mlflow-s3-access"

  role_policy_arns = {
    s3_policy = aws_iam_policy.mlflow_s3_access.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["ml-platform:mlflow"]
    }
  }
}

# S3 Access Policy
resource "aws_iam_policy" "mlflow_s3_access" {
  name        = "${var.project_name}-mlflow-s3-access"
  description = "Allow MLflow pods to access S3 artifacts bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}
```

### variables.tf - 입력 변수

```hcl
# terraform/aws-eks/variables.tf

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "mdpg-mlops"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "mdpg-mlops-eks"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "db_username" {
  description = "RDS master username"
  type        = string
  default     = "mlflow"
  sensitive   = true
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}
```

### outputs.tf - 출력 값

```hcl
# terraform/aws-eks/outputs.tf

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_oidc_issuer_url" {
  description = "OIDC issuer URL"
  value       = module.eks.cluster_oidc_issuer_url
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.mlflow.endpoint
  sensitive   = true
}

output "rds_connection_string" {
  description = "PostgreSQL connection string"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.mlflow.endpoint}/${aws_db_instance.mlflow.db_name}"
  sensitive   = true
}

output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "mlflow_irsa_role_arn" {
  description = "MLflow IRSA role ARN"
  value       = module.mlflow_irsa.iam_role_arn
}

output "alb_controller_role_arn" {
  description = "ALB Controller IRSA role ARN"
  value       = module.alb_controller_irsa.iam_role_arn
}
```

### terraform.tfvars - 변수 값

```hcl
# terraform/aws-eks/terraform.tfvars
# ⚠️ 주의: 이 파일은 .gitignore에 추가하세요!

aws_region   = "us-west-2"
environment  = "production"
project_name = "mdpg-mlops"

# Database Credentials (Use AWS Secrets Manager in production!)
db_username = "mlflow"
db_password = "CHANGE_THIS_SECURE_PASSWORD_123!"  # 반드시 변경!
```

---

## 단계별 배포 가이드

### 1. Terraform Backend 준비 (State 저장)

```bash
# S3 버킷 생성 (Terraform state 저장용)
aws s3api create-bucket \
  --bucket mlops-terraform-state \
  --region us-west-2 \
  --create-bucket-configuration LocationConstraint=us-west-2

# Versioning 활성화
aws s3api put-bucket-versioning \
  --bucket mlops-terraform-state \
  --versioning-configuration Status=Enabled

# DynamoDB 테이블 생성 (State Locking)
aws dynamodb create-table \
  --table-name terraform-state-lock \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-west-2
```

### 2. Terraform 초기화

```bash
cd terraform/aws-eks

# Terraform 초기화 (provider 다운로드)
terraform init
```

### 3. 변수 설정

```bash
# terraform.tfvars 생성 (템플릿 복사)
cp terraform.tfvars.example terraform.tfvars

# 에디터로 편집 (특히 db_password 반드시 변경!)
vim terraform.tfvars
```

### 4. 배포 계획 검토

```bash
# Terraform plan 실행
terraform plan -out=tfplan

# 생성될 리소스 확인:
# - VPC, Subnets, NAT Gateways, Security Groups
# - EKS Cluster, Node Groups
# - RDS PostgreSQL
# - S3 Bucket
# - IAM Roles (IRSA)
# 총 50-60개 리소스
```

### 5. 배포 실행

```bash
# Terraform apply 실행 (15-20분 소요)
terraform apply tfplan

# 배포 완료 후 출력 확인
terraform output

# cluster_endpoint = "https://XXXXXXXX.gr7.us-west-2.eks.amazonaws.com"
# rds_endpoint = "mdpg-mlops-mlflow-db.xxxxx.us-west-2.rds.amazonaws.com:5432"
# s3_bucket_name = "mdpg-mlops-mlflow-artifacts"
```

### 6. kubectl 설정

```bash
# kubeconfig 업데이트
aws eks update-kubeconfig --name mdpg-mlops-eks --region us-west-2

# 클러스터 접속 확인
kubectl get nodes
# NAME                                         STATUS   ROLES    AGE   VERSION
# ip-10-0-11-xxx.us-west-2.compute.internal   Ready    <none>   5m    v1.28.x
# ip-10-0-12-xxx.us-west-2.compute.internal   Ready    <none>   5m    v1.28.x

# Context 확인
kubectl config current-context
# arn:aws:eks:us-west-2:123456789:cluster/mdpg-mlops-eks
```

### 7. EBS CSI Driver 설치 (PVC 지원)

```bash
# Helm repo 추가
helm repo add aws-ebs-csi-driver https://kubernetes-sigs.github.io/aws-ebs-csi-driver
helm repo update

# EBS CSI Driver 설치
helm install aws-ebs-csi-driver aws-ebs-csi-driver/aws-ebs-csi-driver \
  --namespace kube-system \
  --set controller.serviceAccount.create=true \
  --set controller.serviceAccount.name=ebs-csi-controller-sa \
  --set controller.serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$(terraform output -raw ebs_csi_irsa_role_arn)"

# 설치 확인
kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-ebs-csi-driver
```

### 8. ALB Ingress Controller 설치

```bash
# Helm repo 추가
helm repo add eks https://aws.github.io/eks-charts
helm repo update

# ALB Ingress Controller 설치
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName=mdpg-mlops-eks \
  --set serviceAccount.create=true \
  --set serviceAccount.name=aws-load-balancer-controller \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$(terraform output -raw alb_controller_role_arn)"

# 설치 확인
kubectl get deployment -n kube-system aws-load-balancer-controller
```

---

## 검증 및 테스트

### 1. 네트워크 검증

```bash
# VPC 확인
aws ec2 describe-vpcs --filters "Name=tag:Name,Values=mdpg-mlops-vpc"

# Subnets 확인
aws ec2 describe-subnets --filters "Name=tag:Name,Values=mdpg-mlops-vpc-*"

# NAT Gateways 확인 (3개 있어야 함 - Multi-AZ)
aws ec2 describe-nat-gateways --filter "Name=state,Values=available"
```

### 2. EKS 클러스터 검증

```bash
# Cluster 상태 확인
aws eks describe-cluster --name mdpg-mlops-eks --query "cluster.status"
# "ACTIVE"

# Node 상태 확인
kubectl get nodes -o wide

# CoreDNS 정상 확인
kubectl get pods -n kube-system -l k8s-app=kube-dns
```

### 3. RDS 검증

```bash
# RDS 인스턴스 상태 확인
aws rds describe-db-instances \
  --db-instance-identifier mdpg-mlops-mlflow-db \
  --query "DBInstances[0].DBInstanceStatus"
# "available"

# EKS에서 RDS 연결 테스트 (임시 Pod 사용)
kubectl run postgres-test --rm -it --image=postgres:15 -- bash

# Pod 내부에서:
psql -h mdpg-mlops-mlflow-db.xxxxx.us-west-2.rds.amazonaws.com \
     -U mlflow \
     -d mlflow
# Password: (terraform.tfvars에 설정한 비밀번호)

# 연결 성공 시:
mlflow=> \dt
# (아직 테이블 없음)
mlflow=> \q
exit
```

### 4. S3 검증

```bash
# S3 버킷 확인
aws s3 ls | grep mlflow-artifacts

# 버킷 설정 확인
aws s3api get-bucket-versioning --bucket mdpg-mlops-mlflow-artifacts
# "Status": "Enabled"

aws s3api get-bucket-encryption --bucket mdpg-mlops-mlflow-artifacts
# "SSEAlgorithm": "AES256"
```

### 5. IRSA 검증

```bash
# MLflow IRSA Role 확인
aws iam get-role --role-name mdpg-mlops-mlflow-s3-access

# Policy 확인
aws iam list-attached-role-policies --role-name mdpg-mlops-mlflow-s3-access

# 테스트 Pod로 S3 접근 확인 (다음 단계: MLflow 배포 후)
```

---

## 문제 해결

### 문제 1: Terraform apply 실패

**증상**: `Error creating EKS Cluster: ResourceNotFoundException`

**해결**:
```bash
# AWS 자격증명 확인
aws sts get-caller-identity

# IAM 권한 확인 (EKS, EC2, RDS, S3 등)
# 필요시 AdministratorAccess 또는 custom policy 적용
```

### 문제 2: kubectl 연결 실패

**증상**: `error: You must be logged in to the server (Unauthorized)`

**해결**:
```bash
# kubeconfig 다시 생성
aws eks update-kubeconfig --name mdpg-mlops-eks --region us-west-2

# Context 확인
kubectl config get-contexts

# IAM 인증 확인
aws eks get-token --cluster-name mdpg-mlops-eks
```

### 문제 3: RDS 연결 실패

**증상**: `could not connect to server: Connection timed out`

**해결**:
```bash
# Security Group 확인
aws ec2 describe-security-groups --filters "Name=tag:Name,Values=mdpg-mlops-rds-sg"

# EKS Node Security Group 확인
kubectl get nodes -o json | jq '.items[0].spec.providerID'

# Security Group 규칙 수동 추가 (필요시)
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxxxxx \
  --protocol tcp \
  --port 5432 \
  --source-group sg-yyyyyy  # EKS Node SG
```

### 문제 4: ALB Ingress Controller 에러

**증상**: `reconciler error: WebIdentityErr`

**해결**:
```bash
# IRSA Role ARN 확인
terraform output alb_controller_role_arn

# ServiceAccount annotation 확인
kubectl get sa aws-load-balancer-controller -n kube-system -o yaml

# OIDC Provider 확인
aws eks describe-cluster --name mdpg-mlops-eks \
  --query "cluster.identity.oidc.issuer"
```

---

## 비용 최적화

### 1. Spot Instances 사용 (GPU 노드)

```hcl
# terraform/aws-eks/main.tf
eks_managed_node_groups = {
  gpu_workers = {
    capacity_type = "SPOT"  # 70% 비용 절감
    instance_types = ["p3.2xlarge"]
  }
}
```

### 2. Auto-scaling 조정

```hcl
min_size     = 0  # 사용하지 않을 때 0으로
max_size     = 5
desired_size = 0
```

### 3. RDS Scheduled Scaling (개발 환경)

```bash
# 야간/주말 RDS 중지 (개발 환경만)
aws rds stop-db-instance --db-instance-identifier mdpg-mlops-mlflow-db

# 아침에 재시작
aws rds start-db-instance --db-instance-identifier mdpg-mlops-mlflow-db
```

### 4. S3 Lifecycle Policy 활용

```hcl
# terraform/aws-eks/s3.tf
lifecycle_rule {
  id     = "move-to-glacier"
  status = "Enabled"

  transition {
    days          = 90
    storage_class = "GLACIER"  # 장기 보관 비용 절감
  }
}
```

### 5. CloudWatch 로그 보존 기간 단축

```bash
# 30일로 제한 (기본: 무제한)
aws logs put-retention-policy \
  --log-group-name /aws/eks/mdpg-mlops-eks/cluster \
  --retention-in-days 30
```

### 예상 월 비용 (최적화 후)

| 항목 | 비용 (최적화 전) | 비용 (최적화 후) |
|------|-----------------|-----------------|
| EKS Control Plane | $73 | $73 |
| CPU Nodes (t3.medium × 2) | $60 | $60 |
| GPU Nodes (Spot, 0-5) | $0 (미사용) | $0 |
| RDS (db.t3.small) | $30 | $15 (scheduled stop) |
| S3 | $10 | $5 (Glacier) |
| ALB | $17 | $17 |
| NAT Gateway (3개) | $100 | $35 (Single NAT) |
| **총계** | **$290** | **$205** |

---

## 다음 단계

인프라 배포가 완료되면:

1. **MLflow 서버 배포**: [docs/mlflow_remote_setup.md](mlflow_remote_setup.md)
2. **클라이언트 설정**: [docs/vscode_setup.md](vscode_setup.md)
3. **배포 자동화**: [docs/deployment_scripts.md](deployment_scripts.md)

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
**문의**: mlops@example.com
