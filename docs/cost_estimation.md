# AWS 비용 상세 분석

**작성일**: 2025-10-21
**버전**: 1.0

---

## 월간 비용 (기본 운영)

### Phase 5: EKS + MLflow ($190/월)

| 항목 | 사양 | 월 비용 | 연 비용 | 비고 |
|------|------|---------|---------|------|
| **Compute** |
| EKS Control Plane | - | $73 | $876 | 클러스터당 고정 |
| Worker Nodes (CPU) | t3.medium × 2 | $60 | $720 | 24/7 운영 |
| **Database** |
| RDS PostgreSQL | db.t3.small Multi-AZ | $30 | $360 | 20GB 스토리지 포함 |
| **Storage** |
| S3 (Standard) | 100GB | $2 | $24 | 실험 아티팩트 |
| S3 (Requests) | 10,000 PUT/GET | $1 | $12 | API 호출 |
| EBS (Worker Nodes) | gp3 50GB × 2 | $8 | $96 | 노드 루트 볼륨 |
| **Network** |
| ALB (Application Load Balancer) | - | $17 | $204 | HTTPS 지원 |
| Data Transfer | 10GB/월 | $1 | $12 | 인터넷 아웃바운드 |
| NAT Gateway | 3 AZ | $100 | $1,200 | ⚠️ 최적화 대상 |
| **총계** | | **$292** | **$3,504** | |

### Phase 6: Ray Tune 추가 ($18-50/월)

| 항목 | 사양 | 시간당 | 월 사용 | 월 비용 |
|------|------|--------|---------|---------|
| GPU Worker Nodes | p3.2xlarge Spot (1 GPU) | $0.90 | 20시간 | $18 |
| GPU Worker Nodes | p3.2xlarge Spot (5 GPU) | $4.50 | 10시간 | $45 |

**시나리오**:
- 주간 튜닝 (주 1회, 5시간): $18/월
- 집중 튜닝 (주 2회, 10시간): $36/월

---

## 비용 최적화 전략

### 1. NAT Gateway 최적화 ($100 → $35)

**현재**: Multi-AZ (3개 NAT Gateway) = $32.85 × 3 = ~$100

**최적화**: Single NAT Gateway = $32.85 × 1 = ~$35

```hcl
# terraform/aws-eks/main.tf
module "vpc" {
  single_nat_gateway = true  # false → true 변경
}
```

**절감**: **$65/월** ($780/년)

**Trade-off**:
- ❌ 하나의 AZ 장애 시 전체 아웃바운드 중단
- ✅ MLflow 서버는 Private Subnet에서 실행 (영향 최소)
- ✅ 개발/스테이징 환경에 적합

### 2. RDS Scheduled Scaling (개발 환경)

**현재**: 24/7 운영 = $30/월

**최적화**: 야간/주말 중지 (50% 가동률) = $15/월

```bash
# 평일 18:00 중지
aws rds stop-db-instance --db-instance-identifier mdpg-mlops-mlflow-db

# 평일 09:00 시작
aws rds start-db-instance --db-instance-identifier mdpg-mlops-mlflow-db

# Lambda + EventBridge로 자동화
```

**절감**: **$15/월** ($180/년)

**적용 대상**: 개발/스테이징 환경만 (프로덕션 제외)

### 3. S3 Lifecycle Policy

**현재**: Standard 스토리지만 사용

**최적화**:
- 30일 후 → Intelligent-Tiering
- 90일 후 → Glacier Flexible Retrieval

```hcl
# terraform/aws-eks/s3.tf
lifecycle_rule {
  id     = "intelligent-tiering"
  status = "Enabled"

  transition {
    days          = 30
    storage_class = "INTELLIGENT_TIERING"
  }

  transition {
    days          = 90
    storage_class = "GLACIER"
  }
}
```

**절감**: **$5-10/월** (사용량에 따라)

### 4. Spot Instances (GPU)

**현재**: GPU 사용 시 On-Demand

**최적화**: Spot Instances (70% 할인)

```hcl
# terraform/aws-eks/main.tf
eks_managed_node_groups = {
  gpu_workers = {
    capacity_type = "SPOT"  # 추가
    instance_types = ["p3.2xlarge"]
  }
}
```

**비용 비교**:
- On-Demand: $3.06/시간
- Spot: $0.90/시간 (평균 70% 할인)

**절감**: 20시간/월 사용 시 **$43/월**

### 5. EKS Worker Node Right-Sizing

**현재**: t3.medium (2 vCPU, 4GB RAM) × 2

**대안 1**: t3.small (2 vCPU, 2GB RAM) × 2
- 비용: $30/월 (50% 절감)
- 적용: 개발 환경, 트래픽 적은 경우

**대안 2**: t3.large (2 vCPU, 8GB RAM) × 1
- 비용: $60/월 (동일)
- 장점: 단일 노드로 리소스 효율성 증가

### 최적화 요약

| 최적화 | 절감액/월 | 절감액/년 | 위험도 |
|--------|-----------|-----------|--------|
| Single NAT Gateway | $65 | $780 | 낮음 |
| RDS Scheduled Stop (dev) | $15 | $180 | 없음 |
| S3 Lifecycle | $7 | $84 | 없음 |
| GPU Spot Instances | $43 | $516 | 중간 |
| Worker Node Sizing | $30 | $360 | 낮음 |
| **총계** | **$160** | **$1,920** | |

**최적화 후 월 비용**: $292 - $160 = **$132/월**

---

## 비용 시뮬레이션

### 시나리오 1: 최소 운영 (개발 환경)

```
EKS Control Plane:        $73
Worker Nodes (t3.small):  $30
RDS (50% 가동):           $15
S3 + Network:             $10
ALB:                      $17
NAT Gateway (Single):     $35
─────────────────────────────
총계:                    $180/월
```

### 시나리오 2: 표준 운영 (스테이징)

```
EKS Control Plane:        $73
Worker Nodes (t3.medium): $60
RDS (24/7):               $30
S3 + Network:             $15
ALB:                      $17
NAT Gateway (Single):     $35
─────────────────────────────
총계:                    $230/월
```

### 시나리오 3: 프로덕션 (HA)

```
EKS Control Plane:        $73
Worker Nodes (t3.large):  $120
RDS (Multi-AZ):           $60
S3 + Network:             $25
ALB:                      $17
NAT Gateway (Multi-AZ):   $100
─────────────────────────────
총계:                    $395/월
```

---

## ECS vs EKS 비용 비교 (12개월)

### 시나리오 1: ECS → EKS 마이그레이션

```
ECS 운영 (6개월):
  - Fargate (2 vCPU, 4GB): $35/월 × 6 = $210
  - ALB: $17/월 × 6 = $102
  - RDS: $30/월 × 6 = $180
  - S3: $10/월 × 6 = $60
  소계: $552

마이그레이션 비용:
  - 인력 (2주, 2명): $15,000

EKS 운영 (6개월):
  - 표준 운영: $230/월 × 6 = $1,380
  소계: $1,380

총계: $16,932
```

### 시나리오 2: EKS 직접 구축

```
EKS 운영 (12개월):
  - 표준 운영: $230/월 × 12 = $2,760

총계: $2,760
```

### 절감액: $14,172

---

## 비용 모니터링

### AWS Cost Explorer 설정

```bash
# Tag 기반 비용 추적
aws ce get-cost-and-usage \
  --time-period Start=2025-10-01,End=2025-10-31 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=TAG,Key=Project
```

### Budget Alerts 설정

```bash
# $300 예산 알림
aws budgets create-budget \
  --account-id 123456789012 \
  --budget file://budget.json
```

**budget.json**:
```json
{
  "BudgetName": "MLOps-Platform-Monthly",
  "BudgetLimit": {
    "Amount": "300",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

---

## 참고

- [AWS Pricing Calculator](https://calculator.aws/)
- [EKS Pricing](https://aws.amazon.com/eks/pricing/)
- [RDS Pricing](https://aws.amazon.com/rds/postgresql/pricing/)
- [EC2 Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/)

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
