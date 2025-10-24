# TODO - 추후 작업 목록

**최종 업데이트**: 2025-10-24
**버전**: 2.1

이 문서는 현재 프로젝트에서 완전히 구현되지 않았거나, 추후 클라우드 환경에서 수행해야 하는 작업들을 정리한 목록입니다.

## 최근 완료 항목 (Phase 4.6)

### ✅ Ray Tune 하이퍼파라미터 최적화 (2025-10-24)

**완료 항목**:
- ✅ Ray Tune 코어 모듈 구현 (`src/tuning/ray_tune.py`)
- ✅ MLflow 완전 통합 (배치 레벨 + 에포크 레벨 메트릭)
- ✅ ASHA 스케줄러 및 HyperOpt 검색 알고리즘
- ✅ 하이퍼파라미터 검색 공간 정의 (learning_rate, weight_decay, momentum)
- ✅ Makefile 명령어 (`make tune`, `make tune-quick`, `make tune-extensive`)
- ✅ 통합 테스트 (`test_ray_tune.py`)
- ✅ 문서화 (plan.md, README.md, TODO.md)

**주요 기능**:
- 분산 하이퍼파라미터 탐색 (Ray Tune)
- MLflow 자동 로깅 (batch-level + epoch-level 메트릭)
- Best trial 자동 선택 및 기록
- 체크포인트 저장 및 복구

**향후 계획**:
- Phase 5에서 EKS + Ray Cluster로 확장
- GPU 워커 노드를 활용한 대규모 튜닝
- Ray Train과 통합하여 DDP 분산 학습 지원

---

## 📌 프로젝트 방향 변경 (2025-10-21)

**새로운 우선순위**: Phase 5-7 (EKS 기반 MLOps 플랫폼) 구축이 최우선 과제로 변경되었습니다.

기존의 로컬 환경 최적화(Phase 3.3 DDP 테스트, 모델 최적화 등)는 EKS 인프라 구축 이후로 연기되었습니다.

---

## 목차

### 현재 우선순위 (Phase 5-7)
1. [Phase 5: AWS EKS 인프라 구축](#phase-5-aws-eks-인프라-구축)
2. [Phase 6: Ray Tune 하이퍼파라미터 최적화](#phase-6-ray-tune-하이퍼파라미터-최적화)
3. [Phase 7: DDP 분산 학습 + Airflow](#phase-7-ddp-분산-학습--airflow)

### 추후 작업 (Phase 5-7 이후)
4. [모델 최적화](#모델-최적화)
5. [데이터 파이프라인 개선](#데이터-파이프라인-개선)
6. [MLflow 고급 기능](#mlflow-고급-기능)
7. [CI/CD 개선](#cicd-개선)
8. [모니터링 및 관찰성](#모니터링-및-관찰성)

---

## Phase 5: AWS EKS 인프라 구축

**우선순위**: Critical (최우선)
**예상 기간**: 4-5주
**예상 비용**: ~$190/월 (기본 운영)

**목표**: 중앙화된 MLflow 서버를 EKS에 배포하여 멀티 유저 환경 구축

**상세 문서**: [docs/eks_infrastructure.md](docs/eks_infrastructure.md) (작성 예정)

---

### 5.1 AWS 기본 인프라 (1주)

**우선순위**: Critical

#### Terraform 프로젝트 구조
- [ ] `terraform/aws-eks/` 디렉토리 생성
- [ ] `terraform/aws-eks/main.tf` - EKS 클러스터 정의
  - [ ] VPC 설정 (3 AZ, Public/Private 서브넷)
  - [ ] EKS 클러스터 (Kubernetes 1.28+)
  - [ ] Node Group (t3.medium × 2, Auto-scaling)
- [ ] `terraform/aws-eks/rds.tf` - PostgreSQL 데이터베이스
  - [ ] db.t3.small (2 vCPU, 2GB RAM)
  - [ ] Multi-AZ 배포
  - [ ] 암호화 활성화
- [ ] `terraform/aws-eks/s3.tf` - S3 버킷
  - [ ] Versioning 활성화
  - [ ] Lifecycle policy (90일 후 Glacier)
- [ ] `terraform/aws-eks/iam.tf` - IRSA 권한
  - [ ] MLflow Pod → S3 접근 권한
  - [ ] Ray Cluster → S3 접근 권한
- [ ] `terraform/aws-eks/variables.tf` - 변수 정의
- [ ] `terraform/aws-eks/outputs.tf` - 출력 정의

#### 배포 및 검증
- [ ] `terraform init` 실행
- [ ] `terraform plan` 검토
- [ ] `terraform apply` 실행
- [ ] `aws eks update-kubeconfig --name mdpg-mlops` 실행
- [ ] `kubectl get nodes` 확인

**예상 비용**:
- EKS Control Plane: $73/월
- Worker Nodes (t3.medium × 2): $60/월
- RDS (db.t3.small): $30/월
- S3: ~$5/월
- **총계**: ~$168/월

---

### 5.2 MLflow 서버 배포 (1주)

**우선순위**: Critical

#### Helm Chart 작성
- [ ] `charts/mlflow/` 디렉토리 생성
- [ ] `charts/mlflow/Chart.yaml`
- [ ] `charts/mlflow/values.yaml` - 설정 정의
  ```yaml
  replicaCount: 2
  image:
    repository: ghcr.io/[username]/mlflow-study-mlflow
    tag: latest

  env:
    BACKEND_STORE_URI: postgresql://...
    ARTIFACT_ROOT: s3://mlflow-artifacts/

  ingress:
    enabled: true
    className: alb
    hosts:
      - host: mlflow.mdpg.ai

  hpa:
    minReplicas: 2
    maxReplicas: 5
  ```

- [ ] `charts/mlflow/templates/deployment.yaml`
- [ ] `charts/mlflow/templates/service.yaml`
- [ ] `charts/mlflow/templates/ingress.yaml` (ALB Ingress Controller)
- [ ] `charts/mlflow/templates/hpa.yaml` (Horizontal Pod Autoscaler)
- [ ] `charts/mlflow/templates/secret.yaml` (DB 자격증명)

#### MLflow Authentication 설정
- [ ] MLflow Authentication 활성화
  ```bash
  mlflow server \
    --backend-store-uri postgresql://... \
    --default-artifact-root s3://... \
    --app-name basic-auth
  ```
- [ ] 사용자 계정 생성 (MLOps × 2, ML Engineer × 1)
- [ ] 권한 설정 (READ/EDIT/MANAGE)
- [ ] Kubernetes Secret으로 자격증명 저장

#### SSL/TLS 설정
- [ ] AWS Certificate Manager에서 인증서 발급
- [ ] ALB Ingress에 HTTPS 적용
- [ ] HTTP → HTTPS 리다이렉트 설정

#### 배포 및 검증
- [ ] `helm install mlflow ./charts/mlflow`
- [ ] `kubectl get pods -n ml-platform` 확인
- [ ] `kubectl logs -f deployment/mlflow` 확인
- [ ] MLflow UI 접속 (https://mlflow.mdpg.ai)
- [ ] 사용자 로그인 테스트

---

### 5.3 클라이언트 마이그레이션 (1일)

**우선순위**: High

#### VSCode 환경 설정 가이드
- [ ] `.env.remote` 템플릿 생성
  ```bash
  # Remote MLflow Server
  MLFLOW_TRACKING_URI=https://mlflow.mdpg.ai
  MLFLOW_TRACKING_USERNAME=ml_engineer_1
  MLFLOW_TRACKING_PASSWORD=***

  # S3 Configuration (IRSA로 자동 처리)
  # AWS_REGION=us-west-2
  ```

- [ ] VSCode settings.json 예제 작성
- [ ] Python 환경 설정 가이드

#### 로컬 → 원격 전환 테스트
- [ ] 기존 학습 코드 수정 없이 실험 실행
  ```bash
  cp .env.remote .env
  poetry run python -m src.training.train
  ```
- [ ] MLflow UI에서 실험 결과 확인
- [ ] 아티팩트가 S3에 업로드되었는지 확인
  ```bash
  aws s3 ls s3://mlflow-artifacts/
  ```

**상세 가이드**: [docs/vscode_setup.md](docs/vscode_setup.md) (작성 예정)

---

### 5.4 배포 자동화 스크립트 (2-3일)

**우선순위**: High (휴먼 에러 최소화)

#### Setup Scripts
- [ ] `scripts/setup/01-setup-aws.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "🔧 AWS CLI 설정 확인"
  aws sts get-caller-identity

  echo "✅ AWS 설정 완료"
  ```

- [ ] `scripts/setup/02-deploy-eks.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "🚀 EKS 클러스터 배포"
  cd terraform/aws-eks
  terraform init
  terraform plan -out=tfplan

  read -p "Deploy? (yes/no): " confirm
  [[ "$confirm" != "yes" ]] && exit 0

  terraform apply tfplan

  echo "📝 kubeconfig 업데이트"
  aws eks update-kubeconfig --name mdpg-mlops

  echo "✅ EKS 배포 완료"
  kubectl get nodes
  ```

- [ ] `scripts/setup/03-deploy-mlflow.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "🚀 MLflow 서버 배포"
  helm install mlflow ./charts/mlflow \
    --namespace ml-platform \
    --create-namespace \
    --values charts/mlflow/values-production.yaml

  echo "⏳ Pod 시작 대기"
  kubectl wait --for=condition=ready pod -l app=mlflow -n ml-platform --timeout=300s

  echo "✅ MLflow 배포 완료"
  kubectl get pods -n ml-platform
  ```

- [ ] `scripts/setup/04-setup-users.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "👥 MLflow 사용자 계정 생성"
  kubectl exec -it deployment/mlflow -n ml-platform -- \
    mlflow server create-user \
      --username mlops_engineer_1 \
      --password [SECURE_PASSWORD]

  # 추가 사용자...
  echo "✅ 사용자 계정 생성 완료"
  ```

- [ ] `scripts/setup/05-test-connection.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "🔍 연결 테스트"
  export MLFLOW_TRACKING_URI=https://mlflow.mdpg.ai
  export MLFLOW_TRACKING_USERNAME=mlops_engineer_1

  python -c "
  import mlflow
  mlflow.set_experiment('connection-test')
  with mlflow.start_run():
      mlflow.log_param('test', 'success')
  print('✅ MLflow 연결 성공')
  "
  ```

- [ ] `scripts/setup/06-verify-all.sh`
  ```bash
  #!/bin/bash
  set -e

  echo "🔍 전체 시스템 검증"

  # EKS 확인
  kubectl get nodes

  # MLflow 확인
  kubectl get pods -n ml-platform

  # RDS 확인
  kubectl exec -it deployment/mlflow -n ml-platform -- \
    psql $BACKEND_STORE_URI -c "SELECT version();"

  # S3 확인
  aws s3 ls s3://mlflow-artifacts/

  echo "✅ 전체 검증 완료"
  ```

#### Operations Scripts
- [ ] `scripts/ops/backup-mlflow.sh` - PostgreSQL 백업
- [ ] `scripts/ops/restore-mlflow.sh` - 복원
- [ ] `scripts/ops/scale-mlflow.sh` - 수동 스케일링
- [ ] `scripts/ops/logs-mlflow.sh` - 로그 수집

**상세 가이드**: [docs/deployment_scripts.md](docs/deployment_scripts.md) (작성 예정)

---

### Phase 5 성공 기준

- [ ] ✅ MLflow 서버가 EKS에서 안정적으로 실행
- [ ] ✅ RDS PostgreSQL 연결 성공
- [ ] ✅ S3 아티팩트 저장 성공
- [ ] ✅ MLflow Authentication 동작 (최소 3명 사용자)
- [ ] ✅ HTTPS 접속 가능 (https://mlflow.mdpg.ai)
- [ ] ✅ 모든 배포 스크립트 정상 작동
- [ ] ✅ 로컬 개발 환경에서 원격 MLflow 서버 접속 성공
- [ ] ✅ HPA (Horizontal Pod Autoscaler) 동작 확인

---

## Phase 6: Ray Cluster 분산 학습 확장

**우선순위**: High
**예상 기간**: 2-3주
**선행 조건**: Phase 5 완료
**현재 상태**: Phase 4.6에서 로컬 Ray Tune 완료 ✅

**목표**: EKS + Ray Cluster에서 대규모 하이퍼파라미터 튜닝 및 분산 학습

**상세 문서**: [docs/ray_tune_guide.md](docs/ray_tune_guide.md) (작성 예정)

**참고**:
- Ray Tune 코어 기능은 Phase 4.6에서 완료됨
- Phase 6는 EKS 환경에서 GPU 워커 노드를 활용한 확장에 집중

---

### 6.1 Ray Cluster 배포 (1주)

**우선순위**: High

#### KubeRay Operator 설치
- [ ] KubeRay Operator YAML 다운로드
  ```bash
  kubectl create -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/default/operator.yaml
  ```
- [ ] Operator 정상 동작 확인
  ```bash
  kubectl get pods -n ray-system
  ```

#### Ray Cluster Helm Chart
- [ ] `charts/ray-cluster/Chart.yaml`
- [ ] `charts/ray-cluster/values.yaml`
  ```yaml
  head:
    cpu: 2
    memory: 4Gi
    replicas: 1

  worker:
    minReplicas: 0
    maxReplicas: 5
    cpu: 4
    memory: 16Gi
    gpu: 1
    nodeType: p3.2xlarge  # Spot Instance
  ```
- [ ] `charts/ray-cluster/templates/raycluster.yaml`

#### GPU 노드 그룹 추가
- [ ] Terraform에 GPU 노드 그룹 추가
  ```hcl
  resource "aws_eks_node_group" "gpu_nodes" {
    node_group_name = "gpu-workers"
    instance_types  = ["p3.2xlarge"]
    capacity_type   = "SPOT"  # 비용 절감

    scaling_config {
      min_size     = 0
      desired_size = 0
      max_size     = 5
    }
  }
  ```
- [ ] `terraform apply` 실행

#### Ray Dashboard 접근
- [ ] Port-forward 설정
  ```bash
  kubectl port-forward svc/ray-head 8265:8265 -n ray-system
  ```
- [ ] 또는 Ingress 설정 (선택)

---

### 6.2 Ray Tune 코드 작성 (1주)

**우선순위**: High

#### Ray Tune 통합
- [ ] `src/tuning/__init__.py`
- [ ] `src/tuning/ray_tune.py`
  ```python
  from ray import tune
  from ray.tune.integration.mlflow import MLflowLoggerCallback

  def train_cifar10(config):
      # PyTorch 학습 코드
      model = VisionModel(
          model_name=config["model_name"],
          num_classes=10
      )

      # 학습 루프
      for epoch in range(config["epochs"]):
          train_loss = train_epoch(model, ...)
          val_acc = validate(model, ...)

          # Ray Tune에 메트릭 보고
          tune.report(val_accuracy=val_acc, train_loss=train_loss)

  # 탐색 공간 정의
  config = {
      "model_name": tune.choice(["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"]),
      "learning_rate": tune.loguniform(1e-4, 1e-1),
      "batch_size": tune.choice([32, 64, 128]),
      "epochs": 20,
  }

  # Ray Tune 실행
  analysis = tune.run(
      train_cifar10,
      config=config,
      num_samples=100,  # 100 trials
      resources_per_trial={"gpu": 1},
      callbacks=[MLflowLoggerCallback(
          tracking_uri="https://mlflow.mdpg.ai",
          experiment_name="ray-tune-cifar10"
      )]
  )
  ```

#### Search Algorithms
- [ ] `src/tuning/search_algorithms.py`
  ```python
  from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
  from ray.tune.search.optuna import OptunaSearch

  # ASHA (Asynchronous Successive Halving)
  scheduler = ASHAScheduler(
      time_attr='training_iteration',
      metric='val_accuracy',
      mode='max',
      max_t=20,
      grace_period=5,
      reduction_factor=3
  )

  # Optuna Bayesian Optimization
  search_alg = OptunaSearch(
      metric="val_accuracy",
      mode="max"
  )
  ```

#### GPU 스케줄링
- [ ] `resources_per_trial={"gpu": 1}` 설정
- [ ] Fractional GPU 실험 (필요시)
  ```python
  resources_per_trial={"gpu": 0.5}  # 1 GPU에 2 trials
  ```

---

### 6.3 실험 실행 (3-5일)

**우선순위**: High

#### 100 Trials 하이퍼파라미터 탐색
- [ ] Ray Tune 스크립트 실행
  ```bash
  poetry run python -m src.tuning.ray_tune --num-samples 100
  ```
- [ ] Ray Dashboard에서 진행 상황 모니터링
- [ ] MLflow UI에서 실험 결과 확인

#### 목표
- [ ] Test Accuracy 90%+ 달성
- [ ] 최적 하이퍼파라미터 조합 발견
- [ ] MLflow에 모든 trial 자동 기록

#### 최적 모델 등록
- [ ] MLflow Model Registry에 Best Model 등록
  ```python
  best_model_uri = analysis.best_checkpoint
  mlflow.register_model(best_model_uri, "vision-classifier")
  ```
- [ ] Production 스테이지로 전환

---

### Phase 6 성공 기준

- [ ] ✅ Ray Cluster EKS에서 안정 실행
- [ ] ✅ GPU auto-scaling 동작 (0 → 5 → 0)
- [ ] ✅ 100 trials 완료
- [ ] ✅ CIFAR-10 정확도 90%+ 달성
- [ ] ✅ MLflow에 모든 실험 자동 기록
- [ ] ✅ GPU 비용 $20 이하 (Spot Instance 활용)

---

## Phase 7: DDP 분산 학습 + Airflow

**우선순위**: Medium
**예상 기간**: 2-3주
**선행 조건**: Phase 5, 6 완료

**목표**: PyTorch DDP 멀티 GPU 학습 및 Airflow 파이프라인 자동화

**상세 문서**: [docs/distributed_training.md](docs/distributed_training.md) (작성 예정)

---

### 7.1 PyTorch DDP 구현 (1주)

**우선순위**: Medium

#### DDP 유틸리티
- [ ] `src/training/distributed.py`
  ```python
  import torch.distributed as dist

  def setup_distributed(rank, world_size, backend='nccl'):
      dist.init_process_group(backend, rank=rank, world_size=world_size)

  def cleanup_distributed():
      dist.destroy_process_group()
  ```

#### DDP 학습 스크립트
- [ ] `src/training/train_ddp.py`
  ```python
  from torch.nn.parallel import DistributedDataParallel as DDP

  def main(rank, world_size):
      setup_distributed(rank, world_size)

      model = VisionModel(...).to(rank)
      model = DDP(model, device_ids=[rank])

      # 학습 루프
      for epoch in range(epochs):
          train_epoch(model, ...)

          # Rank 0만 MLflow 로깅
          if rank == 0:
              mlflow.log_metrics(...)

      cleanup_distributed()
  ```

#### Docker 이미지 업데이트
- [ ] Dockerfile에 NCCL 라이브러리 추가
- [ ] SSH 설정 (멀티 노드 DDP)

---

### 7.2 DDP 테스트 (3-5일)

**우선순위**: Medium

#### Ray Train으로 DDP 실행
- [ ] Single-node multi-GPU (p3.8xlarge, 4 GPUs)
  ```python
  from ray.train.torch import TorchTrainer

  trainer = TorchTrainer(
      train_func=train_ddp,
      scaling_config=ScalingConfig(num_workers=4, use_gpu=True)
  )
  trainer.fit()
  ```

- [ ] Multi-node multi-GPU (p3.2xlarge × 4)
  ```python
  scaling_config=ScalingConfig(num_workers=4, resources_per_worker={"GPU": 1})
  ```

#### 성능 벤치마크
- [ ] 학습 시간 비교표 작성

  | 설정 | 학습 시간 (20 epochs) | Speedup |
  |------|---------------------|---------|
  | 1 GPU | ??? | 1.0x |
  | 4 GPUs (DDP) | ??? | ???x |

- [ ] 목표: 4 GPUs로 3x 이상 speedup

---

### 7.3 Airflow 파이프라인 (1주)

**우선순위**: Medium

#### Airflow Helm Chart 배포
- [ ] `charts/airflow/values.yaml`
  ```yaml
  executor: KubernetesExecutor
  dags:
    gitSync:
      enabled: true
      repo: https://github.com/[username]/mlflow-study.git
      branch: main
      subPath: dags/
  ```
- [ ] `helm install airflow apache-airflow/airflow`

#### DAG 작성
- [ ] `dags/daily_training.py`
  ```python
  from airflow import DAG
  from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

  with DAG('daily_training', schedule_interval='@daily') as dag:
      train_task = KubernetesPodOperator(
          task_id='train_model',
          image='ghcr.io/[username]/mlflow-study:latest',
          cmds=['poetry', 'run', 'python', '-m', 'src.training.train'],
          env_vars={
              'MLFLOW_TRACKING_URI': 'https://mlflow.mdpg.ai',
              'EXPERIMENT_NAME': 'daily-training',
              'EPOCHS': '20'
          }
      )
  ```

- [ ] `dags/hyperparameter_tuning.py` - 주간 Ray Tune 실행
- [ ] `dags/model_evaluation.py` - 모델 평가

#### MLflow 통합
- [ ] 모든 DAG에서 MLflow 자동 로깅
- [ ] 최고 성능 모델 자동 등록

---

### Phase 7 성공 기준

- [ ] ✅ DDP 멀티 GPU 학습 성공
- [ ] ✅ 4 GPUs로 학습 시간 3x 단축
- [ ] ✅ Airflow DAG 자동 실행 (일일 스케줄)
- [ ] ✅ MLflow에 모든 파이프라인 결과 기록
- [ ] ✅ 다중 실험 동시 실행 (5개+)

---

## 추후 작업 (Phase 5-7 이후)

아래 작업들은 EKS 인프라 구축 및 Ray Tune, DDP 구현 이후에 수행합니다.

---

## Phase 3.3 DDP 테스트 (클라우드 필요)

### 배경 및 제약사항

**현재 상황**:
- Phase 3.3에서 DDP (Distributed Data Parallel) **코드 구조 완성**
- `src/training/distributed.py`, `src/training/train_distributed.py` 구현 완료
- 로컬 CPU 2-process 기본 동작 검증 완료

**제약사항**:
- ❌ MacBook M2 MPS backend는 PyTorch DDP 미지원
- ❌ 로컬 CPU DDP는 학습 속도가 매우 느림 (실용성 없음)
- ❌ Multi-GPU 환경 없음 (실제 분산 학습 테스트 불가)

**결론**:
- ✅ DDP 코드 구조는 완성
- ⏳ **실제 multi-GPU 테스트는 클라우드 환경에서 수행 필요**

---

### 필요 환경

#### Option 1: AWS EC2 (추천)
```
인스턴스 타입: p3.2xlarge
- GPU: 1x NVIDIA V100 (16GB)
- vCPU: 8
- RAM: 61 GB
- 비용: $3.06/hour (온디맨드)
- 비용: ~$0.90/hour (Spot Instance, 약 70% 할인)

인스턴스 타입: p3.8xlarge (4-GPU 테스트용)
- GPU: 4x NVIDIA V100 (16GB each)
- vCPU: 32
- RAM: 244 GB
- 비용: $12.24/hour (온디맨드)
- 비용: ~$3.60/hour (Spot Instance)
```

#### Option 2: GCP Compute Engine
```
머신 타입: n1-standard-8 + 2x T4
- GPU: 2x NVIDIA T4 (16GB each)
- vCPU: 8
- RAM: 30 GB
- 비용: ~$2.50/hour
```

#### Option 3: Google Colab Pro+ (간단 테스트용)
```
- GPU: A100 or V100 (가변)
- 비용: $49.99/month (무제한 실행 시간)
- 제약: 단일 GPU만 지원, DDP 테스트 제한적
```

**추천**: AWS p3.2xlarge Spot Instance (비용 효율적)

---

### 환경 설정 가이드

#### 1. AWS EC2 인스턴스 생성

```bash
# AWS CLI로 Spot Instance 요청
aws ec2 request-spot-instances \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification \
    InstanceType=p3.2xlarge,\
    ImageId=ami-0c55b159cbfafe1f0,\  # Deep Learning AMI
    KeyName=my-key-pair,\
    SecurityGroupIds=sg-xxxxxx
```

#### 2. CUDA 및 PyTorch 설치

```bash
# Deep Learning AMI 사용 시 이미 설치되어 있음
# 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 수동 설치가 필요한 경우
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3. 프로젝트 코드 배포

```bash
# Git clone
git clone https://github.com/[username]/mlflow-study.git
cd mlflow-study

# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
poetry install

# MLflow 인프라 시작 (Docker Compose)
make up
```

---

### 작업 목록

#### 1. Multi-GPU 학습 테스트

**우선순위**: High
**예상 소요**: 4-6시간
**예상 비용**: ~$5-10 (Spot Instance 사용 시)

- [ ] **Single GPU 베이스라인 측정**
  ```bash
  # 환경 변수 설정
  export DISTRIBUTED=false
  export EPOCHS=10
  export DATASET=CIFAR10

  # 학습 실행
  poetry run python -m src.training.train

  # 기록할 메트릭:
  # - 학습 시간 (초)
  # - Throughput (images/sec)
  # - GPU 메모리 사용량
  # - 최종 정확도
  ```

- [ ] **2-GPU DDP 테스트 (p3.2xlarge 2대 또는 p3.8xlarge 1대)**
  ```bash
  # 환경 변수 설정
  export DISTRIBUTED=true
  export BACKEND=nccl
  export MASTER_ADDR=localhost
  export MASTER_PORT=12355

  # DDP 학습 실행
  torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/training/train_distributed.py

  # 검증 항목:
  # - 2개 프로세스 모두 시작
  # - Rank 0, 1 할당 확인
  # - Gradient synchronization 동작
  # - MLflow 로깅 (main process만)
  # - 최종 모델 일치 (all processes)
  ```

- [ ] **4-GPU DDP 테스트 (p3.8xlarge 필요)**
  ```bash
  torchrun --nproc_per_node=4 src/training/train_distributed.py
  ```

- [ ] **Gradient 동기화 검증**
  - [ ] 각 GPU에서 계산된 gradient가 동일한지 확인
  - [ ] All-reduce 후 모델 파라미터 일치 검증
  - [ ] Loss 수렴 패턴 확인

- [ ] **손실 수렴 확인**
  - [ ] Single GPU vs DDP 손실 곡선 비교
  - [ ] 최종 정확도 비교 (동일해야 함)

#### 2. 성능 벤치마크

**우선순위**: High
**예상 소요**: 2-4시간

- [ ] **학습 속도 비교**

  | 설정 | Throughput (images/sec) | 학습 시간 (10 epochs) | GPU 메모리 | Speedup |
  |------|------------------------|---------------------|-----------|---------|
  | 1-GPU | ??? | ??? | ??? | 1.0x |
  | 2-GPU DDP | ??? | ??? | ??? | ???x |
  | 4-GPU DDP | ??? | ??? | ??? | ???x |

  ```bash
  # 측정 스크립트
  # scripts/benchmark_ddp.sh

  #!/bin/bash
  for nproc in 1 2 4; do
    echo "Testing with $nproc GPUs"

    if [ $nproc -eq 1 ]; then
      # Single GPU
      time poetry run python -m src.training.train
    else
      # DDP
      time torchrun --nproc_per_node=$nproc src/training/train_distributed.py
    fi

    # 메트릭 기록
    # ...
  done
  ```

- [ ] **GPU 메모리 사용량 분석**
  ```bash
  # nvidia-smi로 모니터링
  watch -n 1 nvidia-smi
  ```

- [ ] **통신 오버헤드 측정**
  - [ ] All-reduce 시간 측정
  - [ ] Gradient 동기화 오버헤드
  - [ ] 네트워크 대역폭 영향

- [ ] **배치 크기 확장성 테스트**
  - [ ] 1-GPU: batch_size=64
  - [ ] 2-GPU: batch_size=128 (2x)
  - [ ] 4-GPU: batch_size=256 (4x)
  - [ ] Effective batch size 증가에 따른 정확도 변화

#### 3. 최적화

**우선순위**: Medium
**예상 소요**: 4-8시간

- [ ] **Gradient Accumulation 구현**
  ```python
  # src/training/train.py

  accumulation_steps = 4
  optimizer.zero_grad()

  for i, (inputs, targets) in enumerate(train_loader):
      outputs = model(inputs)
      loss = criterion(outputs, targets) / accumulation_steps
      loss.backward()

      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```

- [ ] **All-reduce 전략 최적화**
  - [ ] Bucket size 조정
  - [ ] Gradient compression 시도
  - [ ] 비동기 all-reduce 실험

- [ ] **Mixed Precision Training (AMP) 통합**
  ```python
  from torch.cuda.amp import autocast, GradScaler

  scaler = GradScaler()

  for inputs, targets in train_loader:
      with autocast():
          outputs = model(inputs)
          loss = criterion(outputs, targets)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```

- [ ] **DataLoader 성능 튜닝**
  - [ ] `num_workers` 최적화 (CPU 코어 수 고려)
  - [ ] `pin_memory=True` 효과 측정
  - [ ] `prefetch_factor` 조정

- [ ] **모델 병렬화 실험** (매우 큰 모델용)
  - [ ] Pipeline parallelism
  - [ ] Tensor parallelism

#### 4. 문서화

**우선순위**: High
**예상 소요**: 2-3시간

- [ ] **벤치마크 결과 정리**
  - [ ] `docs/ddp_benchmark_results.md` 생성
  - [ ] 성능 그래프 및 표
  - [ ] 분석 및 결론

- [ ] **클라우드 설정 가이드 작성**
  - [ ] `docs/cloud_setup_guide.md` 생성
  - [ ] AWS/GCP 단계별 설정 방법
  - [ ] 비용 최적화 팁

- [ ] **DDP 문서 업데이트**
  - [ ] `docs/distributed_training.md` 업데이트
  - [ ] 실제 테스트 결과 반영
  - [ ] Best practices 정리

- [ ] **비용 최적화 문서화**
  - [ ] Spot Instance 활용법
  - [ ] 자동 종료 스크립트
  - [ ] 예산 알림 설정

---

### 예상 비용 (AWS 기준)

#### Spot Instance 사용 시 (추천)

| 작업 | 인스턴스 | GPU | 예상 시간 | 시간당 비용 | 총 비용 |
|------|----------|-----|----------|-----------|---------|
| 환경 설정 | p3.2xlarge | 1x V100 | 1시간 | $0.90 | $0.90 |
| 1-GPU 베이스라인 | p3.2xlarge | 1x V100 | 1시간 | $0.90 | $0.90 |
| 2-GPU DDP 테스트 | p3.8xlarge | 4x V100 | 2시간 | $3.60 | $7.20 |
| 4-GPU DDP 테스트 | p3.8xlarge | 4x V100 | 2시간 | $3.60 | $7.20 |
| 최적화 실험 | p3.8xlarge | 4x V100 | 4시간 | $3.60 | $14.40 |
| **총계** | - | - | **10시간** | - | **$30.60** |

#### 비용 절감 팁

1. **Spot Instance 활용**: 70% 비용 절감
2. **작업 자동 종료**: 유휴 시간 제거
   ```bash
   # 학습 완료 후 자동 종료
   poetry run python -m src.training.train && sudo shutdown -h now
   ```
3. **EBS 볼륨 최소화**: 100GB → 50GB
4. **S3 데이터 캐싱**: 데이터 다운로드 반복 방지

---

### 우선순위 및 일정

**우선순위**: **Medium**

Phase 3 완료 후 또는 실제 프로덕션 배포 전에 수행

**선행 조건**:
- ✅ Phase 3.3 DDP 코드 완성
- ✅ 로컬 CPU DDP 기본 동작 검증

**블로커**:
- ⏳ 클라우드 환경 미구축
- ⏳ 예산 승인 필요 (~$30-50)

**예상 일정**:
- 클라우드 환경 준비: 2025-11월 (Phase 3 완료 후)
- DDP 테스트 수행: 2일 (집중 작업)
- 결과 문서화: 1일

---

## 모델 최적화

### Quantization (양자화)

**우선순위**: Medium
**목적**: 모델 크기 축소, 추론 속도 향상

- [ ] **PyTorch Dynamic Quantization**
  ```python
  import torch.quantization as quantization

  model_fp32 = torch.load("model.pth")
  model_int8 = quantization.quantize_dynamic(
      model_fp32, {torch.nn.Linear}, dtype=torch.qint8
  )

  # 크기 비교
  # 정확도 비교
  # 추론 속도 비교
  ```

- [ ] **Post-Training Static Quantization**
  - [ ] Calibration dataset 준비
  - [ ] 정확도 손실 측정 (목표: <1%)

- [ ] **Quantization-Aware Training (QAT)**
  - [ ] 학습 중 quantization 시뮬레이션
  - [ ] 정확도 향상 가능성 탐색

**목표**:
- 모델 크기: 10MB → 2-3MB (INT8)
- 추론 속도: 2-3x 향상
- 정확도 손실: <1%

---

### Pruning (가지치기)

**우선순위**: Low
**목적**: 모델 경량화, 추론 속도 향상

- [ ] **비구조화 Pruning (Unstructured)**
  ```python
  import torch.nn.utils.prune as prune

  # L1 unstructured pruning
  prune.l1_unstructured(model.conv1, name="weight", amount=0.3)
  ```

- [ ] **구조화 Pruning (Structured)**
  - [ ] Channel pruning
  - [ ] Filter pruning

**목표**:
- 파라미터 수: 30% 감소
- 정확도 유지: >89%

---

### Knowledge Distillation

**우선순위**: Low
**목적**: 작은 모델로 큰 모델의 성능 재현

- [ ] **Teacher 모델 학습** (ResNet18)
- [ ] **Student 모델 설계** (MobileNetV3-Small)
- [ ] **Distillation 손실 구현**
  ```python
  # Soft targets from teacher
  # Hard targets from ground truth
  # Temperature scaling
  ```

**목표**:
- Student 정확도: Teacher의 95% 수준
- 모델 크기: 50% 축소
- 추론 속도: 2x 향상

---

### ONNX 변환 및 최적화

**우선순위**: Medium
**목적**: 다양한 프레임워크에서 모델 사용

- [ ] **PyTorch → ONNX 변환**
  ```python
  import torch.onnx

  dummy_input = torch.randn(1, 3, 32, 32)
  torch.onnx.export(
      model,
      dummy_input,
      "model.onnx",
      input_names=["image"],
      output_names=["logits"],
      dynamic_axes={"image": {0: "batch_size"}}
  )
  ```

- [ ] **ONNX Runtime 추론 테스트**
- [ ] **TensorRT 최적화** (GPU)
- [ ] **OpenVINO 최적화** (CPU)

---

## 데이터 파이프라인 개선

### Dataset Caching

**우선순위**: Low
**목적**: 반복 학습 시 데이터 로딩 속도 향상

- [ ] **메모리 캐싱 (작은 데이터셋)**
  ```python
  class CachedDataset(Dataset):
      def __init__(self, dataset):
          self.cache = [dataset[i] for i in range(len(dataset))]

      def __getitem__(self, idx):
          return self.cache[idx]
  ```

- [ ] **디스크 캐싱 (큰 데이터셋)**
  - [ ] h5py or zarr 사용

---

### Prefetching 최적화

**우선순위**: Low

- [ ] **CUDA Stream 활용**
- [ ] **DataLoader prefetch_factor 튜닝**
- [ ] **Multi-process data loading 최적화**

---

### 고급 Augmentation

**우선순위**: Medium
**목적**: 정확도 향상

- [ ] **AutoAugment 구현**
- [ ] **RandAugment 구현**
- [ ] **CutMix / MixUp**

---

## MLflow 고급 기능

### MLflow Projects

**우선순위**: Low
**목적**: 재현 가능한 실험 실행

- [ ] **MLproject 파일 작성**
  ```yaml
  name: vision-training

  conda_env: conda.yaml

  entry_points:
    main:
      parameters:
        epochs: {type: int, default: 20}
        batch_size: {type: int, default: 64}
      command: "python -m src.training.train"
  ```

- [ ] **MLflow Project 실행 테스트**
  ```bash
  mlflow run . -P epochs=50
  ```

---

### MLflow Models Serving

**우선순위**: Medium
**목적**: REST API로 모델 서빙

- [ ] **Model signature 정의**
  ```python
  from mlflow.models.signature import infer_signature

  signature = infer_signature(input_sample, output_sample)
  mlflow.pytorch.log_model(model, "model", signature=signature)
  ```

- [ ] **Local serving 테스트**
  ```bash
  mlflow models serve -m "models:/vision-classifier/Production" -p 8080
  ```

- [ ] **FastAPI wrapper 작성**
- [ ] **추론 최적화** (배치 처리)

---

## CI/CD 개선

### GPU Runner 설정

**우선순위**: Low
**목적**: GPU 테스트 자동화

- [ ] **Self-hosted runner 설정**
  - [ ] AWS EC2 GPU 인스턴스
  - [ ] GitHub Actions self-hosted runner 등록

- [ ] **GPU 테스트 workflow 추가**
  ```yaml
  # .github/workflows/gpu-test.yml
  jobs:
    gpu-test:
      runs-on: [self-hosted, gpu]
      steps:
        - name: Run GPU tests
          run: poetry run pytest tests/test_gpu.py
  ```

---

### Nightly 전체 테스트

**우선순위**: Low

- [ ] **매일 밤 slow 테스트 포함 실행**
  ```yaml
  on:
    schedule:
      - cron: '0 2 * * *'  # 매일 오전 2시 (UTC)
  ```

---

### Performance Regression 테스트

**우선순위**: Low

- [ ] **벤치마크 결과 저장**
- [ ] **성능 저하 감지**
- [ ] **알림 설정**

---

## 모니터링 및 관찰성

### Prometheus Metrics

**우선순위**: Medium (Phase 6-7)

- [ ] **학습 메트릭 노출**
  ```python
  from prometheus_client import Counter, Gauge

  training_loss = Gauge('training_loss', 'Current training loss')
  training_accuracy = Gauge('training_accuracy', 'Current training accuracy')
  ```

- [ ] **Prometheus 서버 설정**
- [ ] **Metrics endpoint** (`/metrics`)

---

### Grafana 대시보드

**우선순위**: Medium (Phase 6-7)

- [ ] **Grafana 설치 및 설정**
- [ ] **대시보드 구축**
  - [ ] 학습 진행 상황
  - [ ] GPU 사용률
  - [ ] 메모리 사용량
  - [ ] MLflow runs 통계

---

### Alert 설정

**우선순위**: Low

- [ ] **학습 실패 알림** (Slack, Email)
- [ ] **성능 저하 알림**
- [ ] **리소스 부족 알림**

---

## 인프라 확장

### Kubernetes 배포 (Phase 7)

**우선순위**: Low (Phase 7)

- [ ] **Helm Chart 작성**
- [ ] **MLflow server 배포**
- [ ] **PostgreSQL StatefulSet**
- [ ] **MinIO 배포**

---

### Airflow 통합 (Phase 8)

**우선순위**: Low (Phase 8)

- [ ] **DAG 작성** (일일 학습 파이프라인)
- [ ] **DockerOperator 사용**
- [ ] **실패 재시도 로직**

---

## 참고

이 TODO 목록은 지속적으로 업데이트됩니다.

**우선순위 정의**:
- **High**: Phase 3 직후 또는 프로덕션 배포 전 필수
- **Medium**: 성능 개선 또는 기능 확장 시 필요
- **Low**: 선택적, 시간 여유 시 수행

**관련 문서**:
- [Phase 3 상세 계획](docs/phase3_plan.md)
- [전체 계획](plan.md)
- [분산 학습 가이드](docs/distributed_training.md)

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-10-18
