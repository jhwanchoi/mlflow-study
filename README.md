# MLflow Vision Training System

프로덕션급 ML 파이프라인을 위한 MLflow 기반 비전 모델 학습 시스템입니다. 이 프로젝트는 로컬 개발 환경에서 시작하여 Kubernetes + Airflow 환경으로 확장 가능한 코어 엔진으로 설계되었습니다.

## 목차

- [주요 특징](#주요-특징)
- [아키텍처](#아키텍처)
- [요구사항](#요구사항)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [사용법](#사용법)
- [테스트](#테스트)
- [MLflow 추적](#mlflow-추적)
- [확장 가이드](#확장-가이드)
- [AI 엔지니어 협업 워크플로우](#ai-엔지니어-협업-워크플로우)

## 주요 특징

- **프로덕션급 아키텍처**: Pydantic 설정 관리, 구조화된 로깅, 타입 힌팅
- **MLflow 통합**: 완전한 실험 추적, 모델 버전 관리, 아티팩트 저장
- **확장 가능한 인프라**: Docker Compose (로컬) → Terraform → Kubernetes (프로덕션)
- **경량 비전 모델**: MobileNetV3-Small (M2 Mac 최적화)
- **공공 데이터셋**: CIFAR-10 (60,000 이미지, 10 클래스)
- **자동화된 평가**: Confusion matrix, per-class metrics 시각화
- **완전한 테스트 커버리지**: 52개 테스트, 56% 커버리지, MLflow 통합 테스트 포함

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Script                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PyTorch Model (MobileNetV3) + CIFAR-10 Data       │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │ MLflow Client                          │
└─────────────────────┼──────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MLflow Tracking Server                      │
│                   (Docker Container)                         │
└───────┬──────────────────────────────────────┬──────────────┘
        │                                      │
        ▼                                      ▼
┌───────────────────┐              ┌──────────────────────────┐
│   PostgreSQL      │              │   MinIO (S3-compatible)  │
│  (Metadata Store) │              │   (Artifact Store)       │
└───────────────────┘              └──────────────────────────┘
```

### 인프라 계층

1. **로컬 개발** (현재): Docker Compose
2. **IaC 관리** (선택): Terraform with Docker provider
3. **프로덕션** (추후 확장): Kubernetes + Helm + Terraform

## 요구사항

- **MacBook M2 Air** (또는 Apple Silicon Mac)
- **Python 3.9+** (3.13.2 권장)
- **Docker & Docker Compose**
- **Poetry** (의존성 관리)
- **Make** (편의 명령어)

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (또는 기존 디렉토리 사용)
cd mlflow-study

# Poetry 설치 (없는 경우)
curl -sSL https://install.python-poetry.org | python3 -

# 의존성 설치
make install

# 환경 변수 설정
make setup
# .env 파일을 확인하고 필요시 수정하세요
```

### 2. MLflow 인프라 시작

**방법 A: Docker Compose (권장)**

```bash
# 인프라 시작
make start

# 상태 확인
make status

# 로그 확인
make logs
```

**방법 B: Terraform (선택사항)**

```bash
# Terraform 초기화
make tf-init

# 인프라 배포
make tf-apply
```

### 3. MLflow UI 확인

```bash
# 브라우저에서 열기
make mlflow-ui

# 또는 직접 접속
open http://localhost:5000
```

### 4. 모델 학습

```bash
# 로컬에서 학습 (M2 GPU 활용)
make train

# 또는 Docker 컨테이너에서 학습
make train-docker
```

### 5. 모델 평가

```bash
# MLflow UI에서 Run ID 복사 후
make evaluate RUN_ID=<your_run_id>
```

## 프로젝트 구조

```
mlflow-study/
├── docker-compose.yml           # MLflow 인프라 오케스트레이션
├── Dockerfile                   # 학습 환경 컨테이너
├── Makefile                     # 편의 명령어
├── pyproject.toml               # Poetry 의존성 관리
├── .env.example                 # 환경 변수 템플릿
│
├── terraform/                   # 인프라 코드
│   └── local/
│       ├── main.tf              # Docker provider 설정
│       ├── outputs.tf           # Output 변수
│       └── README.md            # K8s 확장 가이드
│
├── src/
│   ├── config/
│   │   └── settings.py          # Pydantic 설정 관리
│   ├── data/
│   │   └── dataset.py           # CIFAR-10 데이터 로더
│   ├── models/
│   │   └── vision_model.py      # MobileNetV3 모델 래퍼
│   └── training/
│       ├── train.py             # MLflow 통합 학습 스크립트
│       └── evaluate.py          # 평가 및 메트릭 로깅
│
├── notebooks/                   # Jupyter 실험용
├── tests/                       # 테스트 코드 (52개 테스트)
│   ├── conftest.py              # Pytest fixtures 및 설정
│   ├── test_config.py           # 설정 관리 테스트 (13개)
│   ├── test_data.py             # 데이터 로딩 테스트 (8개)
│   ├── test_models.py           # 모델 아키텍처 테스트 (21개)
│   ├── test_training.py         # 학습 파이프라인 테스트 (7개)
│   └── test_e2e.py              # E2E 통합 테스트 (3개 + 5개 slow)
├── data/                        # 데이터셋 (자동 다운로드)
└── checkpoints/                 # 모델 체크포인트
```

## 사용법

### 기본 명령어

```bash
# 도움말 보기
make help

# 코드 포맷팅
make format

# 코드 린팅
make lint

# 테스트 실행
make test

# Jupyter 노트북 시작
make jupyter
```

## 테스트

이 프로젝트는 완전한 자동화 테스트 스위트를 포함하고 있습니다.

### 테스트 실행

```bash
# 전체 테스트 실행 (빠른 테스트만, ~8초)
make test

# 빠른 테스트만 (slow 마커 제외)
make test-fast

# 느린 테스트 포함 전체 실행
poetry run pytest tests/ -v

# 커버리지 리포트 생성
make test-coverage
```

### 테스트 구성

**총 52개 테스트, 56.17% 코드 커버리지**

| 테스트 파일 | 테스트 수 | 커버리지 | 설명 |
|-------------|-----------|----------|------|
| `test_config.py` | 13 | 94.12% | 설정 관리 및 유효성 검증 |
| `test_data.py` | 8 | 51.79% | 데이터 로딩 및 전처리 |
| `test_models.py` | 21 | 100% | 모델 아키텍처 및 동작 |
| `test_training.py` | 7 | 52% | 학습 파이프라인 |
| `test_e2e.py` | 3 + 5 slow | - | E2E 통합 테스트 |

### MLflow 통합 테스트

테스트는 **격리된 환경**에서 실행되어 프로덕션 MLflow 서버를 오염시키지 않습니다:

```python
# tests/conftest.py - 임시 MLflow URI 생성
@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory):
    tracking_dir = tmp_path_factory.mktemp("mlflow")
    return f"file://{tracking_dir}"  # 테스트 종료 시 자동 삭제
```

**검증 항목**:
- ✅ MLflow 실험 생성 및 추적
- ✅ 파라미터/메트릭 로깅
- ✅ 아티팩트 저장 및 조회
- ✅ Run ID 조회 및 검증

```bash
# MLflow 통합 테스트 실행 (느림, 격리됨)
poetry run pytest tests/test_e2e.py::TestMLflowIntegrationE2E -v
```

**중요**: 테스트는 임시 디렉토리(`/tmp/pytest-xxx/mlflow`)를 사용하며, 테스트 종료 시 모든 데이터가 자동 삭제됩니다. 프로덕션 MLflow 서버(`http://localhost:5001`)는 영향받지 않습니다.

더 자세한 내용은 [TESTING.md](TESTING.md)를 참고하세요.

### 고급 설정

#### 하이퍼파라미터 조정

`.env` 파일 수정:

```bash
# Model Configuration
MODEL_NAME=mobilenet_v3_small  # or mobilenet_v3_large, resnet18
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=20

# Data Configuration
DATASET=CIFAR10  # or CIFAR100, FashionMNIST
USE_AUGMENTATION=true
```

#### 다른 모델 사용

```python
# src/config/settings.py
model_name: Literal["mobilenet_v3_small", "mobilenet_v3_large", "resnet18"]
```

지원 모델:
- `mobilenet_v3_small`: 경량 (2.5M params, M2 최적화)
- `mobilenet_v3_large`: 중간 (5.5M params)
- `resnet18`: 표준 (11.7M params)

## MLflow 추적

### 자동 로깅 항목

**파라미터**:
- 모델 아키텍처 (model_name, num_classes)
- 학습 설정 (batch_size, learning_rate, epochs)
- 데이터 설정 (dataset, augmentation)
- 모델 파라미터 수 (total, trainable, frozen)

**메트릭**:
- Epoch 메트릭: train/val loss, accuracy, learning rate
- Batch 메트릭: 매 N 배치마다 로깅
- 최종 메트릭: best_val_accuracy, test_accuracy, test_f1

**아티팩트**:
- PyTorch 모델 (`.pth`)
- Confusion matrix (시각화)
- Per-class accuracy (시각화)
- 체크포인트 파일

### MLflow UI 탐색

```bash
# UI 열기
make mlflow-ui

# MinIO 콘솔 (아티팩트 확인)
make minio-ui
# Credentials: minio / minio123
```

## 확장 가이드

### Kubernetes + Airflow로 확장

이 시스템은 다음과 같이 확장 가능하도록 설계되었습니다:

#### Phase 1: 로컬 개발 (현재)
```yaml
# Docker Compose
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.2
    ...
```

#### Phase 2: Kubernetes 배포
```yaml
# Helm Chart (추후)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  template:
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.10.2
        env:
        - name: BACKEND_STORE_URI
          valueFrom:
            secretKeyRef:
              name: mlflow-secrets
              key: backend-uri
```

#### Phase 3: Airflow 통합
```python
# Airflow DAG (추후)
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG('vision_training_pipeline') as dag:
    train_task = DockerOperator(
        task_id='train_model',
        image='mlflow-vision-training:latest',
        environment={
            'MLFLOW_TRACKING_URI': '{{ var.value.mlflow_uri }}',
            'EXPERIMENT_NAME': 'production-training'
        }
    )
```

### Terraform 마이그레이션

```hcl
# terraform/kubernetes/main.tf (추후)
provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

resource "helm_release" "mlflow" {
  name       = "mlflow"
  repository = "https://charts.community.dev"
  chart      = "mlflow"

  values = [
    file("${path.module}/values.yaml")
  ]
}
```

## AI 엔지니어 협업 워크플로우

### 역할 분담

**AI 엔지니어/데이터 사이언티스트**:
1. `notebooks/`에서 탐색적 실험 수행
2. MLflow로 실험 자동 추적
3. 최적 하이퍼파라미터 선정
4. 요구사항 문서화

**ML 엔지니어 (당신)**:
1. `src/training/`으로 프로덕션 코드화
2. Docker 이미지 빌드 및 최적화
3. Terraform 인프라 관리
4. Airflow DAG 작성 (추후)
5. 모델 배포 파이프라인 구축

### 협업 시나리오

```bash
# 1. AI 엔지니어: 실험
jupyter notebook notebooks/experiment.ipynb
# → MLflow UI에서 실험 결과 공유

# 2. ML 엔지니어: 프로덕션화
make train  # 검증
make train-docker  # 컨테이너화
make evaluate RUN_ID=abc123  # 평가

# 3. 배포 (추후)
# Airflow DAG 트리거 → K8s Job 실행 → MLflow 모델 서빙
```

## 모니터링 및 디버깅

### 로그 확인

```bash
# 인프라 로그
make logs

# 특정 서비스 로그
docker-compose logs -f mlflow

# 학습 로그 (로컬)
tail -f logs/training.log
```

### 일반적인 문제 해결

**문제: MLflow 서버에 연결할 수 없음**
```bash
# 인프라 상태 확인
make status

# 재시작
make restart
```

**문제: M2 GPU 사용 안 됨**
```bash
# .env 파일 확인
DEVICE=mps  # M2 GPU용

# PyTorch MPS 지원 확인
python -c "import torch; print(torch.backends.mps.is_available())"
```

**문제: 메모리 부족**
```bash
# 배치 사이즈 감소
BATCH_SIZE=32  # .env에서 수정
```

## 정리

```bash
# 임시 파일 정리
make clean

# 인프라 중지
make stop

# 모든 데이터 삭제 (주의!)
make clean-all
```

## 다음 단계

1. **모델 최적화**: Quantization, pruning 적용
2. **CI/CD 파이프라인**: GitHub Actions 통합
3. **Kubernetes 배포**: Helm chart 작성
4. **Airflow DAG**: 학습 파이프라인 자동화
5. **모델 서빙**: MLflow Models + FastAPI
6. **모니터링**: Prometheus + Grafana

## 참고 자료

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Terraform Docker Provider](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.

---

**Version**: 0.1.0
**Last Updated**: 2025-01-16
**Maintainer**: Jihwan Choi
