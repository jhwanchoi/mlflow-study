# MLflow Vision Training System - 프로젝트 진행 현황 및 계획

**작성일**: 2025-10-18
**최종 업데이트**: 2025-10-24
**버전**: 3.2

---

## 🔄 프로젝트 방향 변경 (2025-10-21)

### 배경
초기 계획은 로컬 환경에서의 개발과 최적화(Phase 3-4)에 집중했으나, 프로젝트 목표가 **전사 확장 가능한 MLOps 플랫폼 구축**으로 재정의되었습니다.

### 새로운 요구사항
- **멀티 유저 지원**: MLOps 엔지니어 2명, ML 엔지니어 1명 (향후 확장)
- **중앙화된 MLflow 서버**: 실험 추적 및 모델 레지스트리 공유
- **분산 하이퍼파라미터 최적화**: 100+ trials 병렬 실행
- **확장 가능한 인프라**: 추가 서비스 통합 예정 (6개월 내)

### 아키텍처 결정
| 항목 | 선택 | 대안 | 결정 근거 |
|------|------|------|-----------|
| 컨테이너 오케스트레이션 | AWS EKS | AWS ECS Fargate | 마이그레이션 비용 $14.5k 절감, 팀 K8s 경험 보유 |
| 하이퍼파라미터 튜닝 | Ray Tune | Optuna | 분산 실행, GPU 자동 스케줄링, 100+ trials 지원 |
| 인프라 관리 | Terraform + Scripts | 수동 배포 | 휴먼 에러 최소화, 재현성 보장 |
| 개발 환경 | VSCode 중심 | SageMaker | ML 엔지니어 선호도, 비용 효율성 |

### 비용 분석 (12개월 기준)
```
시나리오 1: ECS → EKS 마이그레이션
  - ECS 운영 (6개월): $420
  - 마이그레이션 비용: $15,000 (인력 2주)
  - EKS 운영 (6개월): $1,140
  - 총계: $16,560

시나리오 2: EKS 직접 구축
  - EKS 운영 (12개월): $2,280
  - 총계: $2,280

절감액: $14,280
```

### 월 운영 비용 (EKS 기반)
- EKS Control Plane: $73
- Worker Nodes (t3.medium × 2): $60
- RDS PostgreSQL (db.t3.small): $30
- S3 + ALB: $27
- **기본 운영: ~$190/월**
- GPU 사용 (p3.2xlarge Spot, 20시간/월): ~$18-20

---

## ✅ 완료된 작업 (Phase 1-2)

### Phase 1: 로컬 환경 구축 ✅

#### 1.1 기본 인프라
- [x] Docker Compose로 MLflow 서버 구성
- [x] PostgreSQL (메타데이터 저장) - **동시성 지원**
- [x] MinIO (S3 호환 아티팩트 저장)
- [x] MLflow Tracking Server with boto3

#### 1.2 학습 코드 개발
- [x] PyTorch 기반 비전 모델 (MobileNetV3-Small, Large, ResNet18)
- [x] CIFAR-10/100, Fashion-MNIST 데이터셋 지원
- [x] MLflow 완전 통합 (실험 추적, 메트릭 로깅, 아티팩트 저장)
- [x] Pydantic 기반 타입 안전 설정 관리

#### 1.3 테스트 스위트
- [x] **52개 테스트** (유닛 + 통합 + E2E)
- [x] **56.61% 코드 커버리지** (목표 50% 초과)
- [x] MLflow 격리 테스트 환경 (프로덕션 오염 방지)
- [x] pytest 마커 기반 테스트 분류 (slow, integration, unit)

### Phase 2: Docker 표준화 ✅ (2025-10-18)

#### 2.1 Python 버전 의존성 해결
- [x] pyproject.toml: `python = ">=3.9,<3.14"` (유연한 버전 범위)
- [x] Dockerfile: Python 3.11 고정 (일관된 환경)
- [x] torch, torchvision, numpy 의존성 명시

#### 2.2 Docker 워크플로우 구축
- [x] Multi-stage Dockerfile (production + development)
- [x] Docker 기반 테스트 실행 환경
- [x] Makefile 명령어 추가:
  - `make test-docker`: Docker 컨테이너에서 전체 테스트
  - `make test-docker-fast`: 빠른 테스트만 실행
  - `make train-docker`: Docker 컨테이너에서 학습

#### 2.3 문서 정리
- [x] README.md 간결화 (Docker 워크플로우 강조)
- [x] QUICKSTART.md 제거 → README 통합
- [x] Docker vs 로컬 워크플로우 명확히 구분

### Phase 3: PostgreSQL 마이그레이션 ✅ (이전 완료)

- [x] SQLite → PostgreSQL 변경
- [x] 동시성 문제 해결 (MVCC 지원)
- [x] boto3 설치로 S3 아티팩트 저장 활성화
- [x] 프로덕션급 백엔드 스토어 구축

---

## 📋 현재 상태 요약

### 현재 아키텍처 (로컬 개발 환경)
```
┌─────────────────────────────────────────────────────────────┐
│            Training & Tuning (Docker or Local)               │
│  ┌──────────────────┐      ┌───────────────────────────┐   │
│  │ Single Training  │      │ Ray Tune (Hyperparameter) │   │
│  │ PyTorch Model    │      │ - ASHA Scheduler          │   │
│  │ CIFAR-10 Data    │      │ - HyperOpt Search         │   │
│  └────────┬─────────┘      │ - Batch-level logging     │   │
│           │                └──────────┬────────────────┘   │
│           │ MLflow Client             │ MLflow Client       │
└───────────┼───────────────────────────┼───────────────────┘
            │                           │
            └────────────┬──────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              MLflow Tracking Server (Docker)                 │
│              ✅ Python 3.11 + boto3 + psycopg2              │
└───────┬──────────────────────────────────────┬──────────────┘
        │                                      │
        ▼                                      ▼
┌───────────────────┐              ┌──────────────────────────┐
│   PostgreSQL      │              │   MinIO (S3-compatible)  │
│  (Metadata Store) │              │   (Artifact Store)       │
│   ✅ MVCC 지원    │              │   ✅ boto3 연결          │
└───────────────────┘              └──────────────────────────┘
                                                │
                      ┌─────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              BentoML Serving (Docker, port 3000)             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  VisionClassifier Service                            │  │
│  │  - Model Loader (Alias/Stage/Version/Run ID)        │  │
│  │  - REST API (predict, batch, info, health)          │  │
│  │  - Device mapping (MPS→CPU)                         │  │
│  └──────────────────┬────────────────────────────────────┘  │
│                     │ Model Registry Integration             │
└─────────────────────┼──────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  HTTP Client  │
              │  (REST API)   │
              └───────────────┘
```

### 목표 아키텍처 (Phase 5-7: EKS 기반 MLOps 플랫폼)
```
┌──────────────────────────────────────────────────────────────────┐
│                       Client Environments                         │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ VSCode        │  │ Jupyter Hub  │  │ CI/CD Pipelines      │  │
│  │ (Local Dev)   │  │ (Notebooks)  │  │ (GitHub Actions)     │  │
│  └───────┬───────┘  └──────┬───────┘  └──────┬───────────────┘  │
│          │ MLflow Client + Ray Client │      │                   │
└──────────┼──────────────────┼──────────────────┼──────────────────┘
           │                  │                  │
           └──────────────────┼──────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       AWS EKS Cluster                             │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  MLflow Tracking Server (HPA: 2-5 pods)                    │  │
│  │  - MLflow Authentication (READ/EDIT/MANAGE)                │  │
│  │  - Ingress (ALB) with SSL                                  │  │
│  └────────┬───────────────────────────────────┬────────────────┘  │
│           │                                   │                   │
│  ┌────────┴────────────┐         ┌───────────┴────────────────┐  │
│  │  Ray Cluster        │         │  Airflow (Future)          │  │
│  │  - Head Node        │         │  - Scheduler               │  │
│  │  - Workers (GPU)    │         │  - Workers                 │  │
│  │  - Auto-scaling     │         │  - DAG Execution           │  │
│  └─────────────────────┘         └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
           │                                   │
           ▼                                   ▼
┌─────────────────────────┐     ┌──────────────────────────────────┐
│  AWS RDS PostgreSQL     │     │  AWS S3 Bucket                   │
│  - db.t3.small          │     │  - Versioned artifacts           │
│  - Multi-AZ (HA)        │     │  - Lifecycle policies            │
│  - Encrypted            │     │  - IRSA for pod access           │
└─────────────────────────┘     └──────────────────────────────────┘
```

### 테스트 커버리지
| 모듈 | 커버리지 | 상태 |
|------|----------|------|
| `src/config/settings.py` | 94.12% | ✅ 우수 |
| `src/models/vision_model.py` | 100.00% | ✅ 완벽 |
| `src/data/dataset.py` | 51.79% | ⚠️ 양호 |
| `src/training/train.py` | 52.00% | ⚠️ 양호 |
| `src/training/evaluate.py` | 18.02% | ❌ 개선 필요 |
| **전체** | **56.61%** | ✅ 목표 초과 |

### 핵심 기능
- ✅ Docker 표준화 (Python 버전 무관)
- ✅ PostgreSQL 동시성 지원
- ✅ MLflow 완전 통합 (실험 추적, 메트릭, 아티팩트)
- ✅ **BentoML 모델 서빙** (MLflow Model Registry 통합)
- ✅ **Model Alias 기반 버전 관리** (champion/challenger)
- ✅ 자동화 테스트 (52개, 56.61% 커버리지)
- ✅ CI/CD 파이프라인 (GitHub Actions)
- ✅ 코드 품질 자동화 (Black, isort, flake8, mypy)
- ✅ 보안 스캔 (Trivy, Bandit)
- ✅ M2 GPU 지원 (MPS backend)
- ✅ 3개 모델 지원 (MobileNetV3-S/L, ResNet18)
- ✅ 3개 데이터셋 지원 (CIFAR-10/100, Fashion-MNIST)
- ✅ REST API 서빙 (4개 엔드포인트)

---

## 🎯 다음 단계 (Phase 5-7: EKS 기반 MLOps 플랫폼)

### Phase 4: CI/CD 파이프라인 구축 ✅ (2025-10-18 완료)

#### 4.1 GitHub Actions 워크플로우
- [x] 테스트 워크플로우 ([.github/workflows/test.yml](.github/workflows/test.yml))
  - Docker 기반 테스트 실행 (52개 테스트, 56.61% 커버리지)
  - 코드 품질 검사 (Black, isort, flake8, mypy)
  - 보안 스캔 (Trivy 파일시스템 스캔)
  - Codecov 통합 및 HTML 리포트 아티팩트
  - 디스크 공간 최적화 (~25-30GB 확보)
- [x] Docker 빌드 워크플로우 ([.github/workflows/docker.yml](.github/workflows/docker.yml))
  - **수동 실행으로 변경** (디스크 공간 및 리소스 절약)
  - Production/Development 이미지 빌드 (workflow_dispatch)
  - MLflow Server 이미지 빌드
  - GitHub Container Registry 푸시 가능
- [x] Release 워크플로우 ([.github/workflows/release.yml](.github/workflows/release.yml))
  - 태그 기반 릴리스 자동화 (v*.*.*)
  - Changelog 자동 생성
  - E2E 테스트 실행

#### 4.2 코드 품질 개선
- [x] Black 포맷팅 적용 (7개 파일)
- [x] isort import 정렬
- [x] flake8 린팅 이슈 해결
  - F401 (unused imports) 제거
  - E501 (line too long) 수정
  - F841 (unused variables) 처리
- [x] mypy 타입 체킹 이슈 해결
  - 외부 라이브러리 타입 ignore 추가
  - numpy array 타입 개선
  - 함수 반환 타입 명시
- [x] .gitignore 수정 (src/data/ 추적 가능하도록)

#### 4.3 로컬 개발 도구
- [x] Pre-commit hook 설정 ([.pre-commit-config.yaml](.pre-commit-config.yaml))
  - Black, isort, flake8, mypy
  - Bandit 보안 검사
  - Hadolint (Dockerfile 린팅)
- [x] Makefile 명령어 추가
  - `make pre-commit-install`
  - `make pre-commit-run`

#### 4.4 문서화
- [x] [CICD.md](CICD.md): 완전한 CI/CD 가이드 (200+ 줄)
- [x] README.md: CI/CD 섹션 추가
- [x] pyproject.toml: pytest markers, bandit 설정

#### 4.5 해결한 문제들
- [x] Black 포맷팅 CI 실패 → Docker 환경 사용
- [x] Security Scan 권한 에러 → `security-events: write` 추가
- [x] GitHub Actions 디스크 부족 → 25-30GB 확보
- [x] Docker 빌드 디스크 부족 → 자동 빌드 비활성화
- [x] flake8/mypy 에러 → 전체 코드 품질 개선

### Phase 4.5: BentoML 모델 서빙 통합 ✅ (2025-10-24 완료)

**목표**: MLflow Model Registry와 통합된 프로덕션 모델 서빙 시스템 구축

**상세 문서**: [docs/bentoml_serving_guide.md](docs/bentoml_serving_guide.md)

#### 4.5.1 BentoML 서비스 구현
- [x] **BentoML 1.2+ 최신 API 적용**
  - `@bentoml.service` decorator 기반 서비스 클래스
  - REST API 엔드포인트 구현
    - `POST /predict_image`: 단일 이미지 예측
    - `POST /predict_batch`: 배치 예측
    - `POST /get_model_info`: 모델 메타데이터 조회
    - `POST /health`: 서비스 상태 확인

- [x] **MLflow 모델 로더** (`src/serving/model_loader.py`)
  - 4가지 모델 로딩 방식 지원:
    1. **Run ID** (개발/디버깅용)
    2. **Model Alias** (권장 - "champion", "challenger")
    3. **Model Stage** (deprecated - "Production", "Staging")
    4. **Model Version** (특정 버전 번호)

- [x] **Cross-platform 호환성**
  - MPS device → CPU mapping (M2 Mac → Docker Linux)
  - `map_location="cpu"` 강제 적용으로 모델 이식성 보장

#### 4.5.2 MLflow Model Registry 고도화
- [x] **Stage 기반 관리** (Deprecated, MLflow 2.9.0+)
  - Version 1: Staging stage
  - Version 2: Production stage
  - Stage transition API 사용법 문서화

- [x] **Alias 기반 관리** (권장)
  - Version 1: "challenger" alias
  - Version 2: "champion" alias
  - MLflow 2.9.0+ 권장 방식 적용
  - Alias 설정 API 사용법 문서화

#### 4.5.3 Docker Compose 통합
- [x] **BentoML 서비스 추가** (port 3000)
  - 환경 변수: MODEL_RUN_ID, MODEL_NAME, MODEL_VERSION, MODEL_STAGE, MODEL_ALIAS
  - MLflow, MinIO와 네트워크 통합
  - Volume mounting (소스 코드 hot-reload)

#### 4.5.4 테스트 및 검증
- [x] **테스트 이미지 자동 생성**
  - `create_test_image.py`: CIFAR-10에서 10개 클래스 샘플 추출

- [x] **Makefile 자동화 명령어**
  - `make serve`: BentoML 서버 시작
  - `make serve-stop`: 서버 중지
  - `make serve-logs`: 로그 확인
  - `make serve-test`: Health check + Model info API 테스트
  - `make serve-test-predict`: 이미지 예측 테스트
  - `make serve-test-all`: 전체 테스트 실행

#### 4.5.5 MLflow Evaluation 개선
- [x] **AWS 자격증명 수정** (`src/training/evaluate.py`)
  - boto3 환경 변수 설정으로 S3 액세스 문제 해결

- [x] **Evaluation Table 로깅**
  - MLflow UI에서 per-class 메트릭 시각화
  - pandas DataFrame → `mlflow.log_table()`

#### 4.5.6 기술 성과
- ✅ **MLflow Stages Deprecation 대응**: MLflow 2.9.0+ Alias 시스템으로 마이그레이션
- ✅ **Pydantic Serialization 이슈 해결**: Protobuf RepeatedScalarContainer → list 변환
- ✅ **Device Compatibility**: MPS 학습 모델 → CPU Docker 환경 이식성 확보
- ✅ **Zero-downtime 모델 업데이트**: Alias 변경만으로 서빙 모델 전환 가능

#### 성공 기준
- ✅ BentoML 서버가 Docker Compose에서 안정적으로 실행
- ✅ MLflow Model Registry Alias를 통한 모델 로딩 성공
- ✅ Champion alias (Version 2) 모델 서빙 검증
- ✅ Test accuracy 51.27% 모델 REST API로 서빙
- ✅ 예측 정확도 검증 (test_cat.png → "cat", confidence 0.307)

### Phase 4.6: Ray Tune 하이퍼파라미터 최적화 통합 ✅ (2025-10-24 완료)

**목표**: MLflow 통합된 자동 하이퍼파라미터 튜닝 시스템 구축

#### 4.6.1 Ray Tune 핵심 기능 구현
- [x] **Trainable 함수** (`src/tuning/ray_tune.py`)
  - Training loop를 Ray Tune과 통합
  - MLflow 자동 로깅 (batch-level + epoch-level)
  - Hyperparameter 주입 (learning_rate, weight_decay, momentum)

- [x] **Search Space 설정**
  - `create_search_space()`: 검색 공간 정의
  - Log-uniform 분포: learning_rate (1e-4 ~ 1e-2)
  - Log-uniform 분포: weight_decay (1e-5 ~ 1e-3)
  - Uniform 분포: momentum (0.8 ~ 0.99)

- [x] **Scheduler & Search Algorithm**
  - ASHA Scheduler: Early stopping으로 비효율적인 trial 조기 종료
  - HyperOpt (optional): Bayesian optimization
  - Grid search 지원: 간단한 테스트용

#### 4.6.2 MLflow 통합 고도화
- [x] **Batch-level 세밀한 로깅**
  - 매 10 batch마다 metrics 로깅
  - `batch_train_loss`, `batch_train_accuracy`
  - `batch_val_loss`, `batch_val_accuracy`
  - Global step counter로 연속적인 학습 곡선 생성

- [x] **Epoch-level 요약 로깅**
  - `epoch_train_loss`, `epoch_train_accuracy`
  - `epoch_val_loss`, `epoch_val_accuracy`
  - `learning_rate`, `epoch`

- [x] **Ray Tune 호환 metrics**
  - `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
  - `training_iteration` (Ray Tune 표준)
  - MLflowLoggerCallback 대체로 full control 확보

- [x] **Hyperparameter & Tags 자동 로깅**
  - MLflow params: learning_rate, weight_decay, momentum, epochs, batch_size
  - Tags: framework="ray-tune", task="hyperparameter-tuning", trial_id

- [x] **Best Trial 자동 추적**
  - 최적 trial config 및 metrics MLflow에 별도 run으로 저장
  - Tag: best_trial=True

#### 4.6.3 Makefile 자동화
- [x] `make tune`: 기본 튜닝 (10 trials)
- [x] `make tune-quick`: 빠른 테스트 (5 trials)
- [x] `make tune-extensive`: 대규모 튜닝 (50 trials)
- [x] `make tune-results`: 결과 요약 확인
- [x] `make tune-clean`: Ray Tune 결과 정리

#### 4.6.4 테스트 및 검증
- [x] **Integration Test** (`test_ray_tune.py`)
  - 2 trials × 3 epochs
  - Grid search: learning_rate [0.001, 0.005]
  - Batch-level 로깅 검증
  - MLflow UI 시각화 확인

#### 4.6.5 기술 성과
- ✅ **MLflowLoggerCallback 제거**: 중복 run 문제 해결
- ✅ **Batch-level 세밀한 추적**: 수백 개 step으로 부드러운 학습 곡선
- ✅ **Cross-platform**: MPS 학습 환경에서도 정상 동작
- ✅ **Modular Design**: 향후 Ray Train (분산 학습) 통합 준비 완료

#### 성공 기준
- ✅ Ray Tune이 로컬에서 안정적으로 실행
- ✅ 각 trial이 MLflow에 개별 run으로 생성 (중복 없음)
- ✅ Batch-level metrics가 MLflow UI에서 부드러운 곡선으로 시각화
- ✅ Epoch-level validation metrics 정상 로깅
- ✅ Best trial 자동 선택 및 로깅
- ✅ Makefile commands로 쉬운 실행 가능

### Phase 5: AWS EKS 인프라 구축 (4-5주, 우선순위: Critical)

**목표**: 중앙화된 MLflow 서버를 EKS에 배포하여 멀티 유저 환경 구축

**상세 문서**: [docs/eks_infrastructure.md](docs/eks_infrastructure.md) (작성 예정)

#### 5.1 AWS 기본 인프라 (1주)
- [ ] Terraform 프로젝트 구조 생성
  - [ ] `terraform/aws-eks/main.tf` - EKS 클러스터
  - [ ] `terraform/aws-eks/rds.tf` - PostgreSQL 데이터베이스
  - [ ] `terraform/aws-eks/s3.tf` - S3 버킷 + 라이프사이클 정책
  - [ ] `terraform/aws-eks/iam.tf` - IRSA (IAM Roles for Service Accounts)
- [ ] EKS 클러스터 배포
  - [ ] Kubernetes 1.28+
  - [ ] t3.medium × 2 worker nodes (CPU 워크로드)
  - [ ] Auto-scaling 그룹 설정
- [ ] RDS PostgreSQL 배포
  - [ ] db.t3.small (2 vCPU, 2GB RAM)
  - [ ] Multi-AZ 배포 (고가용성)
  - [ ] 암호화 활성화
- [ ] S3 버킷 생성
  - [ ] Versioning 활성화
  - [ ] Lifecycle policy (오래된 버전 정리)
  - [ ] IRSA 권한 설정

#### 5.2 MLflow 서버 배포 (1주)
- [ ] Helm Chart 작성
  - [ ] `charts/mlflow/values.yaml` - 설정 정의
  - [ ] `charts/mlflow/templates/deployment.yaml` - MLflow 서버
  - [ ] `charts/mlflow/templates/service.yaml` - ClusterIP 서비스
  - [ ] `charts/mlflow/templates/ingress.yaml` - ALB Ingress
  - [ ] `charts/mlflow/templates/hpa.yaml` - Horizontal Pod Autoscaler (2-5 pods)
- [ ] MLflow Authentication 설정
  - [ ] 기본 인증 (사용자명/비밀번호)
  - [ ] 권한 관리 (READ/EDIT/MANAGE)
  - [ ] Kubernetes Secret으로 자격증명 관리
- [ ] SSL/TLS 설정
  - [ ] AWS Certificate Manager 인증서 생성
  - [ ] ALB Ingress에 HTTPS 적용
- [ ] 배포 및 검증
  - [ ] `helm install mlflow ./charts/mlflow`
  - [ ] Health check 확인
  - [ ] 로그 확인 (`kubectl logs`)

#### 5.3 클라이언트 마이그레이션 (1일)
- [ ] VSCode 환경 설정 가이드 작성
  - [ ] `.env.remote` 템플릿 생성
  - [ ] 환경 변수 설정 (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`)
- [ ] 로컬 → 원격 전환 테스트
  - [ ] 기존 학습 코드로 실험 실행
  - [ ] MLflow UI에서 결과 확인
  - [ ] 아티팩트 S3 업로드 검증

#### 5.4 배포 자동화 스크립트 (2-3일)
- [ ] `scripts/setup/01-setup-aws.sh` - AWS 리소스 생성
- [ ] `scripts/setup/02-deploy-eks.sh` - EKS 클러스터 배포
- [ ] `scripts/setup/03-deploy-mlflow.sh` - MLflow 서버 배포
- [ ] `scripts/setup/04-setup-users.sh` - 사용자 계정 생성
- [ ] `scripts/setup/05-test-connection.sh` - 연결 테스트
- [ ] `scripts/setup/06-verify-all.sh` - 전체 검증
- [ ] `scripts/ops/backup-mlflow.sh` - 백업 스크립트
- [ ] `scripts/ops/restore-mlflow.sh` - 복원 스크립트

#### 성공 기준
- ✅ MLflow 서버가 EKS에서 안정적으로 실행
- ✅ RDS PostgreSQL 연결 성공
- ✅ S3 아티팩트 저장 성공
- ✅ MLflow Authentication 동작 (최소 3명 사용자)
- ✅ HTTPS 접속 가능
- ✅ 모든 배포 스크립트 정상 작동

### Phase 6: Ray Tune 하이퍼파라미터 최적화 (2-3주)

**목표**: 분산 하이퍼파라미터 튜닝으로 모델 정확도 90%+ 달성

**상세 문서**: [docs/ray_tune_guide.md](docs/ray_tune_guide.md) (작성 예정)

#### 6.1 Ray Cluster 배포 (1주)
- [ ] KubeRay Operator 설치
  - [ ] `kubectl apply -f kuberay-operator.yaml`
- [ ] Ray Cluster Helm Chart 작성
  - [ ] `charts/ray-cluster/values.yaml`
  - [ ] Head Node (1개, 고정)
  - [ ] Worker Nodes (GPU, 0-5개 auto-scaling)
- [ ] GPU 노드 그룹 추가
  - [ ] p3.2xlarge Spot Instances
  - [ ] Auto-scaling 설정 (최대 5개)
- [ ] Ray Dashboard 접근 설정
  - [ ] Ingress 또는 Port-forward

#### 6.2 Ray Tune 코드 작성 (1주)
- [ ] `src/tuning/ray_tune.py` - Ray Tune 통합
  - [ ] 탐색 공간 정의 (learning_rate, batch_size, epochs, etc.)
  - [ ] PyTorch Lightning Trainer 래퍼
  - [ ] MLflow 콜백 (모든 trial 자동 기록)
- [ ] `src/tuning/search_algorithms.py`
  - [ ] ASHA (Asynchronous Successive Halving)
  - [ ] Hyperband
  - [ ] Bayesian Optimization (Optuna backend)
- [ ] GPU 스케줄링 설정
  - [ ] `resources_per_trial={"gpu": 1}`
  - [ ] Fractional GPU (필요시)

#### 6.3 실험 실행 (3-5일)
- [ ] 100 trials 하이퍼파라미터 탐색
  - [ ] CIFAR-10 기준
  - [ ] 목표: Test Accuracy 90%+
- [ ] MLflow에 모든 trial 기록
- [ ] 최적 모델 Model Registry 등록

#### 성공 기준
- ✅ Ray Cluster EKS에서 안정 실행
- ✅ GPU auto-scaling 동작 (0 → 5 → 0)
- ✅ 100 trials 완료
- ✅ CIFAR-10 정확도 90%+ 달성
- ✅ MLflow에 모든 실험 기록

### Phase 7: DDP 분산 학습 + Airflow (2-3주)

**목표**: PyTorch DDP 멀티 GPU 학습 및 Airflow 파이프라인 자동화

**상세 문서**: [docs/distributed_training.md](docs/distributed_training.md) (작성 예정)

#### 7.1 PyTorch DDP 구현 (1주)
- [ ] `src/training/distributed.py` - DDP 유틸리티
  - [ ] `setup_distributed()` - 분산 초기화
  - [ ] `cleanup_distributed()` - 정리
- [ ] `src/training/train_ddp.py` - DDP 학습 스크립트
  - [ ] `torch.distributed.launch` 지원
  - [ ] Gradient accumulation
  - [ ] MLflow 로깅 (rank 0만)
- [ ] Docker 이미지 업데이트
  - [ ] NCCL 라이브러리 포함
  - [ ] SSH 설정 (멀티 노드 DDP)

#### 7.2 DDP 테스트 (3-5일)
- [ ] Ray Train으로 DDP 실행
  - [ ] Single-node multi-GPU (p3.8xlarge, 4 GPUs)
  - [ ] Multi-node multi-GPU (p3.2xlarge × 4, 4 GPUs)
- [ ] 성능 벤치마크
  - [ ] 학습 시간 비교 (단일 GPU vs 4 GPUs)
  - [ ] Scaling efficiency 측정

#### 7.3 Airflow 파이프라인 (1주)
- [ ] Airflow Helm Chart 배포
  - [ ] `charts/airflow/values.yaml`
  - [ ] KubernetesExecutor 설정
- [ ] DAG 작성
  - [ ] `dags/daily_training.py` - 일일 학습 파이프라인
  - [ ] `dags/hyperparameter_tuning.py` - 주간 튜닝
  - [ ] `dags/model_evaluation.py` - 모델 평가
- [ ] MLflow 통합
  - [ ] 학습 결과 자동 기록
  - [ ] 최고 성능 모델 자동 등록

#### 성공 기준
- ✅ DDP 멀티 GPU 학습 성공
- ✅ 4 GPUs로 학습 시간 3x 단축
- ✅ Airflow DAG 자동 실행 (일일 스케줄)
- ✅ MLflow에 모든 파이프라인 결과 기록

---

## 📊 성공 기준

### Phase 4: CI/CD (완료 ✅)
- [x] 전체 테스트 커버리지 56%+ 달성
- [x] GitHub Actions 자동화
- [x] 코드 품질 검사 (Black, isort, flake8, mypy)

### Phase 5: EKS 인프라
- [ ] MLflow 서버 EKS 배포 성공
- [ ] 멀티 유저 인증 동작 (3명+)
- [ ] HTTPS 접속 가능
- [ ] 월 운영 비용 $200 이하

### Phase 6: Ray Tune
- [ ] CIFAR-10 test accuracy 90%+
- [ ] 100 trials 완료
- [ ] GPU auto-scaling 동작
- [ ] MLflow에 모든 실험 자동 기록

### Phase 7: DDP + Airflow
- [ ] 멀티 GPU 학습 성공
- [ ] 학습 시간 3x 단축 (vs 단일 GPU)
- [ ] Airflow 일일 파이프라인 자동 실행
- [ ] 다중 실험 동시 실행 (5개+)

---

## 🔗 참고 자료

### 프로젝트 문서 (Phase 0-4: 로컬 개발)
- [README.md](README.md): 메인 문서
- [TESTING.md](TESTING.md): 테스트 가이드 (52개 테스트, 56% 커버리지)
- [CICD.md](CICD.md): CI/CD 파이프라인 가이드

### 프로젝트 문서 (Phase 5-7: EKS 확장, 작성 예정)
- [docs/eks_infrastructure.md](docs/eks_infrastructure.md): EKS 인프라 배포 가이드
- [docs/mlflow_remote_setup.md](docs/mlflow_remote_setup.md): MLflow 서버 설정 및 인증
- [docs/ray_tune_guide.md](docs/ray_tune_guide.md): Ray Tune 사용 가이드
- [docs/vscode_setup.md](docs/vscode_setup.md): VSCode 개발 환경 설정
- [docs/deployment_scripts.md](docs/deployment_scripts.md): 배포 스크립트 사용법
- [docs/cost_estimation.md](docs/cost_estimation.md): AWS 비용 상세 분석
- [docs/migration_guide.md](docs/migration_guide.md): 로컬 → EKS 마이그레이션

### 외부 문서
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Authentication](https://mlflow.org/docs/latest/auth/index.html)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [PyTorch DDP](https://pytorch.org/docs/stable/distributed.html)
- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

---

## 🎉 주요 성과

### Phase 0-4.5 완료 (로컬 개발 환경 + 모델 서빙)
1. **Docker 표준화** → Python 버전 의존성 완전 해결
2. **PostgreSQL 마이그레이션** → 동시성 문제 해결
3. **52개 자동화 테스트** → 코드 품질 보장 (56% 커버리지)
4. **CI/CD 파이프라인** → GitHub Actions 자동화
5. **BentoML 모델 서빙** → MLflow Model Registry 통합
6. **Model Alias 시스템** → MLflow 2.9.0+ 권장 방식 적용
7. **Cross-platform 호환성** → MPS → CPU 자동 변환

### Phase 5-7 계획 (EKS 기반 MLOps 플랫폼)
1. **아키텍처 재설계** → ECS 대신 EKS 직접 구축으로 $14.5k 절감
2. **멀티 유저 지원** → MLflow Authentication + HTTPS
3. **분산 최적화** → Ray Tune으로 100+ trials 병렬 실행
4. **자동화 스크립트** → Terraform + Bash로 휴먼 에러 최소화

---

## 🚀 다음 작업

### 즉시 시작 가능 (문서화 완료 후)
**Phase 5.1**: AWS 기본 인프라 (1주)
- Terraform 코드 작성 (EKS, RDS, S3)
- AWS 리소스 배포
- kubeconfig 설정

**예상 총 기간**: Phase 5-7 완료까지 8-11주

**월 운영 비용**: ~$190 (GPU 사용량 별도)

---

**최종 업데이트**: 2025-10-24
**버전**: 3.1
**상태**: Phase 4.5 완료, Phase 5 계획 중
