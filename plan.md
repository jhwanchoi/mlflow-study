# MLflow Vision Training System - 프로젝트 진행 현황 및 계획

**작성일**: 2025-10-18
**최종 업데이트**: 2025-10-21
**버전**: 3.0

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
│                 Training (Docker or Local)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PyTorch Model + CIFAR-10 Data                      │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │ MLflow Client                          │
└─────────────────────┼──────────────────────────────────────┘
                      │
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
- ✅ 자동화 테스트 (52개, 56.61% 커버리지)
- ✅ CI/CD 파이프라인 (GitHub Actions)
- ✅ 코드 품질 자동화 (Black, isort, flake8, mypy)
- ✅ 보안 스캔 (Trivy, Bandit)
- ✅ M2 GPU 지원 (MPS backend)
- ✅ 3개 모델 지원 (MobileNetV3-S/L, ResNet18)
- ✅ 3개 데이터셋 지원 (CIFAR-10/100, Fashion-MNIST)

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

### Phase 0-4 완료 (로컬 개발 환경)
1. **Docker 표준화** → Python 버전 의존성 완전 해결
2. **PostgreSQL 마이그레이션** → 동시성 문제 해결
3. **52개 자동화 테스트** → 코드 품질 보장 (56% 커버리지)
4. **CI/CD 파이프라인** → GitHub Actions 자동화

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

**최종 업데이트**: 2025-10-21
**버전**: 3.0
**상태**: Phase 5 문서화 진행 중
