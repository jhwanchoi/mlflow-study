# MLflow Vision Training System - 프로젝트 진행 현황 및 계획

**작성일**: 2025-10-18
**버전**: 2.0

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

### 시스템 아키텍처
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

## 🎯 다음 단계 (Phase 3, 5-6)

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

### Phase 3: 학습 파이프라인 고도화 (계획 수립 완료)

**상세 계획**: [docs/phase3_plan.md](docs/phase3_plan.md)
**추후 작업**: [TODO.md](TODO.md)

#### 개요
- 테스트 커버리지 개선 (56% → 75%+)
- 타입 안전성 강화 (mypy strict)
- MLflow Model Registry 통합
- Optuna 하이퍼파라미터 튜닝
- DDP 분산 학습 (코드 구조, 테스트는 추후)

#### 우선순위 및 예상 기간
1. **Phase 3.1**: 테스트 커버리지 개선 (1-2일)
2. **Phase 3.2**: 타입 안전성 강화 (1일)
3. **Phase 3.5**: MLflow Model Registry (2-3일)
4. **Phase 3.4**: Optuna 튜닝 (3-4일)
5. **Phase 3.3**: DDP 코드 구조 (2-3일)

**총 예상 기간**: 3-4주

#### 진행 상황
- [ ] **Phase 3.1**: 테스트 커버리지 개선
  - [ ] test_evaluate_extended.py 생성
  - [ ] test_dataset_extended.py 생성
  - [ ] test_training_extended.py 생성
  - [ ] 전체 커버리지 75% 달성
- [ ] **Phase 3.2**: 타입 안전성 강화
  - [ ] types.py 생성 (TypedDict, Protocol)
  - [ ] 모든 함수 타입 힌트 추가
  - [ ] mypy strict 통과
- [ ] **Phase 3.5**: MLflow Model Registry
  - [ ] registry.py 생성
  - [ ] train.py 통합 (자동 등록)
  - [ ] test_registry.py 생성
  - [ ] docs/model_registry.md 작성
- [ ] **Phase 3.4**: Optuna 하이퍼파라미터 튜닝
  - [ ] tuning.py 생성
  - [ ] Optuna-MLflow 통합
  - [ ] 50 trials 실행
  - [ ] 90%+ 정확도 달성
  - [ ] docs/hyperparameter_tuning.md 작성
- [ ] **Phase 3.3**: DDP 분산 학습
  - [ ] distributed.py 생성
  - [ ] train_distributed.py 생성
  - [ ] 로컬 CPU 테스트
  - [ ] docs/distributed_training.md 작성
  - [ ] TODO.md에 클라우드 테스트 항목 추가

#### 성공 기준
- ✅ 전체 커버리지 75%+
- ✅ mypy strict 모드 통과
- ✅ MLflow Model Registry 동작
- ✅ Optuna 50 trials 완료
- ✅ DDP 코드 구조 완성
- 🎯 CIFAR-10 정확도 90%+ (Optuna 튜닝 후)

#### 제약사항
- ⚠️ M2 Mac: DDP multi-GPU 테스트 불가
- ⚠️ DDP 실제 테스트는 클라우드 환경 필요 → [TODO.md](TODO.md) 참조

### Phase 5: 모델 개선 및 실험

#### 5.1 하이퍼파라미터 튜닝
- [ ] MLflow로 그리드 서치 실행
- [ ] 학습률 스케줄러 실험 (CosineAnnealing, ReduceLROnPlateau)
- [ ] 데이터 증강 전략 비교

#### 5.2 모델 최적화
- [ ] Quantization (INT8)
- [ ] Pruning (구조화/비구조화)
- [ ] Knowledge Distillation

#### 5.3 고급 기능
- [ ] Early stopping 개선 (patience, min_delta)
- [ ] Gradient clipping
- [ ] Mixed precision training (AMP)

### Phase 6: 프로덕션 준비

#### 6.1 CI/CD 파이프라인
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Docker tests
        run: make test-docker
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

#### 6.2 모델 서빙
- [ ] MLflow Models 서빙 (REST API)
- [ ] FastAPI 래퍼 작성
- [ ] 추론 최적화 (배치 처리)

#### 6.3 모니터링
- [ ] Prometheus metrics 노출
- [ ] Grafana 대시보드 구축
- [ ] 로그 집계 (ELK Stack)

---

## 🚀 Kubernetes + Airflow 확장 계획 (Phase 7-8)

### Phase 7: Kubernetes 배포

#### 7.1 Helm Chart 작성
```yaml
# helm/mlflow/values.yaml
mlflow:
  image: mlflow-server:v2.10.2-boto3
  replicas: 2
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

postgresql:
  enabled: true
  persistence:
    size: 10Gi

minio:
  enabled: true
  replicas: 4  # Distributed mode
  persistence:
    size: 50Gi
```

#### 7.2 Terraform IaC
```hcl
# terraform/kubernetes/main.tf
resource "helm_release" "mlflow" {
  name       = "mlflow"
  chart      = "../../helm/mlflow"
  namespace  = "ml-platform"

  values = [
    file("${path.module}/values-prod.yaml")
  ]
}
```

### Phase 8: Airflow 통합

#### 8.1 DAG 작성
```python
# airflow/dags/vision_training_pipeline.py
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG('vision_training_daily') as dag:
    train_task = DockerOperator(
        task_id='train_model',
        image='mlflow-vision-training:latest',
        environment={
            'MLFLOW_TRACKING_URI': '{{ var.value.mlflow_uri }}',
            'EXPERIMENT_NAME': 'production-training',
            'EPOCHS': '50',
        },
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
    )

    evaluate_task = DockerOperator(
        task_id='evaluate_model',
        image='mlflow-vision-training:latest',
        command='python -m src.training.evaluate {{ ti.xcom_pull("train_model")["run_id"] }}',
    )

    train_task >> evaluate_task
```

---

## 📊 성공 기준

### Phase 4 (코드 품질)
- [ ] 전체 테스트 커버리지 70% 이상
- [ ] mypy strict 모드 통과
- [ ] CI/CD 파이프라인 자동화

### Phase 5 (모델 개선)
- [ ] CIFAR-10 test accuracy > 90%
- [ ] 모델 크기 < 5MB (양자화 후)
- [ ] 추론 속도 < 10ms/image (M2 GPU)

### Phase 6 (프로덕션)
- [ ] MLflow 서빙 API 응답시간 < 50ms
- [ ] Prometheus 메트릭 수집 활성화
- [ ] 로그 검색 기능 구현

### Phase 7-8 (K8s + Airflow)
- [ ] Kubernetes 클러스터 배포 성공
- [ ] Airflow DAG 일일 자동 실행
- [ ] 다중 실험 동시 실행 (5개+)

---

## 🔗 참고 자료

### 프로젝트 문서
- [README.md](README.md): 메인 문서
- [TESTING.md](TESTING.md): 테스트 가이드

### 외부 문서
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Airflow Documentation](https://airflow.apache.org/docs/)

---

## 🎉 주요 성과

1. **Docker 표준화** → Python 버전 의존성 완전 해결
2. **PostgreSQL 마이그레이션** → 동시성 문제 해결
3. **52개 자동화 테스트** → 코드 품질 보장
4. **프로덕션급 아키텍처** → K8s 확장 준비 완료

---

**다음 작업**: Phase 3 시작 (학습 파이프라인 고도화)
