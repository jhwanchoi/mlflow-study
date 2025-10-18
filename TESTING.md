# Testing Guide

## 테스트 요약

### ✅ 테스트 결과
- **총 52개 테스트** - 모두 통과 ✅
- **코드 커버리지**: 56.17% (최소 50% 요구사항 충족)
- **실행 시간**: ~8초 (빠른 테스트만)
- **E2E 테스트**: 3개 빠른 테스트 + 5개 느린 테스트 (MLflow 통합 포함)

### 🔒 MLflow 테스트 격리

**중요**: 모든 테스트는 **격리된 환경**에서 실행되어 프로덕션 MLflow 서버를 오염시키지 않습니다.

```python
# tests/conftest.py - 임시 MLflow URI 자동 생성
@pytest.fixture(scope="session")
def mlflow_tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create temporary MLflow tracking URI."""
    tracking_dir = tmp_path_factory.mktemp("mlflow")
    return f"file://{tracking_dir}"  # 테스트 종료 시 자동 삭제
```

**격리 메커니즘**:
1. Pytest가 각 테스트 세션마다 임시 디렉토리 생성 (`/tmp/pytest-of-<user>/pytest-<number>/mlflow<number>/`)
2. MLflow 클라이언트가 임시 디렉토리를 추적 URI로 사용
3. 테스트 종료 시 Pytest가 자동으로 임시 디렉토리 삭제
4. 프로덕션 MLflow 서버(`http://localhost:5001`)는 영향 없음

**검증 방법**:
```bash
# 1. 프로덕션 MLflow 서버 확인
docker-compose ps  # mlflow-server가 실행 중이어야 함

# 2. MLflow 통합 테스트 실행
poetry run pytest tests/test_e2e.py::TestMLflowIntegrationE2E -v

# 3. 프로덕션 MLflow UI 확인
open http://localhost:5001  # 테스트 run이 나타나지 않음 ✅

# 4. 임시 디렉토리 확인 (테스트 종료 후)
ls -la /tmp/pytest-of-*/  # 디렉토리가 자동 삭제됨 ✅
```

### 📊 모듈별 커버리지

| 모듈 | 커버리지 | 설명 |
|------|----------|------|
| `src/config/settings.py` | **94.12%** | 설정 관리 (우수) |
| `src/models/vision_model.py` | **100.00%** | 모델 아키텍처 (완벽) |
| `src/data/dataset.py` | **51.79%** | 데이터 로딩 (양호) |
| `src/training/train.py` | **52.00%** | 학습 파이프라인 (양호) |
| `src/training/evaluate.py` | **18.02%** | 평가 모듈 (개선 필요) |

## 테스트 구조

```
tests/
├── conftest.py              # Pytest fixtures 및 설정
├── test_config.py           # 설정 관리 테스트 (13개)
├── test_data.py             # 데이터 로딩 테스트 (8개)
├── test_models.py           # 모델 아키텍처 테스트 (21개)
├── test_training.py         # 학습 파이프라인 테스트 (7개)
└── test_e2e.py              # 엔드투엔드 통합 테스트 (3개 빠른 + 5개 slow)
```

## 테스트 실행 방법

### 기본 실행

```bash
# 모든 테스트 실행
make test

# 또는 Poetry로 직접 실행
poetry run pytest tests/
```

### 빠른 테스트만 실행 (slow 제외)

```bash
poetry run pytest tests/ -m "not slow"
```

### 커버리지 리포트 포함

```bash
poetry run pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # 브라우저에서 커버리지 확인
```

### 특정 테스트 파일 실행

```bash
# Configuration 테스트만
poetry run pytest tests/test_config.py -v

# Model 테스트만
poetry run pytest tests/test_models.py -v

# 특정 테스트 클래스
poetry run pytest tests/test_models.py::TestVisionModel -v

# 특정 테스트 함수
poetry run pytest tests/test_models.py::TestVisionModel::test_forward_pass -v
```

### 상세 출력 옵션

```bash
# 자세한 출력
poetry run pytest tests/ -v

# 실패 시 자세한 traceback
poetry run pytest tests/ -v --tb=long

# Print문 출력 포함
poetry run pytest tests/ -v -s
```

## 테스트 카테고리

테스트는 pytest 마커로 분류됩니다:

### 마커 목록

- `@pytest.mark.slow`: 느린 테스트 (데이터 다운로드 포함)
- `@pytest.mark.integration`: 통합 테스트
- `@pytest.mark.unit`: 유닛 테스트
- `@pytest.mark.requires_gpu`: GPU 필요
- `@pytest.mark.requires_data`: 데이터셋 필요

### 마커 사용 예시

```bash
# 빠른 테스트만 실행
poetry run pytest -m "not slow"

# 통합 테스트만 실행
poetry run pytest -m "integration"

# GPU 테스트 제외
poetry run pytest -m "not requires_gpu"
```

## 테스트 세부 내용

### 1. Configuration 테스트 (test_config.py)

**13개 테스트**, 94% 커버리지

- ✅ 기본 설정 로드
- ✅ 환경 변수 기반 설정
- ✅ 유효성 검증 (모델명, 배치 크기, 학습률 등)
- ✅ 데이터셋별 이미지 크기 및 정규화 파라미터
- ✅ 설정 캐싱 동작

**예시**:
```python
def test_invalid_batch_size(self):
    """배치 크기가 0 이하면 ValidationError 발생"""
    monkeypatch.setenv("BATCH_SIZE", "0")
    with pytest.raises(ValidationError):
        get_settings()
```

### 2. Data 테스트 (test_data.py)

**8개 테스트**, 52% 커버리지

- ✅ 데이터 변환 (augmentation 포함/제외)
- ✅ 클래스 이름 확인 (CIFAR-10, CIFAR-100, Fashion-MNIST)
- ✅ 변환 출력 shape 검증
- ✅ 잘못된 데이터셋 이름 처리

**예시**:
```python
def test_cifar10_class_names(self):
    """CIFAR-10 클래스 이름이 10개이고 정확한지 확인"""
    classes = get_class_names("CIFAR10")
    assert len(classes) == 10
    assert "airplane" in classes
```

### 3. Model 테스트 (test_models.py)

**21개 테스트**, **100% 커버리지** ⭐

- ✅ 모델 초기화 (MobileNetV3-Small/Large, ResNet18)
- ✅ Forward pass 및 출력 shape
- ✅ 파라미터 카운팅
- ✅ Backbone freezing/unfreezing
- ✅ Gradient flow 검증
- ✅ 다양한 배치 크기 및 이미지 크기 지원

**예시**:
```python
def test_forward_pass(self, test_device, sample_image_batch):
    """Forward pass가 정상 작동하고 출력 shape이 올바른지 확인"""
    model = VisionModel("mobilenet_v3_small", num_classes=10)
    output = model(sample_image_batch)
    assert output.shape == (4, 10)  # (batch_size, num_classes)
```

### 4. Training 테스트 (test_training.py)

**7개 테스트**, 52% 커버리지

- ✅ Train epoch 기본 동작
- ✅ 학습 중 가중치 업데이트 확인
- ✅ Validation epoch (gradient 계산 없음)
- ✅ 체크포인트 저장/로드
- ✅ Early stopping 로직
- ✅ MLflow 로깅 호출 확인

**예시**:
```python
def test_train_epoch_updates_weights(self, dummy_model, dummy_dataloader):
    """학습 후 모델 가중치가 변경되는지 확인"""
    initial_weights = [p.clone() for p in dummy_model.parameters()]
    train_epoch(dummy_model, ...)
    # 가중치 변경 여부 확인
    assert weights_changed
```

### 5. MLflow 통합 테스트 (test_e2e.py)

**8개 E2E 테스트** (3개 빠른 + 5개 느린)

**✅ 빠른 테스트 (기본 실행)**:
1. `test_quick_training_iteration` - 단일 학습 iteration 검증
2. `test_evaluation_predictions` - 모델 예측 생성 및 검증
3. `test_augmentation_differences` - Train/Test transform 차이 확인

**🐌 느린 테스트 (--slow 마커)**:
4. `test_full_training_cycle` - 실제 CIFAR-10으로 2 epoch 학습
5. `test_model_saves_and_loads` - 모델 저장/로드 일관성
6. `test_full_evaluation_with_metrics` - 완전한 평가 파이프라인
7. **`test_mlflow_experiment_tracking`** - **MLflow 통합 테스트** ⭐
8. `test_data_loading_and_preprocessing` - 실제 데이터셋 로딩

**MLflow 통합 테스트 세부 내용** (`test_mlflow_experiment_tracking`):

```python
@pytest.mark.slow
def test_mlflow_experiment_tracking(
    self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test complete MLflow experiment tracking flow."""
    import mlflow

    # 임시 MLflow URI 설정 - 프로덕션 서버와 완전히 분리
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlflow")

    get_settings.cache_clear()
    settings = get_settings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    # MLflow run 시작 및 로깅
    with mlflow.start_run(run_name="test_run") as run:
        # 1. 파라미터 로깅
        mlflow.log_params({
            "model_name": "mobilenet_v3_small",
            "batch_size": 8,
            "learning_rate": 0.01,
        })

        # 2. 메트릭 로깅 (여러 epoch)
        for epoch in range(1, 3):
            mlflow.log_metrics({
                "train_loss": 2.0 / epoch,
                "train_accuracy": 50.0 * epoch,
                "val_loss": 2.5 / epoch,
                "val_accuracy": 45.0 * epoch,
            }, step=epoch)

        # 3. 모델 아티팩트 로깅
        model = create_model()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            mlflow.log_artifact(f.name, "model")

        run_id = run.info.run_id

    # 4. MLflow Client로 검증
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    run_data = client.get_run(run_id)

    # 파라미터 확인
    assert run_data.data.params["model_name"] == "mobilenet_v3_small"
    assert run_data.data.params["batch_size"] == "8"

    # 메트릭 확인
    assert "train_loss" in run_data.data.metrics
    assert "val_accuracy" in run_data.data.metrics

    # 아티팩트 확인
    artifacts = client.list_artifacts(run_id)
    assert len(artifacts) > 0  # 최소 1개 아티팩트 로깅됨
```

**검증 항목**:
- ✅ MLflow 실험 생성 (`mlflow.set_experiment`)
- ✅ Run 시작 및 종료 (`mlflow.start_run`)
- ✅ 파라미터 로깅 (`mlflow.log_params`)
- ✅ 메트릭 로깅 with step (`mlflow.log_metrics`)
- ✅ 아티팩트 로깅 (`mlflow.log_artifact`)
- ✅ MLflow Client를 통한 데이터 조회
- ✅ Run ID 및 메타데이터 검증

**실행 방법**:
```bash
# MLflow 통합 테스트만 실행
poetry run pytest tests/test_e2e.py::TestMLflowIntegrationE2E::test_mlflow_experiment_tracking -v

# 실행 결과: PASSED (테스트 데이터는 임시 디렉토리에만 저장되고 자동 삭제됨)
```

## CI/CD 통합

### GitHub Actions 예시

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Poetry
        run: curl -sSL https://install.python-poetry.org | python3 -
      - name: Install dependencies
        run: poetry install
      - name: Run tests
        run: poetry run pytest tests/ -v --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 테스트 작성 가이드

### 새 테스트 추가하기

1. **적절한 테스트 파일 선택**
   - Config 관련: `test_config.py`
   - Data 관련: `test_data.py`
   - Model 관련: `test_models.py`
   - Training 관련: `test_training.py`

2. **테스트 클래스 작성**
```python
class TestNewFeature:
    """Test new feature functionality."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = new_feature()
        assert result is not None
```

3. **Fixtures 활용**
```python
def test_with_fixture(self, test_device, sample_image_batch):
    """conftest.py의 fixtures 사용"""
    model = create_model()
    output = model(sample_image_batch.to(test_device))
    assert output.shape[0] == 4
```

4. **마커 추가** (필요시)
```python
@pytest.mark.slow
def test_full_training():
    """전체 학습 실행 (느림)"""
    pass
```

## 커버리지 개선 계획

현재 미달된 영역:

### 1. evaluate.py (18% → 목표 70%)
- [ ] 평가 함수 전체 실행 테스트
- [ ] Confusion matrix 생성 테스트
- [ ] Per-class metrics 계산 테스트
- [ ] 시각화 저장 테스트

### 2. train.py (52% → 목표 70%)
- [ ] 전체 학습 루프 실행 (integration test)
- [ ] MLflow 로깅 전체 플로우
- [ ] 체크포인트 저장/로드 통합 테스트

### 3. dataset.py (52% → 목표 70%)
- [ ] DataLoader 생성 완전 테스트
- [ ] Train/Val split 정확도 검증
- [ ] 다양한 데이터셋 로딩 테스트

## 문제 해결

### 일반적인 오류

**1. ModuleNotFoundError**
```bash
# 해결: 의존성 재설치
poetry install
```

**2. Coverage 낮음**
```bash
# 해결: coverage 임계값 조정 (pytest.ini)
--cov-fail-under=50  # 기본값에서 조정
```

**3. 느린 테스트**
```bash
# 해결: 빠른 테스트만 실행
poetry run pytest -m "not slow"
```

**4. GPU 관련 테스트 실패**
```bash
# 해결: GPU 테스트 제외
poetry run pytest -m "not requires_gpu"
```

## CI/CD 통합

### GitHub Actions 자동 테스트

모든 push와 PR에 대해 자동으로 테스트가 실행됩니다:

```yaml
# .github/workflows/test.yml
- Docker 기반 테스트 실행
- 코드 품질 검사 (Black, isort, flake8, mypy)
- 보안 스캔 (Trivy)
- Codecov 자동 업로드
```

**로컬에서 CI/CD 환경 재현**:
```bash
# Docker 환경에서 테스트 (CI/CD와 동일)
make test-docker

# Pre-commit hooks 설치 (코드 품질 자동 검사)
make pre-commit-install
```

### 테스트 실행 체크리스트

커밋 전 (로컬):
- [ ] `make test` 또는 `poetry run pytest` 실행
- [ ] 모든 테스트 통과 확인
- [ ] 커버리지 50% 이상 확인
- [ ] 새 기능에 대한 테스트 추가
- [ ] Pre-commit hooks 통과 (설치한 경우)

PR 전 (CI/CD 확인):
- [ ] GitHub Actions 워크플로우 통과 확인
- [ ] 전체 테스트 스위트 실행 확인
- [ ] 코드 품질 검사 통과 (Black, isort, flake8, mypy)
- [ ] 보안 스캔 통과
- [ ] Codecov 리포트 확인

## 추가 자료

- [Pytest 공식 문서](https://docs.pytest.org/)
- [Coverage.py 문서](https://coverage.readthedocs.io/)
- [pytest-cov 플러그인](https://pytest-cov.readthedocs.io/)

---

**Last Updated**: 2025-01-16
**Test Framework**: pytest 8.4.2
**Coverage Tool**: coverage 7.11.0

## E2E 테스트 상세

### 엔드투엔드 통합 테스트 (test_e2e.py)

실제 학습과 평가 기능이 제대로 동작하는지 검증하는 통합 테스트입니다.

#### ✅ 빠른 E2E 테스트 (기본 실행에 포함)

1. **test_quick_training_iteration**
   - 단일 학습 iteration 실행
   - 가중치 업데이트 및 gradient 계산 검증
   - 실행 시간: ~6초

2. **test_evaluation_predictions**
   - 모델 예측 생성 및 검증
   - 예측값 범위 확인 (0-9 for CIFAR-10)
   - 실행 시간: ~4초

3. **test_augmentation_differences**
   - Train/Test transform 차이 검증
   - Augmentation 동작 확인
   - 실행 시간: <1초

#### 🐌 느린 E2E 테스트 (--slow 마커로 별도 실행)

4. **test_full_training_cycle**
   - 실제 CIFAR-10 데이터로 2 에포크 학습
   - Train/Val loss 및 accuracy 계산
   - Loss 감소 검증
   - 실행 시간: ~5분

5. **test_model_saves_and_loads**
   - 모델 저장 및 로드 검증
   - 예측 일관성 확인
   - 실행 시간: ~30초

6. **test_full_evaluation_with_metrics**
   - 완전한 평가 파이프라인 실행
   - Confusion matrix 생성
   - Per-class accuracy 시각화
   - 실행 시간: ~1분

7. **test_mlflow_experiment_tracking**
   - MLflow 실험 추적 전체 플로우
   - 파라미터/메트릭/아티팩트 로깅
   - Run 생성 및 검증
   - 실행 시간: ~2분

8. **test_data_loading_and_preprocessing**
   - 실제 데이터셋 다운로드 및 로딩
   - 전처리 파이프라인 검증
   - 배치 shape 및 값 범위 확인
   - 실행 시간: ~3분 (최초 다운로드 시)

### E2E 테스트 실행 방법

```bash
# 빠른 E2E 테스트만 (기본)
poetry run pytest tests/test_e2e.py -v -m "not slow"

# 모든 E2E 테스트 (느린 테스트 포함)
poetry run pytest tests/test_e2e.py -v

# 특정 E2E 테스트만
poetry run pytest tests/test_e2e.py::TestEndToEndTraining::test_quick_training_iteration -v

# 전체 학습 사이클 테스트 (실제 데이터 사용)
poetry run pytest tests/test_e2e.py::TestEndToEndTraining::test_full_training_cycle -v -s
```

### E2E 테스트가 검증하는 것

✅ **학습 기능**
- 실제 데이터로 모델 학습 가능
- Loss가 감소하는지 확인
- Gradient 계산 및 가중치 업데이트
- 학습/검증 정확도 계산

✅ **평가 기능**
- 모델 예측 생성
- Confusion matrix 생성
- Per-class metrics 계산
- 시각화 저장

✅ **MLflow 통합**
- 실험 생성 및 추적
- 파라미터/메트릭 로깅
- 모델 아티팩트 저장
- Run 조회 및 검증

✅ **데이터 파이프라인**
- 데이터셋 다운로드 및 로딩
- Train/Val/Test split
- Augmentation 적용
- 배치 생성

### 실제 동작 검증 예시

```python
# test_e2e.py에서
def test_quick_training_iteration(self, test_device):
    """한 번의 학습 iteration이 실제로 동작하는지 검증"""
    
    # 1. 모델 생성
    model = create_model()
    
    # 2. 더미 데이터 생성
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    
    # 3. 학습 실행
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # 4. 검증: Gradient가 계산되었는지
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads  # ✅ 학습이 실제로 동작함
```

