# Phase 3: 학습 파이프라인 고도화 상세 계획

**작성일**: 2025-10-18
**버전**: 1.0
**상태**: 계획 수립 완료, 실행 대기

---

## 목차

1. [개요 및 목표](#1-개요-및-목표)
2. [현재 환경 분석](#2-현재-환경-분석)
3. [아키텍처 설계](#3-아키텍처-설계)
4. [단계별 실행 계획](#4-단계별-실행-계획)
5. [설정 파일 변경](#5-설정-파일-변경)
6. [최종 파일 구조](#6-최종-파일-구조)
7. [검증 및 성공 기준](#7-검증-및-성공-기준)
8. [문서화 계획](#8-문서화-계획)
9. [진행 상황 추적](#9-진행-상황-추적)
10. [참고 자료](#10-참고-자료)

---

## 1. 개요 및 목표

### 1.1 Phase 3의 목적

Phase 4 (CI/CD 파이프라인)를 완료한 시점에서, Phase 3은 **학습 파이프라인의 품질과 기능을 고도화**하여 프로덕션급 ML 시스템으로 발전시키는 것을 목표로 합니다.

### 1.2 전체 목표

1. **품질 강화**: 테스트 커버리지 75%+, mypy strict 모드 통과
2. **MLflow 고급 기능**: Model Registry를 통한 모델 생명주기 관리
3. **학습 최적화**: Optuna를 통한 하이퍼파라미터 자동 튜닝
4. **확장성 준비**: DDP 분산 학습 코드 구조 완성

### 1.3 우선순위 및 일정

| 우선순위 | Phase | 작업 내용 | 예상 기간 |
|---------|-------|----------|----------|
| 1 | 3.1 | 테스트 커버리지 개선 | 1-2일 |
| 2 | 3.2 | 타입 안전성 강화 | 1일 |
| 3 | 3.5 | MLflow Model Registry | 2-3일 |
| 4 | 3.4 | Optuna 하이퍼파라미터 튜닝 | 3-4일 |
| 5 | 3.3 | DDP 분산 학습 (코드만) | 2-3일 |

**총 예상 기간**: 3-4주

### 1.4 성공 기준 요약

- ✅ 전체 테스트 커버리지 75% 이상
- ✅ mypy strict 모드 100% 통과
- ✅ MLflow Model Registry 통한 모델 자동 관리
- ✅ Optuna로 CIFAR-10 정확도 90%+ 달성
- ✅ DDP 코드 구조 완성 (테스트는 추후 클라우드에서)

---

## 2. 현재 환경 분석

### 2.1 강점

**인프라 및 아키텍처**:
- ✅ Pydantic 기반 타입 안전 설정 관리
- ✅ MLflow 완전 통합 (PostgreSQL + MinIO)
- ✅ Docker 표준화 완료 (Python 버전 무관)
- ✅ CI/CD 파이프라인 구축 완료 (GitHub Actions)
- ✅ 모듈화된 코드 구조 (`src/config`, `src/models`, `src/training`, `src/data`)

**테스트 및 품질**:
- ✅ 52개 자동화 테스트
- ✅ 56.61% 코드 커버리지 (목표 50% 초과)
- ✅ 코드 품질 자동화 (Black, isort, flake8, mypy)
- ✅ 보안 스캔 (Trivy, Bandit)

### 2.2 개선 필요 사항

**테스트 커버리지**:
- ⚠️ `src/training/evaluate.py`: 18.02% (개선 필요)
- ⚠️ `src/training/train.py`: 52.00% (양호)
- ⚠️ `src/data/dataset.py`: 51.79% (양호)

**타입 안전성**:
- ⚠️ 일부 함수에 타입 힌트 누락
- ⚠️ mypy strict 모드 비활성화 상태
- ⚠️ 복잡한 타입에 대한 명시적 정의 부족

**기능**:
- ❌ 모델 버전 관리 수동
- ❌ 하이퍼파라미터 수동 설정
- ❌ 단일 프로세스 학습만 지원

### 2.3 환경 제약사항

**하드웨어**:
- MacBook M2 (MPS backend, single GPU)
- MPS는 PyTorch DDP 미지원
- 로컬 CPU DDP는 학습 속도가 매우 느림

**개발 환경**:
- 로컬 개발 중심
- 클라우드 환경 미구축 (Kubernetes 확장은 Phase 7-8)

**DDP 제약**:
- M2 환경에서 실제 multi-GPU 테스트 불가
- DDP 코드 구조만 완성, 실제 테스트는 추후 클라우드에서 수행

---

## 3. 아키텍처 설계

### 3.1 전체 모듈 구조

```
src/
├── config/
│   ├── __init__.py
│   └── settings.py          # 수정: Optuna, DDP 설정 추가
├── data/
│   ├── __init__.py
│   └── dataset.py           # 기존
├── models/
│   ├── __init__.py
│   └── vision_model.py      # 기존
├── training/
│   ├── __init__.py
│   ├── types.py             # 신규: TypedDict, Protocol 정의
│   ├── train.py             # 수정: Registry, DDP 통합
│   ├── evaluate.py          # 기존
│   ├── registry.py          # 신규: MLflow Model Registry
│   ├── tuning.py            # 신규: Optuna 튜닝
│   ├── distributed.py       # 신규: DDP 유틸리티
│   └── train_distributed.py # 신규: DDP 진입점
└── utils/                   # 신규 (필요시)
    └── __init__.py

tests/
├── test_config.py
├── test_data.py
├── test_models.py
├── test_training.py
├── test_e2e.py
├── test_evaluate_extended.py   # 신규: Phase 3.1
├── test_dataset_extended.py    # 신규: Phase 3.1
├── test_training_extended.py   # 신규: Phase 3.1
├── test_registry.py            # 신규: Phase 3.5
├── test_tuning.py              # 신규: Phase 3.4
└── test_distributed.py         # 신규: Phase 3.3

docs/
├── phase3_plan.md              # 본 문서
├── model_registry.md           # 신규: Phase 3.5
├── hyperparameter_tuning.md    # 신규: Phase 3.4
└── distributed_training.md     # 신규: Phase 3.3
```

### 3.2 주요 컴포넌트 설계

#### 3.2.1 TypedDict 및 Protocol (Phase 3.2)

```python
# src/training/types.py
from typing import TypedDict, Protocol, Any
from torch import nn, optim

class TrainingMetrics(TypedDict):
    """Training metrics for one epoch."""
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float

class EvaluationMetrics(TypedDict):
    """Evaluation metrics from evaluate module."""
    accuracy: float
    confusion_matrix: Any  # numpy array
    per_class_metrics: dict[str, dict[str, float]]

class OptimizerProtocol(Protocol):
    """Protocol for PyTorch optimizers."""
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...

class LRSchedulerProtocol(Protocol):
    """Protocol for learning rate schedulers."""
    def step(self, metrics: float | None = None) -> None: ...
    def get_last_lr(self) -> list[float]: ...
```

#### 3.2.2 MLflow Model Registry (Phase 3.5)

```python
# src/training/registry.py
from typing import Optional, Literal
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

Stage = Literal["None", "Staging", "Production", "Archived"]

class ModelRegistry:
    """MLflow Model Registry manager for model lifecycle."""

    def __init__(self, model_name: str = "vision-classifier"):
        """Initialize Model Registry.

        Args:
            model_name: Registered model name in MLflow
        """
        self.client = MlflowClient()
        self.model_name = model_name

    def register_model(
        self,
        run_id: str,
        model_path: str = "model",
        description: Optional[str] = None
    ) -> ModelVersion:
        """Register model from MLflow run.

        Args:
            run_id: MLflow run ID
            model_path: Artifact path of the model
            description: Model description

        Returns:
            Registered ModelVersion
        """
        model_uri = f"runs:/{run_id}/{model_path}"

        # Register model
        mv = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name
        )

        # Update description if provided
        if description:
            self.client.update_model_version(
                name=self.model_name,
                version=mv.version,
                description=description
            )

        return mv

    def promote_model(
        self,
        version: int,
        stage: Stage,
        archive_existing: bool = True
    ) -> None:
        """Promote model to a stage.

        Args:
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Archive existing models in target stage
        """
        if archive_existing and stage in ["Staging", "Production"]:
            # Archive existing models in target stage
            existing = self.client.get_latest_versions(
                name=self.model_name,
                stages=[stage]
            )
            for mv in existing:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=mv.version,
                    stage="Archived"
                )

        # Promote new model
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )

    def get_latest_model(
        self,
        stage: Stage = "Production"
    ) -> Optional[ModelVersion]:
        """Get latest model version in a stage.

        Args:
            stage: Model stage

        Returns:
            Latest ModelVersion or None
        """
        versions = self.client.get_latest_versions(
            name=self.model_name,
            stages=[stage]
        )
        return versions[0] if versions else None

    def compare_models(
        self,
        versions: list[int],
        metric_name: str = "val_accuracy"
    ) -> dict[int, float]:
        """Compare models by metric.

        Args:
            versions: List of version numbers
            metric_name: Metric to compare

        Returns:
            Dictionary mapping version to metric value
        """
        results = {}
        for version in versions:
            mv = self.client.get_model_version(
                name=self.model_name,
                version=str(version)
            )
            run = self.client.get_run(mv.run_id)
            metric_value = run.data.metrics.get(metric_name, 0.0)
            results[version] = metric_value

        return results

    def rollback_to_version(
        self,
        version: int,
        stage: Stage = "Production"
    ) -> None:
        """Rollback to a specific model version.

        Args:
            version: Version to rollback to
            stage: Target stage
        """
        self.promote_model(version, stage, archive_existing=True)
```

#### 3.2.3 Optuna 하이퍼파라미터 튜닝 (Phase 3.4)

```python
# src/training/tuning.py
from typing import Optional
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow

from src.config import get_settings
from src.training.train import train_model

class OptunaObjective:
    """Optuna objective for hyperparameter tuning."""

    def __init__(self, n_epochs: int = 10):
        """Initialize objective.

        Args:
            n_epochs: Number of epochs per trial
        """
        self.n_epochs = n_epochs
        self.settings = get_settings()

    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna.

        Args:
            trial: Optuna trial

        Returns:
            Validation accuracy
        """
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "adamw"])

        # Update settings for this trial
        self.settings.learning_rate = lr
        self.settings.batch_size = batch_size
        self.settings.weight_decay = weight_decay
        self.settings.epochs = self.n_epochs

        # Train model
        try:
            # Create nested MLflow run for this trial
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                # Log trial params
                mlflow.log_params(trial.params)

                # Train (returns run_id, but we get metrics from current run)
                train_model()

                # Get validation accuracy from current run
                run = mlflow.active_run()
                if run:
                    client = mlflow.tracking.MlflowClient()
                    metrics = client.get_run(run.info.run_id).data.metrics
                    val_accuracy = metrics.get("val_accuracy", 0.0)
                else:
                    val_accuracy = 0.0

                return val_accuracy

        except Exception as e:
            # Log error and return 0
            mlflow.log_param("error", str(e))
            return 0.0

def run_hyperparameter_search(
    n_trials: int = 50,
    n_epochs_per_trial: int = 10,
    study_name: Optional[str] = None,
    storage: str = "sqlite:///optuna.db"
) -> optuna.Study:
    """Run Optuna hyperparameter search.

    Args:
        n_trials: Number of trials to run
        n_epochs_per_trial: Epochs per trial
        study_name: Optuna study name
        storage: Optuna storage URL

    Returns:
        Completed Optuna study
    """
    settings = get_settings()

    if study_name is None:
        study_name = f"{settings.experiment_name}-tuning"

    # Create MLflow callback
    mlflc = MLflowCallback(
        tracking_uri=settings.mlflow_tracking_uri,
        metric_name="val_accuracy"
    )

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization within MLflow parent run
    with mlflow.start_run(run_name=f"optuna-{study_name}"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("n_epochs_per_trial", n_epochs_per_trial)

        # Optimize
        study.optimize(
            OptunaObjective(n_epochs=n_epochs_per_trial),
            n_trials=n_trials,
            n_jobs=1,  # M2에서는 1 권장
            callbacks=[mlflc]
        )

        # Log best params
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_accuracy", study.best_value)

        # Save optimization history plot
        try:
            import matplotlib.pyplot as plt
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            mlflow.log_figure(fig, "optimization_history.png")
            plt.close(fig)

            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            mlflow.log_figure(fig, "param_importances.png")
            plt.close(fig)
        except Exception:
            pass  # Visualization errors are non-critical

    return study

if __name__ == "__main__":
    study = run_hyperparameter_search(n_trials=50)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")
```

#### 3.2.4 DDP 분산 학습 유틸리티 (Phase 3.3)

```python
# src/training/distributed.py
import os
import torch
import torch.distributed as dist
from typing import Optional

def setup_distributed(
    backend: Optional[str] = None,
    init_method: str = "env://"
) -> None:
    """Initialize distributed training.

    Args:
        backend: Backend to use (nccl, gloo, mpi). Auto-detect if None.
        init_method: Initialization method
    """
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")

    # Auto-detect backend if not specified
    if backend is None:
        if torch.cuda.is_available():
            backend = "nccl"  # Best for GPU
        elif torch.backends.mps.is_available():
            # MPS doesn't support DDP, use gloo on CPU
            backend = "gloo"
            print("⚠️  WARNING: MPS backend doesn't support DDP. Using CPU with gloo backend.")
        else:
            backend = "gloo"  # CPU fallback

    # Get rank and world size from environment variables
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )

    print(f"Initialized DDP: rank={rank}, world_size={world_size}, backend={backend}")

def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()

def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    return not is_distributed() or dist.get_rank() == 0

def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if is_distributed() else 1

def reduce_dict(input_dict: dict[str, float]) -> dict[str, float]:
    """Reduce dictionary values across all processes.

    Args:
        input_dict: Dictionary with float values

    Returns:
        Averaged dictionary
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys]

    # Convert to tensor
    values_tensor = torch.tensor(values, dtype=torch.float32)

    # All-reduce
    dist.all_reduce(values_tensor, op=dist.ReduceOp.SUM)
    values_tensor /= world_size

    # Convert back to dict
    return {k: v.item() for k, v in zip(keys, values_tensor)}
```

---

## 4. 단계별 실행 계획

### Phase 3.1: 테스트 커버리지 개선

**목표**: 전체 커버리지 56.61% → 75%+

**예상 소요**: 1-2일

#### 작업 1: `tests/test_evaluate_extended.py` 생성

**목적**: `src/training/evaluate.py` 커버리지 18% → 70%+

**테스트 케이스**:
```python
# tests/test_evaluate_extended.py
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.training.evaluate import evaluate_model

class TestEvaluateExtended:
    """Extended tests for evaluate module."""

    def test_confusion_matrix_generation(self, tmp_path):
        """Test confusion matrix is generated correctly."""
        # Mock model, dataloader, etc.
        # ...

    def test_per_class_metrics_calculation(self):
        """Test per-class precision/recall/f1 calculation."""
        # ...

    def test_visualization_saving(self, tmp_path):
        """Test that confusion matrix visualization is saved."""
        # ...

    def test_mlflow_metrics_logging(self):
        """Test that metrics are logged to MLflow."""
        with patch('mlflow.log_metric') as mock_log:
            # ...
            assert mock_log.called
```

#### 작업 2: `tests/test_dataset_extended.py` 생성

**목적**: `src/data/dataset.py` 커버리지 51% → 80%+

**테스트 케이스**:
```python
# tests/test_dataset_extended.py
class TestDatasetExtended:
    """Extended tests for dataset module."""

    def test_train_val_split_ratio(self):
        """Test train/val split ratio is correct."""
        # ...

    def test_dataloader_num_workers(self):
        """Test DataLoader with different num_workers."""
        # ...

    def test_augmentation_pipeline(self):
        """Test data augmentation is applied correctly."""
        # ...

    def test_multiple_datasets(self):
        """Test loading CIFAR-10, CIFAR-100, FashionMNIST."""
        # ...
```

#### 작업 3: `tests/test_training_extended.py` 생성

**목적**: `src/training/train.py` 커버리지 52% → 80%+

**테스트 케이스**:
```python
# tests/test_training_extended.py
class TestTrainingExtended:
    """Extended tests for training module."""

    def test_early_stopping_trigger(self):
        """Test early stopping is triggered correctly."""
        # ...

    def test_checkpoint_save_load(self, tmp_path):
        """Test model checkpoint save and load."""
        # ...

    def test_learning_rate_scheduler(self):
        """Test LR scheduler updates learning rate."""
        # ...
```

#### 검증 방법

```bash
# 확장 테스트 실행
poetry run pytest tests/test_*_extended.py -v

# 전체 커버리지 확인
poetry run pytest --cov=src --cov-report=term --cov-report=html

# 성공 기준
# - Overall coverage ≥ 75%
# - evaluate.py ≥ 70%
# - dataset.py ≥ 80%
# - train.py ≥ 80%
```

---

### Phase 3.2: 타입 안전성 강화

**목표**: mypy strict 모드 100% 통과

**예상 소요**: 1일

#### 작업 1: `src/training/types.py` 생성

위의 [3.2.1 TypedDict 및 Protocol](#321-typeddict-및-protocol-phase-32) 참조

#### 작업 2: 모든 함수에 타입 힌트 추가

**파일 목록**:
- `src/training/train.py`
- `src/training/evaluate.py`
- `src/data/dataset.py`
- `src/models/vision_model.py`
- `src/config/settings.py`

**예시**:
```python
# Before
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    ...

# After
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    ...
```

#### 작업 3: `pyproject.toml` mypy 설정

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "seaborn.*",
    "sklearn.*",
    "mlflow.*",
]
ignore_missing_imports = true
```

#### 작업 4: CI/CD 업데이트

`.github/workflows/test.yml`:
```yaml
- name: Run mypy (type checking)
  run: |
    poetry run mypy src/
  continue-on-error: false  # strict 모드, 실패 시 CI 중단
```

#### 검증 방법

```bash
# 로컬 mypy 검사
poetry run mypy src/ --strict

# 성공 기준
# - mypy: 0 errors
# - 모든 public 함수 타입 힌트 존재
# - CI/CD mypy strict 통과
```

---

### Phase 3.5: MLflow Model Registry

**목표**: 모델 생명주기 자동 관리

**예상 소요**: 2-3일

#### 작업 내용

1. **`src/training/registry.py` 생성**
   위의 [3.2.2 MLflow Model Registry](#322-mlflow-model-registry-phase-35) 참조

2. **`src/training/train.py` 수정**
   학습 완료 후 자동 모델 등록:
   ```python
   from src.training.registry import ModelRegistry

   def train_model() -> str:
       # ... 학습 코드 ...

       # 학습 완료 후
       run_id = mlflow.active_run().info.run_id

       # 모델 등록
       registry = ModelRegistry(model_name="vision-classifier")
       mv = registry.register_model(
           run_id=run_id,
           description=f"Model trained on {settings.dataset}"
       )

       # 최고 성능이면 Staging으로 승격
       if best_val_acc > 0.85:  # 임계값
           registry.promote_model(
               version=int(mv.version),
               stage="Staging"
           )

       return run_id
   ```

3. **`tests/test_registry.py` 생성**
   ```python
   class TestModelRegistry:
       def test_register_model(self):
           """Test model registration."""
           # ...

       def test_promote_to_staging(self):
           """Test promoting model to Staging."""
           # ...

       def test_get_latest_production_model(self):
           """Test retrieving latest production model."""
           # ...
   ```

4. **`docs/model_registry.md` 문서 생성**

#### MLflow UI 활용

```bash
# MLflow 서버 시작
make up

# 브라우저에서 확인
open http://localhost:5001

# Models 탭:
# - 등록된 모델 목록
# - 버전별 메트릭 비교
# - Stage 전환 이력
# - 모델 다운로드
```

#### 검증 방법

```bash
# 학습 실행
make train

# MLflow UI 확인
# - Models 탭에서 "vision-classifier" 확인
# - 최신 버전 확인
# - Stage 확인 (None 또는 Staging)

# 테스트
poetry run pytest tests/test_registry.py -v
```

---

### Phase 3.4: Optuna 하이퍼파라미터 튜닝

**목표**: 자동화된 하이퍼파라미터 최적화로 정확도 90%+ 달성

**예상 소요**: 3-4일

#### 작업 1: 의존성 추가

```bash
poetry add optuna optuna-integration[mlflow]
```

#### 작업 2: `src/training/tuning.py` 생성

위의 [3.2.3 Optuna 하이퍼파라미터 튜닝](#323-optuna-하이퍼파라미터-튜닝-phase-34) 참조

#### 작업 3: `src/config/settings.py` 수정

```python
class Settings(BaseSettings):
    # ... 기존 설정 ...

    # Optuna 설정
    optuna_n_trials: int = Field(
        default=50,
        description="Number of Optuna trials",
        gt=0,
    )
    optuna_n_epochs_per_trial: int = Field(
        default=10,
        description="Epochs per Optuna trial",
        gt=0,
    )
    optuna_study_name: str = Field(
        default="vision-tuning",
        description="Optuna study name",
    )
    optuna_storage: str = Field(
        default="sqlite:///optuna.db",
        description="Optuna storage URL",
    )
```

#### 작업 4: Makefile 명령어 추가

```makefile
.PHONY: tune
tune: ## Run Optuna hyperparameter tuning
	@echo "Starting Optuna hyperparameter tuning..."
	poetry run python -m src.training.tuning
```

#### 작업 5: `tests/test_tuning.py` 생성

```python
class TestOptunaTuning:
    def test_objective_function(self):
        """Test OptunaObjective callable."""
        # ...

    def test_mlflow_integration(self):
        """Test MLflow logging in trials."""
        # ...

    def test_best_params_extraction(self):
        """Test best params are logged."""
        # ...
```

#### 작업 6: `docs/hyperparameter_tuning.md` 문서 생성

#### 검증 방법

```bash
# Optuna 튜닝 실행
make tune

# MLflow UI에서 확인
# - "optuna-vision-tuning" parent run
# - 50개 nested runs (각 trial)
# - 최적 파라미터 확인
# - Optimization history 시각화 확인

# 성공 기준
# - 50 trials 완료
# - 최적 파라미터 발견
# - CIFAR-10 val_accuracy 90%+ (목표)
```

---

### Phase 3.3: DDP 분산 학습

**목표**: DDP 코드 구조 완성, 테스트는 추후 클라우드에서

**예상 소요**: 2-3일

**⚠️ 중요**: M2 Mac 제약으로 실제 multi-GPU 테스트는 [TODO.md](../TODO.md)에 명시

#### 작업 1: `src/training/distributed.py` 생성

위의 [3.2.4 DDP 분산 학습 유틸리티](#324-ddp-분산-학습-유틸리티-phase-33) 참조

#### 작업 2: `src/training/train.py` 수정

```python
from src.training.distributed import (
    is_distributed,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
)

def train_model() -> str:
    settings = get_settings()

    # Setup distributed if enabled
    if settings.distributed:
        setup_distributed()

    try:
        # ... 기존 학습 코드 ...

        # MLflow 로깅은 main process만
        if is_main_process():
            mlflow.log_metric("loss", loss)
            mlflow.log_metric("accuracy", acc)

        # ... 학습 계속 ...

    finally:
        # Cleanup distributed
        if settings.distributed:
            cleanup_distributed()

    return run_id
```

#### 작업 3: `src/training/train_distributed.py` 생성

```python
# src/training/train_distributed.py
"""Distributed training entry point."""

from src.training.train import train_model
from src.training.distributed import setup_distributed, cleanup_distributed

def main():
    """Main function for distributed training."""
    setup_distributed()

    try:
        train_model()
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
```

#### 작업 4: `src/config/settings.py` 수정

```python
class Settings(BaseSettings):
    # ... 기존 설정 ...

    # DDP 설정
    distributed: bool = Field(
        default=False,
        description="Enable distributed training",
    )
    backend: Literal["nccl", "gloo", "mpi"] = Field(
        default="nccl",
        description="Distributed backend",
    )
```

#### 작업 5: Makefile 명령어 추가

```makefile
.PHONY: train-ddp
train-ddp: ## Run distributed training (local CPU test only)
	@echo "⚠️  WARNING: M2 MPS doesn't support DDP"
	@echo "Using CPU gloo backend for basic testing only"
	@echo "For actual multi-GPU training, see TODO.md"
	torchrun --nproc_per_node=2 src/training/train_distributed.py
```

#### 작업 6: `tests/test_distributed.py` 생성

```python
class TestDistributed:
    """Basic distributed utility tests."""

    def test_setup_cleanup(self):
        """Test distributed setup and cleanup."""
        # Basic initialization test only
        # Actual DDP training test in TODO.md
        # ...
```

#### 작업 7: `docs/distributed_training.md` 문서 생성

**포함 내용**:
- DDP 코드 구조 설명
- 로컬 CPU 테스트 방법
- **TODO 섹션**: 클라우드 GPU 테스트 계획

#### 작업 8: `TODO.md` 업데이트

DDP 클라우드 테스트 항목 추가 (자세한 내용은 아래 [TODO.md](#todomd) 섹션 참조)

#### 검증 방법

```bash
# 로컬 CPU DDP 테스트 (기본 동작만)
make train-ddp

# 성공 기준
# - 2 processes 시작
# - 각 process가 rank 0, 1 할당
# - 기본 학습 동작
# - 에러 없이 종료

# ⚠️ 실제 multi-GPU 테스트는 TODO.md 참조
```

---

## 5. 설정 파일 변경

### 5.1 pyproject.toml

```toml
[tool.poetry.dependencies]
python = ">=3.9,<3.14"
# ... 기존 의존성 ...

# Phase 3.4: Optuna
optuna = "^3.5.0"
optuna-integration = {extras = ["mlflow"], version = "^3.5.0"}

[tool.poetry.group.dev.dependencies]
# ... 기존 dev 의존성 ...

# Phase 3.2: mypy strict 설정
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

# External libraries without type stubs
[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "seaborn.*",
    "sklearn.*",
    "mlflow.*",
    "optuna.*",
]
ignore_missing_imports = true

# Phase 3.1: 커버리지 임계값 상향
[tool.pytest.ini_options]
minversion = "7.0"
addopts = """
  --cov=src
  --cov-report=html
  --cov-report=term
  --cov-report=xml
  --cov-fail-under=75
  -v
"""
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

### 5.2 .github/workflows/test.yml

```yaml
# Phase 3.2: mypy strict 검사 활성화
- name: Run mypy (type checking)
  run: |
    poetry run mypy src/
  continue-on-error: false  # strict 모드이므로 실패 시 CI 중단
```

### 5.3 Makefile

```makefile
# ... 기존 명령어 ...

# Phase 3.1: 확장 테스트
.PHONY: test-extended
test-extended: ## Run extended tests with coverage
	poetry run pytest tests/test_*_extended.py -v --cov=src

# Phase 3.4: Optuna 튜닝
.PHONY: tune
tune: ## Run Optuna hyperparameter tuning
	@echo "Starting Optuna hyperparameter tuning..."
	@echo "This will run $(OPTUNA_N_TRIALS) trials (default: 50)"
	poetry run python -m src.training.tuning

# Phase 3.3: DDP (로컬 테스트용)
.PHONY: train-ddp
train-ddp: ## Run distributed training (local CPU test only)
	@echo "⚠️  WARNING: M2 MPS doesn't support DDP"
	@echo "Using CPU gloo backend for basic testing only"
	@echo "For actual multi-GPU training, see TODO.md"
	torchrun --nproc_per_node=2 src/training/train_distributed.py
```

---

## 6. 최종 파일 구조

### 6.1 전체 프로젝트 구조

```
mlflow-study/
├── .github/
│   └── workflows/
│       ├── test.yml (수정: mypy strict)
│       ├── docker.yml
│       └── release.yml
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py (수정: Optuna, DDP 설정)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── vision_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── types.py (신규: TypedDict, Protocol)
│   │   ├── train.py (수정: Registry, DDP 통합)
│   │   ├── evaluate.py
│   │   ├── registry.py (신규: Model Registry)
│   │   ├── tuning.py (신규: Optuna)
│   │   ├── distributed.py (신규: DDP 유틸)
│   │   └── train_distributed.py (신규: DDP 진입점)
│   └── utils/ (신규, 필요시)
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_e2e.py
│   ├── test_evaluate_extended.py (신규: 3.1)
│   ├── test_dataset_extended.py (신규: 3.1)
│   ├── test_training_extended.py (신규: 3.1)
│   ├── test_registry.py (신규: 3.5)
│   ├── test_tuning.py (신규: 3.4)
│   └── test_distributed.py (신규: 3.3)
├── docs/
│   ├── phase3_plan.md (신규: 본 문서)
│   ├── model_registry.md (신규: 3.5)
│   ├── hyperparameter_tuning.md (신규: 3.4)
│   └── distributed_training.md (신규: 3.3)
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.mlflow
├── Makefile (수정: tune, train-ddp 추가)
├── pyproject.toml (수정: Optuna, mypy strict)
├── pytest.ini
├── README.md (수정: 문서 링크 추가)
├── plan.md (수정: Phase 3 요약)
├── TODO.md (신규: DDP 테스트 등)
├── TESTING.md
└── CICD.md
```

### 6.2 새로 생성되는 파일 목록

**코드 파일 (8개)**:
1. `src/training/types.py`
2. `src/training/registry.py`
3. `src/training/tuning.py`
4. `src/training/distributed.py`
5. `src/training/train_distributed.py`
6. `tests/test_evaluate_extended.py`
7. `tests/test_dataset_extended.py`
8. `tests/test_training_extended.py`
9. `tests/test_registry.py`
10. `tests/test_tuning.py`
11. `tests/test_distributed.py`

**문서 파일 (5개)**:
1. `docs/phase3_plan.md` (본 문서)
2. `docs/model_registry.md`
3. `docs/hyperparameter_tuning.md`
4. `docs/distributed_training.md`
5. `TODO.md`

**수정되는 파일 (6개)**:
1. `src/training/train.py`
2. `src/config/settings.py`
3. `pyproject.toml`
4. `Makefile`
5. `.github/workflows/test.yml`
6. `README.md`
7. `plan.md`

---

## 7. 검증 및 성공 기준

### 7.1 Phase별 검증

#### Phase 3.1: 테스트 커버리지

**검증 명령어**:
```bash
make test
# 또는
poetry run pytest --cov=src --cov-report=term
```

**성공 기준**:
- ✅ Overall coverage: ≥ 75%
- ✅ `src/training/evaluate.py`: ≥ 70%
- ✅ `src/data/dataset.py`: ≥ 80%
- ✅ `src/training/train.py`: ≥ 80%
- ✅ CI/CD 통과

#### Phase 3.2: 타입 안전성

**검증 명령어**:
```bash
poetry run mypy src/ --strict
```

**성공 기준**:
- ✅ mypy: 0 errors
- ✅ 모든 public 함수 타입 힌트 존재
- ✅ CI/CD mypy strict 통과

#### Phase 3.5: Model Registry

**검증 명령어**:
```bash
# 학습 실행
make train

# MLflow UI 확인
open http://localhost:5001

# 테스트
poetry run pytest tests/test_registry.py -v
```

**성공 기준**:
- ✅ 학습 완료 후 모델 자동 등록
- ✅ MLflow UI Models 탭에서 모델 버전 확인 가능
- ✅ Stage 전환 (None → Staging → Production) 동작
- ✅ 모델 비교 및 롤백 기능 동작
- ✅ `tests/test_registry.py` 모두 통과

#### Phase 3.4: Optuna 튜닝

**검증 명령어**:
```bash
# Optuna 실행
make tune

# MLflow UI 확인
open http://localhost:5001
```

**성공 기준**:
- ✅ 50 trials 자동 실행 완료
- ✅ 각 trial이 MLflow nested run으로 기록
- ✅ 최적 파라미터 자동 로깅
- ✅ CIFAR-10 정확도 향상 (목표: 90%+)
- ✅ Optuna 시각화 생성 (optimization_history.png 등)
- ✅ `tests/test_tuning.py` 통과

#### Phase 3.3: DDP

**검증 명령어**:
```bash
# 로컬 CPU DDP 테스트
make train-ddp

# 기본 테스트
poetry run pytest tests/test_distributed.py -v
```

**성공 기준**:
- ✅ DDP 코드 구조 완성
- ✅ `setup_distributed()` / `cleanup_distributed()` 동작
- ✅ `is_main_process()` 올바르게 동작
- ✅ 로컬 CPU 2-process 기본 실행 성공
- ✅ `tests/test_distributed.py` 기본 테스트 통과
- ⏳ `TODO.md`에 클라우드 테스트 항목 추가
- ⏳ 실제 multi-GPU 테스트는 보류

### 7.2 전체 Phase 3 성공 기준

**필수 항목** (모두 충족 필요):
- ✅ 전체 테스트 커버리지 75% 이상
- ✅ mypy strict 모드 0 errors
- ✅ 52개 + 신규 테스트 모두 통과
- ✅ CI/CD 파이프라인 통과
- ✅ MLflow Model Registry 동작
- ✅ Optuna 50 trials 완료

**목표 항목** (최선의 노력):
- 🎯 CIFAR-10 정확도 90%+ (Optuna 튜닝 후)
- 🎯 evaluate.py 커버리지 70%+
- 🎯 dataset.py 커버리지 80%+
- 🎯 train.py 커버리지 80%+

**추후 항목** (TODO.md):
- ⏳ DDP multi-GPU 클라우드 테스트
- ⏳ 학습 속도 벤치마크
- ⏳ Gradient accumulation 검증

---

## 8. 문서화 계획

### 8.1 신규 생성 문서

#### `docs/model_registry.md`

**목적**: MLflow Model Registry 사용 가이드

**내용**:
- Model Registry 개요
- 모델 등록 방법
- Stage 관리 (None, Staging, Production, Archived)
- 모델 비교 및 선택
- 롤백 방법
- MLflow UI 사용법
- CLI 사용 예시

#### `docs/hyperparameter_tuning.md`

**목적**: Optuna 하이퍼파라미터 튜닝 가이드

**내용**:
- Optuna 개요
- MLflow 통합 방식
- 튜닝 실행 방법 (`make tune`)
- 튜닝 파라미터 커스터마이징
- 결과 분석 및 시각화
- 최적 파라미터 적용 방법
- 성능 개선 팁

#### `docs/distributed_training.md`

**목적**: DDP 분산 학습 가이드 (+ TODO)

**내용**:
- DDP 개요 및 장점
- 코드 구조 설명
- 로컬 테스트 방법 (CPU)
- **TODO 섹션**: 클라우드 GPU 테스트 계획
  - 필요 환경 (AWS p3.2xlarge 등)
  - 설정 방법
  - 벤치마크 계획
  - 예상 비용
- 문제 해결 가이드

### 8.2 업데이트 문서

#### `plan.md`

**추가 내용**:
- Phase 3 요약 섹션
- Phase 3.1 ~ 3.5 체크리스트
- `docs/phase3_plan.md` 링크

#### `README.md`

**추가 내용**:
- 문서 섹션에 Phase 3 계획 링크
- TODO.md 링크
- 새로운 기능 소개 (Model Registry, Optuna)

#### `TESTING.md`

**추가 내용** (필요시):
- 확장 테스트 설명
- 커버리지 75% 목표 명시

---

## 9. 진행 상황 추적

### 9.1 Phase 3.1: 테스트 커버리지 개선

- [ ] `tests/test_evaluate_extended.py` 생성
  - [ ] Confusion matrix 생성 테스트
  - [ ] Per-class metrics 계산 테스트
  - [ ] 시각화 저장 테스트
  - [ ] MLflow 메트릭 로깅 테스트
- [ ] `tests/test_dataset_extended.py` 생성
  - [ ] Train/Val split 비율 검증
  - [ ] DataLoader workers 테스트
  - [ ] Augmentation pipeline 테스트
  - [ ] 다중 데이터셋 로딩 테스트
- [ ] `tests/test_training_extended.py` 생성
  - [ ] Early stopping 동작 테스트
  - [ ] Checkpoint save/load 테스트
  - [ ] Learning rate scheduler 테스트
- [ ] 커버리지 75% 달성 확인
- [ ] CI/CD 통과 확인

### 9.2 Phase 3.2: 타입 안전성 강화

- [ ] `src/training/types.py` 생성
  - [ ] TrainingMetrics TypedDict
  - [ ] EvaluationMetrics TypedDict
  - [ ] OptimizerProtocol
  - [ ] LRSchedulerProtocol
- [ ] 모든 함수 타입 힌트 추가
  - [ ] `src/training/train.py`
  - [ ] `src/training/evaluate.py`
  - [ ] `src/data/dataset.py`
  - [ ] `src/models/vision_model.py`
- [ ] `pyproject.toml` mypy strict 설정
- [ ] `.github/workflows/test.yml` 업데이트
- [ ] mypy strict 통과 확인

### 9.3 Phase 3.5: MLflow Model Registry

- [ ] `src/training/registry.py` 생성
  - [ ] ModelRegistry 클래스
  - [ ] register_model 메서드
  - [ ] promote_model 메서드
  - [ ] get_latest_model 메서드
  - [ ] compare_models 메서드
  - [ ] rollback_to_version 메서드
- [ ] `src/training/train.py` 통합
  - [ ] 학습 후 자동 등록
  - [ ] 조건부 Staging 승격
- [ ] `tests/test_registry.py` 생성
  - [ ] 모델 등록 테스트
  - [ ] Stage 전환 테스트
  - [ ] 버전 관리 테스트
- [ ] `docs/model_registry.md` 작성
- [ ] MLflow UI에서 동작 확인

### 9.4 Phase 3.4: Optuna 하이퍼파라미터 튜닝

- [ ] 의존성 추가
  - [ ] `poetry add optuna optuna-integration`
- [ ] `src/training/tuning.py` 생성
  - [ ] OptunaObjective 클래스
  - [ ] run_hyperparameter_search 함수
  - [ ] MLflow 통합
  - [ ] 시각화 생성
- [ ] `src/config/settings.py` 수정
  - [ ] Optuna 설정 추가
- [ ] Makefile `tune` 명령어 추가
- [ ] `tests/test_tuning.py` 생성
- [ ] `docs/hyperparameter_tuning.md` 작성
- [ ] 50 trials 실행 및 결과 확인
- [ ] 90%+ 정확도 달성 확인

### 9.5 Phase 3.3: DDP 분산 학습

- [ ] `src/training/distributed.py` 생성
  - [ ] setup_distributed 함수
  - [ ] cleanup_distributed 함수
  - [ ] is_main_process 함수
  - [ ] get_rank, get_world_size 함수
  - [ ] reduce_dict 함수
- [ ] `src/training/train_distributed.py` 생성
- [ ] `src/training/train.py` 수정
  - [ ] DDP 통합
  - [ ] Main process 로깅
- [ ] `src/config/settings.py` 수정
  - [ ] DDP 설정 추가
- [ ] Makefile `train-ddp` 명령어 추가
- [ ] `tests/test_distributed.py` 생성 (기본 테스트만)
- [ ] `docs/distributed_training.md` 작성
- [ ] `TODO.md` 업데이트 (클라우드 테스트 항목)
- [ ] 로컬 CPU DDP 기본 동작 확인

### 9.6 문서화 및 마무리

- [ ] `docs/phase3_plan.md` 작성 (본 문서)
- [ ] `TODO.md` 작성
- [ ] `plan.md` 업데이트
- [ ] `README.md` 업데이트
- [ ] 전체 테스트 실행 및 통과 확인
- [ ] CI/CD 파이프라인 통과 확인
- [ ] Phase 3 완료 커밋 및 푸시

---

## 10. 참고 자료

### 10.1 공식 문서

- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna MLflow Integration](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [typing — Support for type hints](https://docs.python.org/3/library/typing.html)

### 10.2 내부 문서

- [README.md](../README.md): 프로젝트 개요
- [plan.md](../plan.md): 전체 Phase 계획
- [TESTING.md](../TESTING.md): 테스트 가이드
- [CICD.md](../CICD.md): CI/CD 파이프라인 가이드
- [TODO.md](../TODO.md): 추후 작업 목록

### 10.3 관련 이슈 및 참고 자료

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [DDP on Apple Silicon](https://github.com/pytorch/pytorch/issues/77764): M2에서 DDP 제약
- [Optuna Pruners](https://optuna.readthedocs.io/en/stable/reference/pruners.html): Early stopping for trials
- [MLflow Autologging](https://mlflow.org/docs/latest/tracking/autolog.html)

---

## 요약

Phase 3은 **학습 파이프라인의 품질과 기능을 한 단계 끌어올리는** 작업입니다.

**핵심 목표**:
1. ✅ 테스트 커버리지 75%+ (신뢰성 확보)
2. ✅ mypy strict 모드 (타입 안전성 강화)
3. ✅ MLflow Model Registry (모델 생명주기 자동화)
4. ✅ Optuna 튜닝 (90%+ 정확도 달성)
5. ✅ DDP 코드 구조 (확장성 준비)

**예상 기간**: 3-4주

**제약사항**: M2 Mac에서 DDP multi-GPU 테스트 불가 → 코드만 완성, 실제 테스트는 클라우드에서 (TODO.md)

**다음 단계**: Phase 3 완료 후 Phase 5 (모델 개선) 또는 Phase 7 (Kubernetes 배포)

---

**문서 버전**: 1.0
**최종 업데이트**: 2025-10-18
**작성자**: MLflow Vision Training System Team
