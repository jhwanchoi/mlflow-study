# TODO - 추후 작업 목록

**최종 업데이트**: 2025-10-18

이 문서는 현재 프로젝트에서 완전히 구현되지 않았거나, 추후 클라우드 환경에서 수행해야 하는 작업들을 정리한 목록입니다.

---

## 목차

1. [Phase 3.3 DDP 테스트 (클라우드 필요)](#phase-33-ddp-테스트-클라우드-필요)
2. [모델 최적화](#모델-최적화)
3. [데이터 파이프라인 개선](#데이터-파이프라인-개선)
4. [MLflow 고급 기능](#mlflow-고급-기능)
5. [CI/CD 개선](#cicd-개선)
6. [모니터링 및 관찰성](#모니터링-및-관찰성)
7. [인프라 확장](#인프라-확장)

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
